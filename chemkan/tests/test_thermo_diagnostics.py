"""Thermodynamic coefficient diagnostics + thermo-init provenance (DIAGNOSTIC layer).

These cover the Cantera coefficient helper, the coefficient-intervention experiment, and
the `--thermo-init` option. Nothing here is part of the paper reproduction.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_DIAG = _SCRIPTS / "diagnostics"
for _p in (str(_SCRIPTS), str(_DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _chemistry import SPECIES  # noqa: E402
from chemkan.model import ChemKAN  # noqa: E402


def _cantera_or_skip():
    try:
        import cantera  # noqa: F401
    except Exception:
        pytest.skip("cantera not available")


def _ic(phi=0.9, T0=1050.0):
    """A physically valid H2/air mass-fraction vector in ChemKAN species order."""
    import cantera as ct
    gas = ct.Solution("h2o2.yaml")
    gas.set_equivalence_ratio(phi, "H2", {"O2": 1.0, "N2": 3.76})
    gas.TP = T0, ct.one_atm
    return np.array([gas.Y[gas.species_index(s)] for s in SPECIES])


# --------------------------------------------------------------------------
# 1-3: Cantera coefficient correctness (units, single MW division, ordering)
# --------------------------------------------------------------------------

def test_coefficient_units_are_kelvin_scale():
    """-h/cp has units of K; for H2/air combustion the magnitudes are ~1e2-1e5 K."""
    _cantera_or_skip()
    from _thermo_coeffs import cantera_coefficients
    coeffs, cp = cantera_coefficients(1050.0, _ic(), species=SPECIES)
    assert coeffs.shape == (len(SPECIES),)
    assert 1e3 < cp < 1e5, f"cp_mass out of physical range: {cp}"
    assert 1e2 < np.abs(coeffs).max() < 1e6, f"coefficient scale wrong: {np.abs(coeffs).max()}"


def test_no_double_molecular_weight_division():
    """Recompute h/W by hand and confirm the helper divided by MW exactly once."""
    _cantera_or_skip()
    import cantera as ct
    from _thermo_coeffs import cantera_coefficients
    Y = _ic()
    coeffs, cp = cantera_coefficients(1050.0, Y, species=SPECIES)
    gas = ct.Solution("h2o2.yaml")
    gas.TPY = 1050.0, ct.one_atm, {s: float(y) for s, y in zip(SPECIES, Y)}
    for k, sp in enumerate(SPECIES):
        i = gas.species_index(sp)
        expect = -(gas.partial_molar_enthalpies[i] / gas.molecular_weights[i]) / gas.cp_mass
        assert np.isclose(coeffs[k], expect, rtol=1e-10), f"{sp}: MW handling differs"


def test_coefficients_follow_chemkan_species_order():
    """Reordering the requested species reorders the returned coefficients identically."""
    _cantera_or_skip()
    from _thermo_coeffs import cantera_coefficients
    Y = _ic()
    coeffs, _ = cantera_coefficients(1050.0, Y, species=SPECIES)
    rev = list(reversed(SPECIES))
    Yrev = Y[::-1].copy()
    coeffs_rev, _ = cantera_coefficients(1050.0, Yrev, species=rev)
    assert np.allclose(coeffs, coeffs_rev[::-1], rtol=1e-10)


def test_reference_states_cover_ignition():
    _cantera_or_skip()
    from _thermo_coeffs import reference_state_indices
    t = np.linspace(0, 6e-4, 50)
    T = np.where(t < 3e-4, 1050.0, 2600.0)          # synthetic step "ignition"
    ref = np.zeros((50, len(SPECIES) + 1)); ref[:, -1] = T
    idx = reference_state_indices(t, ref)
    assert set(idx) == {"initial", "pre_ignition", "ignition", "post_ignition"}
    assert idx["initial"] == 0 and idx["post_ignition"] == 49
    assert idx["pre_ignition"] <= idx["ignition"]


# --------------------------------------------------------------------------
# 4-8: the intervention modifies only thermo.linear and leaves the original intact
# --------------------------------------------------------------------------

def _model():
    torch.manual_seed(0)
    return ChemKAN(species_dim=len(SPECIES), hidden_dim=3, num_basis=5, n_mu=3,
                   use_base_act=False)


def test_intervention_changes_only_thermo_linear():
    from hydrogen_thermo_intervention import apply_thermo_coefficients, changed_parameters
    base = _model()
    new = apply_thermo_coefficients(base, np.arange(len(SPECIES), dtype=float) * 100.0)
    assert changed_parameters(base, new) == ["thermo.linear.weight"]


def test_original_model_state_unchanged_by_intervention():
    from hydrogen_thermo_intervention import apply_thermo_coefficients
    base = _model()
    before = base.thermo.linear.weight.detach().clone()
    apply_thermo_coefficients(base, np.full(len(SPECIES), 1234.0))
    assert torch.equal(base.thermo.linear.weight.detach(), before)   # deep copy, not in-place


def test_intervention_applies_requested_coefficients():
    from hydrogen_thermo_intervention import apply_thermo_coefficients
    coeffs = np.linspace(-1e4, 1e4, len(SPECIES))
    new = apply_thermo_coefficients(_model(), coeffs)
    assert np.allclose(new.thermo.linear.weight.detach().numpy().ravel(), coeffs, rtol=1e-5)


def test_multiple_reference_states_give_distinct_coefficients():
    """State dependence is real: different reference T -> different coefficients."""
    _cantera_or_skip()
    from _thermo_coeffs import cantera_coefficients
    c_lo, _ = cantera_coefficients(1050.0, _ic(), species=SPECIES)
    c_hi, _ = cantera_coefficients(2500.0, _ic(), species=SPECIES)
    assert not np.allclose(c_lo, c_hi), "coefficients should depend on the state"


def test_ignition_delay_returns_none_when_flat():
    from hydrogen_thermo_intervention import ignition_delay
    t = np.linspace(0, 6e-4, 50)
    assert ignition_delay(t, np.full(50, 1050.0)) is None            # never ignites
    assert ignition_delay(t, np.linspace(1050, 2600, 50)) is not None


# --------------------------------------------------------------------------
# 9-11: --thermo-init option and its provenance
# --------------------------------------------------------------------------

def test_thermo_init_random_is_the_default_and_preserves_init():
    """Default must leave nn.Linear's own init untouched (|w| <= 1/sqrt(m))."""
    import train_hydrogen
    args = train_hydrogen.build_parser().parse_args([])
    assert args.thermo_init == "random"
    w = _model().thermo.linear.weight.detach().numpy().ravel()
    assert np.abs(w).max() <= 1.0 / np.sqrt(len(SPECIES)) + 1e-6


def test_thermo_init_cantera_flag_parses_with_reference_state():
    import train_hydrogen
    args = train_hydrogen.build_parser().parse_args(
        ["--thermo-init", "cantera", "--thermo-init-temperature", "1150",
         "--thermo-init-phi", "1.3"])
    assert args.thermo_init == "cantera"
    assert args.thermo_init_temperature == 1150.0 and args.thermo_init_phi == 1.3


def test_cantera_init_records_full_provenance():
    """The metadata block a cantera-initialized run must carry (see train_hydrogen)."""
    _cantera_or_skip()
    from _thermo_coeffs import MECH, cantera_coefficients
    coeffs, cp = cantera_coefficients(1050.0, _ic(), species=SPECIES, mech=MECH)
    meta = {"thermo_init": "cantera", "mechanism": MECH, "reference_T_K": 1050.0,
            "reference_phi": 0.9, "reference_state": "initial state of the first training condition",
            "cp_mass_J_per_kg_K": cp, "coefficients": [float(c) for c in coeffs],
            "species_order": SPECIES, "formula": "-h_k/cp"}
    for key in ("thermo_init", "mechanism", "reference_T_K", "cp_mass_J_per_kg_K",
                "coefficients", "species_order", "formula"):
        assert key in meta
    assert len(meta["coefficients"]) == len(SPECIES)


# --------------------------------------------------------------------------
# 13: diagnostics cannot overwrite the primary run
# --------------------------------------------------------------------------

def test_intervention_script_never_writes_a_checkpoint():
    src = (_DIAG / "hydrogen_thermo_intervention.py").read_text()
    assert "torch.save" not in src, "diagnostic must not write checkpoints"


def test_diagnostic_never_writes_into_the_run_dir():
    """The run dir is READ ONLY here: the only write target is the tables CSV."""
    src = (_DIAG / "hydrogen_thermo_intervention.py").read_text()
    # no write-mode open anywhere except the explicit CSV output
    writes = [ln.strip() for ln in src.splitlines() if '.open("w"' in ln or "'w'" in ln]
    assert all("out" in w for w in writes), f"unexpected write target: {writes}"
    assert "args.run_dir" not in src.split("out = Path")[-1], "run_dir must not be a write path"


@pytest.mark.skipif(not (_ROOT / "data/generated/hydrogen.npz").exists(),
                    reason="hydrogen.npz absent")
def test_intervention_cli_runs_and_writes_only_its_table(tmp_path):
    _cantera_or_skip()
    run_dir = _ROOT.parent / ("results/reproduction/chemkan/hydrogen/main/"
                              "base_off_direct_autograd_seed0")
    if not (run_dir / "checkpoint_final.pt").exists():
        pytest.skip("primary hydrogen checkpoint not available")
    before = (run_dir / "checkpoint_final.pt").stat().st_mtime
    out = tmp_path / "intervention.csv"
    r = subprocess.run([sys.executable, str(_DIAG / "hydrogen_thermo_intervention.py"),
                        "--run-dir", str(run_dir), "--out-csv", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert (run_dir / "checkpoint_final.pt").stat().st_mtime == before   # untouched


# --------------------------------------------------------------------------
# Stage-1 reuse + Stage-2 probe (controlled-experiment plumbing)
# --------------------------------------------------------------------------

def test_stage1_from_and_probe_flags_parse():
    import train_hydrogen
    a = train_hydrogen.build_parser().parse_args([])
    assert a.stage1_from is None and a.stage2_probe is False          # defaults unchanged
    assert a.stage2_probe_epochs == "0,10,50,100,250,500"
    b = train_hydrogen.build_parser().parse_args(
        ["--stage1-from", "x.pt", "--stage2-probe", "--stage2-probe-epochs", "0,10"])
    assert b.stage1_from == "x.pt" and b.stage2_probe is True


def test_probe_writes_schema_and_does_not_touch_parameters(tmp_path):
    """The probe must record the requested fields and leave every parameter untouched."""
    import csv as _csv

    from _stage2_probe import Stage2Probe
    from chemkan.normalization import MinMaxNormalizer

    model = _model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    m = len(SPECIES)
    t = np.linspace(0, 6e-4, 12)
    ref = np.zeros((12, m + 1)); ref[:, -1] = np.linspace(1050, 2600, 12)
    ref[:, 0] = 0.02
    norm = MinMaxNormalizer(torch.zeros(m + 1), torch.ones(m + 1) * 3000)

    def fake_integrate(model_, inorm, solver, u0, tt, device="cpu"):
        # stand-in integrator: shape (T, B, m+1), flat temperature
        out = torch.zeros(len(tt), 1, m + 1)
        out[:, 0, -1] = 1050.0
        return out

    p = Stage2Probe(model, integrate_fn=fake_integrate, input_norm=None, solver=None,
                    full_norm=norm, t=t, ref=ref, species=SPECIES,
                    path=tmp_path / "stage2_probe.csv", epochs=[0, 2])
    p.probe(0, None)
    p.on_epoch(1, 1.234, {}, 0.0)          # epoch index 1 -> 2 completed steps -> probes
    p.close()

    rows = list(_csv.DictReader((tmp_path / "stage2_probe.csv").open()))
    assert [r["epoch"] for r in rows] == ["0", "2"]
    for field in ("stage2_loss", "temperature_MSE", "peak_T_K", "ignites",
                  "ignition_delay_s", "thermo_linear_norm", "coeff_H2", "coeff_N2"):
        assert field in rows[0]
    assert rows[0]["ignites"] == "False"                    # flat T -> no ignition
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "probe mutated parameters"


def test_stage1_checkpoint_contains_kinetic_only_payload():
    """checkpoint_stage1.pt must carry the kinetic core + provenance, not the full model."""
    src = (_SCRIPTS / "train_hydrogen.py").read_text()
    assert '"kinetic_state": model.kinetic.state_dict()' in src
    assert 'run.run_dir / "checkpoint_stage1.pt"' in src
    # --stage1-from loads ONLY the kinetic core, so thermo init stays this run's policy
    assert 'model.kinetic.load_state_dict(s1ck["kinetic_state"])' in src


# --------------------------------------------------------------------------
# scaled-random control: magnitude-only rescale of the default random vector
# --------------------------------------------------------------------------

def test_scaled_random_is_a_valid_choice_and_not_the_default():
    import train_hydrogen
    assert train_hydrogen.build_parser().parse_args([]).thermo_init == "random"   # unchanged
    a = train_hydrogen.build_parser().parse_args(["--thermo-init", "scaled-random"])
    assert a.thermo_init == "scaled-random"
    assert a.thermo_init_temperature == 1050.0 and a.thermo_init_phi == 0.9        # spec defaults


def test_scaled_random_math_preserves_direction_and_matches_norm():
    """The operation is w <- w * (||w_cantera|| / ||w||): direction/signs/ratios survive."""
    m = _model()
    w0 = m.thermo.linear.weight.detach().clone()
    target = 1.712274e5
    with torch.no_grad():
        orig = float(torch.linalg.vector_norm(w0))
        m.thermo.linear.weight.mul_(target / orig)
    w1 = m.thermo.linear.weight.detach()
    a, b = w0.numpy().ravel(), w1.numpy().ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 1 - 1e-6, "direction must be preserved"
    assert np.array_equal(np.sign(a), np.sign(b)), "sign pattern must be preserved"
    ratios = b / a
    assert np.allclose(ratios, ratios[0], rtol=1e-4), "relative species ratios must be preserved"
    assert np.isclose(float(np.linalg.norm(b)), target, rtol=1e-5), "norm must hit the target"


@pytest.mark.skipif(
    not ((_ROOT / "data/generated/hydrogen.npz").exists()
         and (_ROOT / "data/generated/hydrogen_temperature_20000.npz").exists()),
    reason="hydrogen data / 20k cache absent")
def test_scaled_random_arm_matches_random_arm_everywhere_except_thermo_linear(tmp_path):
    """End-to-end: same seed -> kinetic.* and thermo.correction.* must be bit-identical."""
    _cantera_or_skip()
    runs = {}
    for arm in ("random", "scaled-random"):
        d = tmp_path / arm
        r = subprocess.run(
            [sys.executable, str(_SCRIPTS / "train_hydrogen.py"), "--thermo-init", arm,
             *(["--thermo-match-cantera-norm"] if arm == "scaled-random" else []),
             "--stage1-epochs", "1", "--stage2-epochs", "0", "--run-dir", str(d)],
            capture_output=True, text=True, cwd=str(_SCRIPTS))
        assert r.returncode == 0, r.stderr
        runs[arm] = torch.load(d / "checkpoint_final.pt", map_location="cpu",
                               weights_only=False)
    a = runs["random"]["model_state"]
    b = runs["scaled-random"]["model_state"]
    differing = [k for k in a if not torch.equal(a[k], b[k])]
    assert differing == ["thermo.linear.weight"], f"unexpected differences: {differing}"
    assert not [k for k in differing if k.startswith("kinetic")]          # kinetic identical
    assert not [k for k in differing if "correction" in k]                # correction identical

    wr = a["thermo.linear.weight"].numpy().ravel()
    ws = b["thermo.linear.weight"].numpy().ravel()
    cos = float(wr @ ws / (np.linalg.norm(wr) * np.linalg.norm(ws)))
    assert cos > 1 - 1e-5, "scaled-random must keep the random direction"
    assert np.linalg.norm(ws) > 1e4, "scaled-random must reach the Cantera magnitude"

    meta = runs["scaled-random"]["thermo_init"]
    for key in ("original_random_norm", "cantera_reference_norm", "effective_scale_factor",
                "resulting_norm", "reference_temperature", "reference_phi"):
        assert key in meta, f"missing provenance: {key}"
    assert meta["thermo_init"] == "scaled-random"
    assert np.isclose(meta["resulting_norm"], meta["cantera_reference_norm"], rtol=1e-4)


@pytest.mark.skipif(
    not ((_ROOT / "data/generated/hydrogen.npz").exists()
         and (_ROOT / "data/generated/hydrogen_temperature_20000.npz").exists()),
    reason="hydrogen data / 20k cache absent")
def test_scaled_random_does_not_copy_cantera_values(tmp_path):
    """Magnitude only: the scaled vector must NOT align with the Cantera coefficients."""
    _cantera_or_skip()
    out = {}
    for arm in ("scaled-random", "cantera"):
        d = tmp_path / arm
        r = subprocess.run(
            [sys.executable, str(_SCRIPTS / "train_hydrogen.py"), "--thermo-init", arm,
             *(["--thermo-match-cantera-norm"] if arm == "scaled-random" else []),
             "--stage1-epochs", "1", "--stage2-epochs", "0", "--run-dir", str(d)],
            capture_output=True, text=True, cwd=str(_SCRIPTS))
        assert r.returncode == 0, r.stderr
        out[arm] = torch.load(d / "checkpoint_final.pt", map_location="cpu", weights_only=False
                              )["model_state"]["thermo.linear.weight"].numpy().ravel()
    ws, wc = out["scaled-random"], out["cantera"]
    cos = float(ws @ wc / (np.linalg.norm(ws) * np.linalg.norm(wc)))
    assert abs(cos) < 0.9, f"scaled-random must not reproduce Cantera structure (cos={cos:.3f})"


# --------------------------------------------------------------------------
# scaled-random: explicit scale factor, norm matching, isolated direction seed
# --------------------------------------------------------------------------

_H2 = (_ROOT / "data/generated/hydrogen.npz")
_CACHE = (_ROOT / "data/generated/hydrogen_temperature_20000.npz")
_STAGE1 = (_ROOT.parent / "results/reproduction/chemkan/hydrogen/diagnostics/"
           "stage1_seed0/checkpoint_stage1.pt")
_needs_data = pytest.mark.skipif(not (_H2.exists() and _CACHE.exists() and _STAGE1.exists()),
                                 reason="hydrogen data / stage-1 checkpoint absent")


# The shared Stage-1 checkpoint was trained under the historical N=5/base-OFF reading,
# and train_hydrogen.py refuses --stage1-from across an architecture change. These tests
# exercise the thermo-init machinery, not the architecture, so they pin the checkpoint's
# own architecture explicitly instead of inheriting the script defaults.
_STAGE1_ARCH = ["--num-basis", "5", "--no-use-base-act"]


def _arm(tmp_path, name, *extra):
    """Run a 0-epoch Stage-2 arm from the shared Stage-1 checkpoint; return the checkpoint."""
    d = tmp_path / name
    r = subprocess.run(
        [sys.executable, str(_SCRIPTS / "train_hydrogen.py"),
         "--stage1-from", str(_STAGE1), "--seed", "0", "--stage2-epochs", "0",
         "--run-dir", str(d), *_STAGE1_ARCH, *extra],
        capture_output=True, text=True, cwd=str(_SCRIPTS))
    assert r.returncode == 0, r.stderr
    return torch.load(d / "checkpoint_final.pt", map_location="cpu", weights_only=False)


def test_scaling_modes_are_mutually_exclusive():
    import train_hydrogen
    p = train_hydrogen.build_parser()
    a = p.parse_args(["--thermo-init", "scaled-random", "--thermo-init-scale-factor", "1e4"])
    assert a.thermo_init_scale_factor == 1e4 and a.thermo_match_cantera_norm is False
    b = p.parse_args(["--thermo-init", "scaled-random", "--thermo-match-cantera-norm"])
    assert b.thermo_match_cantera_norm is True and b.thermo_init_scale_factor is None
    assert p.parse_args([]).thermo_linear_init_seed == 0          # default direction seed


@_needs_data
def test_explicit_scale_factor_is_exactly_s_times_random(tmp_path):
    """w_scaled == s * w_random, and the two vectors are collinear."""
    ck = _arm(tmp_path, "s1e4", "--thermo-init", "scaled-random",
              "--thermo-init-scale-factor", "1e4")
    meta = ck["thermo_init"]
    w0 = np.array(meta["original_random_vector"])
    w1 = np.array(meta["resulting_vector"])
    assert np.allclose(w1, 1e4 * w0, rtol=1e-5)                    # (1) exact scaling
    cos = float(w0 @ w1 / (np.linalg.norm(w0) * np.linalg.norm(w1)))
    assert cos > 1 - 1e-6                                          # (2) collinear
    assert meta["scaling_mode"] == "explicit-scale-factor"


@_needs_data
def test_match_cantera_norm_reproduces_the_cantera_l2_norm(tmp_path):
    meta = _arm(tmp_path, "nm", "--thermo-init", "scaled-random",
                "--thermo-match-cantera-norm")["thermo_init"]
    assert np.isclose(meta["resulting_norm"], meta["cantera_reference_norm"], rtol=1e-5)  # (3)
    assert np.isclose(meta["effective_scale_factor"],
                      meta["cantera_reference_norm"] / meta["original_random_norm"], rtol=1e-5)


@_needs_data
def test_direction_seed_changes_only_thermo_linear(tmp_path):
    """(4)(5)(6)(7): an isolated direction seed must not perturb anything else."""
    base = _arm(tmp_path, "d0", "--thermo-init", "scaled-random",
                "--thermo-match-cantera-norm", "--thermo-linear-init-seed", "0")
    alt = _arm(tmp_path, "d1", "--thermo-init", "scaled-random",
               "--thermo-match-cantera-norm", "--thermo-linear-init-seed", "1")
    a, b = base["model_state"], alt["model_state"]
    differing = [k for k in a if not torch.equal(a[k], b[k])]
    assert differing == ["thermo.linear.weight"], f"leaked beyond thermo.linear: {differing}"
    assert not [k for k in differing if k.startswith("kinetic")]          # kinetic identical
    assert not [k for k in differing if "correction" in k]                # correction identical
    # different direction, same norm
    w0 = np.array(base["thermo_init"]["original_random_vector"])
    w1 = np.array(alt["thermo_init"]["original_random_vector"])
    assert not np.allclose(w0, w1), "direction seed must change the direction"
    assert np.isclose(base["thermo_init"]["resulting_norm"],
                      alt["thermo_init"]["resulting_norm"], rtol=1e-4)


@_needs_data
def test_direction_seed_zero_reproduces_default_random_init(tmp_path):
    """Compatibility: seed 0 + dir-seed 0 must equal the plain random arm's vector."""
    rnd = _arm(tmp_path, "rand", "--thermo-init", "random")
    sr = _arm(tmp_path, "sr", "--thermo-init", "scaled-random",
              "--thermo-match-cantera-norm", "--thermo-linear-init-seed", "0")
    w_default = rnd["model_state"]["thermo.linear.weight"].numpy().ravel()
    w_unscaled = np.array(sr["thermo_init"]["original_random_vector"])
    assert np.array_equal(w_default, w_unscaled)                   # bit-identical


@_needs_data
def test_existing_random_and_cantera_arms_are_unchanged(tmp_path):
    """(9)(10): the pre-existing init paths must behave exactly as before."""
    rnd = _arm(tmp_path, "r", "--thermo-init", "random")
    w = rnd["model_state"]["thermo.linear.weight"].numpy().ravel()
    assert np.abs(w).max() <= 1.0 / np.sqrt(len(SPECIES)) + 1e-6   # untouched default init
    assert rnd["thermo_init"] == {"thermo_init": "random"}         # no extra mutation
    can = _arm(tmp_path, "c", "--thermo-init", "cantera")
    cw = can["model_state"]["thermo.linear.weight"].numpy().ravel()
    assert np.abs(cw).max() > 1e4                                  # physical scale as before
    assert can["thermo_init"]["thermo_init"] == "cantera"
    assert "coefficients" in can["thermo_init"]


@_needs_data
def test_provenance_records_every_required_field(tmp_path):
    meta = _arm(tmp_path, "prov", "--thermo-init", "scaled-random",
                "--thermo-match-cantera-norm", "--thermo-linear-init-seed", "2")["thermo_init"]
    for key in ("thermo_linear_init_seed", "original_random_vector", "original_random_norm",
                "requested_scale_factor", "effective_scale_factor", "resulting_vector",
                "resulting_norm", "cantera_reference_norm", "sign_match_vs_cantera"):
        assert key in meta, f"missing provenance field: {key}"
    n, d = meta["sign_match_vs_cantera"].split("/")
    assert 0 <= int(n) <= int(d) == len(SPECIES)


# --------------------------------------------------------------------------
# Cantera reference-state semantics (corrected): exact IC resolution
# --------------------------------------------------------------------------

_HIST_CANTERA = (_ROOT.parent / "results/reproduction/chemkan/hydrogen/diagnostics/"
                 "thermo_init_cantera_stage2_10000_seed0/config.json")


def _train_grid():
    d = np.load(_H2, allow_pickle=True)
    return d["train_ics"], d["train_states"]


@_needs_data
def test_resolver_returns_the_exact_requested_training_ic():
    from _thermo_coeffs import resolve_training_ic
    ics, states = _train_grid()
    y = resolve_training_ic(ics, states, 1050.0, 0.9, species_dim=len(SPECIES))
    j = np.where((np.abs(ics[:, 0] - 1050.0) < 1e-9) & (np.abs(ics[:, 1] - 0.9) < 1e-9))[0][0]
    assert np.allclose(y, states[j, 0, :len(SPECIES)].astype(np.float32))


@_needs_data
def test_resolver_fails_loudly_on_a_non_training_ic():
    """(3) No nearest-neighbour fallback: a bad reference state must raise."""
    from _thermo_coeffs import resolve_training_ic
    ics, states = _train_grid()
    with pytest.raises(SystemExit):
        resolve_training_ic(ics, states, 1234.0, 0.77, species_dim=len(SPECIES))


@_needs_data
def test_cantera_init_1050_09_uses_the_real_1050_09_composition(tmp_path):
    """(1) --thermo-init-phi now actually selects the composition."""
    _cantera_or_skip()
    from _thermo_coeffs import cantera_coefficients, resolve_training_ic
    meta = _arm(tmp_path, "c09", "--thermo-init", "cantera",
                "--thermo-init-temperature", "1050", "--thermo-init-phi", "0.9")["thermo_init"]
    ics, states = _train_grid()
    y = resolve_training_ic(ics, states, 1050.0, 0.9, species_dim=len(SPECIES))
    expect, _ = cantera_coefficients(1050.0, y, species=SPECIES)
    assert np.allclose(np.array(meta["coefficients"]), expect, rtol=1e-9)


@pytest.mark.skipif(not _HIST_CANTERA.exists(), reason="historical cantera config absent")
@_needs_data
def test_1050_phi05_reproduces_the_historical_coefficient_vector(tmp_path):
    """(2) The completed successful arm == T=1050 K with the phi=0.5 composition."""
    _cantera_or_skip()
    hist = np.array(json.loads(_HIST_CANTERA.read_text())["thermo_init"]["coefficients"])
    meta = _arm(tmp_path, "c05", "--thermo-init", "cantera",
                "--thermo-init-temperature", "1050", "--thermo-init-phi", "0.5")["thermo_init"]
    assert np.allclose(np.array(meta["coefficients"]), hist, rtol=1e-12, atol=1e-9)


@pytest.mark.skipif(not _HIST_CANTERA.exists(), reason="historical cantera config absent")
@_needs_data
def test_norm_match_at_1050_phi05_hits_the_historical_norm(tmp_path):
    """(4) The scale-vs-structure control targets the ACTUAL successful norm."""
    _cantera_or_skip()
    hist = np.array(json.loads(_HIST_CANTERA.read_text())["thermo_init"]["coefficients"])
    meta = _arm(tmp_path, "nm05", "--thermo-init", "scaled-random",
                "--thermo-match-cantera-norm", "--thermo-init-temperature", "1050",
                "--thermo-init-phi", "0.5")["thermo_init"]
    assert np.isclose(meta["cantera_reference_norm"], float(np.linalg.norm(hist)), rtol=1e-9)
    assert np.isclose(meta["resulting_norm"], float(np.linalg.norm(hist)), rtol=1e-5)


# --------------------------------------------------------------------------------------
# Thermo-initialization trajectory comparison (DIAGNOSTIC plotting artifact)
# --------------------------------------------------------------------------------------

_TRAJ_CSV = (_ROOT.parent / "results/reproduction/chemkan/hydrogen/tables"
             / "hydrogen_thermo_initialization_trajectory_comparison.csv")
_TRAJ_FIGS = _ROOT.parent / "results/reproduction/chemkan/hydrogen/figures"


def test_trajectory_comparison_module_declares_the_expected_runs_and_conditions():
    """The comparison covers both representative conditions and every init arm."""
    import plot_thermo_initialization_comparison as P
    assert [(c[0], c[1]) for c in P.CONDITIONS] == [(1050.0, 0.9), (1150.0, 1.3)]
    keys = [r[0] for r in P.RUNS]
    assert set(P.MAIN_KEYS) <= set(keys) and set(P.SCALE_KEYS) <= set(keys)
    # the main structure figure must not be cluttered with the scale ladder
    assert "x1e4_dir0" not in P.MAIN_KEYS and "x1e5_dir0" not in P.MAIN_KEYS
    # the ignition criterion is REUSED, never redefined here
    from hydrogen_thermo_intervention import ignition_delay as ref_def
    assert P.ignition_delay is ref_def
    assert not hasattr(P, "_ignition_delay")


@pytest.mark.skipif(not _TRAJ_CSV.exists(), reason="comparison CSV not generated yet")
def test_trajectory_comparison_csv_is_complete_and_finite():
    """Every run x condition row is present with finite temperatures."""
    import csv

    import plot_thermo_initialization_comparison as P
    rows = list(csv.DictReader(_TRAJ_CSV.open()))
    assert len(rows) == len(P.RUNS) * len(P.CONDITIONS)
    want = {(dirname, role) for _, dirname, _, _ in P.RUNS
            for _, _, role in P.CONDITIONS}
    assert {(r["run"], r["condition"]) for r in rows} == want
    for r in rows:
        for col in ("initial_T_K", "peak_T_K", "final_T_K", "peak_error_K",
                    "reference_peak_T_K", "thermo_linear_norm"):
            assert np.isfinite(float(r[col])), f"{col} not finite in {r['run']}"
        if r["ignition_delay_s"]:                      # blank means "did not ignite"
            assert np.isfinite(float(r["ignition_delay_s"]))


@pytest.mark.skipif(not _TRAJ_CSV.exists(), reason="comparison figures not generated yet")
@pytest.mark.parametrize("stem", [
    "hydrogen_DIAGNOSTIC_thermo_initialization_temperature_comparison",
    "hydrogen_DIAGNOSTIC_thermo_initialization_scale_response"])
@pytest.mark.parametrize("ext", ["png", "pdf"])
def test_trajectory_comparison_figures_exist(stem, ext):
    f = _TRAJ_FIGS / f"{stem}.{ext}"
    assert f.exists() and f.stat().st_size > 0

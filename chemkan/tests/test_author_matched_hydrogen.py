"""Author-matched hydrogen script (chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py).

Fast tests on a small synthetic hydrogen-shaped problem (9 species + T, 2 trajectories,
6 observation times) in float64, so resume/branch checks are exact.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(HERE / "author_repo_match"))
sys.path.insert(0, str(HERE))

import train_author_matched_hydrogen as amh  # noqa: E402
from _chemistry import ATOMIC_WEIGHTS, ELEMENT_COUNTS, MOLAR_WEIGHTS  # noqa: E402

from chemkan.losses import element_conservation_loss  # noqa: E402
from chemkan.normalization import MinMaxNormalizer  # noqa: E402
from chemkan.temperature import ObservedTemperature  # noqa: E402

M = amh.N_SPECIES


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _problem(seed: int = 0) -> dict:
    """Hydrogen-shaped synthetic data: smooth species/temperature curves, 2 trajectories."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 6e-4, 6)
    s = (t / 6e-4).view(-1, 1, 1)
    base = torch.rand(1, 2, M + 1, generator=g) * 0.5 + 0.1
    slope = torch.rand(1, 2, M + 1, generator=g) * 0.2 - 0.1
    full = base + slope * s                                          # (T, B, m+1) physical
    full[..., M] = 1000.0 + 500.0 * s[..., 0] + torch.tensor([0.0, 50.0])
    u_min, u_max = full.amin(dim=(0, 1)), full.amax(dim=(0, 1))
    norm = MinMaxNormalizer(u_min, u_max)
    t_dense = torch.linspace(0.0, 6e-4, 20)
    T_dense = (1000.0 + 500.0 * (t_dense / 6e-4)).view(-1, 1, 1) + torch.tensor([0.0, 50.0]).view(1, 2, 1)
    return {"t": t, "u_hat": norm.normalize(full), "t_dense": t_dense, "full": full,
            "T_dense": T_dense,
            "T_dense_hat": norm.subset(slice(M, M + 1)).normalize(T_dense), "norm": norm,
            "dataset": {"path": "synthetic", "split": "train", "sha256": "synthetic-v1",
                        "n_points": 6, "n_conditions": 2},
            "dense_temperature": {"path": "synthetic", "sha256": "dense-v1", "n_points": 20},
            "species": ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2"]}


def _args(**kw) -> argparse.Namespace:
    base = dict(epochs=3, seed=0, t_ref=amh.T_REF_DEFAULT, sensitivity="direct_autograd",
                dtype="float64", solve_mode="batched", checkpoint_every=1, threads=None,
                resume=False,
                stage1_archive=None, warmup_epochs=0)
    base.update(kw)
    return argparse.Namespace(**base)


def _rows(path: Path) -> list[list[str]]:
    return [line.split(",")[:5] for line in path.read_text().splitlines()]   # drop elapsed


def _stage1(tmp_path, name="s1", **kw):
    problem = _problem()
    out = amh.run_stage1(_args(**kw), problem, tmp_path / name)
    return problem, out


# --------------------------------------------------------------------------- model

def test_parameter_count_is_344():
    model = amh.build_model()
    assert sum(p.numel() for p in model.parameters()) == 344


def test_rbf_formula_centers_and_layer_normalizers():
    model = amh.build_model()
    x = torch.linspace(-1.3, 1.3, 7).view(1, 7).repeat(1, 10)[:, :10]
    c = torch.linspace(-1.0, 1.0, 5)
    for edges in amh.kan_edges(model):
        assert torch.equal(edges.centers, c)
    manual = lambda z: torch.exp(-(((z.unsqueeze(-1) - c) / 0.5) ** 2))          # noqa: E731
    for edges, tanh in ((model.kinetic.add.edges, False), (model.thermo.correction.edges, False)):
        z = torch.tanh(x) if tanh else x
        want = torch.einsum("bik,oik->boi", manual(z), edges.w_rbf)
        assert torch.allclose(edges(x), want, atol=1e-14)
    lean_in = torch.linspace(-2.0, 2.0, 3).view(1, 3)
    lean = model.kinetic.lean.edges
    want = torch.einsum("bik,oik->boi", manual(torch.tanh(lean_in)), lean.w_rbf)
    assert torch.allclose(lean(lean_in), want, atol=1e-14)                     # layer 2: tanh


def test_glorot_bounds_and_constant_thermo_init():
    model = amh.build_model()
    g = torch.Generator().manual_seed(3)
    amh.init_kinetic_(model, g)
    amh.init_thermo_(model, g)
    for edges in amh.kan_edges(model):
        out, n_in, gg = edges.w_rbf.shape
        a = math.sqrt(6.0 / (gg * n_in + out))
        assert edges.w_rbf.abs().max() <= a
        assert edges.w_rbf.abs().max() > 0.8 * a                             # uses the range
    assert model.thermo.linear.bias is None
    assert torch.equal(model.thermo.linear.weight, torch.full((1, M), 1e-5))


def test_normalize_denormalize_round_trip():
    p = _problem()
    phys = p["norm"].denormalize(p["u_hat"])
    assert torch.allclose(p["norm"].normalize(phys), p["u_hat"], atol=1e-12)


def test_rhs_normalized_units_single_t_ref():
    p = _problem()
    model = amh.build_model()
    amh.init_kinetic_(model, torch.Generator().manual_seed(0))
    amh.init_thermo_(model, torch.Generator().manual_seed(1))
    u = p["u_hat"][2]
    for t_ref in (6e-4, 50.0, 1.0):
        dyn = amh.Stage2Dynamics(model, t_ref)
        assert torch.allclose(dyn(torch.tensor(0.0), u) * t_ref, model(u), atol=1e-12)
    raw = model(u)
    r = model.kinetic(u)
    assert torch.allclose(raw[:, M:], model.thermo.linear(r) + model.thermo.correction(u))
    temp = ObservedTemperature(p["t_dense"], p["T_dense_hat"])
    d1 = amh.Stage1Dynamics(model.kinetic, temp, 6e-4)
    t0 = torch.tensor(1e-4)
    want = model.kinetic(torch.cat([u[:, :M], temp(t0)], -1)) / 6e-4
    assert torch.allclose(d1(t0, u[:, :M]), want)


def test_loss_eq18_sum_convention_and_pinn_on_physical_species():
    p = _problem()
    target = p["u_hat"]
    loss_fn = amh.make_loss_fn(target, p["norm"])
    pred = target + 0.1                                            # constant offset
    total, comp = loss_fn(pred)
    n_t = target.shape[0]
    assert torch.allclose(comp["mse"], torch.tensor(n_t * 0.01), atol=1e-12)       # SUM over t
    assert torch.allclose(comp["mse_time_avg"], torch.tensor(0.01), atol=1e-12)
    y_phys = p["norm"].subset(slice(0, M)).denormalize(pred[..., :M])
    want = element_conservation_loss(y_phys, ELEMENT_COUNTS.double(), ATOMIC_WEIGHTS.double(),
                                     MOLAR_WEIGHTS.double())
    assert torch.allclose(comp["pinn"], want)
    assert torch.allclose(total, comp["mse"] + 1e-4 * comp["pinn"])


# --------------------------------------------------------------------------- stages

def test_stage1_resume_equivalence_and_extension_archives(tmp_path):
    problem, full = _stage1(tmp_path, "full", epochs=4)
    amh.run_stage1(_args(epochs=4), problem, tmp_path / "part", stop_after=2)
    amh.run_stage1(_args(epochs=4, resume=True), problem, tmp_path / "part")
    assert _rows(tmp_path / "full/stage1_history.csv") == _rows(tmp_path / "part/stage1_history.csv")
    a = torch.load(tmp_path / "full/stage1_final.pt", weights_only=False)["model_state"]
    b = torch.load(tmp_path / "part/stage1_final.pt", weights_only=False)["model_state"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    # extension: the epoch-4 archive stays, a new epoch-6 archive is written
    old = full["archive"]
    sha_old = amh.checkpoint_sha256(old)
    ext = amh.run_stage1(_args(epochs=6, resume=True), problem, tmp_path / "full")
    assert old.exists() and amh.checkpoint_sha256(old) == sha_old
    assert ext["archive"].name == "stage1_full_epoch6.pt"


def test_stage2_warmup_freezes_and_gradients_reach_linear(tmp_path):
    problem, s1 = _stage1(tmp_path, epochs=2)
    run = tmp_path / "s2"
    amh.run_stage2(_args(epochs=0, warmup_epochs=2, stage1_archive=s1["archive"]), problem, run)
    after = torch.load(run / "stage2_working.pt", weights_only=False)["model_state"]
    arch = torch.load(s1["archive"], weights_only=False)["model_state"]
    for k in arch:
        if k.startswith("kinetic."):
            assert torch.equal(after[k], arch[k]), k                  # kinetic frozen
    fresh = amh.build_model()
    amh.init_thermo_(fresh, torch.Generator().manual_seed(0))
    assert torch.equal(after["thermo.correction.edges.w_rbf"],
                       fresh.thermo.correction.edges.w_rbf)            # KAN_cor frozen
    assert not torch.equal(after["thermo.linear.weight"], torch.full((1, M), 1e-5))
    # the gradient reaches thermo.linear through the trajectory with the rest frozen
    model = amh.build_model()
    model.load_state_dict(after)
    for prm in list(model.kinetic.parameters()) + list(model.thermo.correction.parameters()):
        prm.requires_grad_(False)
    loss_fn = amh.make_loss_fn(problem["u_hat"], problem["norm"])
    loss, _ = amh.loss_and_gradients(amh.Stage2Dynamics(model, amh.T_REF_DEFAULT),
                                     problem["u_hat"][0], problem["t"],
                                     [model.thermo.linear.weight], loss_fn,
                                     amh._solver("direct_autograd"))
    assert model.thermo.linear.weight.grad is not None
    assert model.thermo.linear.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.kinetic.parameters())


def test_stage2_resume_mid_warmup_equivalence(tmp_path):
    problem, s1 = _stage1(tmp_path, epochs=2)
    kw = dict(epochs=2, warmup_epochs=3, stage1_archive=s1["archive"])
    amh.run_stage2(_args(**kw), problem, tmp_path / "full")
    amh.run_stage2(_args(**kw), problem, tmp_path / "part", stop_after=2)      # mid-warm-up
    amh.run_stage2(_args(**{**kw, "resume": True}), problem, tmp_path / "part")
    for f in ("warmup_history.csv", "stage2_history.csv"):
        assert _rows(tmp_path / "full" / f) == _rows(tmp_path / "part" / f)
    a = torch.load(tmp_path / "full/stage2_final.pt", weights_only=False)["model_state"]
    b = torch.load(tmp_path / "part/stage2_final.pt", weights_only=False)["model_state"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_archive_unchanged_and_two_branches(tmp_path):
    problem, s1 = _stage1(tmp_path, epochs=2)
    sha = amh.checkpoint_sha256(s1["archive"])
    arch = torch.load(s1["archive"], weights_only=False)["model_state"]
    for name, seed in (("branch_a", 0), ("branch_b", 1)):
        amh.run_stage2(_args(epochs=2, seed=seed, stage1_archive=s1["archive"]), problem,
                       tmp_path / name)
    assert amh.checkpoint_sha256(s1["archive"]) == sha                 # byte-identical
    a = torch.load(tmp_path / "branch_a/stage2_final.pt", weights_only=False)
    b = torch.load(tmp_path / "branch_b/stage2_final.pt", weights_only=False)
    assert a["config"]["stage1_archive"]["sha256"] == b["config"]["stage1_archive"]["sha256"] == sha
    k = "kinetic.add.edges.w_rbf"
    assert not torch.equal(a["model_state"][k], arch[k])               # own evolving copy
    c = "thermo.correction.edges.w_rbf"
    assert not torch.equal(a["model_state"][c], b["model_state"][c])   # separate branches


@pytest.mark.parametrize("key, change", [
    ("t_ref", {"value_s": 50.0}),
    ("dataset", {"sha256": "other"}),
    ("normalization", {"u_min": [0.0] * 10}),
    ("architecture", {"num_basis": 4}),
    ("state_space", {"stage2": "physical"}),
])
def test_stage1_import_rejects_mismatch(tmp_path, key, change):
    problem, s1 = _stage1(tmp_path, epochs=1)
    cfg = amh.base_config(_args(), problem)
    amh.check_stage1_import(cfg, cfg)                                   # identical: accepted
    bad = {**cfg, key: {**cfg[key], **change}}
    with pytest.raises(SystemExit, match=key):
        amh.check_stage1_import(bad, cfg)


def test_tolerances_match_current_hydrogen_trainer():
    import train_hydrogen
    d = train_hydrogen.build_parser().parse_args([])
    assert (d.solver_method, d.rtol, d.atol) == ("tsit5", amh.RTOL, amh.ATOL)


def test_fsa_backend_one_epoch_each_stage(tmp_path):
    problem = _problem()
    s1 = amh.run_stage1(_args(epochs=1, sensitivity="fsa"), problem, tmp_path / "s1")
    amh.run_stage2(_args(epochs=1, warmup_epochs=1, sensitivity="fsa",
                         stage1_archive=s1["archive"]), problem, tmp_path / "s2")
    assert (tmp_path / "s2/stage2_final.pt").exists()


# --------------------------------------------------------------------------- additions

def _tight():
    from chemkan.solver import SolverConfig
    return SolverConfig(method="tsit5", rtol=1e-12, atol=1e-14, sensitivity="direct_autograd")


@pytest.mark.parametrize("stage", [1, 2])
def test_per_trajectory_matches_batched_loss_and_gradient(stage):
    p = _problem()
    model = amh.build_model()
    g = torch.Generator().manual_seed(0)
    amh.init_kinetic_(model, g)
    amh.init_thermo_(model, g)
    make = amh.stage1_units if stage == 1 else amh.stage2_units
    params = list(model.kinetic.parameters()) if stage == 1 else list(model.parameters())
    out = {}
    for mode in ("batched", "per_trajectory"):
        for prm in params:
            prm.grad = None
        vals = amh.epoch_gradients(make(model, p, amh.T_REF_DEFAULT, mode), p["t"], params,
                                   _tight())
        out[mode] = (vals, torch.cat([prm.grad.reshape(-1) for prm in params]).clone())
    (vb, gb), (vp, gp) = out["batched"], out["per_trajectory"]
    for k in ("loss", "mse", "pinn", "mse_time_avg", "kin_out_max_raw"):
        assert vp[k] == pytest.approx(vb[k], rel=1e-9), k
    assert (gp - gb).norm() <= 1e-8 * gb.norm()


def test_history_extra_columns_and_config(tmp_path):
    problem, s1 = _stage1(tmp_path, epochs=1)
    head = (tmp_path / "s1/stage1_history.csv").read_text().splitlines()[0].split(",")
    assert head[-2:] == amh.KINETIC_COLUMNS
    amh.run_stage2(_args(epochs=1, warmup_epochs=1, stage1_archive=s1["archive"]), problem,
                   tmp_path / "s2")
    import csv
    for f in ("warmup_history.csv", "stage2_history.csv"):
        row = next(csv.DictReader(open(tmp_path / "s2" / f)))
        assert list(row)[-10:] == amh.THERMO_COLUMNS
        raw, scaled = float(row["kin_out_max_raw"]), float(row["kin_out_max_scaled"])
        assert scaled == pytest.approx(raw / amh.T_REF_DEFAULT, rel=1e-12)
    row = next(csv.DictReader(open(tmp_path / "s2/warmup_history.csv")))
    assert all(float(row[f"w_{s}"]) == 1e-5 for s in amh.SPECIES_NAMES)   # starting weights
    assert float(row["w_norm"]) == pytest.approx(3e-5)
    import json
    cfg = json.loads((tmp_path / "s2/config.json").read_text())
    assert cfg["solver"]["solve_mode"] == "batched"


@pytest.mark.parametrize("sensitivity", ["direct_autograd", "fsa"])
def test_per_trajectory_runs_end_to_end(tmp_path, sensitivity):
    problem = _problem()
    kw = dict(epochs=1, solve_mode="per_trajectory", sensitivity=sensitivity)
    s1 = amh.run_stage1(_args(**kw), problem, tmp_path / "s1")
    amh.run_stage2(_args(**kw, warmup_epochs=1, stage1_archive=s1["archive"]), problem,
                   tmp_path / "s2")
    assert (tmp_path / "s2/stage2_final.pt").exists()


# --------------------------------------------------------------------------- evaluator

def _reference(p):
    return {"t": p["t"].numpy(), "states": p["full"].permute(1, 0, 2).numpy(),
            "ics": torch.tensor([[1000.0, 0.5], [950.0, 0.9]]).numpy(),
            "is_test": torch.tensor([False, True]).numpy(),
            "t_cache": p["t_dense"].numpy(), "T_cache": p["T_dense"][..., 0].numpy(),
            "t_diag": torch.linspace(0.0, 6e-4, 31).numpy(),
            "dataset_sha256": "synthetic-v1", "dense_sha256": "dense-v1"}


@pytest.mark.parametrize("stage", [1, 2])
def test_evaluator_on_tiny_checkpoint_and_physical_round_trip(tmp_path, stage):
    import json

    import evaluate_author_matched_hydrogen as ev
    problem, s1 = _stage1(tmp_path, epochs=1)
    path = s1["archive"]
    if stage == 2:
        amh.run_stage2(_args(epochs=1, stage1_archive=path), problem, tmp_path / "s2")
        path = tmp_path / "s2/stage2_final.pt"
    ref = _reference(problem)
    res = ev.evaluate(torch.load(path, weights_only=False), ref, return_arrays=True)
    arr = res.pop("_arrays")
    json.dumps(res)                                                  # JSON-serializable
    assert [r["split"] for r in res["conditions"]] == ["train", "test"]
    assert all(r["finite"] and r["eq18_mse"] >= 0 for r in res["conditions"])
    n = amh.N_SPECIES if stage == 1 else amh.N_SPECIES + 1
    back = arr["norm"].subset(slice(0, n)).normalize(torch.as_tensor(arr["pred_phys"]))
    assert torch.allclose(back, torch.as_tensor(arr["pred_hat"]), atol=1e-10)
    assert torch.allclose(torch.as_tensor(arr["pred_phys"][0]),
                          problem["full"][0, :, :n], atol=1e-9)      # t = 0 is the data IC
    if stage == 2:
        r = res["conditions"][0]
        assert r["in_fig8b_set"] and r["status"] in ("evaluated", "argmax_at_window_edge")
        assert r["reference_temperature_rise_K"] == pytest.approx(500.0, rel=1e-6)
        assert len(res["summary"]["t0_950K"]) == 1
        assert res["summary"]["fig8b_30"]["n"] == 1


def test_evaluator_refuses_other_dataset(tmp_path):
    import evaluate_author_matched_hydrogen as ev
    problem, s1 = _stage1(tmp_path, epochs=1)
    ref = {**_reference(problem), "dataset_sha256": "different"}
    with pytest.raises(SystemExit, match="sha256"):
        ev.evaluate(torch.load(s1["archive"], weights_only=False), ref)

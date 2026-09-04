import torch

from chemkan.model import ChemKAN, KineticCore, ThermodynamicSuperstructure

# Paper benchmark configurations live in the TESTS (not in reusable code): these are
# regression checks on specific published configs, not restrictions on the model.
BIODIESEL = {"species_dim": 6, "hidden_dim": 4, "num_basis": 3, "n_mu": 2}
HYDROGEN = {"species_dim": 9, "hidden_dim": 3, "num_basis": 5, "n_mu": 3}


def _count(module):
    return sum(p.numel() for p in module.parameters())


# --- shapes -----------------------------------------------------------------------

def test_kinetic_core_shape():
    core = KineticCore(species_dim=3, hidden_dim=4, num_basis=6, n_mu=2, use_base_act=False)
    assert core(torch.randn(5, 4)).shape == (5, 3)         # (B, m+1) -> (B, m)


def test_thermo_superstructure_shape_and_no_bias():
    m = 3
    thermo = ThermodynamicSuperstructure(species_dim=m, num_basis=6, use_base_act=False)
    assert thermo(torch.randn(5, m + 1), torch.randn(5, m)).shape == (5, 1)
    assert thermo.linear.bias is None                      # nn.Linear(m, 1, bias=False)


def test_full_chemkan_shape():
    model = ChemKAN(species_dim=3, hidden_dim=4, num_basis=6, n_mu=2, use_base_act=False)
    assert model(torch.randn(5, 4)).shape == (5, 4)        # (B, m+1) -> (B, m+1)


# --- generic architecture (no experiment-specific dims) ---------------------------

def test_generic_arbitrary_dimensions_run():
    model = ChemKAN(species_dim=4, hidden_dim=7, num_basis=6, n_mu=2, use_base_act=False)
    out = model(torch.randn(3, 5))                         # (B, m+1) -> (B, m+1)
    assert out.shape == (3, 5)
    core = KineticCore(species_dim=4, hidden_dim=7, num_basis=6, n_mu=2, use_base_act=False)
    assert core(torch.randn(3, 5)).shape == (3, 4)


# --- parameter-count regressions (ChemKAN reported counts) ------------------------

def test_biodiesel_param_count_156():
    assert _count(KineticCore(**BIODIESEL, use_base_act=False)) == 156


def test_hydrogen_param_count_344():
    assert _count(ChemKAN(**HYDROGEN, use_base_act=False)) == 344


def test_biodiesel_base_on_param_count_208():
    assert _count(KineticCore(**BIODIESEL, use_base_act=True)) == 208


def test_hydrogen_base_on_param_count_411():
    assert _count(ChemKAN(**HYDROGEN, use_base_act=True)) == 411


# --- checkpoint round-trip (reconstruct from dataset dim + stored architecture) ---

def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    arch = {"hidden_dim": 5, "num_basis": 4, "n_mu": 2, "use_base_act": False}
    species_dim = 4
    model = ChemKAN(species_dim=species_dim, **arch)
    ckpt = {
        "model_state": model.state_dict(),
        "architecture": arch,
        "data": {"species": ["a", "b", "c", "d"], "species_dim": species_dim},
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    loaded = torch.load(path, weights_only=False)
    m = species_dim                                        # "inferred from dataset"
    rebuilt = ChemKAN(species_dim=m, **loaded["architecture"])
    rebuilt.load_state_dict(loaded["model_state"])
    rebuilt.eval()
    model.eval()

    x = torch.randn(6, species_dim + 1)
    assert torch.allclose(model(x), rebuilt(x), atol=1e-6)


# --- the two matched-count interpretations of the reported 344 --------------------
#
# The paper reports hidden=3, n_mu=3, 344 total parameters for hydrogen but does NOT
# state the grid size N. Eq. 11 writes every edge as N RBF coefficients PLUS one
# learnable base (Swish) coefficient. Two configurations reproduce 344 exactly:
#
#   A  N=5, base OFF  -- historical inferred interpretation (drops the Eq. 11 base term)
#   B  N=4, base ON   -- Eq. 11-aligned inferred interpretation (keeps it)
#
# NEITHER N is paper-explicit. Both are inferred from the reported count.
HYDROGEN_BASE_ON_N4 = {"species_dim": 9, "hidden_dim": 3, "num_basis": 4, "n_mu": 3}


def test_hydrogen_n4_base_on_param_count_344():
    """Interpretation B must also hit the paper's reported 344."""
    assert _count(ChemKAN(**HYDROGEN_BASE_ON_N4, use_base_act=True)) == 344


def test_both_interpretations_agree_block_by_block():
    """A and B match not only in total but in every block -- they are indistinguishable
    from the parameter count alone."""
    a = ChemKAN(**HYDROGEN, use_base_act=False)             # N=5, base OFF
    b = ChemKAN(**HYDROGEN_BASE_ON_N4, use_base_act=True)   # N=4, base ON
    for block in ("kinetic.add", "kinetic.lean", "thermo.linear", "thermo.correction"):
        ca = sum(p.numel() for n, p in a.named_parameters() if n.startswith(block))
        cb = sum(p.numel() for n, p in b.named_parameters() if n.startswith(block))
        assert ca == cb, f"{block}: {ca} != {cb}"
    assert _count(a) == _count(b) == 344


def test_n4_base_on_decomposition_follows_eq11():
    """Every edge carries num_basis RBF weights + exactly one base weight."""
    model = ChemKAN(**HYDROGEN_BASE_ON_N4, use_base_act=True)
    counts = {n: p.numel() for n, p in model.named_parameters()}
    N = HYDROGEN_BASE_ON_N4["num_basis"]
    assert counts["kinetic.add.edges.w_rbf"] == 3 * 10 * N       # 120
    assert counts["kinetic.add.edges.w_base"] == 3 * 10          # 30
    assert counts["kinetic.lean.edges.w_rbf"] == 9 * 3 * N       # 108
    assert counts["kinetic.lean.edges.w_base"] == 9 * 3          # 27
    assert counts["thermo.linear.weight"] == 9
    assert counts["thermo.correction.edges.w_rbf"] == 1 * 10 * N  # 40
    assert counts["thermo.correction.edges.w_base"] == 1 * 10     # 10
    assert sum(counts.values()) == 344


def test_base_weights_start_at_zero_so_the_base_path_starts_inert():
    """w_base is zero-initialized: at epoch 0 the base pathway contributes nothing."""
    model = ChemKAN(**HYDROGEN_BASE_ON_N4, use_base_act=True)
    for n, p in model.named_parameters():
        if n.endswith("w_base"):
            assert torch.count_nonzero(p) == 0, n


def test_base_off_creates_no_base_parameter():
    model = ChemKAN(**HYDROGEN, use_base_act=False)
    assert not any(n.endswith("w_base") for n, _ in model.named_parameters())
    assert model.kinetic.add.edges.w_base is None


# --- the SCRIPT DEFAULTS, not just the library ------------------------------------
#
# Since 2026-09-03 the hydrogen script defaults to interpretation B (N=4, base ON).
# These tests pin the CLI defaults themselves, so a silent regression in argparse is
# caught even though the library keeps `use_base_act` a required argument.

def _hydrogen_parser():
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import train_hydrogen
    return train_hydrogen.build_parser()


def test_hydrogen_script_defaults_are_the_eq11_aligned_reading():
    args = _hydrogen_parser().parse_args([])
    assert (args.num_basis, args.n_mu, args.hidden_dim) == (4, 3, 3)
    assert args.use_base_act is True
    model = ChemKAN(species_dim=9, hidden_dim=args.hidden_dim, num_basis=args.num_basis,
                    n_mu=args.n_mu, use_base_act=args.use_base_act)
    assert _count(model) == 344


def test_hydrogen_script_can_still_express_the_historical_base_off_reading():
    """Existing N=5/base-OFF runs must remain reproducible from the CLI."""
    args = _hydrogen_parser().parse_args(["--no-use-base-act", "--num-basis", "5"])
    assert args.use_base_act is False and args.num_basis == 5
    model = ChemKAN(species_dim=9, hidden_dim=args.hidden_dim, num_basis=args.num_basis,
                    n_mu=args.n_mu, use_base_act=args.use_base_act)
    assert _count(model) == 344


def test_biodiesel_script_defaults_are_unchanged_by_the_hydrogen_decision():
    """The paper states biodiesel's grid explicitly ("three-point grids"), and 156 params
    only match with the base path OFF -- so biodiesel keeps N=3/base-OFF."""
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import train_biodiesel
    args = train_biodiesel.build_parser().parse_args([])
    assert (args.num_basis, args.n_mu, args.hidden_dim) == (3, 2, 4)
    assert args.use_base_act is False
    assert _count(KineticCore(species_dim=6, hidden_dim=args.hidden_dim,
                              num_basis=args.num_basis, n_mu=args.n_mu,
                              use_base_act=args.use_base_act)) == 156

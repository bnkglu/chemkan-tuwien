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

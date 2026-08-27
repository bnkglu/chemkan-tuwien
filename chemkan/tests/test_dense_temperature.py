"""Dense precomputed Cantera Stage-1 temperature approach (supervisor-approved).

Covers the temperature-only generator mode, the dense loader's validations, the
reuse of ObservedTemperature (exact-at-node / linear-between), the invariants that
must NOT change (50-point output grid, species targets, parameter counts, Stage-2
independence), and the checkpoint metadata for both temperature sources.

Tests that need Cantera or the generated archives skip cleanly when unavailable.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# conftest puts src/ and scripts/ on the path; the generators live one level deeper.
_DG = Path(__file__).resolve().parents[1] / "scripts" / "data_gen"
if str(_DG) not in sys.path:
    sys.path.insert(0, str(_DG))

from _data import DATA_DIR, load_hydrogen, load_hydrogen_temperature  # noqa: E402
from chemkan.temperature import ObservedTemperature  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _cantera_or_skip():
    try:
        import cantera  # noqa: F401
    except Exception:
        pytest.skip("cantera not available")


def _canonical_or_skip():
    if not (DATA_DIR / "hydrogen.npz").exists():
        pytest.skip("canonical hydrogen.npz not available")
    return load_hydrogen(split="train")


def _tempcfg(n_points):
    import cantera as ct
    return SimpleNamespace(grid="coarse", t_end=0.6e-3, n_points=n_points, phis=None,
                           pressure=ct.one_atm, rtol=1e-9, atol=1e-15, seed=0,
                           ignition_points=n_points)


@pytest.fixture
def small_dense_cache():
    """Generate a small temperature-only cache next to the canonical archive."""
    _cantera_or_skip()
    if not (DATA_DIR / "hydrogen.npz").exists():
        pytest.skip("canonical hydrogen.npz not available")
    import generate_hydrogen as gh

    n = 48
    path = DATA_DIR / f"hydrogen_temperature_{n}.npz"
    pre_existing = path.exists()
    np.savez_compressed(path, **gh.generate_temperature_only(_tempcfg(n)))
    try:
        yield n
    finally:
        if not pre_existing and path.exists():
            path.unlink()


# --------------------------------------------------------------------------
# 1 / 2 / 3  generator (temperature-only) behavior
# --------------------------------------------------------------------------

def test_temperature_only_shapes():
    _cantera_or_skip()
    import generate_hydrogen as gh
    n = 32
    out = gh.generate_temperature_only(_tempcfg(n))
    assert out["t"].shape == (n,)
    assert out["train_T"].shape == (n, 35, 1)
    assert out["test_T"].shape == (n, 1, 1)
    # no dense species trajectories saved
    assert "states" not in out and "train_states" not in out and "test_states" not in out


def test_normal_generate_unchanged_without_flag():
    _cantera_or_skip()
    import generate_hydrogen as gh
    cfg = _tempcfg(8)
    full = gh.generate(cfg)
    assert "states" in full and "train_states" in full
    assert full["train_states"].shape[0] == 35
    assert full["states"].shape[-1] == 10  # 9 species + T (unchanged layout)


def test_temperature_only_ic_order_matches_canonical():
    _cantera_or_skip()
    if not (DATA_DIR / "hydrogen.npz").exists():
        pytest.skip("canonical hydrogen.npz not available")
    import generate_hydrogen as gh
    out = gh.generate_temperature_only(_tempcfg(16))
    d = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    assert np.allclose(out["train_ics"], d["train_ics"])
    assert np.allclose(out["test_ics"], d["test_ics"])


# --------------------------------------------------------------------------
# 4  loader rejects mismatched IC ordering
# --------------------------------------------------------------------------

def test_dense_loader_success(small_dense_cache):
    n = small_dense_cache
    dense = load_hydrogen_temperature(split="train", n_points=n)
    assert dense["t_dense"].shape == (n,)
    assert dense["T_dense_TB1"].shape == (n, 35, 1)
    assert dense["ics"].shape == (35, 2)


def test_dense_loader_rejects_ic_mismatch(small_dense_cache):
    n = small_dense_cache
    path = DATA_DIR / f"hydrogen_temperature_{n}.npz"
    d = dict(np.load(path, allow_pickle=True))
    d["train_ics"] = d["train_ics"][::-1].copy()   # break the canonical ordering
    np.savez_compressed(path, **d)
    with pytest.raises(ValueError):
        load_hydrogen_temperature(split="train", n_points=n)


# --------------------------------------------------------------------------
# 5 / 8 / 9  ObservedTemperature reuse with a dense grid
# --------------------------------------------------------------------------

def test_dense_provider_returns_B1():
    n, b = 100, 35
    t = torch.linspace(0.0, 0.6e-3, n)
    T = torch.rand(n, b, 1) * 100 + 1000
    prov = ObservedTemperature(t, T)
    assert prov(torch.tensor(0.3e-3)).shape == (b, 1)


def test_dense_provider_exact_at_stored_time():
    n, b = 50, 3
    t = torch.linspace(0.0, 0.6e-3, n)
    T = torch.rand(n, b, 1)
    prov = ObservedTemperature(t, T)
    j = 17
    assert torch.allclose(prov(t[j]), T[j], atol=1e-6)


def test_dense_provider_linear_between():
    t = torch.tensor([0.0, 1.0, 2.0])
    T = torch.tensor([[[0.0]], [[10.0]], [[20.0]]])  # (3, 1, 1)
    prov = ObservedTemperature(t, T)
    assert torch.allclose(prov(torch.tensor(0.25)), torch.tensor([[2.5]]), atol=1e-6)


# --------------------------------------------------------------------------
# 6 / 7  Stage-1 output grid + species target invariants (unchanged)
# --------------------------------------------------------------------------

def test_stage1_output_grid_is_50_points():
    data = _canonical_or_skip()
    assert data["t"].shape == (50,)


def test_stage1_species_target_shape():
    data = _canonical_or_skip()
    assert data["species_TBm"].shape == (50, 35, 9)


# --------------------------------------------------------------------------
# 10  training-data source reproduces the original provider
# --------------------------------------------------------------------------

def test_training_data_source_matches_original_provider():
    data = _canonical_or_skip()
    prov = ObservedTemperature(data["t"], data["T_obs_TB1"])
    for j in (0, 10, 49):
        assert torch.allclose(prov(data["t"][j]), data["T_obs_TB1"][j], atol=1e-3)


# --------------------------------------------------------------------------
# 11  Stage 2 does not depend on any temperature provider / dense cache
# --------------------------------------------------------------------------

def test_stage2_dynamics_take_no_temperature_provider():
    from chemkan.dynamics import ChemKANDynamics
    params = inspect.signature(ChemKANDynamics.__init__).parameters
    assert "temperature" not in params and "temp" not in params


# --------------------------------------------------------------------------
# 12  parameter counts unchanged
# --------------------------------------------------------------------------

def test_parameter_counts_unchanged():
    from chemkan.model import ChemKAN
    model = ChemKAN(species_dim=9, hidden_dim=3, num_basis=5, n_mu=3, use_base_act=False)
    assert sum(p.numel() for p in model.parameters()) == 344
    assert sum(p.numel() for p in model.kinetic.parameters()) == 285


# --------------------------------------------------------------------------
# 13  checkpoint metadata for both temperature sources
# --------------------------------------------------------------------------

def test_stage1_temperature_metadata_records_both_sources():
    import train_hydrogen as th
    dense = th.stage1_temperature_metadata(
        "dense_cantera", 20000, cache_file="hydrogen_temperature_20000.npz")
    assert dense == {
        "source": "dense_cantera", "n_points": 20000,
        "provider": "ObservedTemperature", "interpolation": "linear",
        "cache_file": "hydrogen_temperature_20000.npz",
    }
    orig = th.stage1_temperature_metadata("training_data", 50)
    assert orig == {
        "source": "training_data", "n_points": 50,
        "provider": "ObservedTemperature", "interpolation": "linear",
    }

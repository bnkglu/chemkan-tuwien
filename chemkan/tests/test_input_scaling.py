"""Raw-temperature saturation vs. min-max input-scaling regressions (the fix).

These prove the practical consequence of the pre-KAN input scaling: the RAW physical
path cannot represent temperature dependence (tanh saturates), while the min-max
scaled path restores a nonzero d(output)/d(T_physical).
"""

import pytest
import torch
from _data import input_scaling_meta, load_input_scaling

from chemkan.dynamics import ChemKANDynamics
from chemkan.model import ChemKAN, KineticCore
from chemkan.normalization import MinMaxNormalizer

CPU = torch.device("cpu")
F64 = torch.float64


# --- §18 raw-temperature saturation witness (CPU / float64) ------------------------

def test_tanh_saturates_on_raw_kelvin_bit_exact():
    assert torch.tanh(torch.tensor(323.0, dtype=F64)).item() == 1.0
    assert torch.tanh(torch.tensor(343.0, dtype=F64)).item() == 1.0


def test_raw_input_model_is_temperature_blind_witness():
    torch.manual_seed(0)
    core = KineticCore(species_dim=2, hidden_dim=3, num_basis=4, n_mu=2,
                       use_base_act=False).double()          # base-off, float64
    Y = torch.tensor([[0.1, 0.2]], dtype=F64)
    u1 = torch.cat([Y, torch.tensor([[323.0]], dtype=F64)], dim=-1)  # raw physical [Y,T]
    u2 = torch.cat([Y, torch.tensor([[343.0]], dtype=F64)], dim=-1)
    # input_normalizer=None path == feeding u directly to the model; internal tanh
    # saturates so the two temperatures collapse to identical model outputs.
    assert torch.equal(core(u1), core(u2))


def test_raw_input_temperature_gradient_is_exactly_zero():
    torch.manual_seed(0)
    core = KineticCore(species_dim=2, hidden_dim=3, num_basis=4, n_mu=2,
                       use_base_act=False).double()
    Y = torch.tensor([[0.1, 0.2]], dtype=F64)
    T = torch.tensor([[323.0]], dtype=F64, requires_grad=True)
    core(torch.cat([Y, T], dim=-1)).sum().backward()
    assert torch.equal(T.grad, torch.zeros_like(T.grad))    # d out / d T == 0 exactly


# --- §19 scaled-input temperature sensitivity witness -----------------------------

def test_minmax_separates_temperatures_before_tanh():
    # train T range includes 323, 343
    norm = MinMaxNormalizer(torch.tensor([0.0]), torch.tensor([400.0])).double()
    n323 = norm.normalize(torch.tensor([323.0], dtype=F64))
    n343 = norm.normalize(torch.tensor([343.0], dtype=F64))
    assert not torch.equal(n323, n343)
    assert not torch.equal(torch.tanh(n323), torch.tanh(n343))


def test_scaled_input_temperature_gradient_is_nonzero():
    torch.manual_seed(0)
    m = 2
    model = ChemKAN(species_dim=m, hidden_dim=3, num_basis=4, n_mu=2,
                    use_base_act=False).double()
    u_min = torch.tensor([0.0, 0.0, 300.0], dtype=F64)
    u_max = torch.tensor([1.0, 1.0, 400.0], dtype=F64)      # T range 300..400 K
    norm = MinMaxNormalizer(u_min, u_max)
    dyn = ChemKANDynamics(model, input_normalizer=norm)

    Y = torch.tensor([[0.2, 0.3]], dtype=F64)
    T = torch.tensor([[330.0]], dtype=F64, requires_grad=True)
    u = torch.cat([Y, T], dim=-1)                           # physical [Y, T]
    dyn(torch.tensor(0.0, dtype=F64), u).sum().backward()
    assert T.grad is not None and T.grad.abs().sum() > 0    # temperature now matters


# --- §20 train-only stats, NO clipping --------------------------------------------

def test_no_clipping_outside_training_extrema():
    norm = MinMaxNormalizer(torch.tensor([0.0]), torch.tensor([1.0]))
    assert norm.normalize(torch.tensor([1.5])).item() > 1.0   # held-out above max
    assert norm.normalize(torch.tensor([-0.5])).item() < 0.0  # held-out below min


# --- §21 checkpoint round-trip of the input-scaling metadata ----------------------

def _full_norm():
    return MinMaxNormalizer(torch.tensor([0.0, 0.0, 300.0]),
                            torch.tensor([1.0, 1.0, 2000.0]))


def test_minmax_scaling_roundtrips_through_checkpoint():
    norm = _full_norm()
    ckpt = {"state_representation": "physical",
            "input_scaling": input_scaling_meta("minmax", norm)}
    rebuilt = load_input_scaling(ckpt, CPU)
    x = torch.tensor([[0.5, 0.5, 1000.0]])
    assert torch.equal(norm.normalize(x), rebuilt.normalize(x))


def test_minmax_dynamics_outputs_identical_after_reload():
    torch.manual_seed(0)
    model = ChemKAN(species_dim=2, hidden_dim=3, num_basis=4, n_mu=2, use_base_act=False)
    norm = _full_norm()
    ckpt = {"state_representation": "physical",
            "input_scaling": input_scaling_meta("minmax", norm)}
    rebuilt = load_input_scaling(ckpt, CPU)
    u = torch.tensor([[0.2, 0.3, 900.0]])
    a = ChemKANDynamics(model, input_normalizer=norm)(torch.tensor(0.0), u)
    b = ChemKANDynamics(model, input_normalizer=rebuilt)(torch.tensor(0.0), u)
    assert torch.equal(a, b)


def test_none_scaling_roundtrips_to_none():
    ckpt = {"state_representation": "physical",
            "input_scaling": input_scaling_meta("none", _full_norm())}
    assert ckpt["input_scaling"] == {"method": "none"}
    assert load_input_scaling(ckpt, CPU) is None


def test_missing_representation_metadata_raises():
    with pytest.raises(ValueError):
        load_input_scaling({"input_scaling": {"method": "none"}}, CPU)


def test_missing_input_scaling_metadata_raises():
    with pytest.raises(ValueError):
        load_input_scaling({"state_representation": "physical"}, CPU)


def test_non_physical_representation_raises():
    with pytest.raises(ValueError):
        load_input_scaling({"state_representation": "normalized",
                            "input_scaling": {"method": "none"}}, CPU)


def test_unknown_scaling_method_raises():
    with pytest.raises(ValueError):
        load_input_scaling({"state_representation": "physical",
                            "input_scaling": {"method": "zscore"}}, CPU)

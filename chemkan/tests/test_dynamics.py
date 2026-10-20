import pytest
import torch

from chemkan.dynamics import ChemKANDynamics, KineticDynamics
from chemkan.model import ChemKAN, KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.temperature import ConstantTemperature


def test_kinetic_dynamics_preserves_state_shape():
    m = 6
    core = KineticCore(species_dim=m, hidden_dim=4, num_basis=3, n_mu=2, use_base_act=False)
    temp = ConstantTemperature(torch.full((5,), 330.0))
    dyn = KineticDynamics(core, temp, input_normalizer=None)   # raw-input ablation
    Y = torch.randn(5, m)                                 # state (B, m)
    assert dyn(torch.tensor(0.0), Y).shape == (5, m)      # derivative (B, m)


def test_chemkan_dynamics_preserves_state_shape():
    m = 9
    model = ChemKAN(species_dim=m, hidden_dim=3, num_basis=5, n_mu=3, use_base_act=False)
    dyn = ChemKANDynamics(model, input_normalizer=None)
    u = torch.randn(4, m + 1)                             # state (B, m+1)
    assert dyn(torch.tensor(0.0), u).shape == (4, m + 1)  # derivative (B, m+1)


def test_input_normalizer_is_required():
    m = 3
    core = KineticCore(species_dim=m, hidden_dim=4, num_basis=3, n_mu=2, use_base_act=False)
    temp = ConstantTemperature(torch.full((2,), 330.0))
    with pytest.raises(TypeError):
        KineticDynamics(core, temp)                       # input_normalizer required
    with pytest.raises(TypeError):
        ChemKANDynamics(ChemKAN(species_dim=m, hidden_dim=4, num_basis=3, n_mu=2,
                                use_base_act=False))


def test_scaling_changes_model_input_but_not_state_or_output_shape():
    m = 3
    model = ChemKAN(species_dim=m, hidden_dim=4, num_basis=3, n_mu=2, use_base_act=False)
    u_min = torch.zeros(m + 1)
    u_max = torch.tensor([1.0, 1.0, 1.0, 2000.0])         # T range up to 2000 K
    norm = MinMaxNormalizer(u_min, u_max)
    raw = ChemKANDynamics(model, input_normalizer=None)
    scaled = ChemKANDynamics(model, input_normalizer=norm)
    u = torch.tensor([[0.2, 0.3, 0.5, 1000.0]])           # physical [Y, T]
    # same physical state in; scaling changes the derivative (different model input),
    # but both remain physical (B, m+1) with no external rescale of the output.
    assert raw(torch.tensor(0.0), u).shape == scaled(torch.tensor(0.0), u).shape == (1, m + 1)
    assert not torch.equal(raw(torch.tensor(0.0), u), scaled(torch.tensor(0.0), u))

import pytest
import torch

from chemkan.kan.layers import AddKANLayer, LeanKANLayer


def test_addkan_output_shape():
    layer = AddKANLayer(4, 7, num_basis=8, use_base_act=False)
    assert layer(torch.randn(5, 4)).shape == (5, 7)


def test_addkan_equals_sum_of_edges():
    layer = AddKANLayer(4, 7, num_basis=8, use_base_act=False)
    x = torch.randn(5, 4)
    assert torch.allclose(layer(x), layer.edges(x).sum(dim=-1))


def test_leankan_multiplies_first_nmu_inputs():
    torch.manual_seed(0)
    lean = LeanKANLayer(2, 1, n_mu=2, num_basis=8, use_base_act=False)
    x = torch.randn(5, 2)
    g = lean.edges(x)                                  # (5, 1, 2)
    expected = g[..., 0] * g[..., 1]                   # pure product, no add term
    assert torch.allclose(lean(x), expected, atol=1e-6)


def test_leankan_partitions_inputs_mult_then_add():
    torch.manual_seed(0)
    lean = LeanKANLayer(3, 2, n_mu=1, num_basis=8, use_base_act=False)
    x = torch.randn(5, 3)
    g = lean.edges(x)                                  # (5, 2, 3)
    expected = g[..., :1].prod(dim=-1) + g[..., 1:].sum(dim=-1)
    assert torch.allclose(lean(x), expected, atol=1e-6)


def test_leankan_nmu0_equals_addkan():
    torch.manual_seed(0)
    lean = LeanKANLayer(4, 7, n_mu=0, num_basis=8, use_base_act=False)
    add = AddKANLayer(4, 7, num_basis=8, use_base_act=False)
    add.load_state_dict(lean.state_dict())
    x = torch.randn(5, 4)
    assert torch.allclose(lean(x), add(x), atol=1e-6)


def test_leankan_empty_product_is_not_one():
    torch.manual_seed(0)
    lean = LeanKANLayer(3, 4, n_mu=0, num_basis=8, use_base_act=False)
    add = AddKANLayer(3, 4, num_basis=8, use_base_act=False)
    add.load_state_dict(lean.state_dict())
    assert torch.allclose(lean(torch.zeros(2, 3)), add(torch.zeros(2, 3)), atol=1e-6)


@pytest.mark.parametrize("n_mu", [-1, 5])
def test_leankan_invalid_nmu_raises(n_mu):
    with pytest.raises(ValueError):
        LeanKANLayer(4, 7, n_mu=n_mu, num_basis=8, use_base_act=False)


def test_addkan_requires_num_basis():
    with pytest.raises(TypeError):
        AddKANLayer(4, 7, use_base_act=False)             # no num_basis default


def test_leankan_requires_n_mu_and_num_basis():
    with pytest.raises(TypeError):
        LeanKANLayer(4, 7, use_base_act=False)            # n_mu AND num_basis required
    with pytest.raises(TypeError):
        LeanKANLayer(4, 7, n_mu=2, use_base_act=False)    # num_basis still required

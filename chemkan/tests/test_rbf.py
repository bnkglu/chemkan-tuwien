import pytest
import torch

from chemkan.kan._common import gaussian
from chemkan.kan.rbf import RBFEdgeFunctions


def test_gaussian_uses_2_h_squared():
    x = torch.tensor([0.3])
    centers = torch.tensor([0.0, 0.5])
    h = 0.4
    expected = torch.exp(-(x.unsqueeze(-1) - centers) ** 2 / (2 * h ** 2))
    assert torch.allclose(gaussian(x, centers, h), expected)


def test_centers_are_registered_buffers():
    edges = RBFEdgeFunctions(3, 2, num_basis=5, use_base_act=False)
    assert "centers" in dict(edges.named_buffers())
    assert "centers" in edges.state_dict()


def test_buffers_move_with_dtype_cast():
    edges = RBFEdgeFunctions(3, 2, num_basis=5, use_base_act=False).double()
    assert edges.centers.dtype == torch.float64          # buffer followed .to(...)


def test_edge_output_shape():
    edges = RBFEdgeFunctions(4, 7, num_basis=6, use_base_act=False)
    out = edges(torch.randn(5, 4))
    assert out.shape == (5, 7, 4)                         # (B, out, in)


def test_gradients_reach_trainable_params():
    edges = RBFEdgeFunctions(3, 2, num_basis=5, use_base_act=True)
    edges(torch.randn(4, 3)).sum().backward()
    assert edges.w_rbf.grad is not None
    assert edges.w_base.grad is not None


def test_base_off_has_no_trainable_w_base():
    edges = RBFEdgeFunctions(3, 2, num_basis=5, use_base_act=False)
    assert edges.w_base is None
    assert "w_base" not in dict(edges.named_parameters())


def test_base_on_has_trainable_w_base():
    edges = RBFEdgeFunctions(3, 2, num_basis=5, use_base_act=True)
    assert "w_base" in dict(edges.named_parameters())


def test_use_base_act_is_required():
    with pytest.raises(TypeError):
        RBFEdgeFunctions(3, 2, 5)       # missing required keyword-only use_base_act


def test_num_basis_is_required():
    with pytest.raises(TypeError):
        RBFEdgeFunctions(3, 2, use_base_act=False)   # no num_basis default anymore


def test_num_basis_min_two():
    with pytest.raises(ValueError):
        RBFEdgeFunctions(3, 2, num_basis=1, use_base_act=False)

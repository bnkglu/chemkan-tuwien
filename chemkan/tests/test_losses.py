import pytest
import torch

from chemkan.losses import (
    chemkan_loss,
    element_conservation_loss,
    trajectory_mse,
)


def test_trajectory_mse_zero_when_equal():
    x = torch.rand(10, 4, 3)
    assert trajectory_mse(x, x).item() == 0.0


def test_trajectory_mse_matches_documented_reduction():
    pred = torch.zeros(2, 3, 2)          # (T=2, B=3, n*=2)
    target = torch.ones(2, 3, 2)
    # per-state mean = 1.0 ; sum over T=2 -> 2.0 ; mean over B -> 2.0
    assert torch.isclose(trajectory_mse(pred, target), torch.tensor(2.0))


def test_element_conservation_zero_when_constant_in_time():
    Y = torch.rand(1, 4, 3).repeat(6, 1, 1)              # identical across timesteps
    ec = element_conservation_loss(Y, torch.rand(2, 3), torch.rand(2), torch.rand(3) + 1)
    assert torch.isclose(ec, torch.tensor(0.0), atol=1e-6)


def test_element_conservation_positive_when_drifting():
    Y = torch.rand(6, 4, 3)                               # varies across timesteps
    ec = element_conservation_loss(Y, torch.rand(2, 3), torch.rand(2), torch.rand(3) + 1)
    assert ec.item() > 0.0


def test_chemkan_loss_adds_weighted_pinn():
    pred = torch.rand(6, 4, 3)
    target = torch.rand(6, 4, 3)
    mse = trajectory_mse(pred, target)
    total = chemkan_loss(
        pred, target, use_pinn=True, alpha_pinn=1e-4,
        Y_phys=pred, element_counts=torch.rand(2, 3),
        atomic_weights=torch.rand(2), molar_weights=torch.rand(3) + 1,
    )
    assert total.item() >= mse.item()                    # PINN term is non-negative


def test_chemkan_loss_requires_use_pinn():
    pred = torch.rand(6, 4, 3)
    with pytest.raises(TypeError):
        chemkan_loss(pred, pred)                          # use_pinn is required


def test_chemkan_loss_pinn_without_alpha_raises():
    pred = torch.rand(6, 4, 3)
    with pytest.raises(ValueError):
        chemkan_loss(pred, pred, use_pinn=True,           # alpha_pinn missing
                     Y_phys=pred, element_counts=torch.rand(2, 3),
                     atomic_weights=torch.rand(2), molar_weights=torch.rand(3) + 1)


def test_chemkan_loss_mse_only_when_pinn_off():
    pred, target = torch.rand(6, 4, 3), torch.rand(6, 4, 3)
    assert torch.isclose(chemkan_loss(pred, target, use_pinn=False),
                         trajectory_mse(pred, target))

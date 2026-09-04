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


# --- Eq. 18 verified against a LITERAL transcription of the printed formula -------------
#
#   L = (1/n*) sum_{j=1..Nt} sum_{k=1..n*} ( u_hat^pred_k(t_j) - u_hat^obs_k(t_j) )^2
#     + alpha_PINN * sum_{i=1..Ne} sum_{j=1..Nt} | sum_{k=1..m} N_i^k W_i
#                                                  (Y^pred_{k,j} - Y^pred_{k,1}) / W_k |
#
# Written with explicit Python loops so the test cannot inherit a mistake from the
# vectorized implementation. Guards the reduction order, the placement of 1/n*, the
# absolute value AFTER the species sum, and the use of the PREDICTED initial state.

def _eq18_literal(u_pred, u_obs, Y_phys, N, W_i, W_k, alpha, b):
    Nt, _, n_star = u_pred.shape
    m = Y_phys.shape[-1]
    mse = 0.0
    for j in range(Nt):
        for k in range(n_star):
            mse += (u_pred[j, b, k] - u_obs[j, b, k]) ** 2
    mse = mse / n_star
    pinn = 0.0
    for i in range(N.shape[0]):
        for j in range(Nt):
            inner = 0.0
            for k in range(m):
                inner += N[i, k] * W_i[i] * (Y_phys[j, b, k] - Y_phys[0, b, k]) / W_k[k]
            pinn += abs(inner)
    return mse + alpha * pinn


def test_chemkan_loss_matches_literal_eq18_transcription():
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from _chemistry import ATOMIC_WEIGHTS, ELEMENT_COUNTS, MOLAR_WEIGHTS, SPECIES

    torch.manual_seed(0)
    Nt, B, m = 5, 3, len(SPECIES)
    n_star, alpha = m + 1, 1e-4
    u_pred = torch.rand(Nt, B, n_star, dtype=torch.float64)
    u_obs = torch.rand(Nt, B, n_star, dtype=torch.float64)
    Y_phys = torch.rand(Nt, B, m, dtype=torch.float64) * 0.1
    N = ELEMENT_COUNTS.double(); W_i = ATOMIC_WEIGHTS.double(); W_k = MOLAR_WEIGHTS.double()

    expected = sum(_eq18_literal(u_pred, u_obs, Y_phys, N, W_i, W_k, alpha, b)
                   for b in range(B)) / B          # Eq. 18 is per trajectory; we mean over B
    actual = chemkan_loss(u_pred, u_obs, use_pinn=True, alpha_pinn=alpha, Y_phys=Y_phys,
                          element_counts=N, atomic_weights=W_i, molar_weights=W_k)
    assert torch.isclose(actual, torch.as_tensor(expected), rtol=0, atol=1e-12)


def test_pinn_reference_is_the_PREDICTED_initial_state_not_the_observation():
    """Eq. 18's element term differences Y^pred_{k,j} against Y^pred_{k,1}."""
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from _chemistry import ATOMIC_WEIGHTS, ELEMENT_COUNTS, MOLAR_WEIGHTS

    torch.manual_seed(1)
    Y = torch.rand(4, 2, ELEMENT_COUNTS.shape[1], dtype=torch.float64) * 0.1
    loss_a = element_conservation_loss(Y, ELEMENT_COUNTS.double(), ATOMIC_WEIGHTS.double(),
                                       MOLAR_WEIGHTS.double())
    shifted = Y + 0.05                      # shifting ALL timesteps leaves the drift unchanged
    loss_b = element_conservation_loss(shifted, ELEMENT_COUNTS.double(),
                                       ATOMIC_WEIGHTS.double(), MOLAR_WEIGHTS.double())
    assert torch.isclose(loss_a, loss_b, rtol=0, atol=1e-12)

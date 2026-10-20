r"""ChemKAN loss terms (ChemKAN Eq. 18):  L = L_MSE + alpha_PINN * L_PINN.

Two inputs live on DIFFERENT scales, deliberately:

* trajectory_mse            -- uses [0, 1]-NORMALIZED states (Eq. 18 MSE term).
* element_conservation_loss -- uses PHYSICAL (denormalized) species mass fractions.

Reduction (PyTorch implementation assumption, documented)
---------------------------------------------------------
Eq. 18's MSE divides the state sum by ``n*`` and shows a sum over the ``Nt``
timesteps. We implement that literally per trajectory -- ``(1/n*) sum_k`` (a mean
over the state axis) then ``sum`` over timesteps -- and then AVERAGE over the batch
of trajectories. The PINN term is summed over elements and timesteps and likewise
averaged over the batch, so BOTH terms share the same per-trajectory-then-mean
normalization before ``alpha_PINN`` weights them.
"""

from __future__ import annotations

import torch


def trajectory_mse(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    r"""Eq. 18 MSE on normalized states.  pred/target: (T, B, n*) -> scalar.

    ``n*`` (= m for Stage 1, m+1 for Stage 2) is implicit in the last axis.
    """
    if pred_norm.shape != target_norm.shape:
        raise ValueError(f"shape mismatch: {pred_norm.shape} vs {target_norm.shape}")
    per_state = ((pred_norm - target_norm) ** 2).mean(dim=-1)   # (1/n*) sum_k -> (T, B)
    per_traj = per_state.sum(dim=0)                             # sum over timesteps -> (B,)
    return per_traj.mean()                                      # mean over trajectories


def element_conservation_loss(Y_phys: torch.Tensor, element_counts: torch.Tensor,
                              atomic_weights: torch.Tensor, molar_weights: torch.Tensor
                              ) -> torch.Tensor:
    r"""Eq. 18 PINN term on PHYSICAL species mass fractions.

    Elemental mass fraction of element i:  z_i = sum_k N_i^k * W_i * Y_k / W_k.
    Penalise its drift from the initial state, summed over elements and timesteps,
    averaged over trajectories.

        Y_phys         : (T, B, m)  physical (denormalized) mass fractions
        element_counts : (Ne, m)    N_i^k  (atoms of element i in species k)
        atomic_weights : (Ne,)      W_i
        molar_weights  : (m,)       W_k
    """
    coeff = element_counts * atomic_weights[:, None] / molar_weights[None, :]   # (Ne, m)
    z = torch.einsum("tbk,ik->tbi", Y_phys, coeff)             # (T, B, Ne) elemental mass
    drift = (z - z[0:1]).abs()                                 # vs initial state -> (T,B,Ne)
    return drift.sum(dim=(0, 2)).mean()                        # sum time+elements, mean batch


def chemkan_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor, *,
                 use_pinn: bool, alpha_pinn: float | None = None,
                 Y_phys: torch.Tensor | None = None,
                 element_counts: torch.Tensor | None = None,
                 atomic_weights: torch.Tensor | None = None,
                 molar_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Total Eq. 18 loss.

    ``use_pinn`` is required (this reusable code never silently decides whether the
    physics term is active). ``alpha_pinn`` (a hydrogen experiment choice, e.g. 1e-4)
    and the element/weight tensors are required only when ``use_pinn`` is True; when
    False they are ignored.
    """
    loss = trajectory_mse(pred_norm, target_norm)
    if use_pinn:
        if alpha_pinn is None:
            raise ValueError("use_pinn=True requires an explicit alpha_pinn")
        if any(v is None for v in (Y_phys, element_counts, atomic_weights, molar_weights)):
            raise ValueError("use_pinn=True requires Y_phys and the element/weight tensors")
        loss = loss + alpha_pinn * element_conservation_loss(
            Y_phys, element_counts, atomic_weights, molar_weights)
    return loss

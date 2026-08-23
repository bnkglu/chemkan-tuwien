"""The single shared Gaussian RBF primitive (ChemKAN Eq. 12).

Both ``RBFActivation`` (scalar, educational) and ``RBFEdgeFunctions`` (vectorized,
production) call this, so the formula lives in exactly one place and the two cannot
drift apart.
"""

from __future__ import annotations

import torch


def gaussian(x: torch.Tensor, centers: torch.Tensor, h: float) -> torch.Tensor:
    r"""Gaussian radial basis  psi(r) = exp(-r^2 / (2 h^2)),  r = x - c_k.

    A trailing basis axis is appended, so this serves inputs of any rank:

        x (batch,)   -> (batch, num_basis)
        x (B, in)    -> (B, in, num_basis)

    The denominator is ``2 * h**2`` exactly (ChemKAN Eq. 12).
    """
    r = x.unsqueeze(-1) - centers                 # broadcast x against the centers grid
    return torch.exp(-r ** 2 / (2 * h ** 2))

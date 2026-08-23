"""KAN layers built by COMPOSITION over ``RBFEdgeFunctions`` (no RBF math duplicated).

    AddKANLayer   -- additive node (ChemKAN Eq. 7; LeanKAN Eq. 10 with n_mu=0)
    LeanKANLayer  -- multiply the first n_mu inputs, add the rest (LeanKAN Eq. 8-10)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .rbf import RBFEdgeFunctions


class AddKANLayer(nn.Module):
    r"""Additive KAN layer:  y_i = sum_j phi_{i,j}(x_j)  (ChemKAN Eq. 7)."""

    def __init__(self, in_features: int, out_features: int, num_basis: int,
                 grid: tuple[float, float] = (-1.0, 1.0), *, use_base_act: bool):
        super().__init__()
        self.edges = RBFEdgeFunctions(in_features, out_features, num_basis, grid,
                                      use_base_act=use_base_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in) -> (B, out)
        return self.edges(x).sum(dim=-1)                  # sum over inputs j


class LeanKANLayer(nn.Module):
    r"""LeanKAN layer (Eq. 8-10): product of the first ``n_mu`` INPUTS + sum of the rest.

        y_i = ( prod_{j<n_mu} phi_{i,j}(x_j) ) + ( sum_{j>=n_mu} phi_{i,j}(x_j) )

    ``n_mu`` partitions the INPUT axis, not the outputs. n_mu=0 reduces exactly to
    AddKAN -- and we return the additive term ALONE in that case, because an empty
    ``prod`` over the last axis yields 1 in PyTorch, which would wrongly add a
    constant 1 to every output.
    """

    def __init__(self, in_features: int, out_features: int, n_mu: int,
                 num_basis: int, grid: tuple[float, float] = (-1.0, 1.0),
                 *, use_base_act: bool):
        super().__init__()
        if not 0 <= n_mu <= in_features:
            raise ValueError(
                f"n_mu must satisfy 0 <= n_mu <= in_features ({in_features}); got {n_mu}"
            )
        self.n_mu = n_mu
        self.edges = RBFEdgeFunctions(in_features, out_features, num_basis, grid,
                                      use_base_act=use_base_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in) -> (B, out)
        g = self.edges(x)                                 # (B, out, in)
        add = g[..., self.n_mu:].sum(dim=-1)              # sum the remaining inputs
        if self.n_mu == 0:
            return add                                    # explicit: no multiplicative term
        mult = g[..., :self.n_mu].prod(dim=-1)            # product of first n_mu inputs
        return mult + add

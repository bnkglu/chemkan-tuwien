r"""RBF activations (ChemKAN Eq. 11-12).

    phi(x) = sum_i w^psi_i * psi(|x - c_i|)  [ + w^b * b(x) ]

Two concepts live here:

* ``RBFActivation``     -- ONE learnable univariate phi(x); for teaching, tests,
                           and plotting a single learned edge function.
* ``RBFEdgeFunctions``  -- the vectorized production block that evaluates ALL
                           (out x in) edge functions of a layer at once.

Base-path ambiguity (documented, not silently resolved)
-------------------------------------------------------
ChemKAN Eq. 11 adds a Swish/SiLU base path ``w^b * b(x)`` to every edge. However
the paper's *reported parameter counts* (biodiesel 156, hydrogen 344) only match
if that base path is NOT present (base ON would give 208 / 411). We therefore make
``use_base_act`` a REQUIRED, keyword-only argument with no default:

    use_base_act=False  -> count-matching main reproduction (no w_base created).
    use_base_act=True   -> literal-Eq.-11 sensitivity experiment (SiLU base path).

When ``use_base_act=False`` we ``register_parameter("w_base", None)`` so there is
no trainable parameter to count or to drift during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._common import gaussian


def _grid(num_basis: int, lo: float, hi: float) -> tuple[torch.Tensor, float]:
    if num_basis < 2:
        raise ValueError("num_basis must be >= 2 (need >=2 points for a grid)")
    centers = torch.linspace(lo, hi, num_basis)
    h = (hi - lo) / (num_basis - 1)          # spacing = bump width; kept a Python float
    return centers, h


class RBFActivation(nn.Module):
    """One learnable univariate activation phi(x). Educational / test / plotting use.

    Unlike ``RBFEdgeFunctions`` this does NOT apply ``tanh`` -- it is the raw scalar
    function so you can plot phi over an arbitrary x range.
    """

    def __init__(self, num_basis: int, x_min: float = -1.0, x_max: float = 1.0,
                 *, use_base_act: bool):
        super().__init__()
        centers, self.h = _grid(num_basis, x_min, x_max)
        self.register_buffer("centers", centers)              # fixed model state
        self.w_rbf = nn.Parameter(torch.randn(num_basis) * 0.1)
        if use_base_act:
            self.base = nn.SiLU()
            self.w_base = nn.Parameter(torch.zeros(1))
        else:
            self.base = None
            self.register_parameter("w_base", None)           # not trainable, not counted

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch,) -> (batch,)
        out = gaussian(x, self.centers, self.h) @ self.w_rbf
        if self.w_base is not None:
            out = out + self.w_base * self.base(x)
        return out


class RBFEdgeFunctions(nn.Module):
    r"""Vectorized (out x in) grid of edge functions phi_{i,j} (ChemKAN Eq. 7, 11-12).

    forward:  x (B, in_features)  ->  edge_values (B, out_features, in_features)
    with  edge_values[b, i, j] = phi_{i, j}(x[b, j]).

    The layer input is normalized with ``tanh`` first (KAN internal normalization --
    distinct from dataset min-max normalization) so every input lands on the fixed
    center grid.
    """

    def __init__(self, in_features: int, out_features: int, num_basis: int,
                 grid: tuple[float, float] = (-1.0, 1.0), *, use_base_act: bool):
        super().__init__()
        centers, self.h = _grid(num_basis, grid[0], grid[1])
        self.register_buffer("centers", centers)              # fixed model state
        self.w_rbf = nn.Parameter(torch.randn(out_features, in_features, num_basis) * 0.1)
        if use_base_act:
            self.base = nn.SiLU()
            self.w_base = nn.Parameter(torch.zeros(out_features, in_features))
        else:
            self.base = None
            self.register_parameter("w_base", None)           # not trainable, not counted

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in) -> (B, out, in)
        x = torch.tanh(x)                                     # KAN internal normalization
        psi = gaussian(x, self.centers, self.h)              # (B, in, num_basis)
        edge = torch.einsum("bik,oik->boi", psi, self.w_rbf)  # (B, out, in)
        if self.w_base is not None:
            edge = edge + torch.einsum("bi,oi->boi", self.base(x), self.w_base)
        return edge

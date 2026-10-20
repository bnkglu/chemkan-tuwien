r"""Dataset (state) normalization -- the [0, 1] min-max of ChemKAN Eq. 18.

This is DISTINCT from the KAN-internal ``tanh`` applied inside ``RBFEdgeFunctions``:

* MinMaxNormalizer -- scales physical thermochemical states to [0, 1] using
  TRAINING-set statistics, for the Eq. 18 trajectory MSE.
* tanh             -- squashes a layer's inputs onto the RBF center grid, inside the
  network. Not a dataset statistic.

``MinMaxNormalizer`` is an ``nn.Module`` whose statistics are registered buffers, so
``normalizer.to(device)`` moves them alongside the model and normalization works on
CUDA/MPS. The buffers are NOT trainable parameters.

The repository's data generators already store train-only ``u_min`` / ``u_max`` in
each ``.npz`` (metadata: "train-only min-max (Eq. 18)"); reuse them via ``from_npz``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _as_float(v: torch.Tensor) -> torch.Tensor:
    """Tensor unchanged in device; only non-float dtypes are cast to the default float."""
    t = torch.as_tensor(v)
    if not torch.is_floating_point(t):
        t = t.to(torch.get_default_dtype())
    return t


class MinMaxNormalizer(nn.Module):
    r"""Affine map  x_hat = (x - u_min) / (u_max - u_min)  and its inverse.

    ``u_min`` / ``u_max`` are per-column (shape ``(n_vars,)``), fitted on the TRAINING
    set only, and broadcast over any leading (time, batch) axes. Stored as buffers so
    device/dtype follow normal ``.to(...)`` semantics.
    """

    def __init__(self, u_min: torch.Tensor, u_max: torch.Tensor, eps: float = 1e-12):
        super().__init__()
        u_min = _as_float(u_min)
        u_max = _as_float(u_max)
        if u_min.shape != u_max.shape:
            raise ValueError("u_min and u_max must have the same shape")
        self.register_buffer("u_min", u_min)
        self.register_buffer("u_max", u_max)
        self.register_buffer("range", (u_max - u_min).clamp_min(eps))  # guard zero range

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.u_min) / self.range

    def denormalize(self, x_hat: torch.Tensor) -> torch.Tensor:
        return x_hat * self.range + self.u_min

    def subset(self, columns: slice | list[int]) -> MinMaxNormalizer:
        """A normalizer over a column subset (e.g. species-only for Stage 1).

        Slicing preserves device and dtype of the current buffers.
        """
        return MinMaxNormalizer(self.u_min[columns], self.u_max[columns])

    @classmethod
    def from_npz(cls, npz, u_min_key: str = "u_min", u_max_key: str = "u_max"
                 ) -> MinMaxNormalizer:
        """Build from an ``np.load(...)`` archive; cast to the default float dtype."""
        u_min = torch.as_tensor(npz[u_min_key], dtype=torch.get_default_dtype())
        u_max = torch.as_tensor(npz[u_max_key], dtype=torch.get_default_dtype())
        return cls(u_min, u_max)

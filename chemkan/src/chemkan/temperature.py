r"""Batch-aware temperature providers used by the kinetic dynamics.

Both providers are called by the ODE solver as ``T = provider(t)`` at internal
solver times and MUST return shape ``(B, 1)`` -- one temperature per trajectory in
the batch. A single scalar temperature for the whole batch is never assumed.

* ConstantTemperature  -- isothermal biodiesel: each trajectory has its own fixed T.
* ObservedTemperature  -- hydrogen Stage 1: T(t) read from the training data while
                          only species are integrated.

PyTorch implementation assumption
---------------------------------
The paper does not specify how the observed Stage-1 temperature is evaluated at the
solver's adaptive internal times. ObservedTemperature LINEARLY INTERPOLATES between
saved training times (and clamps to the endpoints). This is our choice, not a
paper-specified scheme.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConstantTemperature(nn.Module):
    """Per-trajectory constant temperature. ``temperature`` is ``(B, 1)`` (or ``(B,)``)."""

    def __init__(self, temperature: torch.Tensor):
        super().__init__()
        temperature = torch.as_tensor(temperature, dtype=torch.get_default_dtype())
        if temperature.ndim == 1:
            temperature = temperature.unsqueeze(-1)          # (B,) -> (B, 1)
        if temperature.ndim != 2 or temperature.shape[-1] != 1:
            raise ValueError("temperature must be (B,) or (B, 1)")
        self.register_buffer("temperature", temperature)

    def forward(self, t: torch.Tensor) -> torch.Tensor:      # t ignored -> (B, 1)
        return self.temperature


class ObservedTemperature(nn.Module):
    r"""Temperature read from data and interpolated at solver time ``t``.

        saved_times  : (T,)      strictly increasing
        temperatures : (T, B, 1) observed T for each trajectory at each saved time

    ``forward(t)`` returns ``(B, 1)`` at a scalar solver time via linear
    interpolation between the two bracketing saved times (endpoints clamped).
    """

    def __init__(self, saved_times: torch.Tensor, temperatures: torch.Tensor):
        super().__init__()
        saved_times = torch.as_tensor(saved_times, dtype=torch.get_default_dtype())
        temperatures = torch.as_tensor(temperatures, dtype=torch.get_default_dtype())
        if temperatures.ndim == 2:                           # (T, B) -> (T, B, 1)
            temperatures = temperatures.unsqueeze(-1)
        if saved_times.ndim != 1:
            raise ValueError("saved_times must be 1-D (T,)")
        if temperatures.ndim != 3 or temperatures.shape[-1] != 1:
            raise ValueError("temperatures must be (T, B) or (T, B, 1)")
        if temperatures.shape[0] != saved_times.shape[0]:
            raise ValueError("temperatures.shape[0] must equal saved_times.shape[0] (T)")
        if not torch.all(saved_times[1:] > saved_times[:-1]):
            raise ValueError("saved_times must be strictly increasing")
        self.register_buffer("saved_times", saved_times)
        self.register_buffer("temperatures", temperatures)

    def forward(self, t: torch.Tensor) -> torch.Tensor:      # scalar t -> (B, 1)
        times = self.saved_times
        t = torch.as_tensor(t, dtype=times.dtype, device=times.device)
        t = t.clamp(times[0], times[-1])
        # right-bracket index in [1, T-1], kept as a tensor to avoid a per-step
        # device->host sync (no int()/.item()); index_select gathers on-device.
        hi = torch.searchsorted(times, t.reshape(1), right=True).clamp(1, times.numel() - 1)
        lo = hi - 1
        t0, t1 = times.index_select(0, lo), times.index_select(0, hi)   # each (1,)
        w = ((t - t0) / (t1 - t0)).view(1, 1, 1)                        # weight in [0, 1]
        T0 = self.temperatures.index_select(0, lo)                      # (1, B, 1)
        T1 = self.temperatures.index_select(0, hi)
        return (T0 + w * (T1 - T0)).squeeze(0)                          # (B, 1)

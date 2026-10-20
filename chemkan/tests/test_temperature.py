import pytest
import torch

from chemkan.temperature import ConstantTemperature, ObservedTemperature


def test_constant_temperature_batch_shape_and_per_trajectory():
    T = torch.tensor([300.0, 350.0, 400.0])          # (B,)
    provider = ConstantTemperature(T)
    out = provider(torch.tensor(1.234))
    assert out.shape == (3, 1)                        # (B, 1)
    assert not torch.allclose(out[0], out[1])         # trajectories differ


def test_observed_temperature_returns_batch_shape():
    times = torch.linspace(0.0, 1.0, 5)
    temps = torch.randn(5, 4, 1)                      # (T, B, 1)
    provider = ObservedTemperature(times, temps)
    assert provider(torch.tensor(0.5)).shape == (4, 1)


def test_observed_temperature_exact_at_saved_time():
    times = torch.tensor([0.0, 1.0, 2.0])
    temps = torch.tensor([[300.0], [400.0], [500.0]]).unsqueeze(1)   # (3, 1, 1)
    provider = ObservedTemperature(times, temps)
    assert torch.allclose(provider(torch.tensor(1.0)), torch.tensor([[400.0]]))


def test_observed_temperature_linear_midpoint():
    times = torch.tensor([0.0, 2.0])
    temps = torch.tensor([300.0, 500.0]).reshape(2, 1, 1)            # (2, 1, 1)
    provider = ObservedTemperature(times, temps)
    assert torch.allclose(provider(torch.tensor(1.0)), torch.tensor([[400.0]]))


def test_observed_temperature_clamps_outside_range():
    times = torch.tensor([0.0, 1.0])
    temps = torch.tensor([300.0, 400.0]).reshape(2, 1, 1)
    provider = ObservedTemperature(times, temps)
    assert torch.allclose(provider(torch.tensor(5.0)), torch.tensor([[400.0]]))


def test_observed_temperature_distinct_per_trajectory():
    times = torch.tensor([0.0, 1.0])
    temps = torch.tensor([[300.0, 600.0], [400.0, 800.0]]).reshape(2, 2, 1)
    provider = ObservedTemperature(times, temps)
    out = provider(torch.tensor(0.5))
    assert not torch.allclose(out[0], out[1])


# --- validation ------------------------------------------------------------------

def test_observed_temperature_accepts_TB_and_adds_last_dim():
    times = torch.tensor([0.0, 1.0])
    temps = torch.tensor([[300.0, 600.0], [400.0, 800.0]])   # (T, B), no trailing 1
    provider = ObservedTemperature(times, temps)
    assert provider.temperatures.shape == (2, 2, 1)          # (T, B) -> (T, B, 1)
    assert provider(torch.tensor(0.5)).shape == (2, 1)       # output (B, 1)


def test_observed_temperature_rejects_bad_final_dim():
    times = torch.tensor([0.0, 1.0])
    temps = torch.zeros(2, 3, 2)                             # last dim != 1
    with pytest.raises(ValueError):
        ObservedTemperature(times, temps)


def test_observed_temperature_rejects_non_monotonic_times():
    times = torch.tensor([0.0, 1.0, 0.5])                    # not strictly increasing
    temps = torch.zeros(3, 2, 1)
    with pytest.raises(ValueError):
        ObservedTemperature(times, temps)


def test_observed_temperature_rejects_length_mismatch():
    times = torch.tensor([0.0, 1.0, 2.0])
    temps = torch.zeros(2, 2, 1)                             # T mismatch
    with pytest.raises(ValueError):
        ObservedTemperature(times, temps)


# --- unresolved reproduction issue: physical T saturates tanh (see ASSUMPTIONS.md) --

def test_raw_physical_temperature_saturates_tanh_DIAGNOSTIC():
    """DIAGNOSTIC (does not change model behavior): physical combustion temperatures
    all collapse to ~1.0 under tanh, so temperature dependence is nearly erased at the
    first KAN layer. This documents an UNRESOLVED reproduction ambiguity."""
    temps = torch.tensor([323.0, 343.0, 1000.0, 1400.0])
    saturated = torch.tanh(temps)
    assert torch.allclose(saturated, torch.ones_like(saturated), atol=1e-6)

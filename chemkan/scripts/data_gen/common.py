"""
Small shared utilities for the ChemKAN reproduction data generators.

This module holds only the little pieces that the biodiesel and hydrogen
generators (and the optional methane extension) genuinely repeat: the RNG
entry point, min/max normalization, the noise helper, a couple of diagnostics,
and .npz I/O. It is intentionally not a framework.

Conventions used across all generators
--------------------------------------
* A "case" is one trajectory produced from one initial condition.
* Trajectories are stored as `states` of shape (n_cases, n_times, n_vars).
  The layout of the last axis differs by system, and every archive records it
  under the key `state_layout`:

  - "species_then_temperature"  (hydrogen, methane): u = [Y_1, ..., Y_m, T],
    temperature last, matching Eqs. 1-2 of Koenig, Kim & Deng (2025).

  - "species_only"  (biodiesel): u = [Y_1, ..., Y_m]. The process is isothermal,
    so T is not integrated. It is stored per case in `train_T` / `test_T`.
    Eq. 13 still feeds the full u = [u~, T] into the kinetic core, so a training
    loader broadcasts T across the time axis -- see `with_temperature` below.
* Each system is saved as one compressed .npz. Every archive also records a
  concise `metadata` string (system, generator, seed, mechanism, species, time
  grid, normalization, library versions). Explanations live in the data-gen
  README, not in the .npz.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def make_rng(seed: int) -> np.random.Generator:
    """Single entry point for randomness. Never use np.random.* directly."""
    return np.random.default_rng(seed)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _versions() -> dict:
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("scipy", "cantera"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"
    return versions


def metadata(**fields) -> str:
    """Concise technical metadata embedded in each .npz, as a JSON string.

    Records only technical facts -- system, generator, seed, mechanism, species,
    time grid, normalization method, library versions, git commit -- not
    explanations. Any discussion of paper ambiguities belongs in the
    data-generation README, not here. Pass the per-system fields as keyword
    arguments; library versions and the git commit are added automatically.
    """
    base = {"git_commit": _git_commit(), "versions": _versions()}
    base.update(fields)
    return json.dumps(base, indent=2, default=str)


# --------------------------------------------------------------------------
# State-vector assembly
# --------------------------------------------------------------------------


def with_temperature(states: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Append a per-case constant temperature as the final state column.

    For the isothermal biodiesel system, temperature is an *input* to the
    kinetic core (Eq. 13 takes u = [u~, T]) but not an integrated state. This
    helper broadcasts the per-case scalar T across the time axis so the model
    input has the same [Y_1, ..., Y_m, T] layout as the combustion cases.

        states : (n_cases, n_times, m)
        T      : (n_cases,)
        ->       (n_cases, n_times, m + 1)

    Do not feed the appended column back through the ODE integrator: its
    derivative is identically zero.
    """
    if states.shape[0] != T.shape[0]:
        raise ValueError(f"case-count mismatch: {states.shape[0]} vs {T.shape[0]}")
    col = np.broadcast_to(T[:, None, None], (*states.shape[:2], 1))
    return np.concatenate([states, col], axis=-1)


# --------------------------------------------------------------------------
# Normalization  (u_hat = (u - min) / (max - min), fitted on TRAIN only)
# --------------------------------------------------------------------------


def fit_minmax(train_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-variable min/max over all training cases and times (train-only).

    Returns (u_min, u_max), each of shape (n_vars,). Guards against a zero
    range for variables that never change (e.g. N2 mass fraction).
    """

    # Compute one min/max pair per state variable using only clean training data.
    # Axis convention: states has shape (n_cases, n_times, n_variables).
    # We reduce over cases and time, not over variables.
    # This avoids test-set leakage and keeps one consistent scale for train/test.

    u_min = train_states.min(axis=(0, 1))
    u_max = train_states.max(axis=(0, 1))

    flat = (u_max - u_min) < 1e-12
    u_max = np.where(flat, u_min + 1.0, u_max)
    return u_min, u_max


def normalize(states: np.ndarray, u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    return (states - u_min) / (u_max - u_min)


def denormalize(states_hat: np.ndarray, u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    return states_hat * (u_max - u_min) + u_min


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


def add_noise(
    states: np.ndarray,
    level: float,
    rng: np.random.Generator,
    mode: str = "multiplicative",
    u_min: np.ndarray | None = None,
    u_max: np.ndarray | None = None,
) -> np.ndarray:
    """Add synthetic noise to the trajectories.

    ChemKAN reports experiments with synthetic noise up to 15%, but the exact
    noise distribution is not explicitly specified. This implementation uses
    multiplicative Gaussian noise as a documented implementation choice.

    mode="multiplicative" (default): u_noisy = u * (1 + level * N(0,1)).
        Noise scales with the local signal, so near-zero species stay near
        zero.

    mode="range": u_noisy = u + level * (u_max - u_min) * N(0,1).
        Noise is a fixed fraction of each variable's dynamic range. Provided
        as an alternative for a sensitivity check; requires u_min/u_max.

    The initial condition (t=0) is left clean in both modes -- it is an input
    to the model, not an observation.
    """
    if level == 0.0:
        return states.copy()

    noisy = states.copy()
    if mode == "multiplicative":
        noisy = states * (1.0 + level * rng.standard_normal(states.shape))
    elif mode == "range":
        if u_min is None or u_max is None:
            raise ValueError("mode='range' requires u_min and u_max")
        noisy = states + level * (u_max - u_min) * rng.standard_normal(states.shape)
    else:
        raise ValueError(f"unknown noise mode: {mode!r}")

    noisy[:, 0, :] = states[:, 0, :]  # keep initial conditions exact
    return noisy


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def ignition_delay(t: np.ndarray, T: np.ndarray) -> float:
    """Ignition delay = time of maximum temperature rise rate (ChemNODE def.).

    Returns np.nan if the mixture does not ignite within the window, defined
    here as a total temperature rise below 100 K.
    """
    if T.max() - T[0] < 100.0:
        return float("nan")
    return float(t[np.argmax(np.gradient(T, t))])


def stiffness_ratio(t: np.ndarray, states: np.ndarray) -> float:
    """Crude stiffness proxy: ratio of slowest to fastest resolved time scale.

    A rough diagnostic only -- the ratio of resolved time scales, not a
    Jacobian eigenvalue spread. Reported to give a rough sense of relative
    stiffness, not as a rigorous stiffness measure.
    """
    d = np.abs(np.gradient(states, t, axis=0))
    scale = np.abs(states).max(axis=0) + 1e-30
    tau = scale / (d.max(axis=0) + 1e-30)
    return float(tau.max() / tau.min())


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def save(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    size = path.stat().st_size / 1e6
    print(f"  wrote {path}  ({size:.1f} MB)")
    return path


def load(path: str | Path) -> dict:
    """Load every array from a generated .npz into a plain dict.

    Inspect `d["state_layout"]` before assuming the last column is temperature,
    `d["mechanism"]` for the Cantera mechanism, and `d["metadata"]` for the
    concise generation metadata (seed, time grid, library versions, ...).
    """
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

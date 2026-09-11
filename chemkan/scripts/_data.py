"""Thin loader for the repository's generated .npz datasets (no data is regenerated).

This is NOT a generic data framework -- it only points at the archives already
produced by ``chemkan/scripts/data_gen`` and reshapes them into the ``(T, B, ...)``
tensors the training code expects. The default location is ``chemkan/data/generated``
(next to this package); override ``DATA_DIR`` via the ``CHEMKAN_DATA_DIR`` environment
variable if your checkout lives elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

# ``chemkan`` is imported as an installed package (``pip install -e .`` from chemkan/);
# no sys.path manipulation is needed.
from chemkan.normalization import MinMaxNormalizer

_DEFAULT = Path(__file__).resolve().parents[1] / "data/generated"
DATA_DIR = Path(os.environ.get("CHEMKAN_DATA_DIR", _DEFAULT))


def resolve_device(name: str = "cpu") -> torch.device:
    """Resolve a --device string to an available torch.device (falls back to cpu)."""
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load(name: str):
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Set CHEMKAN_DATA_DIR or run the repo's data generators."
        )
    return np.load(path, allow_pickle=True)


def _to_TBx(states: np.ndarray) -> torch.Tensor:
    """(cases, times, vars) -> (T, B, vars) as float32."""
    return torch.as_tensor(states, dtype=torch.float32).permute(1, 0, 2).contiguous()


def _check_split(split: str) -> None:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")


def input_scaling_meta(method: str, full_normalizer: MinMaxNormalizer) -> dict:
    """Checkpoint metadata for the pre-KAN input scaling (see dynamics.py).

    'minmax' stores the EXACT full-state train statistics so evaluation reconstructs
    the same transform; 'none' stores only the method.
    """
    if method == "none":
        return {"method": "none"}
    if method == "minmax":
        return {"method": "minmax",
                "u_min": full_normalizer.u_min.detach().cpu(),
                "u_max": full_normalizer.u_max.detach().cpu()}
    raise ValueError(f"unknown input scaling method: {method!r}")


def load_input_scaling(ckpt: dict, device) -> MinMaxNormalizer | None:
    """Reconstruct the input normalizer from checkpoint metadata (never refit).

    Fails loudly if the mandatory representation/scaling metadata is missing or
    unsupported. Returns ``None`` for the 'none' (raw-input) method.
    """
    if "state_representation" not in ckpt or "input_scaling" not in ckpt:
        raise ValueError(
            "checkpoint missing required 'state_representation' / 'input_scaling' metadata")
    if ckpt["state_representation"] != "physical":
        raise ValueError(
            f"unsupported state_representation {ckpt['state_representation']!r} "
            "(only 'physical' is supported)")
    scaling = ckpt["input_scaling"]
    method = scaling.get("method")
    if method == "none":
        return None
    if method == "minmax":
        return MinMaxNormalizer(scaling["u_min"], scaling["u_max"]).to(device)
    raise ValueError(f"unknown input scaling method: {method!r}")


def noise_tag(noise_percent) -> str:
    """Archive key suffix for a noise level given in PERCENT (5 -> 'noise05')."""
    pct = int(round(float(noise_percent)))
    if abs(float(noise_percent) - pct) > 1e-9:
        raise ValueError(f"noise_percent must be a whole percent; got {noise_percent!r}")
    if pct < 0:
        raise ValueError(f"noise_percent must be >= 0; got {noise_percent!r}")
    return f"noise{pct:02d}"


def available_noise_percents() -> list[int]:
    """Whole-percent noise levels actually stored in biodiesel.npz, ascending."""
    d = _load("biodiesel")
    return sorted(int(k[len("train_states_noise"):]) for k in d.files
                  if k.startswith("train_states_noise"))


def load_biodiesel(split: str = "train", noise_percent=None):
    """Isothermal biodiesel (species_only). Returns a dict of tensors.

    ``u_min`` / ``u_max`` are ALWAYS the train-only statistics stored in the archive
    (Eq. 18), regardless of split -- test states are normalized with TRAIN stats.

    ``noise_percent`` selects a stored deterministic noise realization by WHOLE PERCENT
    (0, 1, 2, 3, 5, 7, 10, 15 in the current archive; ``available_noise_percents()`` is
    the authority). Nothing is generated here: the level
    must already exist in the archive, otherwise the loader fails loudly listing what is
    available. ``species_TBm`` always stays the CLEAN trajectory (paper Eq. 22's
    noise-free reference); the selected observations are returned separately as
    ``targets_TBm``, so a caller that ignores the new key keeps its previous behavior.
    ``Y0`` comes from the clean states -- the generator leaves t=0 exact in every noise
    array, and the initial condition is a model input, not an observation.
    """
    _check_split(split)
    d = _load("biodiesel")
    states = _to_TBx(d[f"{split}_states"])              # (T, B, m) clean
    targets = states
    if noise_percent is not None:
        key = f"{split}_states_{noise_tag(noise_percent)}"
        if key not in d.files:
            raise KeyError(
                f"{key} not in biodiesel.npz; stored levels are "
                f"{available_noise_percents()} percent. Add the missing level with "
                f"scripts/data_gen/add_biodiesel_noise_level.py -- never fabricate it here.")
        targets = _to_TBx(d[key])
    return {
        "t": torch.as_tensor(d["t"], dtype=torch.float32),          # (T,)
        "Y0": states[0],                                            # (B, m) always clean
        "species_TBm": states,                                      # (T, B, m) clean truth
        "targets_TBm": targets,                                     # (T, B, m) observations
        "noise_percent": None if noise_percent is None else int(round(float(noise_percent))),
        "T_const": torch.as_tensor(d[f"{split}_T"], dtype=torch.float32),  # (B,)
        "u_min": torch.as_tensor(d["u_min"], dtype=torch.float32),  # (m,) train-only
        "u_max": torch.as_tensor(d["u_max"], dtype=torch.float32),  # (m,) train-only
        "species": list(d["species"]),
    }


def load_hydrogen(split: str = "train"):
    """Hydrogen (species_then_temperature). Returns a dict of tensors.

    ``u_min`` / ``u_max`` are ALWAYS the train-only statistics (see ``load_biodiesel``).
    """
    _check_split(split)
    d = _load("hydrogen")
    full = _to_TBx(d[f"{split}_states"])                # (T, B, m+1)
    m = len(d["species"])
    return {
        "t": torch.as_tensor(d["t"], dtype=torch.float32),          # (T,)
        "m": m,
        "full_TBm1": full,                                          # (T, B, m+1)
        "species_TBm": full[..., :m],                              # (T, B, m)
        "T_obs_TB1": full[..., m:m + 1],                          # (T, B, 1)
        "u_min": torch.as_tensor(d["u_min"], dtype=torch.float32),  # (m+1,) train-only
        "u_max": torch.as_tensor(d["u_max"], dtype=torch.float32),  # (m+1,) train-only
        "species": list(d["species"]),
    }


def load_hydrogen_temperature(split: str = "train", n_points: int = 20000):
    """Dense precomputed Cantera Stage-1 temperature trajectory.

    Loads ``hydrogen_temperature_{n_points}.npz`` (produced by
    ``generate_hydrogen.py --temperature-only``) and returns the dense temperature
    as ``(N, B, 1)`` for feeding ``ObservedTemperature``. This is ONLY the external
    Stage-1 temperature provider: it carries no species trajectories, targets, or
    normalization -- the canonical 50-point ``hydrogen.npz`` remains the sole source
    of training targets and normalization statistics (never refit here).

    Validates against the canonical hydrogen archive and FAILS LOUDLY on any
    mismatch: strictly-increasing / finite times, matching time range, matching
    batch size, and identical initial-condition ordering.
    """
    _check_split(split)
    d = _load(f"hydrogen_temperature_{n_points}")

    t_dense = torch.as_tensor(d["t"], dtype=torch.float32)              # (N,)
    T_dense = torch.as_tensor(d[f"{split}_T"], dtype=torch.float32)     # (N, B, 1)
    ics = torch.as_tensor(d[f"{split}_ics"], dtype=torch.float32)       # (B, 2)

    # --- self-consistency of the dense file -----------------------------------
    if t_dense.ndim != 1:
        raise ValueError("dense temperature: t must be 1-D (N,)")
    if t_dense.shape[0] != n_points:
        raise ValueError(
            f"dense temperature: t has {t_dense.shape[0]} points but n_points={n_points} was "
            f"requested (wrong file for this resolution?)")
    if "n_points" in d.files and int(d["n_points"]) != n_points:
        raise ValueError(
            f"dense temperature: archive records n_points={int(d['n_points'])} but n_points="
            f"{n_points} was requested (mismatched cache).")
    if not torch.isfinite(t_dense).all() or not torch.isfinite(T_dense).all():
        raise ValueError("dense temperature: non-finite values present")
    if not torch.all(t_dense[1:] > t_dense[:-1]):
        raise ValueError("dense temperature: t must be strictly increasing")
    if T_dense.ndim != 3 or T_dense.shape[0] != t_dense.shape[0] or T_dense.shape[-1] != 1:
        raise ValueError(
            f"dense temperature: {split}_T must be (N, B, 1); got {tuple(T_dense.shape)}")

    # --- cross-check against the canonical 50-point hydrogen archive ----------
    canon = _load("hydrogen")
    t_canon = torch.as_tensor(canon["t"], dtype=torch.float32)
    if not (torch.isclose(t_dense[0], t_canon[0]) and torch.isclose(t_dense[-1], t_canon[-1])):
        raise ValueError(
            f"dense temperature time range [{float(t_dense[0]):.3e}, {float(t_dense[-1]):.3e}] "
            f"does not match canonical [{float(t_canon[0]):.3e}, {float(t_canon[-1]):.3e}]")
    canon_ics = torch.as_tensor(canon[f"{split}_ics"], dtype=torch.float32)
    if ics.shape != canon_ics.shape or not torch.allclose(ics, canon_ics):
        raise ValueError(
            f"dense temperature IC ordering/values do not match the canonical hydrogen "
            f"archive for split={split!r}; expected {tuple(canon_ics.shape)} in the same order")
    if T_dense.shape[1] != canon_ics.shape[0]:
        raise ValueError(
            f"dense temperature batch size {T_dense.shape[1]} != canonical {canon_ics.shape[0]}")

    return {
        "t_dense": t_dense,               # (N,)
        "T_dense_TB1": T_dense,           # (N, B, 1)
        "ics": ics,                       # (B, 2) = [T0, phi], canonical order
        "metadata": str(d["metadata"]),
    }

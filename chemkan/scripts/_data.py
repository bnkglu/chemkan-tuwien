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


def load_biodiesel(split: str = "train"):
    """Isothermal biodiesel (species_only). Returns a dict of tensors.

    ``u_min`` / ``u_max`` are ALWAYS the train-only statistics stored in the archive
    (Eq. 18), regardless of split -- test states are normalized with TRAIN stats.
    """
    _check_split(split)
    d = _load("biodiesel")
    states = _to_TBx(d[f"{split}_states"])              # (T, B, m)
    return {
        "t": torch.as_tensor(d["t"], dtype=torch.float32),          # (T,)
        "Y0": states[0],                                            # (B, m)
        "species_TBm": states,                                      # (T, B, m)
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

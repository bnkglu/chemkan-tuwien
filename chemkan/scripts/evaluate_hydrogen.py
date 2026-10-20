r"""Load a trained hydrogen/methane ChemKAN as a full ``[Y, T]`` predictor.

``load_predictor`` rebuilds a Stage-2 checkpoint (``stage2_final.pt`` or a Stage-2 archive
from ``train_author_matched_hydrogen.py``) and integrates it in the trainer's own normalized
coordinates, with its t_ref and solver tolerances, returning physical states. The figure and
evaluation scripts (Fig. 7, ``evaluate_hydrogen_grid.py``, ``evaluate_hydrogen_ignition.py``)
use it so every evaluation integrates the model exactly as it was trained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig, integrate


def load_predictor(path, device="cpu", *, rtol=None, atol=None, float64=False) -> dict:
    """A trained full ``[Y, T]`` model from a run directory or Stage-2 checkpoint file.

    The checkpoint is integrated in its own normalized coordinates with its t_ref and
    tolerances, and returned in physical units. ``predict(u0, t)``: (B, m+1) physical,
    (T,) -> (T, B, m+1) physical. ``rtol`` / ``atol`` replace the checkpoint's training
    tolerances; ``float64`` integrates in double precision (needed for tolerances below
    float32's ~1e-7) and returns the input dtype.
    """
    path = Path(path)
    if path.is_dir():
        if not (path / "stage2_final.pt").exists():
            raise FileNotFoundError(f"{path}: no stage2_final.pt")
        path = path / "stage2_final.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dtype = torch.float64 if float64 else None

    def tolerances(solver):
        return SolverConfig(method=solver.method, rtol=rtol or solver.rtol,
                            atol=atol or solver.atol, sensitivity=solver.sensitivity)

    def cast(fn):
        if dtype is None:
            return fn
        return lambda u0, t: fn(u0.to(dtype), t.to(dtype)).to(u0.dtype)

    sys.path.insert(0, str(Path(__file__).resolve().parent / "author_repo_match"))
    import train_author_matched_hydrogen as amh
    cfg = ckpt["config"]
    if ckpt.get("stage") != 2:
        raise ValueError(f"{path}: a Stage-2 checkpoint is needed for the full [Y, T] state")
    amh.configure_system(cfg.get("system", {}).get("name", "hydrogen"))
    model = amh.build_model().to(device=device, dtype=dtype or torch.float32)
    model.load_state_dict(ckpt["model_state"])
    norm = MinMaxNormalizer(torch.tensor(cfg["normalization"]["u_min"], dtype=dtype),
                            torch.tensor(cfg["normalization"]["u_max"], dtype=dtype)).to(device)
    dyn = amh.Stage2Dynamics(model, cfg["t_ref"]["value_s"])
    solver = tolerances(SolverConfig(method="tsit5", rtol=cfg["solver"]["rtol"],
                                     atol=cfg["solver"]["atol"], sensitivity="direct_autograd"))
    return {"predict": cast(lambda u0, t: norm.denormalize(
                integrate(dyn, norm.normalize(u0), t, solver))),
            "checkpoint": path, "ckpt": ckpt, "solver": solver,
            "architecture": cfg["architecture"]}

r"""Inputs for the Julia benchmark: ChemKAN weights, normalization, initial states, output grid,
tolerances and the PyTorch predictions/reference used to validate the Julia port.

Writes ``inputs_<system>.npz`` next to this file. Nothing is timed here.

    ~/uni_projects/chemkan-venv/bin/python results/inference_benchmark/julia/export_inputs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import benchmark as b  # noqa: E402
import reference_components as am  # noqa: E402


def export(system: str) -> None:
    ckpt = torch.load(b.ROOT / b.SYSTEMS[system]["checkpoint"], map_location="cpu", weights_only=False)
    p = b.BiodieselProblem(ckpt) if system == "biodiesel" else b.GasProblem(system, ckpt)
    sd = {k: v.detach().cpu().numpy() for k, v in ckpt["model_state"].items()}
    out = {"t": p.t, "u_min": p.u_min, "u_max": p.u_max,
           "is_test": p.is_test.astype(np.int64)}
    if system == "biodiesel":
        cfg = ckpt["config"]
        out |= {"W1": sd["add.edges.w_rbf"], "W2": sd["lean.edges.w_rbf"],
                "centers": sd["add.edges.centers"], "h": np.array(am.LIBRARY_H),
                "n_mu": np.array(int(np.ceil(sd["add.edges.w_rbf"].shape[0] / 2))),
                "rate_scale": np.array(1.0 / b.pt.rhs_divisor_from_config(cfg)),
                "u0": np.column_stack([p.y0, p.T]), "float32": np.array(0),
                "n_out": np.array(6)}
    else:
        out |= {"W1": sd["kinetic.add.edges.w_rbf"], "W2": sd["kinetic.lean.edges.w_rbf"],
                "Wlin": sd["thermo.linear.weight"][0], "Wcor": sd["thermo.correction.edges.w_rbf"],
                "centers": sd["kinetic.add.edges.centers"], "h": np.array(b.pt.amh.LIBRARY_H),
                "n_mu": np.array(b.pt.amh.N_MU),
                "rate_scale": np.array(1.0 / ckpt["config"]["t_ref"]["value_s"]),
                "u0": p.u0, "float32": np.array(1), "n_out": np.array(len(p.names)),
                "pressure": np.array(p.pressure), "T0": p.ics[:, 0], "phi": p.ics[:, 1],
                "Y0_full": np.array(p.y0), "keep": p.keep.astype(np.int64)}
    rt, at = p.reference_tol
    reference = np.stack([p.reference(i, rt, at) for i in range(len(p.ics))], axis=1)
    out["reference"] = reference                                      # (T, n, k) physical
    out["chemkan_tols"] = np.array(list(p.chemkan_tols.values()), dtype=float)
    for j, (rtol, atol) in enumerate(p.chemkan_tols.values()):
        out[f"torch_pred_{j}"] = np.concatenate(
            [p.chemkan([i], rtol, atol) for i in range(len(p.ics))], axis=1)
    np.savez(HERE / f"inputs_{system}.npz", **{k: np.asarray(v) for k, v in out.items()})
    print(system, {k: np.asarray(v).shape for k, v in out.items()})


if __name__ == "__main__":
    for s in sys.argv[1:] or ["hydrogen", "biodiesel"]:
        export(s)

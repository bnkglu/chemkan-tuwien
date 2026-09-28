r"""Phase 1: is the configured PyTorch core the SAME function as the released Julia KAN?

Loads the trained Julia coefficients (``julia_reference/``, written by
``export_julia_reference.jl``), maps them into ``_author_match.build_core()``, and compares,
ONE probe state at a time (batch of 1, as the Julia ODE calls ``kan1``):

    layer-1 output (4)  |  KAN output before /50 (6)  |  RHS after /50 (7)

Reports max absolute error, relative L2 error and cosine similarity (over all states, and
the worst single state). Also, as information only, integrates the Julia-trained parameters
with the Phase-2 solver settings and compares losses/predictions with Julia's own solve.

    python check_equivalence.py            # writes <EXP>/equivalence/equivalence.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _author_match import (EXP, N_TEST, N_TRAIN, NormalizedDynamics, build_core,  # noqa: E402
                           julia_mse, load_julia_params, load_julia_reference)
from chemkan.solver import SolverConfig, integrate  # noqa: E402

REL_L2_PASS = 1e-10          # float64 both sides; only rounding may differ


def compare(ours: np.ndarray, ref: np.ndarray) -> dict:
    diff = ours - ref
    per_state_rel = np.linalg.norm(diff, axis=1) / np.linalg.norm(ref, axis=1)
    per_state_cos = (ours * ref).sum(1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(ref, axis=1))
    return {
        "max_abs_error": float(np.abs(diff).max()),
        "rel_l2_error": float(np.linalg.norm(diff) / np.linalg.norm(ref)),
        "cosine_similarity": float((ours * ref).sum() / (np.linalg.norm(ours) * np.linalg.norm(ref))),
        "worst_state_rel_l2_error": float(per_state_rel.max()),
        "worst_state_cosine_similarity": float(per_state_cos.min()),
        "ref_abs_max": float(np.abs(ref).max()),
    }


def main() -> int:
    torch.set_default_dtype(torch.float64)
    ref = load_julia_reference()
    core = build_core().double()
    load_julia_params(core, ref["p"])
    dyn = NormalizedDynamics(core)

    layer1, kan_out, rhs = [], [], []
    with torch.no_grad():
        for z in torch.from_numpy(ref["states"]):
            z = z.reshape(1, 7)                              # one state at a time
            layer1.append(core.add(z)[0].numpy())
            kan_out.append(core(z)[0].numpy())
            rhs.append(dyn(torch.zeros(()), z)[0].numpy())
    result = {
        "n_probe_states": int(ref["states"].shape[0]),
        "probe_states": "50 normalized test states (experiments 21-30 at time indices "
                        "1,5,10,20,30 with normalized T) + 10 uniform states in [-0.2,1.2]^7",
        "layer1_output": compare(np.array(layer1), ref["layer1"]),
        "kan_output_before_div50": compare(np.array(kan_out), ref["kan_out"]),
        "rhs_after_div50": compare(np.array(rhs), ref["rhs"]),
        "pass_threshold_rel_l2": REL_L2_PASS,
    }
    result["rhs_equivalent"] = all(result[k]["rel_l2_error"] < REL_L2_PASS for k in
                                   ("layer1_output", "kan_output_before_div50", "rhs_after_div50"))

    # Information only: Julia-trained parameters integrated with the Phase-2 solver.
    solver = SolverConfig(method="tsit5", rtol=1e-2, atol=1e-6, sensitivity="direct_autograd")
    t = torch.from_numpy(ref["t"])
    u0 = torch.from_numpy(ref["u0_norm"])
    with torch.no_grad():
        pred = torch.stack([integrate(dyn, u0[i:i + 1], t, solver)[:, 0, :6].T
                            for i in range(N_TRAIN + N_TEST)])       # (30, 6, T)
    target = torch.from_numpy(ref["normdata"][:N_TRAIN + N_TEST])
    jpred = ref["pred_final_p"][:N_TRAIN + N_TEST]
    result["julia_params_in_pytorch_solver"] = {
        "solver": "torchdiffeq tsit5, rtol 1e-2, atol 1e-6, one trajectory per solve",
        "train_mse_pytorch": float(julia_mse(pred[:N_TRAIN], target[:N_TRAIN])),
        "val_mse_pytorch": float(julia_mse(pred[N_TRAIN:], target[N_TRAIN:])),
        "train_mse_julia": float(julia_mse(torch.from_numpy(jpred[:N_TRAIN]), target[:N_TRAIN])),
        "val_mse_julia": float(julia_mse(torch.from_numpy(jpred[N_TRAIN:]), target[N_TRAIN:])),
        "prediction_max_abs_diff_vs_julia": float(np.abs(pred.numpy() - jpred).max()),
        "prediction_rel_l2_diff_vs_julia": float(np.linalg.norm(pred.numpy() - jpred)
                                                 / np.linalg.norm(jpred)),
    }
    # The same parameters on an (almost) exact solve of the same ODE: how much of each
    # solver's loss is its own discretization error?
    tight = SolverConfig(method="tsit5", rtol=1e-10, atol=1e-12, sensitivity="direct_autograd")
    with torch.no_grad():
        exact = torch.stack([integrate(dyn, u0[i:i + 1], t, tight)[:, 0, :6].T
                             for i in range(N_TRAIN + N_TEST)])
    result["julia_params_tight_solver"] = {
        "solver": "torchdiffeq tsit5, rtol 1e-10, atol 1e-12, one trajectory per solve",
        "train_mse": float(julia_mse(exact[:N_TRAIN], target[:N_TRAIN])),
        "val_mse": float(julia_mse(exact[N_TRAIN:], target[N_TRAIN:])),
        "julia_solver_prediction_max_abs_error": float(np.abs(jpred - exact.numpy()).max()),
        "pytorch_rtol1e-2_prediction_max_abs_error": float((pred - exact).abs().max()),
    }

    out = EXP / "equivalence"
    out.mkdir(parents=True, exist_ok=True)
    (out / "equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["rhs_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

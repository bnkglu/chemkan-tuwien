r"""Evaluate an author-matched hydrogen Stage-1 or Stage-2 checkpoint on all 36 conditions.

Evaluation only. The checkpoint's own configuration fixes the model, normalization
statistics, t_ref and solver tolerances; the current ``hydrogen.npz`` and dense temperature
cache must have the sha256 the checkpoint was trained on (refused otherwise).

    Stage 1  integrate normalized species with the observed (dense Cantera) temperature
    Stage 2  integrate the full normalized state [Y_hat, T_hat] from the initial condition

Per condition (train/test flagged): Eq. 18 MSE (mean over states, sum over time) and the
time-averaged MSE, on the 100-point grid. Stage 2 also reports peak T, temperature rise,
ignition delay and the largest dT/dt against the reference, all with the existing Fig. 8B
definitions (``evaluate_hydrogen_ignition``): both sides on the same dense 601-point grid,
reference from the dense cache, delay = t[argmax(gradient(T, t))], no rise threshold. The
Fig. 8B summary uses the paper's 30 conditions (T0 >= 1000 K); the six 950 K cases' rises
are listed separately.

    python evaluate_author_matched_hydrogen.py --checkpoint <run>/stage2_final.pt --out eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import train_author_matched_hydrogen as amh  # noqa: E402
from _data import DATA_DIR  # noqa: E402
from _predictions import checkpoint_sha256  # noqa: E402
from evaluate_hydrogen_ignition import (PAPER_FIG8B_T0_K, ignition_delay_index,  # noqa: E402
                                        temperature_rise)

from chemkan.normalization import MinMaxNormalizer  # noqa: E402
from chemkan.solver import SolverConfig, integrate  # noqa: E402
from chemkan.temperature import ObservedTemperature  # noqa: E402

M = amh.N_SPECIES


def load_reference(points: int = 601) -> dict:
    """All 36 conditions of ``hydrogen.npz`` plus the dense reference temperature, in the
    archive's canonical order (the cache is stored split-ordered and reassembled here)."""
    canon = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    cache_path = DATA_DIR / f"hydrogen_temperature_{amh.DENSE_T_POINTS}.npz"
    cache = np.load(cache_path, allow_pickle=True)
    ics, is_test = canon["ics"], canon["is_test"].astype(bool)
    if not (np.allclose(cache["train_ics"], ics[~is_test])
            and np.allclose(cache["test_ics"], ics[is_test])):
        raise SystemExit("dense temperature cache IC ordering does not match hydrogen.npz")
    t_cache = np.asarray(cache["t"], dtype=float)
    T_cache = np.empty((len(t_cache), len(ics)))
    T_cache[:, ~is_test] = cache["train_T"][..., 0]
    T_cache[:, is_test] = cache["test_T"][..., 0]
    t = np.asarray(canon["t"], dtype=float)
    return {"t": t, "states": np.asarray(canon["states"], dtype=float), "ics": ics,
            "is_test": is_test, "t_cache": t_cache, "T_cache": T_cache,
            "t_diag": np.linspace(t[0], t[-1], points),
            "dataset_sha256": checkpoint_sha256(DATA_DIR / "hydrogen.npz"),
            "dense_sha256": checkpoint_sha256(cache_path)}


def _integrate(func, y0, t, solver) -> np.ndarray:
    """Batched solve; on failure, per condition (NaN rows for the ones that still fail)."""
    with torch.no_grad():
        try:
            return integrate(func, y0, t, solver).cpu().numpy()
        except Exception:                                       # noqa: BLE001
            pass
        out = np.full((len(t), y0.shape[0], y0.shape[1]), np.nan)
        for i in range(y0.shape[0]):
            try:
                out[:, i:i + 1] = integrate(func.single(i), y0[i:i + 1], t, solver).cpu().numpy()
            except Exception:                                   # noqa: BLE001
                pass
        return out


class _Stage1(amh.Stage1Dynamics):
    def single(self, i):
        temp = ObservedTemperature(self.temperature.saved_times,
                                   self.temperature.temperatures[:, i:i + 1])
        return _Stage1(self.kinetic, temp, self.t_ref)


class _Stage2(amh.Stage2Dynamics):
    def single(self, i):
        return self


def evaluate(ckpt: dict, ref: dict, *, return_arrays: bool = False) -> dict:
    cfg, stage = ckpt["config"], ckpt["stage"]
    if cfg["dataset"]["sha256"] != ref["dataset_sha256"]:
        raise SystemExit("hydrogen.npz sha256 differs from the one the checkpoint was trained on")
    if cfg["dense_temperature"]["sha256"] != ref["dense_sha256"]:
        raise SystemExit("dense temperature cache sha256 differs from the checkpoint's")
    dtype = getattr(torch, cfg["dtype"])
    model = amh.build_model().to(dtype)
    model.load_state_dict(ckpt["model_state"])
    norm = MinMaxNormalizer(torch.tensor(cfg["normalization"]["u_min"], dtype=dtype),
                            torch.tensor(cfg["normalization"]["u_max"], dtype=dtype))
    t_ref = cfg["t_ref"]["value_s"]
    solver = SolverConfig(method="tsit5", rtol=cfg["solver"]["rtol"], atol=cfg["solver"]["atol"],
                          sensitivity="direct_autograd")
    as_t = lambda a: torch.as_tensor(a, dtype=dtype)             # noqa: E731
    states = as_t(ref["states"]).permute(1, 0, 2)                # (T, 36, m+1) physical
    u_hat = norm.normalize(states)
    t = as_t(ref["t"])
    n_state = M if stage == 1 else M + 1

    if stage == 1:
        T_hat = norm.subset(slice(M, M + 1)).normalize(as_t(ref["T_cache"]).unsqueeze(-1))
        func = _Stage1(model.kinetic, ObservedTemperature(as_t(ref["t_cache"]), T_hat), t_ref)
        pred_hat = _integrate(func, u_hat[0, :, :M], t, solver)
        pred_phys = norm.subset(slice(0, M)).denormalize(as_t(pred_hat)).numpy()
    else:
        func = _Stage2(model, t_ref)
        pred_hat = _integrate(func, u_hat[0], t, solver)
        pred_phys = norm.denormalize(as_t(pred_hat)).numpy()
        diag_hat = _integrate(func, u_hat[0], as_t(ref["t_diag"]), solver)
        T_diag = norm.denormalize(as_t(diag_hat)).numpy()[..., M]              # (P, 36)
        T_ref_diag = np.stack([np.interp(ref["t_diag"], ref["t_cache"], ref["T_cache"][:, i])
                               for i in range(len(ref["ics"]))], axis=1)

    err = (pred_hat - u_hat[..., :n_state].numpy()) ** 2
    eq18 = err.mean(axis=-1).sum(axis=0)                                        # (36,)
    rows = []
    for i, (T0, phi) in enumerate(ref["ics"]):
        row = {"T0_K": float(T0), "phi": float(phi),
               "split": "test" if ref["is_test"][i] else "train",
               "finite": bool(np.isfinite(pred_hat[:, i]).all()),
               "eq18_mse": float(eq18[i]), "mse_time_avg": float(eq18[i] / len(ref["t"]))}
        if stage == 2:
            td, Tp, Tr = ref["t_diag"], T_diag[:, i], T_ref_diag[:, i]
            k_ref = ignition_delay_index(td, Tr)
            k_pred = ignition_delay_index(td, Tp)
            d_ref = float(td[k_ref])
            d_pred = None if k_pred is None else float(td[k_pred])
            row.update({
                "in_fig8b_set": any(np.isclose(T0, x) for x in PAPER_FIG8B_T0_K),
                "reference_peak_T_K": float(Tr.max()), "predicted_peak_T_K":
                    float(Tp.max()) if k_pred is not None else None,
                "reference_temperature_rise_K": temperature_rise(Tr),
                "predicted_temperature_rise_K": temperature_rise(Tp) if k_pred is not None else None,
                "reference_ignition_delay_s": d_ref, "predicted_ignition_delay_s": d_pred,
                "ignition_delay_relative_error": None if d_pred is None else (d_pred - d_ref) / d_ref,
                "reference_max_dTdt_K_per_s": float(np.max(np.gradient(Tr, td))),
                "predicted_max_dTdt_K_per_s":
                    float(np.max(np.gradient(Tp, td))) if k_pred is not None else None,
                "status": ("integration_failed" if k_pred is None else
                           "argmax_at_window_edge" if k_pred in (0, len(td) - 1) else "evaluated"),
            })
        rows.append(row)

    med = lambda xs: float(np.median(xs)) if xs else None      # noqa: E731
    summary = {"n_conditions": len(rows),
               "eq18_mse_median": {s: med([r["eq18_mse"] for r in rows
                                           if r["split"] == s and r["finite"]])
                                   for s in ("train", "test")},
               "n_integration_failed": sum(not r["finite"] for r in rows)}
    if stage == 2:
        fig8b = [r for r in rows if r["in_fig8b_set"] and r["status"] != "integration_failed"]
        summary["fig8b_30"] = {
            "n": len(fig8b),
            "median_predicted_temperature_rise_K": med([r["predicted_temperature_rise_K"] for r in fig8b]),
            "median_reference_temperature_rise_K": med([r["reference_temperature_rise_K"] for r in fig8b]),
            "median_abs_ignition_delay_relative_error":
                med([abs(r["ignition_delay_relative_error"]) for r in fig8b])}
        summary["t0_950K"] = [{"phi": r["phi"],
                               "predicted_temperature_rise_K": r["predicted_temperature_rise_K"],
                               "reference_temperature_rise_K": r["reference_temperature_rise_K"]}
                              for r in rows if np.isclose(r["T0_K"], 950.0)]
    out = {"stage": stage, "run_id": ckpt.get("run_id"),
           "epochs_completed": ckpt.get("epochs_completed", ckpt.get("phase_epochs_completed")),
           "definitions": {"eq18_mse": "mean over states, sum over the 100 observation times",
                           "ignition_delay": "t[argmax(gradient(T, t))] on the dense diagnostic "
                                             "grid, reference from the dense Cantera cache "
                                             "(evaluate_hydrogen_ignition definitions)",
                           "stage1_temperature": "observed input, so no temperature metrics"},
           "diagnostic_grid_points": len(ref["t_diag"]), "summary": summary, "conditions": rows}
    if return_arrays:
        out["_arrays"] = {"pred_hat": pred_hat, "pred_phys": pred_phys, "norm": norm}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--points", type=int, default=601)
    args = ap.parse_args(argv)
    ckpt = torch.load(args.checkpoint, weights_only=False)
    torch.set_default_dtype(getattr(torch, ckpt["config"]["dtype"]))
    result = evaluate(ckpt, load_reference(args.points))
    result["checkpoint"] = {"path": str(args.checkpoint),
                            "sha256": checkpoint_sha256(args.checkpoint)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

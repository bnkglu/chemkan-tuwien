r"""Ignition-delay evaluation for a trained hydrogen ChemKAN (Fig. 8B).

Evaluation only -- nothing is trained and no checkpoint is written.

Paper definition (Sec. III B): ignition delay = the time of maximum temperature-rise
rate. Estimating that from the 50 sparse training/plotting samples would compare a coarse
reference derivative with a fine model derivative, so BOTH sides are put on the SAME dense
diagnostic grid (601 points over 0-0.6 ms by default, the resolution the data generator
already uses for its stored reference delays) and run through the SAME estimator:

    delay = t[argmax(gradient(T, t))]

The reference temperature comes from the dense Cantera cache
``hydrogen_temperature_20000.npz``, linearly interpolated onto the diagnostic grid -- the
same interpolation the Stage-1 temperature provider uses. The model is integrated from
its initial state only, on that identical grid.

A prediction whose temperature never rises by ``--rise-threshold`` inside the window is
recorded as *no ignition in window*, with its delay and error left undefined rather than
reported as the argmax of a nearly flat curve. Conditions whose integration fails are
recorded separately. Every reference-igniting condition is retained either way: the set
of evaluated conditions is fixed by the REFERENCE, never by whether the model succeeds.

    python evaluate_hydrogen_ignition.py \
        --run-dir ../../results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4/random_stage2_10000_seed0 \
        --out ../../results/reproduction/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from _data import DATA_DIR, load_input_scaling, resolve_device
from _predictions import checkpoint_sha256
from evaluate_hydrogen import build_chemkan, solver_from_ckpt
from _run import utc_now

from chemkan.dynamics import ChemKANDynamics
from chemkan.solver import integrate

RISE_THRESHOLD_K = 100.0        # REPRODUCTION CHOICE, matching the data generator's rule


def ignition_delay(t: np.ndarray, T: np.ndarray, rise_threshold: float):
    """Time of maximum dT/dt, or None when the trajectory never rises enough.

    ``np.gradient`` uses second-order central differences inside the window and
    second-order one-sided differences at both endpoints, so an argmax landing on the
    first or last sample is reported as such by the caller rather than silently trusted.
    """
    T = np.asarray(T, dtype=float)
    t = np.asarray(t, dtype=float)
    if not np.isfinite(T).all():
        return None
    if float(T.max() - T[0]) < rise_threshold:
        return None
    return int(np.argmax(np.gradient(T, t)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True, help="directory for the CSV / summary")
    p.add_argument("--label", default=None)
    p.add_argument("--points", type=int, default=601,
                   help="dense diagnostic grid resolution over the 0-0.6 ms window")
    p.add_argument("--cache-points", type=int, default=20000,
                   help="dense reference temperature cache resolution")
    p.add_argument("--rise-threshold", type=float, default=RISE_THRESHOLD_K,
                   help="minimum temperature rise counted as ignition, K")
    p.add_argument("--held-out", default="1150:1.3")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dev = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoint_final.pt"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    canon = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    cache = np.load(DATA_DIR / f"hydrogen_temperature_{args.cache_points}.npz",
                    allow_pickle=True)
    ics, is_test = canon["ics"], canon["is_test"]                  # (36, 2), (36,)
    states = canon["states"]                                       # (36, 50, m+1)
    m = len(canon["species"])

    # Reassemble the cache's split-ordered dense temperature back into canonical order.
    t_cache = np.asarray(cache["t"], dtype=float)
    T_cache = np.empty((len(t_cache), len(ics)))
    T_cache[:, ~is_test] = cache["train_T"][..., 0]
    T_cache[:, is_test] = cache["test_T"][..., 0]
    if not np.allclose(cache["train_ics"], ics[~is_test]) or \
       not np.allclose(cache["test_ics"], ics[is_test]):
        raise SystemExit("dense temperature cache IC ordering does not match hydrogen.npz")

    t_dense = np.linspace(float(canon["t"][0]), float(canon["t"][-1]), args.points)
    T_ref = np.stack([np.interp(t_dense, t_cache, T_cache[:, i]) for i in range(len(ics))],
                     axis=1)                                       # (P, 36)

    model = build_chemkan(ckpt, m, dev)
    dyn = ChemKANDynamics(model, input_normalizer=load_input_scaling(ckpt, dev)).to(dev)
    solver = solver_from_ckpt(ckpt)
    u0 = torch.as_tensor(states[:, 0, :], dtype=torch.float32, device=dev)
    t_t = torch.as_tensor(t_dense, dtype=torch.float32, device=dev)
    try:
        with torch.no_grad():
            pred = integrate(dyn, u0, t_t, solver).cpu().numpy()   # (P, 36, m+1)
    except Exception as exc:
        logging.warning("batched integration raised (%s); retrying per condition",
                        type(exc).__name__)
        pred = np.full((args.points, len(ics), m + 1), np.nan, dtype=np.float32)
        for i in range(len(ics)):
            try:
                with torch.no_grad():
                    pred[:, i:i + 1] = integrate(dyn, u0[i:i + 1], t_t, solver).cpu().numpy()
            except Exception as inner:
                logging.warning("T0=%.0f phi=%.2f integration failed: %s",
                                ics[i, 0], ics[i, 1], type(inner).__name__)
    T_pred = pred[..., -1]                                          # (P, 36)

    T0_h, phi_h = (float(v) for v in args.held_out.split(":"))
    rows, n_ref_ignite, n_model_ignite = [], 0, 0
    for i in range(len(ics)):
        k_ref = ignition_delay(t_dense, T_ref[:, i], args.rise_threshold)
        if k_ref is None:
            continue                       # reference does not ignite in window: out of scope
        n_ref_ignite += 1
        d_ref = float(t_dense[k_ref])
        finite = bool(np.isfinite(T_pred[:, i]).all())
        k_pred = ignition_delay(t_dense, T_pred[:, i], args.rise_threshold) if finite else None
        if not finite:
            status, d_pred, abs_err, rel_err = "integration_failed", "", "", ""
        elif k_pred is None:
            status, d_pred, abs_err, rel_err = "no_ignition_in_window", "", "", ""
        else:
            n_model_ignite += 1
            status = "ignited"
            d_pred = float(t_dense[k_pred])
            abs_err = d_pred - d_ref
            rel_err = abs_err / d_ref
            if k_pred in (0, len(t_dense) - 1):
                status = "ignited_at_window_edge"     # argmax on a one-sided derivative
        rows.append({
            "T0_K": float(ics[i, 0]), "phi": float(ics[i, 1]),
            "reference_delay_s": d_ref, "chemkan_delay_s": d_pred,
            "absolute_error_s": abs_err, "relative_error": rel_err,
            "is_held_out": bool(np.isclose(ics[i, 0], T0_h) and np.isclose(ics[i, 1], phi_h)),
            "reference_peak_T_K": float(T_ref[:, i].max()),
            "chemkan_peak_T_K": float(T_pred[:, i].max()) if finite else "",
            "status": status,
        })
    rows.sort(key=lambda r: (r["T0_K"], r["phi"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or run_dir.name
    csv_path = out / f"hydrogen_ignition_delay_{label}.csv"
    if csv_path.exists() and not args.force:
        raise SystemExit(f"{csv_path} exists -- pass --force to overwrite.")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logging.info("wrote %s", csv_path)

    summary = {
        "run_id": ckpt.get("run_id"), "checkpoint": str(ckpt_path),
        "checkpoint_sha256": checkpoint_sha256(ckpt_path),
        "architecture": ckpt["architecture"],
        "definition": "ignition delay = time of maximum dT/dt (paper Sec. III B)",
        "estimator": "numpy.gradient on the dense diagnostic grid, identical for "
                     "reference and model",
        "diagnostic_grid": {"points": args.points,
                            "t_start_s": float(t_dense[0]), "t_end_s": float(t_dense[-1])},
        "reference_source": f"hydrogen_temperature_{args.cache_points}.npz, linearly "
                            f"interpolated onto the diagnostic grid",
        "rise_threshold_K": args.rise_threshold,
        "conditions_total": int(len(ics)),
        "reference_igniting": n_ref_ignite,
        "model_igniting": n_model_ignite,
        "solver": {"method": solver.method, "rtol": solver.rtol, "atol": solver.atol},
        "evaluated_at": utc_now(),
    }
    (out / f"hydrogen_ignition_delay_{label}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    logging.info("reference ignites in %d/%d conditions; model ignites in %d of those",
                 n_ref_ignite, len(ics), n_model_ignite)


if __name__ == "__main__":
    main()

r"""Evaluate a trained hydrogen ChemKAN over the 441-condition fine grid (Fig. 8A).

Evaluation only -- nothing is trained and no checkpoint is written.

The fine grid (``hydrogen_fine.npz``) is 21 x 21 conditions,
``T0 = linspace(950, 1200, 21)`` x ``phi = linspace(0.5, 1.5, 21)``, RECONSTRUCTED from
the paper's reported 441-condition count and figure spacing; the paper does not tabulate
it. Its composition matches the paper: 35 original training conditions, 1 original
held-out condition and 405 further unseen conditions, so 406 are unseen during training.

Normalization comes from the canonical ``hydrogen.npz`` TRAIN statistics -- the same ones
the checkpoint was trained with. The fine archive's own ``u_min``/``u_max`` are ignored
on purpose: refitting a normalizer on the evaluation grid would change the metric.

All conditions are integrated in ONE batch from their initial states only, matching how
``evaluate_hydrogen.py`` integrates a whole split. The adaptive solver picks its steps
from the joint error norm, so batch composition is part of the evaluation: the batch size
actually used is recorded in the summary. If the batch raises or produces any non-finite
value, the affected conditions are re-integrated individually so one blow-up cannot
destroy the other 440, and those rows say so. Failures are recorded as failures with
their condition identifiers -- never dropped, never replaced by an invented finite MSE. A
trajectory that integrates but does not ignite still gets a trajectory MSE; non-ignition
is a separate diagnostic (see ``evaluate_hydrogen_ignition.py``).

    python evaluate_hydrogen_grid.py \
        --run-dir ../../results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4/random_stage2_10000_seed0 \
        --out ../../results/reproduction/chemkan/hydrogen/generalization
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from _data import DATA_DIR, load_input_scaling, resolve_device
from _predictions import checkpoint_sha256
from evaluate_hydrogen import build_chemkan, solver_from_ckpt
from _run import utc_now

from chemkan.dynamics import ChemKANDynamics
from chemkan.losses import trajectory_mse
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import integrate

_METRIC_CONVENTION = ("normalized trajectory MSE (Eq. 18): mean over modeled state variables "
                      "(full [Y, T] state), summed over observation times, using the "
                      "canonical train-only min-max normalization; reported PER CONDITION "
                      "(never averaged before plotting).")


def _match_rows(ics: np.ndarray, targets: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of rows of ``ics`` that appear in ``targets`` (order-independent)."""
    mask = np.zeros(len(ics), dtype=bool)
    for row in targets:
        hit = np.all(np.isclose(ics, row, atol=tol, rtol=0.0), axis=1)
        mask |= hit
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="directory holding checkpoint_final.pt")
    p.add_argument("--out", required=True, help="directory for the CSV / predictions / summary")
    p.add_argument("--label", default=None,
                   help="artifact name stem (default: the run directory's name)")
    p.add_argument("--grid", default="hydrogen_fine", help="fine-grid archive stem")
    p.add_argument("--held-out", default="1150:1.3",
                   help="original held-out condition, T0:phi (paper Sec. II D 2)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--save-predictions", action="store_true",
                   help="also persist all 441 predicted trajectories (npz)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dev = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoint_final.pt"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    fine = np.load(DATA_DIR / f"{args.grid}.npz", allow_pickle=True)
    canon = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    t = torch.as_tensor(fine["t"], dtype=torch.float32, device=dev)
    states = fine["states"]                       # (441, Nt, m+1)
    ics = fine["ics"]                             # (441, 2) = [T0, phi]
    m = len(fine["species"])

    model = build_chemkan(ckpt, m, dev)
    input_normalizer = load_input_scaling(ckpt, dev)
    solver = solver_from_ckpt(ckpt)
    # Canonical TRAIN statistics -- never the fine grid's own.
    norm = MinMaxNormalizer(torch.as_tensor(canon["u_min"], dtype=torch.float32),
                            torch.as_tensor(canon["u_max"], dtype=torch.float32)).to(dev)
    dyn = ChemKANDynamics(model, input_normalizer=input_normalizer).to(dev)

    train_mask = _match_rows(ics, canon["train_ics"])
    T0_h, phi_h = (float(v) for v in args.held_out.split(":"))
    held_mask = np.all(np.isclose(ics, [T0_h, phi_h], atol=1e-6, rtol=0.0), axis=1)
    logging.info("grid %s: %d conditions | %d training | %d held-out | %d additional",
                 args.grid, len(ics), train_mask.sum(), held_mask.sum(),
                 len(ics) - train_mask.sum() - held_mask.sum())
    if train_mask.sum() != len(canon["train_ics"]):
        raise SystemExit(f"expected {len(canon['train_ics'])} training conditions on the "
                         f"grid, matched {train_mask.sum()}")
    if held_mask.sum() != 1:
        raise SystemExit(f"held-out condition {args.held_out} matched {held_mask.sum()} rows")

    reference = torch.as_tensor(states, dtype=torch.float32, device=dev)   # (N, Nt, m+1)
    u0_all = reference[:, 0, :]                                            # (N, m+1)
    ref_TBx = reference.permute(1, 0, 2)                                   # (Nt, N, m+1)

    def integrate_subset(idx):
        """(T, len(idx), m+1) predictions, or None if the solver raised."""
        try:
            with torch.no_grad():
                return integrate(dyn, u0_all[idx], t, solver)
        except Exception as exc:                   # solver failure is data, not a crash
            logging.warning("integration raised for %d condition(s): %s",
                            len(idx), type(exc).__name__)
            return None

    started = time.perf_counter()
    all_idx = torch.arange(len(ics), device=dev)
    pred_TBx = integrate_subset(all_idx)
    batch_note = f"single batch of {len(ics)}"
    if pred_TBx is None or not torch.isfinite(pred_TBx).all():
        # Retry individually so one blow-up cannot invalidate the whole grid.
        logging.warning("batched integration incomplete -- retrying per condition")
        batch_note = f"single batch of {len(ics)}, then per-condition retry"
        pred_TBx = torch.full_like(ref_TBx, float("nan"))
        for i in range(len(ics)):
            one = integrate_subset(all_idx[i:i + 1])
            if one is not None:
                pred_TBx[:, i:i + 1, :] = one

    rows, n_fail = [], 0
    pred_norm = norm.normalize(pred_TBx)
    ref_norm = norm.normalize(ref_TBx)
    for i in range(len(ics)):
        finite = bool(torch.isfinite(pred_TBx[:, i, :]).all())
        status = "ok" if finite else "integration_failed_or_non_finite"
        mse = (float(trajectory_mse(pred_norm[:, i:i + 1, :], ref_norm[:, i:i + 1, :]))
               if finite else float("nan"))
        if not finite:
            n_fail += 1
            logging.warning("T0=%.1f phi=%.2f -> %s", ics[i, 0], ics[i, 1], status)
        rows.append({
            "index": i, "T0_K": float(ics[i, 0]), "phi": float(ics[i, 1]),
            "trajectory_mse": mse, "status": status,
            "is_training": bool(train_mask[i]), "is_held_out": bool(held_mask[i]),
            "is_additional": bool(not train_mask[i] and not held_mask[i]),
        })
    preds = pred_TBx.permute(1, 0, 2).cpu().numpy()                        # (N, Nt, m+1)
    elapsed = time.perf_counter() - started

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or run_dir.name
    csv_path = out / f"{label}_generalization_441.csv"
    if csv_path.exists() and not args.force:
        raise SystemExit(f"{csv_path} exists -- pass --force to overwrite.")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logging.info("wrote %s", csv_path)

    ok = np.array([r["trajectory_mse"] for r in rows if r["status"] == "ok"])
    summary = {
        "run_id": ckpt.get("run_id"), "checkpoint": str(ckpt_path),
        "checkpoint_sha256": checkpoint_sha256(ckpt_path),
        "architecture": ckpt["architecture"], "grid": args.grid,
        "grid_provenance": "RECONSTRUCTED 21x21: T0=linspace(950,1200,21), "
                           "phi=linspace(0.5,1.5,21); count and spacing inferred from the "
                           "paper's 441-condition figure, not tabulated there",
        "conditions": len(rows), "training": int(train_mask.sum()),
        "held_out": int(held_mask.sum()),
        "additional": int(len(ics) - train_mask.sum() - held_mask.sum()),
        "evaluated_ok": int(len(ok)), "failed": int(n_fail),
        "integration_batching": batch_note,
        "mse_min": None if not len(ok) else float(ok.min()),
        "mse_median": None if not len(ok) else float(np.median(ok)),
        "mse_max": None if not len(ok) else float(ok.max()),
        "metric_convention": _METRIC_CONVENTION,
        "normalization": "canonical hydrogen.npz train-only min-max (not refit on the grid)",
        "solver": {"method": solver.method, "rtol": solver.rtol, "atol": solver.atol},
        "wall_time_s": round(elapsed, 3), "evaluated_at": utc_now(),
    }
    (out / f"{label}_generalization_441.json").write_text(json.dumps(summary, indent=2,
                                                                    default=str))
    logging.info("wrote %s", out / f"{label}_generalization_441.json")
    if args.save_predictions:
        npz = out / f"{label}_generalization_441_predictions.npz"
        if npz.exists() and not args.force:
            raise SystemExit(f"{npz} exists -- pass --force to overwrite.")
        np.savez_compressed(npz, predictions=preds, reference=states,
                            t=fine["t"], ics=ics, species=fine["species"],
                            u_min=canon["u_min"], u_max=canon["u_max"],
                            provenance=np.asarray(json.dumps(summary, default=str)))
        logging.info("wrote %s", npz)

    logging.info("evaluated %d/%d conditions in %.1f s | MSE median %s max %s | %d failed",
                 len(ok), len(rows), elapsed,
                 "n/a" if not len(ok) else f"{np.median(ok):.4e}",
                 "n/a" if not len(ok) else f"{ok.max():.4e}", n_fail)


if __name__ == "__main__":
    main()

r"""Local ChemKAN-vs-Cantera inference benchmark for Table I.

This is a **local PyTorch-vs-Cantera benchmark**, NOT a reproduction of the paper's
timing: the paper's 2.0x speed-up was measured against Arrhenius.jl, on other hardware,
with an unstated timing scope. The two numbers are not directly comparable and must be
reported side by side, never merged.

What is timed, on both sides, is the same task: produce [Y_1..Y_9, T] at the same 50
requested output times over 0-0.6 ms for the same 36 coarse initial conditions.

    ChemKAN : one batched Tsit5 integration under no_grad, using the checkpoint's own
              solver tolerances and input scaling. float32, CPU by default.
    Cantera : 36 IdealGasConstPressureReactor runs advanced to the same output times,
              same mechanism and pressure as the data generator, tolerances configurable
              (the generator's 1e-9/1e-15 are the default -- they are much tighter than
              the model's 1e-6/1e-8, so the ratio is reported for both settings when
              ``--cantera-tolerance-sweep`` is passed).

Accuracy is measured in the same run and reported beside the timing: a model that fails
to reproduce the dynamics is fast for the wrong reason, so the peak-temperature error and
temperature rise and peak dT/dt stay attached to the speed-up -- as continuous numbers,
with no ignited/not verdict.

    python benchmark_inference.py \
        --run-dir ../../../results/reproduction/legacy/hydrogen/chemkan/diagnostics/base_on_n4/random_stage2_10000_seed0 \
        --out ../../../results/reproduction/legacy/hydrogen/tables
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "data_gen"))

from _data import DATA_DIR, load_input_scaling, resolve_device      # noqa: E402
from _predictions import checkpoint_sha256                          # noqa: E402
from _run import utc_now                                            # noqa: E402
from evaluate_hydrogen import build_chemkan, solver_from_ckpt       # noqa: E402

from chemkan.dynamics import ChemKANDynamics                        # noqa: E402
from chemkan.solver import integrate                                # noqa: E402


def repo_relative(path) -> str:
    """Path from the repository root, so provenance never bakes in a local absolute path.

    Falls back to the string as given if no 'results' component is present.
    """
    parts = Path(path).resolve().parts
    if "results" in parts:
        return "/".join(parts[parts.index("results"):])
    return str(path)

FUEL = "H2"
OXIDIZER = {"O2": 1.0, "N2": 3.76}

# The paper's Fig. 8B set, written out explicitly and never derived from a threshold. The
# remaining 950 K row is reported beside it, so no condition is dropped.
PAPER_FIG8B_T0_K = (1000.0, 1050.0, 1100.0, 1150.0, 1200.0)

# Largest tolerated gap (K) between a stored peak-temperature error and the one recomputed
# from a run's committed predictions. The benchmark integrates all 36 conditions in one
# batch while the prediction files were written per split, and torchdiffeq batches share
# one step sequence, so the two differ at the millikelvin level; 0.05 K bounds that noise
# with a wide margin while still catching a different checkpoint or grid.
ACCURACY_CONSISTENCY_TOL_K = 0.05


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def temperature_diagnostics(t, T_pred, T_ref, T0) -> dict:
    """Continuous accuracy diagnostics with no ignited/not verdict.

    ``T_pred`` and ``T_ref`` are (points, conditions). Conditions are grouped by the paper's
    explicit Fig. 8B set and the 950 K row; each group reports temperature rise and peak
    dT/dt for model and reference, so a flat prediction is visible as a number.
    """
    rate = lambda T: np.gradient(T, t, axis=0).max(axis=0)
    rise_p, rise_r = T_pred.max(axis=0) - T_pred[0], T_ref.max(axis=0) - T_ref[0]
    rate_p, rate_r = rate(T_pred), rate(T_ref)
    groups = {"paper_fig8b_set_T0_1000_1200K": np.isin(T0, PAPER_FIG8B_T0_K),
              "T0_950K": np.isclose(T0, 950.0)}
    return {name: {"conditions": int(sel.sum()),
                   "model_temperature_rise_K": _stats(rise_p[sel]),
                   "reference_temperature_rise_K": _stats(rise_r[sel]),
                   "model_peak_dTdt_K_per_s_median": float(np.median(rate_p[sel])),
                   "reference_peak_dTdt_K_per_s_median": float(np.median(rate_r[sel]))}
            for name, sel in groups.items()}


def time_chemkan(dyn, u0, t, solver, warmup: int, reps: int) -> tuple[list[float], np.ndarray]:
    with torch.no_grad():
        for _ in range(warmup):
            pred = integrate(dyn, u0, t, solver)
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            pred = integrate(dyn, u0, t, solver)
            times.append(time.perf_counter() - start)
    return times, pred.cpu().numpy()


def time_cantera(mech, pressure, ics, t, keep, rtol, atol, warmup: int, reps: int):
    from reactor import integrate_case          # data_gen/reactor.py, the generator's own

    def one_pass():
        return np.stack([integrate_case(mech, FUEL, OXIDIZER, float(T0), float(phi),
                                        t, pressure, keep, rtol=rtol, atol=atol)
                         for T0, phi in ics])
    for _ in range(warmup):
        states = one_pass()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        states = one_pass()
        times.append(time.perf_counter() - start)
    return times, states


def recompute_accuracy_only(run_dir: Path, out: Path, label: str) -> None:
    """Replace only the accuracy diagnostics of an existing benchmark JSON.

    Reads the run's committed ``predictions/{train,test}_predictions.npz`` instead of
    re-integrating, so every timing block stays exactly as measured. Refuses to write unless
    the predictions carry the benchmarked checkpoint's SHA-256, their reference equals the
    dataset, and they reproduce the stored peak-temperature error.
    """
    path = out / f"hydrogen_inference_benchmark_{label}.json"
    result = json.loads(path.read_text())
    canon = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    is_test, states, ics = canon["is_test"], canon["states"], canon["ics"]
    t = np.asarray(canon["t"], dtype=float)

    split = {s: np.load(run_dir / "predictions" / f"{s}_predictions.npz", allow_pickle=True)
             for s in ("train", "test")}
    for name, z in split.items():
        sha = json.loads(str(z["provenance"]))["checkpoint_sha256"]
        if sha != result["checkpoint_sha256"]:
            raise SystemExit(f"{name} predictions come from checkpoint {sha[:12]}, not the "
                             f"benchmarked {result['checkpoint_sha256'][:12]}")
    pred = np.empty((len(t), len(ics), states.shape[-1]))
    ref = np.empty_like(pred)
    pred[:, ~is_test], pred[:, is_test] = split["train"]["predictions"], split["test"]["predictions"]
    ref[:, ~is_test], ref[:, is_test] = split["train"]["reference"], split["test"]["reference"]
    if not np.allclose(ref.transpose(1, 0, 2), states, rtol=0, atol=1e-3):
        raise SystemExit("prediction reference does not match hydrogen.npz")

    T_pred, T_ref = pred[..., -1], states[..., -1].T
    peak_err = np.abs(T_pred.max(axis=0) - T_ref.max(axis=0))
    stored = result["accuracy"]["peak_temperature_abs_error_K"]
    gaps = {"median": abs(float(np.median(peak_err)) - stored["median"]),
            "max": abs(float(peak_err.max()) - stored["max"])}
    if max(gaps.values()) > ACCURACY_CONSISTENCY_TOL_K:
        raise SystemExit(f"committed predictions do not reproduce the stored peak-T error "
                         f"(gaps {gaps}); refusing to mix trajectories")

    acc = result["accuracy"]
    for retired in ("reference_igniting_conditions", "chemkan_igniting_within_reference_set",
                    "chemkan_igniting_where_reference_does_not"):
        acc.pop(retired, None)                       # 100 K verdicts, retired
    acc["temperature_diagnostics"] = temperature_diagnostics(t, T_pred, T_ref, ics[:, 0])
    acc["temperature_diagnostics_source"] = {
        "predictions": [repo_relative(run_dir / "predictions" / f"{s}_predictions.npz")
                        for s in ("train", "test")],
        "recomputed_at": utc_now(),
        "timing_blocks": "unchanged from the original measurement",
        "peak_error_consistency_gap_K": gaps,
    }
    path.write_text(json.dumps(result, indent=2, default=str))
    logging.info("rewrote the accuracy block of %s (timings untouched; peak-T gap %.2e K)",
                 path, max(gaps.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--cantera-rtol", type=float, default=1e-9,
                   help="data-generator value; --cantera-tolerance-sweep also times "
                        "the model's own 1e-6/1e-8")
    p.add_argument("--cantera-atol", type=float, default=1e-15)
    p.add_argument("--cantera-tolerance-sweep", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--accuracy-only", action="store_true",
                   help="rewrite only the accuracy block of an existing benchmark JSON from "
                        "the run's committed predictions; no timing is re-measured")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.accuracy_only:
        recompute_accuracy_only(Path(args.run_dir), Path(args.out),
                                args.label or Path(args.run_dir).name)
        return

    dev = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoint_final.pt"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    canon = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    species = [str(s) for s in canon["species"]]
    t_np = np.asarray(canon["t"], dtype=float)
    ics, states_ref = canon["ics"], canon["states"]
    mech, pressure = str(canon["mechanism"]), float(canon["pressure"])

    model = build_chemkan(ckpt, len(species), dev)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    dyn = ChemKANDynamics(model, input_normalizer=load_input_scaling(ckpt, dev)).to(dev)
    solver = solver_from_ckpt(ckpt)
    u0 = torch.as_tensor(states_ref[:, 0, :], dtype=torch.float32, device=dev)
    t_t = torch.as_tensor(t_np, dtype=torch.float32, device=dev)

    logging.info("timing ChemKAN: %d conditions x %d output points, %d warm-up + %d reps",
                 len(ics), len(t_np), args.warmup, args.reps)
    kan_times, kan_pred = time_chemkan(dyn, u0, t_t, solver, args.warmup, args.reps)

    import cantera as ct                                 # imported for the version stamp
    from reactor import species_index
    _, keep = species_index(mech)

    cantera_runs = [(args.cantera_rtol, args.cantera_atol)]
    if args.cantera_tolerance_sweep:
        cantera_runs.append((solver.rtol, solver.atol))  # the model's own tolerances
    cantera_results = []
    for rtol, atol in cantera_runs:
        logging.info("timing Cantera: rtol=%g atol=%g", rtol, atol)
        try:
            ct_times, _ = time_cantera(mech, pressure, ics, t_np, keep, rtol, atol,
                                       args.warmup, args.reps)
        except Exception as exc:
            # A tolerance setting Cantera cannot integrate is a RESULT about that setting,
            # not a reason to lose the measurement at the settings that do work.
            logging.warning("Cantera failed at rtol=%g atol=%g: %s", rtol, atol, exc)
            cantera_results.append({"rtol": rtol, "atol": atol,
                                    "status": f"integration_failed: {type(exc).__name__}",
                                    "note": str(exc).strip().splitlines()[0][:200]})
            continue
        cantera_results.append({
            "rtol": rtol, "atol": atol, "status": "ok",
            "median_s": statistics.median(ct_times), "min_s": min(ct_times),
            "times_s": ct_times,
            "speedup_vs_chemkan": statistics.median(ct_times) / statistics.median(kan_times),
        })
    ok_runs = [r for r in cantera_results if r.get("status") == "ok"]
    if not ok_runs:
        raise SystemExit("Cantera failed at every requested tolerance; no timing to report.")

    # Accuracy, measured in the same run so it cannot be reported apart from the timing.
    T_pred, T_ref = kan_pred[..., -1], states_ref[..., -1].T          # (P, 36)
    peak_err = np.abs(T_pred.max(axis=0) - T_ref.max(axis=0))

    result = {
        "run_id": ckpt.get("run_id"), "checkpoint": repo_relative(ckpt_path),
        "checkpoint_sha256": checkpoint_sha256(ckpt_path),
        "architecture": ckpt["architecture"], "parameter_count": n_params,
        "benchmark": "local PyTorch-vs-Cantera inference benchmark; NOT the paper's "
                     "Arrhenius.jl measurement and not comparable to it directly",
        "task": {"conditions": int(len(ics)), "output_points": int(len(t_np)),
                 "t_end_s": float(t_np[-1]),
                 "outputs": "[Y_1..Y_9, T] at every requested output time"},
        "timing_scope": "integration only, after warm-up; excludes data/checkpoint loading "
                        "and model construction",
        "warmup": args.warmup, "repetitions": args.reps,
        "chemkan": {"median_s": statistics.median(kan_times), "min_s": min(kan_times),
                    "times_s": kan_times, "batched": True,
                    "solver": {"method": solver.method, "rtol": solver.rtol,
                               "atol": solver.atol, "sensitivity": solver.sensitivity},
                    "dtype": str(torch.get_default_dtype()), "device": str(dev),
                    "torch": torch.__version__, "threads": torch.get_num_threads()},
        "cantera": {"mechanism": mech, "pressure_pa": pressure,
                    "version": ct.__version__, "runs": cantera_results,
                    "batched": False,
                    "note": "36 separate reactor integrations, as the generator runs them"},
        "accuracy": {
            "peak_temperature_abs_error_K": {"median": float(np.median(peak_err)),
                                             "max": float(peak_err.max())},
            "temperature_diagnostics": temperature_diagnostics(t_np, T_pred, T_ref, ics[:, 0]),
            "caveat": "timing is only meaningful beside this accuracy; a model that does "
                      "not reproduce the dynamics is not a valid speed-up claim",
        },
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "python": platform.python_version()},
        "measured_at": utc_now(),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or run_dir.name
    path = out / f"hydrogen_inference_benchmark_{label}.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists -- pass --force to overwrite.")
    path.write_text(json.dumps(result, indent=2, default=str))
    logging.info("wrote %s", path)
    logging.info("ChemKAN median %.4f s | Cantera median %.4f s | ratio %.2fx  "
                 "(Cantera rtol=%g atol=%g)",
                 statistics.median(kan_times), ok_runs[0]["median_s"],
                 ok_runs[0]["speedup_vs_chemkan"], ok_runs[0]["rtol"], ok_runs[0]["atol"])
    for r in cantera_results:
        if r.get("status") != "ok":
            logging.info("  Cantera at rtol=%g atol=%g: %s", r["rtol"], r["atol"], r["status"])
    paper = result["accuracy"]["temperature_diagnostics"]["paper_fig8b_set_T0_1000_1200K"]
    logging.info("accuracy: median peak-T error %.1f K | paper set: median temperature rise "
                 "model %.0f K vs reference %.0f K",
                 float(np.median(peak_err)), paper["model_temperature_rise_K"]["median"],
                 paper["reference_temperature_rise_K"]["median"])


if __name__ == "__main__":
    main()

"""
Convergence test for dense precomputed Cantera temperature trajectories.

Compares interpolation from dense Cantera grids (linear and, optionally, PCHIP)
against Cantera evaluated directly at an independent set of off-grid validation
times. Linear is the reference (it matches the Stage-1 ObservedTemperature
provider); PCHIP is included only to see whether a higher-order interpolant buys
meaningful accuracy on the same cache.

This is a diagnostic only:
- does not modify datasets;
- does not train ChemKAN;
- does not save dense caches;
- does not modify Stage 1 or Stage 2.

Run from the repository root:

    python scripts/data_gen/benchmark_temperature_resolution.py

Optional:

    python scripts/data_gen/benchmark_temperature_resolution.py \
        --resolutions 20000 50000 100000 200000 \
        --eval-points 50000

Evaluation times (--times):
    random  (default) independent random off-grid times, model-independent;
    solver  the actual recorded Tsit5 RHS query times from one batched Stage-1
            integration (notebook 9.5b) -- training-relevant, clusters in the
            autoignition window. Uses the trained checkpoint by default:

    python scripts/data_gen/benchmark_temperature_resolution.py \
        --times solver --resolutions 2000 20000 200000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cantera as ct
from scipy.interpolate import PchipInterpolator

# Make the chemkan package + scripts importable regardless of the working dir, so the
# optional "solver" evaluation-times mode can rebuild the Stage-1 machinery (9.5b).
_HERE = Path(__file__).resolve()
_REPO = next((p for p in _HERE.parents if (p / "chemkan" / "src" / "chemkan").is_dir()), None)
if _REPO is not None:
    for _sub in ("chemkan/src", "chemkan/scripts", "chemkan/scripts/data_gen"):
        _d = str(_REPO / _sub)
        if _d not in sys.path:
            sys.path.insert(0, _d)
DEFAULT_CHECKPOINT = str(_REPO / "hydrogen_chemkan.pt") if _REPO else "hydrogen_chemkan.pt"

from reactor import integrate_case, species_index
from generate_hydrogen import (
    MECH,
    DROP,
    FUEL,
    OXIDIZER,
    T0_COARSE,
    PHI_COARSE,
    TEST_IC,
)


def training_conditions():
    """Return the 35 paper training conditions in canonical generator order."""
    conditions = []

    for T0 in T0_COARSE:
        for phi in PHI_COARSE:
            if (
                abs(T0 - TEST_IC[0]) < 1e-12
                and abs(phi - TEST_IC[1]) < 1e-12
            ):
                continue

            conditions.append((float(T0), float(phi)))

    assert len(conditions) == 35
    return conditions


def make_validation_times(
    n_points: int,
    t_end: float,
    seed: int,
) -> np.ndarray:
    """
    Generate fixed, independent, mostly off-grid query times.

    t=0 and t=t_end are included so integrate_case() starts correctly.
    Interior points are randomly distributed and sorted. Using random query
    times avoids systematically coinciding with any particular candidate grid.
    """
    rng = np.random.default_rng(seed)

    interior = rng.uniform(
        np.nextafter(0.0, 1.0),
        np.nextafter(t_end, 0.0),
        size=n_points - 2,
    )

    t = np.concatenate(([0.0], np.sort(interior), [t_end]))

    # Extremely unlikely, but guarantee strict monotonicity.
    t = np.unique(t)

    return t


def make_solver_times(
    checkpoint_path: str,
    model_source: str,
    seed_model: int,
    solver_rtol: float,
    solver_atol: float,
):
    """
    Recorded ODE-solver RHS query times from ONE batched Stage-1 integration.

    This is the notebook 9.5b methodology: build the real Stage-1 dynamics
    (ObservedTemperature + kinetic core), run a single batched integration over
    all 35 conditions, and record every scalar RHS time the adaptive Tsit5 solver
    requested. Those internal times -- which cluster in the stiff autoignition
    window -- are the training-relevant evaluation set.

    The query times depend on the kinetic weights, so by default we load the
    trained hydrogen checkpoint (as 9.5b did). model_source="untrained" uses a
    fresh seeded model instead.

    Returns (t_eval, t_end): sorted unique in-range times (t=0 included) and the
    integration end time, so the dense caches can be built over the same interval.
    """
    import torch

    from _data import load_hydrogen
    from chemkan.dynamics import KineticDynamics
    from chemkan.model import ChemKAN
    from chemkan.normalization import MinMaxNormalizer
    from chemkan.solver import SolverConfig, integrate
    from chemkan.temperature import ObservedTemperature

    data = load_hydrogen(split="train")
    m = data["species_TBm"].shape[-1]

    if model_source == "trained":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = ChemKAN(species_dim=m, **ckpt["architecture"])
        model.load_state_dict(ckpt["model_state"])
    else:
        torch.manual_seed(seed_model)
        model = ChemKAN(species_dim=m, hidden_dim=3, num_basis=5,
                        n_mu=3, use_base_act=False)

    # Same construction as train_hydrogen (minmax input scaling, observed T).
    full_norm = MinMaxNormalizer(data["u_min"], data["u_max"])
    temp = ObservedTemperature(data["t"], data["T_obs_TB1"])
    kin_dyn = KineticDynamics(model.kinetic, temp, input_normalizer=full_norm)

    Y0 = data["species_TBm"][0]                       # (B, m)
    t_grid = data["t"]
    cfg = SolverConfig(method="tsit5", rtol=solver_rtol, atol=solver_atol,
                       sensitivity="direct_autograd")

    recorded = []
    handle = kin_dyn.register_forward_pre_hook(
        lambda mod, args: recorded.append(float(args[0].detach()))
    )
    with torch.no_grad():
        integrate(kin_dyn, Y0, t_grid, cfg)
    handle.remove()

    lo, hi = float(t_grid[0]), float(t_grid[-1])
    t_eval = np.array(sorted({t for t in recorded if lo <= t <= hi} | {lo}))
    return t_eval, hi


def cantera_temperature_matrix(
    conditions,
    t: np.ndarray,
    pressure: float,
    rtol: float,
    atol: float,
    keep: np.ndarray,
) -> np.ndarray:
    """
    Return temperature matrix with shape (n_times, n_conditions).

    Cantera is evaluated directly at every requested time in `t`.
    """
    T = np.empty((len(t), len(conditions)), dtype=np.float64)

    for b, (T0, phi) in enumerate(conditions):
        states = integrate_case(
            MECH,
            FUEL,
            OXIDIZER,
            T0,
            phi,
            t,
            pressure,
            keep,
            rtol,
            atol,
        )

        T[:, b] = states[:, -1]

    return T


def interpolate_temperature(
    t_cache: np.ndarray,
    T_cache: np.ndarray,
    t_query: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate a dense Cantera cache onto the validation times.

    T_cache: (N_cache, B)
    output:  (N_query, B)

    method:
      - "linear": np.interp, equivalent in meaning to ObservedTemperature
        (the actual Stage-1 provider). This is the reference method.
      - "pchip":  scipy PchipInterpolator, shape-preserving cubic Hermite, added
        only to compare a higher-order interpolant's accuracy/cost on the SAME
        cache. It is NOT what the Stage-1 provider uses.

    NumPy/SciPy interpolation is intentionally used here only because this script
    is measuring resolution/convergence, not testing the PyTorch provider
    implementation itself. ObservedTemperature has already been tested separately.
    """
    out = np.empty((len(t_query), T_cache.shape[1]), dtype=np.float64)

    if method == "linear":
        for b in range(T_cache.shape[1]):
            out[:, b] = np.interp(t_query, t_cache, T_cache[:, b])
    elif method == "pchip":
        # All query times lie inside [t_cache[0], t_cache[-1]] (0 and t_end are
        # in both grids), so no extrapolation occurs.
        for b in range(T_cache.shape[1]):
            out[:, b] = PchipInterpolator(t_cache, T_cache[:, b])(t_query)
    else:
        raise ValueError(f"unknown interpolation method: {method!r}")

    return out


def error_metrics(pred: np.ndarray, ref: np.ndarray):
    err = np.abs(pred - ref)

    flat = err.ravel()

    max_idx = np.unravel_index(np.argmax(err), err.shape)

    return {
        "mae": float(np.mean(flat)),
        "rmse": float(np.sqrt(np.mean((pred - ref) ** 2))),
        "max": float(np.max(flat)),
        "max_time_idx": int(max_idx[0]),
        "max_condition_idx": int(max_idx[1]),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[20_000, 50_000, 100_000, 200_000],
    )

    parser.add_argument(
        "--eval-points",
        type=int,
        default=50_000,
        help="Number of random direct-Cantera validation times (--times random).",
    )

    parser.add_argument(
        "--times",
        choices=["random", "solver"],
        default="random",
        help="Evaluation times: 'random' independent off-grid times (default), or "
             "'solver' = recorded ODE-solver RHS query times (notebook 9.5b).",
    )

    parser.add_argument(
        "--solver-model",
        choices=["trained", "untrained"],
        default="trained",
        help="Model whose weights set the RHS query times (--times solver). "
             "'trained' loads the checkpoint (as 9.5b); 'untrained' uses a seeded model.",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Trained hydrogen checkpoint for --times solver --solver-model trained.",
    )

    parser.add_argument(
        "--seed-model",
        type=int,
        default=0,
        help="Seed for the untrained model (--solver-model untrained).",
    )

    parser.add_argument(
        "--solver-rtol",
        type=float,
        default=1e-6,
        help="Tsit5 rtol for the RHS-time recording integration (train_hydrogen default).",
    )

    parser.add_argument(
        "--solver-atol",
        type=float,
        default=1e-8,
        help="Tsit5 atol for the RHS-time recording integration (train_hydrogen default).",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["linear", "pchip"],
        choices=["linear", "pchip"],
        help="Interpolation methods to evaluate on each cache.",
    )

    parser.add_argument(
        "--t-end",
        type=float,
        default=0.6e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--pressure",
        type=float,
        default=ct.one_atm,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-15,
    )

    args = parser.parse_args()

    conditions = training_conditions()

    _, keep = species_index(MECH, DROP)

    # ---------------------------------------------------------------
    # Evaluation times: random off-grid (default) or recorded solver times (9.5b)
    # ---------------------------------------------------------------

    if args.times == "solver":
        t_eval, args.t_end = make_solver_times(
            args.checkpoint,
            args.solver_model,
            args.seed_model,
            args.solver_rtol,
            args.solver_atol,
        )
        times_desc = (
            f"recorded ODE-solver RHS times (9.5b), model={args.solver_model}, "
            f"tsit5 rtol={args.solver_rtol:g} atol={args.solver_atol:g}"
        )
    else:
        t_eval = make_validation_times(
            args.eval_points,
            args.t_end,
            args.seed,
        )
        times_desc = f"random off-grid times (seed {args.seed})"

    print("=" * 78)
    print("Dense Cantera temperature-resolution convergence")
    print("=" * 78)
    print(f"Conditions      : {len(conditions)}")
    print(f"Eval times      : {len(t_eval):,}  [{times_desc}]")
    print(f"Time interval   : 0 -> {args.t_end:.6e} s")
    print(f"Resolutions     : {args.resolutions}")
    print(f"Methods         : {args.methods}")
    print()

    # ---------------------------------------------------------------
    # Direct Cantera reference at the evaluation times
    # ---------------------------------------------------------------

    print(
        f"Generating direct Cantera reference at "
        f"{len(t_eval):,} validation times..."
    )

    start = time.perf_counter()

    T_ref = cantera_temperature_matrix(
        conditions,
        t_eval,
        args.pressure,
        args.rtol,
        args.atol,
        keep,
    )

    reference_time = time.perf_counter() - start

    print(f"Reference generation: {reference_time:.3f} s")
    print()

    results = []

    # ---------------------------------------------------------------
    # Resolution sweep
    # ---------------------------------------------------------------

    for n in args.resolutions:

        print(f"Testing {n:,}-point cache...")

        t_cache = np.linspace(
            0.0,
            args.t_end,
            n,
            dtype=np.float64,
        )

        start = time.perf_counter()

        T_cache = cantera_temperature_matrix(
            conditions,
            t_cache,
            args.pressure,
            args.rtol,
            args.atol,
            keep,
        )

        generation_time = time.perf_counter() - start

        memory_mb = T_cache.nbytes / (1024**2)

        # Same cache, one row per interpolation method (generation cost shared).
        for method in args.methods:

            start_interp = time.perf_counter()

            T_interp = interpolate_temperature(
                t_cache,
                T_cache,
                t_eval,
                method=method,
            )

            interpolation_time = time.perf_counter() - start_interp

            metrics = error_metrics(T_interp, T_ref)

            worst_condition = conditions[metrics["max_condition_idx"]]
            worst_time = t_eval[metrics["max_time_idx"]]

            row = {
                "resolution": n,
                "method": method,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "max": metrics["max"],
                "generation_s": generation_time,
                "interp_s": interpolation_time,
                "memory_mb": memory_mb,
                "worst_T0": worst_condition[0],
                "worst_phi": worst_condition[1],
                "worst_t": worst_time,
            }

            results.append(row)

            print(
                f"  [{method}]\n"
                f"    MAE       : {metrics['mae']:.8f} K\n"
                f"    RMSE      : {metrics['rmse']:.8f} K\n"
                f"    max error : {metrics['max']:.8f} K\n"
                f"    worst IC  : T0={worst_condition[0]:.0f} K, "
                f"phi={worst_condition[1]:.2f}\n"
                f"    worst time: {worst_time:.9e} s\n"
                f"    interpolation: {interpolation_time:.3f} s\n"
            )

        print(
            f"  generation: {generation_time:.3f} s\n"
            f"  T cache   : {memory_mb:.2f} MiB\n"
        )

    # ---------------------------------------------------------------
    # Final table
    # ---------------------------------------------------------------

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    header = (
        f"{'points':>10} "
        f"{'method':>8} "
        f"{'MAE [K]':>14} "
        f"{'RMSE [K]':>14} "
        f"{'max [K]':>14} "
        f"{'gen [s]':>12} "
        f"{'interp [s]':>12} "
        f"{'T MB':>10}"
    )

    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['resolution']:>10,d} "
            f"{r['method']:>8} "
            f"{r['mae']:>14.8f} "
            f"{r['rmse']:>14.8f} "
            f"{r['max']:>14.8f} "
            f"{r['generation_s']:>12.3f} "
            f"{r['interp_s']:>12.3f} "
            f"{r['memory_mb']:>10.2f}"
        )

    print()
    print(
        "Interpretation: choose the smallest resolution after which increasing "
        "the grid provides no practically meaningful reduction in temperature error."
    )


if __name__ == "__main__":
    main()
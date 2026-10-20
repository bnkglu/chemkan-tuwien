r"""Final ChemKAN inference-time benchmark against the detailed-chemistry reference solver.

One system per call (``--system hydrogen | methane | biodiesel``); results go to
``<this folder>/<system>/``. Nothing is trained, no checkpoint or dataset is written.

Identical initial-value problems
    hydrogen / methane  Cantera IdealGasConstPressureReactor (adiabatic, constant pressure,
        0-D), mechanism / pressure / fuel / air from the dataset and generator
        (``data_gen/reactor.py``), initial state (T0, p, Y0) from ``set_equivalence_ratio``.
        ChemKAN integrates the same (T0, Y0) in its normalized coordinates.
    biodiesel  no Cantera mechanism exists: the reference is the known mass-action kinetics
        of the dataset generator (``generate_biodiesel.integrate_case``, released-code
        reaction orders), SciPy LSODA with the analytic Jacobian; isothermal, same
        (y0, T).
    Both sides return the full physical state at the same 100 output times, uniform over the
    dataset's window (hydrogen 0.6 ms, methane 5 ms, biodiesel 30 s).

Timed region (``time.perf_counter``)
    included: per-trajectory initialization (reference: thermo state, reactor and network
        objects or rate constants; ChemKAN: normalization of the initial state and tensor
        creation), the full ODE integration, and conversion of the output back to physical
        units in a NumPy array.
    excluded: checkpoint / mechanism / dataset loading, model construction, reference-data
        generation, accuracy evaluation, plotting, disk I/O.

Protocol: single-threaded everything (BLAS / OpenMP env = 1, torch intra- and inter-op 1);
``--warmup`` untimed calls, then ``--reps`` timed calls, for
    latency     one trajectory at a time, per condition;
    throughput  all conditions: reference sequentially; ChemKAN sequentially and batched
                (one torchdiffeq call; the batch shares one adaptive step sequence).
Medians and IQR (25th-75th percentile). Speed-up = reference time / ChemKAN time.

Accuracy is computed in the same run from the timed solves' outputs, against the reference
at its data-generation tolerance: time-averaged MSE in the model's own min-max normalized
coordinates (mean over states and the 100 times) plus physical diagnostics.

    ~/uni_projects/chemkan-venv/bin/python results/experiments/inference_benchmark/benchmark.py \
        --system hydrogen
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for sub in ("chemkan/scripts", "chemkan/scripts/author_repo_match", "chemkan/scripts/figures"):
    sys.path.insert(0, str(ROOT / sub))

import author_runs  # noqa: E402  (biodiesel generator import without the figures' common.py)
import plot_trajectory as pt  # noqa: E402

from chemkan.solver import SolverConfig, integrate  # noqa: E402

N_POINTS = 100
SYSTEMS = {
    "hydrogen": {
        "checkpoint": "results/experiments/hydrogen/author_matched/main/S2_seed0_200k/stage2_final.pt",
        "fuel": "H2", "drop": ("AR",), "major": ["H2", "O2", "H2O"]},
    # methane: the best 5k Stage-2 snapshot by training-condition full-trajectory MSE (the final
    # 200k model sits on a loss spike); pass --checkpoint to time another one
    "methane": {
        "checkpoint": "results/experiments/methane/author_matched/main/S2_seed0/archive/"
                      "stage2_S2_seed0_epoch180000.pt",
        "fuel": "CH4", "drop": ("AR",), "major": ["CH4", "O2", "CO2", "H2O", "CO"]},
    "biodiesel": {
        "checkpoint": "results/experiments/biodiesel/author_repo_match/figures/fig05/"
                      "chemkan_noise00_seed0/checkpoint_final.pt"},
}
AIR = {"O2": 1.0, "N2": 3.76}
GAS_REFERENCE_TOL = (1e-9, 1e-15)            # generate_hydrogen.py / generate_methane.py
BIODIESEL_REFERENCE_TOL = (1e-10, 1e-12)     # generate_biodiesel.integrate_case


# ----------------------------------------------------------------------------- utilities
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"median": float(med), "q25": float(q1), "q75": float(q3), "iqr": float(q3 - q1),
            "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}


def timed(fn, warmup: int, reps: int):
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return times, out


# ----------------------------------------------------------------------------- problems
class GasProblem:
    """Hydrogen / methane: Cantera reactor vs the author-matched Stage-2 ChemKAN."""

    def __init__(self, system: str, ckpt: dict):
        import cantera as ct
        self.ct, spec = ct, SYSTEMS[system]
        self.model = pt.load_model(ckpt)                     # checks the dataset sha256
        if self.model["temperature"] != "predicted":
            raise SystemExit("not a Stage-2 checkpoint")
        data = np.load(self.model["data"], allow_pickle=True)
        self.species = [str(s) for s in data["species"]]
        self.names = self.species + ["T"]
        self.ics, self.is_test = data["ics"], data["is_test"].astype(bool)
        self.labels = [f"T0={a:g} K, phi={b:g}" for a, b in self.ics]
        self.mech, self.pressure, self.fuel = str(data["mechanism"]), float(data["pressure"]), spec["fuel"]
        self.t = np.linspace(0.0, float(data["t"][-1]), N_POINTS)
        self.dataset_t, self.dataset_states = np.asarray(data["t"], float), data["states"]
        self.gas = ct.Solution(self.mech)                    # mechanism loaded once, untimed
        self.keep = np.array([self.gas.species_index(s) for s in self.species])
        kept = [s for s in self.gas.species_names if s not in spec["drop"]]
        if kept != self.species:
            raise SystemExit("dataset species differ from the mechanism minus the dropped inert")
        self.y0 = []
        for T0, phi in self.ics:                             # (T0, p, Y0) of each IVP
            self.gas.set_equivalence_ratio(phi, self.fuel, AIR)
            self.gas.TP = T0, self.pressure
            self.y0.append(self.gas.Y.copy())
        self.u0 = np.array([np.append(y[self.keep], T0) for y, (T0, _) in zip(self.y0, self.ics)])
        self.u_min, self.u_max = self.model["u_min"], self.model["u_max"]
        self.major = spec["major"]
        self.reference_tol = GAS_REFERENCE_TOL
        trained = self.model["solver"]
        # "biodiesel_tolerance": the released biodiesel example's rtol 1e-2 / atol 1e-6
        self.chemkan_tols = {"trained": (trained.rtol, trained.atol),
                             "biodiesel_tolerance": (1e-2, 1e-6)}
        # The model's own (1e-6, 1e-8) is also tried: Cantera returns NaN rates for some
        # conditions at atol 1e-8 on mass fractions; that failure is recorded, not hidden.
        self.reference_tols = {"generator": GAS_REFERENCE_TOL, "rtol1e-6_atol1e-12": (1e-6, 1e-12),
                               "rtol1e-6_atol1e-10": (1e-6, 1e-10),
                               "chemkan_tolerance": (trained.rtol, trained.atol),
                               "biodiesel_tolerance": (1e-2, 1e-6)}
        self.dtype = self.model["dtype"]
        self.t_torch = torch.as_tensor(self.t, dtype=self.dtype)
        self.precision = {"chemkan": str(self.dtype).replace("torch.", ""), "reference": "float64"}
        self.reference_label = (f"Cantera {ct.__version__} IdealGasConstPressureReactor, "
                                f"{self.mech} (AR dropped from the state), CVODES BDF")

    def reference(self, i: int, rtol: float, atol: float) -> np.ndarray:
        ct, gas = self.ct, self.gas
        gas.TPY = self.ics[i][0], self.pressure, self.y0[i]
        reactor = ct.IdealGasConstPressureReactor(gas)
        net = ct.ReactorNet([reactor])
        net.rtol, net.atol = rtol, atol
        out = np.empty((len(self.t), len(self.keep) + 1))
        out[0, :-1], out[0, -1] = gas.Y[self.keep], gas.T
        for j in range(1, len(self.t)):
            net.advance(self.t[j])
            out[j, :-1], out[j, -1] = reactor.Y[self.keep], reactor.T
        return out

    def chemkan(self, idx, rtol: float, atol: float) -> np.ndarray:
        """Physical states (T, len(idx), k) for the conditions ``idx`` in one solve."""
        span = self.u_max - self.u_min
        z0 = torch.as_tensor((self.u0[idx] - self.u_min) / span, dtype=self.dtype)
        solver = SolverConfig(method="tsit5", rtol=rtol, atol=atol, sensitivity="direct_autograd")
        with torch.no_grad():
            z = integrate(self.model["rhs"], z0, self.t_torch, solver)
        return z.numpy().astype(float) * span + self.u_min

    def dataset_check(self, reference: np.ndarray) -> dict:
        """The benchmark's reference against the stored dataset (same 100-point grid)."""
        same_grid = len(self.dataset_t) == len(self.t) and np.allclose(self.dataset_t, self.t)
        if not same_grid:
            return {"same_grid": False}
        diff = np.abs(reference - np.transpose(self.dataset_states, (1, 0, 2)))
        return {"same_grid": True, "max_abs_diff_species": float(diff[..., :-1].max()),
                "max_abs_diff_T_K": float(diff[..., -1].max())}

    def physical(self, pred: np.ndarray, ref: np.ndarray) -> dict:
        """Per-condition physical diagnostics; arrays are (T, n, k)."""
        t, Tp, Tr = self.t, pred[..., -1], ref[..., -1]
        delay = lambda T: t[np.argmax(np.gradient(T, t, axis=0), axis=0)]   # noqa: E731
        dp, dr = delay(Tp), delay(Tr)
        major = [self.species.index(s) for s in self.major]
        return {
            "peak_T_abs_error_K": np.abs(Tp.max(0) - Tr.max(0)),
            "final_T_abs_error_K": np.abs(Tp[-1] - Tr[-1]),
            "max_abs_T_error_K": np.abs(Tp - Tr).max(0),
            "ignition_delay_reference_s": dr,
            "ignition_delay_abs_error_s": np.abs(dp - dr),
            "ignition_delay_rel_error": np.abs(dp - dr) / dr,
            "max_abs_Y_error_major": np.abs(pred[..., major] - ref[..., major]).max((0, 2)),
            "max_abs_Y_error_all": np.abs(pred[..., :-1] - ref[..., :-1]).max((0, 2)),
            "max_element_imbalance_rel": self.element_imbalance(pred),
        }

    def element_imbalance(self, pred: np.ndarray) -> np.ndarray:
        """Largest relative drift of any element's mass fraction from t = 0 (per condition)."""
        gas, keep = self.gas, self.keep
        W = gas.molecular_weights[keep]
        elems = [e for e in gas.element_names if e != "Ar"]
        a = np.array([[gas.n_atoms(gas.species_names[k], e) for k in keep] for e in elems])
        aw = np.array([gas.atomic_weight(e) for e in elems])
        Z = np.einsum("tnk,ek->tne", pred[..., :-1] / W, a) * aw          # element mass fractions
        Z0 = Z[0]
        rel = np.abs(Z - Z0) / np.where(Z0 > 0, Z0, np.inf)
        return rel.max((0, 2))


class BiodieselProblem:
    """Biodiesel (7b noise-free run): known kinetics (LSODA) vs ChemKAN."""

    def __init__(self, ckpt: dict):
        self.gb = author_runs._generator()
        self.model = pt.load_model(ckpt)
        data = np.load(self.model["data"], allow_pickle=True)
        if author_runs.DATA_SHA256 != pt.checkpoint_sha256(self.model["data"]):
            raise SystemExit("biodiesel_v2.npz differs from the file the 7b runs used")
        self.species = [str(s) for s in data["species"]]
        self.names = list(self.species)
        states = np.concatenate([data["train_states"], data["test_states"]])
        temps = np.concatenate([data["train_T"], data["test_T"]]).astype(float)
        self.is_test = np.arange(len(states)) >= len(data["train_states"])
        self.ics = np.column_stack([states[:, 0, 0], states[:, 0, 1], temps])
        self.labels = [f"TG0={a:.3f}, ROH0={b:.3f}, T={c:.2f} K" for a, b, c in self.ics]
        self.y0, self.T = states[:, 0, :], temps
        self.t = np.linspace(0.0, float(data["t"][-1]), N_POINTS)
        self.dataset_t, self.dataset_states = np.asarray(data["t"], float), states
        self.u_min, self.u_max = self.model["u_min"], self.model["u_max"]   # 6 species + T
        self.order = "released_code"
        self.reference_tol = BIODIESEL_REFERENCE_TOL
        trained = self.model["solver"]
        self.chemkan_tols = {"evaluation_7b": (author_runs.EVAL_RTOL, author_runs.EVAL_ATOL),
                             "trained": (trained.rtol, trained.atol)}
        self.reference_tols = {"generator": BIODIESEL_REFERENCE_TOL, "loose": (1e-6, 1e-8)}
        self.dtype = self.model["dtype"]
        self.t_torch = torch.as_tensor(self.t, dtype=self.dtype)
        self.precision = {"chemkan": "float64", "reference": "float64"}
        import scipy
        self.reference_label = (f"SciPy {scipy.__version__} solve_ivp LSODA with analytic "
                                "Jacobian, known mass-action kinetics (released-code orders)")

    def reference(self, i: int, rtol: float, atol: float) -> np.ndarray:
        gb = self.gb
        k = gb.rate_constants(self.T[i], self.order)
        sol = gb.solve_ivp(gb.rhs, (self.t[0], self.t[-1]), self.y0[i], t_eval=self.t,
                           method="LSODA", jac=gb.jac, args=(k,), rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y.T

    def chemkan(self, idx, rtol: float, atol: float) -> np.ndarray:
        span = self.u_max - self.u_min
        u0 = np.column_stack([self.y0[idx], self.T[idx]])
        z0 = torch.as_tensor((u0 - self.u_min) / span, dtype=self.dtype)
        solver = SolverConfig(method="tsit5", rtol=rtol, atol=atol, sensitivity="direct_autograd")
        with torch.no_grad():
            z = integrate(self.model["rhs"], z0, self.t_torch, solver)
        return z[..., :6].numpy() * span[:6] + self.u_min[:6]

    def dataset_check(self, reference=None) -> dict:
        """Reference re-solved on the dataset's 30-point grid against the stored states."""
        diff = 0.0
        for i in range(len(self.y0)):
            k = self.gb.rate_constants(self.T[i], self.order)
            sol = self.gb.solve_ivp(self.gb.rhs, (0.0, self.dataset_t[-1]), self.y0[i],
                                    t_eval=self.dataset_t, method="LSODA", jac=self.gb.jac,
                                    args=(k,), rtol=self.reference_tol[0], atol=self.reference_tol[1])
            diff = max(diff, float(np.abs(np.clip(sol.y.T, 0, None) - self.dataset_states[i]).max()))
        return {"same_grid": False, "dataset_grid_points": len(self.dataset_t),
                "max_abs_diff_on_dataset_grid_mol_per_L": diff}

    def dataset_grid_mse(self, rtol: float, atol: float) -> dict:
        """ChemKAN on the 30-point dataset grid vs the stored clean data (7b's numbers)."""
        span = self.u_max[:6] - self.u_min[:6]
        t = torch.as_tensor(self.dataset_t, dtype=self.dtype)
        u0 = np.column_stack([self.y0, self.T])
        z0 = torch.as_tensor((u0 - self.u_min) / (self.u_max - self.u_min), dtype=self.dtype)
        solver = SolverConfig(method="tsit5", rtol=rtol, atol=atol, sensitivity="direct_autograd")
        with torch.no_grad():
            z = integrate(self.model["rhs"], z0, t, solver)[..., :6].numpy()
        target = (np.transpose(self.dataset_states, (1, 0, 2)) - self.u_min[:6]) / span
        mse = ((z - target) ** 2).mean(axis=(0, 2))
        return {"train_mean": float(mse[~self.is_test].mean()),
                "test_mean": float(mse[self.is_test].mean())}

    def physical(self, pred: np.ndarray, ref: np.ndarray) -> dict:
        tg, fame = self.species.index("TG"), self.species.index("RCO2R")
        conv = lambda y: 1.0 - y[-1, :, tg] / y[0, :, tg]                    # noqa: E731
        return {
            "max_abs_conc_error_mol_per_L": np.abs(pred - ref).max((0, 2)),
            "final_FAME_abs_error_mol_per_L": np.abs(pred[-1, :, fame] - ref[-1, :, fame]),
            "final_FAME_rel_error": np.abs(pred[-1, :, fame] - ref[-1, :, fame]) / ref[-1, :, fame],
            "final_TG_conversion_abs_error": np.abs(conv(pred) - conv(ref)),
            "min_predicted_conc_mol_per_L": pred.min((0, 2)),
        }


# ----------------------------------------------------------------------------- benchmark
def normalized_mse(problem, pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Time-averaged MSE per condition in the model's normalized coordinates."""
    k = pred.shape[-1]
    span = problem.u_max[:k] - problem.u_min[:k]
    return (((pred - ref) / span) ** 2).mean(axis=(0, 2))


def split_summary(problem, values) -> dict:
    v = np.asarray(values, dtype=float)
    out = {"all": stats(v)}
    out["train"] = stats(v[~problem.is_test])
    if problem.is_test.any():
        out["test"] = stats(v[problem.is_test])
    return out


def run(system: str, warmup: int, reps: int, out_dir: Path, checkpoint: str | None = None) -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = dict(SYSTEMS[system])
    if checkpoint:
        spec["checkpoint"] = checkpoint
    ckpt_path = ROOT / spec["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    problem = BiodieselProblem(ckpt) if system == "biodiesel" else GasProblem(system, ckpt)
    torch.set_default_dtype(problem.dtype)
    n = len(problem.ics)
    allidx = np.arange(n)
    load_start = os.getloadavg()

    # Reference at the generator tolerance: the accuracy target (untimed solve, reused below).
    rt, at = problem.reference_tol
    reference = np.stack([problem.reference(i, rt, at) for i in range(n)], axis=1)  # (T, n, k)

    rows, timing = [], {"latency": {}, "throughput": {}}
    accuracy = {}

    # ---- reference solver at each tolerance
    failed = {}
    for name, (rtol, atol) in problem.reference_tols.items():
        bad = []
        for i in range(n):
            try:
                problem.reference(i, rtol, atol)
            except Exception:  # noqa: BLE001
                bad.append(i)
        if bad:
            failed[f"reference_{name}"] = {"rtol": rtol, "atol": atol, "failed_conditions": bad,
                                           "failed_labels": [problem.labels[i] for i in bad]}
            print(f"reference at rtol {rtol:g} atol {atol:g} fails for conditions {bad}; not timed")
            continue
        lat = []
        for i in range(n):
            times, _ = timed(lambda: problem.reference(i, rtol, atol), warmup, reps)
            lat.append(times)
            rows += [{"system": system, "solver": f"reference_{name}", "condition": i,
                      "rep": r, "seconds": s} for r, s in enumerate(times)]
        seq, pred = timed(lambda: np.stack([problem.reference(i, rtol, atol) for i in allidx], 1),
                          warmup, reps)
        key = f"reference_{name}"
        timing["latency"][key] = {"rtol": rtol, "atol": atol, **stats(np.ravel(lat)),
                                  "per_condition_median_s": [float(np.median(x)) for x in lat]}
        timing["throughput"][f"{key}_sequential"] = {
            "rtol": rtol, "atol": atol, **stats(seq), "trajectories": n,
            "trajectories_per_s": n / float(np.median(seq))}
        accuracy[key] = {"rtol": rtol, "atol": atol,
                         "time_averaged_normalized_mse": split_summary(problem, normalized_mse(problem, pred, reference)),
                         "per_condition_mse": normalized_mse(problem, pred, reference).tolist()}

    # ---- ChemKAN at each tolerance
    for name, (rtol, atol) in problem.chemkan_tols.items():
        lat, single = [], []
        for i in range(n):
            times, out = timed(lambda: problem.chemkan([i], rtol, atol), warmup, reps)
            lat.append(times)
            single.append(out[:, 0])
            rows += [{"system": system, "solver": f"chemkan_{name}", "condition": i,
                      "rep": r, "seconds": s} for r, s in enumerate(times)]
        single = np.stack(single, axis=1)
        seq, _ = timed(lambda: np.concatenate([problem.chemkan([i], rtol, atol) for i in allidx], 1),
                       warmup, reps)
        bat, batched = timed(lambda: problem.chemkan(allidx, rtol, atol), warmup, reps)
        key = f"chemkan_{name}"
        timing["latency"][key] = {"rtol": rtol, "atol": atol, **stats(np.ravel(lat)),
                                  "per_condition_median_s": [float(np.median(x)) for x in lat]}
        timing["throughput"][f"{key}_sequential"] = {
            "rtol": rtol, "atol": atol, **stats(seq), "trajectories": n,
            "trajectories_per_s": n / float(np.median(seq))}
        timing["throughput"][f"{key}_batched"] = {
            "rtol": rtol, "atol": atol, **stats(bat), "trajectories": n,
            "trajectories_per_s": n / float(np.median(bat))}
        mse_single = normalized_mse(problem, single, reference)
        accuracy[key] = {
            "rtol": rtol, "atol": atol,
            "time_averaged_normalized_mse": split_summary(problem, mse_single),
            "per_condition_mse": mse_single.tolist(),
            "batched_time_averaged_normalized_mse": split_summary(
                problem, normalized_mse(problem, batched, reference)),
            "physical": {k: split_summary(problem, v)
                         for k, v in problem.physical(single, reference).items()},
            "max_abs_diff_single_vs_batched_normalized": float(
                np.abs((single - batched) / (problem.u_max - problem.u_min)[:single.shape[-1]]).max()),
        }
        if system == "biodiesel":
            accuracy[key]["dataset_grid_30pt_mse_vs_stored_data"] = problem.dataset_grid_mse(rtol, atol)

    # ---- speed-ups (reference time / ChemKAN time)
    speedup = {}
    for kname in problem.chemkan_tols:
        ck = f"chemkan_{kname}"
        for rname in problem.reference_tols:
            rk = f"reference_{rname}"
            if rk in failed:
                continue
            ref_lat = np.array(timing["latency"][rk]["per_condition_median_s"])
            kan_lat = np.array(timing["latency"][ck]["per_condition_median_s"])
            speedup[f"{rk}/{ck}"] = {
                "latency_median_of_per_condition_ratios": float(np.median(ref_lat / kan_lat)),
                "latency_ratio_of_medians": timing["latency"][rk]["median"] / timing["latency"][ck]["median"],
                "throughput_vs_chemkan_sequential":
                    timing["throughput"][f"{rk}_sequential"]["median"]
                    / timing["throughput"][f"{ck}_sequential"]["median"],
                "throughput_vs_chemkan_batched":
                    timing["throughput"][f"{rk}_sequential"]["median"]
                    / timing["throughput"][f"{ck}_batched"]["median"],
            }

    import cantera
    import scipy
    import torchdiffeq
    result = {
        "system": system,
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": pt.checkpoint_sha256(ckpt_path),
        "checkpoint_epochs": problem.model.get("epochs"),
        "dataset": str(Path(problem.model["data"]).relative_to(ROOT)),
        "dataset_sha256": pt.checkpoint_sha256(problem.model["data"]),
        "conditions": n, "train_conditions": int((~problem.is_test).sum()),
        "test_conditions": int(problem.is_test.sum()),
        "condition_labels": problem.labels,
        "is_test": problem.is_test.tolist(),
        "output_grid": {"points": N_POINTS, "t_start_s": 0.0, "t_end_s": float(problem.t[-1]),
                        "spacing": "uniform"},
        "states": problem.names,
        "reference": problem.reference_label,
        "reference_accuracy_target": {"rtol": problem.reference_tol[0], "atol": problem.reference_tol[1]},
        "reference_vs_dataset": problem.dataset_check(reference),
        "chemkan": {"solver": "torchdiffeq Tsit5 (adaptive, dopri-type error control), no_grad",
                    "parameters": int(sum(p.numel() for p in ckpt["model_state"].values())),
                    "normalization": "min-max with the checkpoint's training statistics"},
        "precision": problem.precision,
        "timing_protocol": {
            "warmup_calls": warmup, "timed_repetitions": reps, "clock": "time.perf_counter",
            "included": "per-trajectory initialization, ODE integration, output in physical units",
            "excluded": "model/mechanism/data loading, training, reference-data generation, "
                        "accuracy evaluation, plotting, disk I/O",
            "threads": {"torch_intra_op": torch.get_num_threads(),
                        "torch_inter_op": torch.get_num_interop_threads(),
                        "blas_openmp_env": 1, "cantera_scipy": "single-threaded"},
            "statistic": "median and IQR (25th-75th percentile)",
            "latency_samples": "conditions x repetitions, one trajectory per call",
        },
        "reference_settings_that_failed": failed,
        "timing": timing,
        "speedup_reference_over_chemkan": speedup,
        "accuracy_vs_reference": accuracy,
        "environment": {
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip() or platform.processor(),
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torchdiffeq": torchdiffeq.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "cantera": cantera.__version__,
            "load_average_start": load_start, "load_average_end": os.getloadavg(),
            "git_commit": git_commit(),
        },
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    with open(out_dir / "latency_samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "per_condition.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(accuracy)
        w.writerow(["condition", "label", "split"]
                   + [f"{k}_latency_median_s" for k in timing["latency"]]
                   + [f"{k}_mse" for k in keys])
        for i in range(n):
            w.writerow([i, problem.labels[i], "test" if problem.is_test[i] else "train"]
                       + [timing["latency"][k]["per_condition_median_s"][i] for k in timing["latency"]]
                       + [accuracy[k]["per_condition_mse"][i] for k in keys])
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=list(SYSTEMS), required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", default=None, help="repository-relative checkpoint (default: SYSTEMS)")
    ap.add_argument("--smoke", action="store_true", help="allow < 20 repetitions (code check only)")
    args = ap.parse_args(argv)
    if args.reps < 20 and not args.smoke:
        raise SystemExit("the protocol needs at least 20 timed repetitions")
    out = args.out_dir or HERE / args.system
    res = run(args.system, args.warmup, args.reps, out, args.checkpoint)
    for k, v in res["timing"]["latency"].items():
        print(f"latency    {k:32s} median {v['median']:.4e} s  IQR {v['iqr']:.2e}")
    for k, v in res["timing"]["throughput"].items():
        print(f"throughput {k:40s} median {v['median']:.4e} s  {v['trajectories_per_s']:.1f} traj/s")
    for k, v in res["speedup_reference_over_chemkan"].items():
        print(f"speed-up   {k}: latency {v['latency_ratio_of_medians']:.3g}x, "
              f"throughput seq {v['throughput_vs_chemkan_sequential']:.3g}x / "
              f"batched {v['throughput_vs_chemkan_batched']:.3g}x")
    for k, v in res["accuracy_vs_reference"].items():
        m = v["time_averaged_normalized_mse"]
        print(f"accuracy   {k:32s} MSE median {m['all']['median']:.3e}"
              + (f"  test {m['test']['median']:.3e}" if "test" in m else ""))
    print("reference vs dataset:", res["reference_vs_dataset"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

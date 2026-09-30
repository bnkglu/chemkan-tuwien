r"""Loading and verification of the biodiesel_v2 figure runs
(``results/experiments/biodiesel/author_repo_match/figures/``, planned by
``chemkan/scripts/author_repo_match/run_figures.py``).

Loss convention. Everything returned here is in the TIME-AVERAGED convention of the
released code (``Flux.mse``: mean over the 6 species and the 30 observation times, then
mean over trajectories). The ChemKAN trainer logs that directly (``train_mse``,
``val_mse``, ``val_mse_noisy``); the DeepONet trainer logs Eq. 18 (sum over time), so its
columns are divided by N_t = 30 here. Eq. 18 = 30 x time-averaged.

    history(run) -> dict of arrays: epoch (1-based), train, test_clean, test_noisy (or None)
    verify()     -> (rows, problems)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "chemkan/scripts/author_repo_match"))
sys.path.insert(0, str(ROOT / "deeponet"))
import run_figures  # noqa: E402

RUNS = run_figures.OUT
DATA = ROOT / run_figures.DATA
DATA_SHA256 = "c79afbfbbee863f50704985aaf658931d8e989f8086d528f97dfd6d680ca7b31"
N_T = 30
CONVENTION = "time-averaged MSE (released-code Flux.mse; Eq. 18 = 30 x this)"
CONVENTION_SHORT = "time-averaged MSE (Eq. 18 / 30)"          # axis labels and titles


def plan() -> list[dict]:
    return run_figures.plan()


def by_name() -> dict[str, dict]:
    return {r["name"]: r for r in plan()}


def _rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def config(run: dict) -> dict:
    return json.loads((run["dir"] / "config.json").read_text())


def history(run: dict) -> dict:
    """Per-epoch losses in the time-averaged convention; epoch is 1-based for both models."""
    rows = _rows(run["dir"] / "history.csv")
    col = lambda k: np.array([float(r[k]) for r in rows]) if rows and k in rows[0] else None  # noqa: E731
    if run["model"] == "chemkan":
        return {"epoch": col("epoch"), "train": col("train_mse"), "test_clean": col("val_mse"),
                "test_noisy": col("val_mse_noisy")}
    scale = 1.0 / N_T                                         # DeepONet logs Eq. 18
    div = lambda a: None if a is None else a * scale         # noqa: E731
    return {"epoch": col("epoch") + 1, "train": div(col("mse_loss")),
            "test_clean": div(col("test_mse_clean")), "test_noisy": div(col("test_mse_noisy"))}


def _expected_settings(run: dict, cfg: dict) -> list[str]:
    """Differences between the run's config.json and run_figures.py's plan."""
    bad = []
    a = run["args"]
    get = lambda flag: a[a.index(flag) + 1] if flag in a else None  # noqa: E731
    noise = get("--noise-percent")
    if cfg.get("seed") != 0:
        bad.append(f"seed {cfg.get('seed')}")
    if cfg.get("epochs") != run["epochs"]:
        bad.append(f"epochs {cfg.get('epochs')} != {run['epochs']}")
    if run["model"] == "chemkan":
        src, sol, arch = cfg["data_source"], cfg["solver"], cfg["architecture"]
        if src.get("path") != run_figures.DATA or src.get("sha256") != DATA_SHA256:
            bad.append(f"data {src.get('path')} {str(src.get('sha256'))[:8]}")
        if (sol.get("sensitivity"), sol.get("per_trajectory_solves")) != ("fsa", False):
            bad.append(f"solver {sol.get('sensitivity')}/{sol.get('per_trajectory_solves')}")
        if str(arch.get("hidden")) != get("--hidden") or arch.get("parameter_count") != 39 * int(get("--hidden")):
            bad.append(f"hidden {arch.get('hidden')} / {arch.get('parameter_count')}")
        if str(src.get("noise_percent")) != str(noise):
            bad.append(f"noise {src.get('noise_percent')} != {noise}")
    else:
        df, arch = cfg.get("dataset_file") or {}, cfg["architecture"]
        if df.get("path") != run_figures.DATA or df.get("sha256") != DATA_SHA256:
            bad.append(f"data {df.get('path')} {str(df.get('sha256'))[:8]}")
        if str(arch.get("w")) != get("--width"):
            bad.append(f"width {arch.get('w')}")
        q = get("--trunk-hidden") or str(int(get("--width")) - 1)
        if str(arch.get("q")) != q or arch["branch_dims"][0] != 3:
            bad.append(f"q {arch.get('q')} branch_in {arch['branch_dims'][0]}")
        got_noise = None if cfg.get("noise") is None else str(cfg["noise"]["percent"])
        if got_noise != noise:
            bad.append(f"noise {got_noise} != {noise}")
        want_eval = 1 if "--eval-every" in a else None
        got_eval = (cfg.get("in_training_evaluation") or {}).get("eval_every")
        if got_eval != want_eval:
            bad.append(f"eval_every {got_eval}")
    return bad


def verify() -> tuple[list[dict], list[str]]:
    """One row per planned run; problems lists everything that is not as planned."""
    rows, problems = [], []
    for run in plan():
        name, d = run["name"], run["dir"]
        row = {"run": name, "model": run["model"], "status": "ok"}
        issues = []
        if not (d / "checkpoint_final.pt").exists():
            issues.append("no checkpoint_final.pt")
        if not (d / "history.csv").exists() or not (d / "config.json").exists():
            issues.append("missing history.csv/config.json")
        else:
            h = history(run)
            row["epochs"] = len(h["epoch"])
            if row["epochs"] != run["epochs"] or not np.array_equal(h["epoch"], np.arange(1, run["epochs"] + 1)):
                issues.append(f"history has {row['epochs']} epochs, expected {run['epochs']}")
            finite = all(np.isfinite(v).all() for v in h.values() if v is not None)
            if not finite:
                issues.append("NaN/inf in history")
            cfg = config(run)
            row["parameters"] = cfg["architecture"]["parameter_count"] if run["model"] == "chemkan" \
                else cfg["parameter_count"]
            issues += _expected_settings(run, cfg)
            row["train_final"] = float(h["train"][-1])
            row["test_clean_final"] = None if h["test_clean"] is None else float(h["test_clean"][-1])
            row["test_noisy_final"] = None if h["test_noisy"] is None else float(h["test_noisy"][-1])
        if issues:
            row["status"] = "; ".join(issues)
            problems.append(f"{name}: {row['status']}")
        rows.append(row)
    return rows, problems


# --------------------------------------------------------------------------- predictions / losses

FIG3_TEST_INDEX = 3            # test row 24 of diesel_u0_samples.txt: TG 1.941, ROH 1.433, T 334.75 K
SPECIES = ["TG", "ROH", "DG", "MG", "GL", "RCO2R"]


def _paths():
    for sub in ("chemkan/scripts", "chemkan/scripts/author_repo_match", "deeponet"):
        if str(ROOT / sub) not in sys.path:
            sys.path.insert(0, str(ROOT / sub))


def _generator():
    """``data_gen/generate_biodiesel.py``, imported with ITS ``common`` module (the figures
    folder has another ``common.py`` that would otherwise shadow it)."""
    import importlib
    if "generate_biodiesel" in sys.modules:
        return sys.modules["generate_biodiesel"]
    shadow = sys.modules.pop("common", None)
    sys.path.insert(0, str(ROOT / "chemkan/scripts/data_gen"))
    try:
        return importlib.import_module("generate_biodiesel")
    finally:
        sys.path.remove(str(ROOT / "chemkan/scripts/data_gen"))
        sys.modules.pop("common", None)
        if shadow is not None:
            sys.modules["common"] = shadow


def dataset() -> dict:
    """The v2 arrays (physical units), loaded directly."""
    with np.load(DATA) as d:
        return {k: d[k] for k in d.files}


def fig3_case(noise_percents=(0, 5, 10, 15), n_dense: int = 301) -> dict:
    """Test row 24: dense ground truth (released-code kinetics) and noisy test observations."""
    gb = _generator()
    d = dataset()
    y0, T = d["test_states"][FIG3_TEST_INDEX, 0], float(d["test_T"][FIG3_TEST_INDEX])
    t_dense = np.linspace(0.0, float(d["t"][-1]), n_dense)
    truth = np.clip(gb.integrate_case(y0, T, t_dense, "released_code"), 0.0, None)
    obs = {p: d[f"test_states_noise{p:02d}"][FIG3_TEST_INDEX] for p in noise_percents}
    return {"t": d["t"], "t_dense": t_dense, "truth": truth, "observations": obs,
            "y0": y0, "T": T, "species": list(d["species"])}


def chemkan_predict(run: dict, y0, T: float, times) -> np.ndarray:
    """Physical species (len(times), 6) from a ChemKAN figure run, integrated with its own
    training solver and normalization."""
    import torch
    _paths()
    from _author_match import NormalizedDynamics, build_core
    from chemkan.solver import SolverConfig, integrate
    cfg = config(run)
    src, sol = cfg["data_source"], cfg["solver"]
    core = build_core(cfg["architecture"]["hidden"]).double()
    state = torch.load(run["dir"] / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    core.load_state_dict(state["model_state"])
    y_min, y_max = np.array(src["species_min"]), np.array(src["species_max"])
    z0 = np.concatenate([(np.asarray(y0) - y_min) / (y_max - y_min),
                         [(T - src["T_min"]) / (src["T_max"] - src["T_min"])]])
    solver = SolverConfig(method=sol["method"], rtol=sol["rtol"], atol=sol["atol"],
                          sensitivity="direct_autograd")
    with torch.no_grad():
        z = integrate(NormalizedDynamics(core), torch.tensor(z0, dtype=torch.float64)[None],
                      torch.as_tensor(np.asarray(times), dtype=torch.float64), solver)
    return z[:, 0, :6].numpy() * (y_max - y_min) + y_min


def deeponet_predict(run: dict, y0, T: float, times) -> np.ndarray:
    """Physical species (len(times), 6) from a DeepONet figure run."""
    import torch
    _paths()
    from biodiesel_deeponet import prepare_inputs
    from evaluate_biodiesel_deeponet import build_model
    from chemkan.normalization import MinMaxNormalizer
    ckpt = torch.load(run["dir"] / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt, "cpu")
    n = ckpt["normalization"]
    full = MinMaxNormalizer(n["u_min"], n["u_max"])
    data = {"Y0": torch.as_tensor(np.asarray(y0), dtype=torch.float32)[None],
            "T_const": torch.tensor([T], dtype=torch.float32),
            "t": torch.as_tensor(np.asarray(times), dtype=torch.float32)}
    branch, tau = prepare_inputs(data, full, float(n["t_end_s"]), branch_in=model.branch_dims[0])
    with torch.no_grad():
        return full.subset(slice(0, 6)).denormalize(model(branch, tau))[:, 0, :].numpy()


def final_losses(run: dict) -> dict:
    """Train / noise-free test loss of the FINAL checkpoint (time-averaged), Fig. 4.

    ChemKAN: metrics.json, evaluated after the last update with the training solver.
    DeepONet: the checkpoint evaluated on biodiesel_v2.npz (its history has no test loss).
    """
    if run["model"] == "chemkan":
        fp = json.loads((run["dir"] / "metrics.json").read_text())["final_parameters"]
        v = fp["training_solver_rtol1e-2"]
        return {"train": v["train_mse"], "test_clean": v["val_mse"]}
    _paths()
    from evaluate_biodiesel_deeponet import evaluate
    ck = run["dir"] / "checkpoint_final.pt"
    tr = evaluate(ck, split="train", data_file=DATA)
    te = evaluate(ck, split="test", data_file=DATA)
    return {"train": tr["mse"] / N_T, "test_clean": te["mse_clean"] / N_T}


def converged(run: dict) -> dict:
    """Converged losses of a Fig. 5 run: the FINAL checkpoint (after the last Adam step of
    the 10,000 epochs), time-averaged -- train (on the run's noisy observations), noisy
    test and noise-free test. At 0 % noise the noisy test set equals the noise-free one.

    ChemKAN: train and noise-free test from metrics.json (the trainer's final evaluation:
    training solver, one solve per trajectory); the noisy test is computed here the same
    way, since metrics.json does not store it. DeepONet: the checkpoint evaluated on
    biodiesel_v2.npz at the run's noise level.
    """
    a = run["args"]
    p = int(a[a.index("--noise-percent") + 1]) if "--noise-percent" in a else 0
    if run["model"] == "chemkan":
        out = final_losses(run)
        out["test_noisy"] = _chemkan_final_noisy_test(run, p) if p else out["test_clean"]
        return out
    _paths()
    from evaluate_biodiesel_deeponet import evaluate
    ck, level = run["dir"] / "checkpoint_final.pt", p or None
    tr = evaluate(ck, split="train", noise_percent=level, data_file=DATA)
    te = evaluate(ck, split="test", noise_percent=level, data_file=DATA)
    return {"train": tr["mse"] / N_T, "test_clean": te["mse_clean"] / N_T,
            "test_noisy": te["mse"] / N_T}


def _chemkan_final_noisy_test(run: dict, percent: int) -> float:
    """Noisy-test loss of a ChemKAN final checkpoint, exactly as the trainer's final
    evaluation (its data loader, training solver, one solve per trajectory, julia_mse)."""
    import torch
    _paths()
    from _author_match import NormalizedDynamics, build_core, julia_mse
    from chemkan.solver import SolverConfig
    from train_author_repo_match import load_canonical_data, predict
    cfg = config(run)
    core = build_core(cfg["architecture"]["hidden"]).double()
    core.load_state_dict(torch.load(run["dir"] / "checkpoint_final.pt", map_location="cpu",
                                    weights_only=False)["model_state"])
    data = load_canonical_data(torch.device("cpu"), DATA, percent)
    sol = cfg["solver"]
    solver = SolverConfig(method=sol["method"], rtol=sol["rtol"], atol=sol["atol"],
                          sensitivity="direct_autograd")
    with torch.no_grad():
        pred = predict(NormalizedDynamics(core), data["u0"][data["test"]], data["t"], solver)
    return float(julia_mse(pred, data["test_noisy_target"]))


def noise_floor(percent: int) -> float:
    """Time-averaged MSE between the noisy and the clean TRAINING targets at ``percent``,
    in the run's normalization: the training-loss increase a perfect model would show."""
    run = by_name()[f"fig05_chemkan_noise{percent:02d}"]
    src = config(run)["data_source"]
    d = dataset()
    rng = np.array(src["species_max"]) - np.array(src["species_min"])
    diff = (d[f"train_states_noise{percent:02d}"] - d["train_states"]) / rng
    return float(np.mean(diff ** 2))


def _fmt(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3e}"


if __name__ == "__main__":
    rows, problems = verify()
    print(f"{'run':26s} {'status':6s} {'epochs':>6s} {'params':>6s} {'train':>10s} {'test clean':>10s} {'test noisy':>10s}")
    for r in rows:
        print(f"{r['run']:26s} {r['status'][:6]:6s} {r.get('epochs', 0):6d} {r.get('parameters', 0):6d} "
              f"{_fmt(r.get('train_final'))} {_fmt(r.get('test_clean_final')):>10s} {_fmt(r.get('test_noisy_final')):>10s}")
    print(f"\n{len(rows)} runs, {len(problems)} with problems ({CONVENTION}; final logged epoch)")
    for p in problems:
        print("PROBLEM", p)

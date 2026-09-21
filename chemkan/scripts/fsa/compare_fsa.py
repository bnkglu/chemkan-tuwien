r"""Evaluate the FSA runs with the EXISTING procedures and compare them with their baselines.

    python fsa/compare_fsa.py [--skip-eval]

1. Runs the repository's evaluation CLIs on every completed FSA run (nothing is trained):
   ``evaluate_biodiesel.py`` / ``evaluate_hydrogen.py`` (train + test metrics.json and
   prediction artifacts), and for hydrogen ``evaluate_hydrogen_ignition.py`` (Fig. 8B
   procedure) and ``evaluate_hydrogen_grid.py`` (Fig. 8A 441-condition procedure).
2. Builds the comparisons

       B0-FSA     vs B0        biodiesel clean train / held-out full-trajectory errors
       Stage-1 FSA vs H_STAGE1 shared Stage-1 kinetic core
       H0-FSA     vs H0        primary hydrogen result
       Hnorm1-FSA vs Hnorm1    separately labelled initialization comparison

   with the hydrogen species / temperature / ignition / thermodynamic-branch / runtime
   diagnostics of ``diagnostics/analyze_base_on_matrix.py`` (``evaluate_checkpoint``,
   ``stability``, ``run_wall_hours``, NFE from the history CSVs).

Outputs: ``results/experiments/fsa/tables/*.csv|json`` and
``results/experiments/fsa/figures/*.png``. These are single-seed, single-configuration
results; they say nothing about noise or seed robustness, and end-to-end hydrogen
differences cannot be attributed to Stage 1 or Stage 2 without a controlled ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _problems as P                                               # noqa: E402
import analyze_base_on_matrix as M                                  # noqa: E402
from _run import git_commit, utc_now                                # noqa: E402
from fsa_runs import OUT as FSA_DIRS                                # noqa: E402

from chemkan.dynamics import KineticDynamics                        # noqa: E402
from chemkan.solver import integrate                                # noqa: E402
from chemkan.temperature import ObservedTemperature                 # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1]
FSA = P.RESULTS / "experiments/fsa"
TABLES, FIGURES = FSA / "tables", FSA / "figures"
BASE_TABLES = P.RESULTS / "reproduction/tables"
BASE_GRID = P.RESULTS / "reproduction/chemkan/hydrogen/generalization"
PAIRS = {"B0": (P.B0_DIR, FSA_DIRS["B0-FSA"]),
         "H_STAGE1": (P.H_STAGE1_DIR, FSA_DIRS["H_STAGE1-FSA"]),
         "H0": (P.H0_DIR, FSA_DIRS["H0-FSA"]),
         "Hnorm1": (P.HNORM1_DIR, FSA_DIRS["Hnorm1-FSA"])}


def rel(p):
    return str(Path(p).resolve().relative_to(P.ROOT))


def done(d: Path, stage1: bool = False) -> bool:
    return (d / ("checkpoint_stage1.pt" if stage1 else "checkpoint_final.pt")).exists()


def cli(*args):
    r = subprocess.run([sys.executable, *args], cwd=SCRIPTS, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(args)} failed:\n{r.stderr[-2000:]}")


def run_evaluations(force: bool):
    for name, (_, d) in PAIRS.items():
        if name == "H_STAGE1" or not done(d):
            continue
        script = "evaluate_biodiesel.py" if name == "B0" else "evaluate_hydrogen.py"
        for split in ("train", "test"):
            if force or not (d / "predictions" / f"{split}_predictions.npz").exists():
                cli(script, "--run-dir", str(d), "--split", split, "--metrics",
                    "--save-predictions", *(["--force"] if force else []))
        if name != "B0":
            label = d.name
            if force or not (TABLES / f"hydrogen_ignition_delay_{label}.json").exists():
                cli("evaluate_hydrogen_ignition.py", "--run-dir", str(d), "--out", str(TABLES), "--force")
            if force or not (TABLES / f"{label}_generalization_441.json").exists():
                cli("evaluate_hydrogen_grid.py", "--run-dir", str(d), "--out", str(TABLES), "--force")
        print(f"evaluated {name}-FSA", flush=True)


def history_summary(d: Path, name: str) -> dict:
    h = pd.read_csv(d / name)
    e = h["elapsed_seconds"].to_numpy(float)
    spe = np.diff(e, prepend=0.0) if "epoch_wall_time_s" not in h else h["epoch_wall_time_s"].to_numpy(float)
    out = {"epochs_logged": int(len(h)), "final_loss": float(h.total_loss.iloc[-1]),
           "min_loss": float(h.total_loss.min()), "min_loss_epoch": int(h.total_loss.idxmin()),
           "training_seconds": float(e[-1]), "median_seconds_per_epoch": float(np.median(spe)),
           "wall_hours_run_log": M.run_wall_hours(d)}
    if "nfe" in h:
        out.update(median_nfe=float(h.nfe.median()), total_nfe=float(h.nfe.sum()))
    return out


# ------------------------------------------------------------------------ biodiesel
def biodiesel_rows() -> list[dict]:
    rows = []
    for label, d in (("B0 (direct autograd)", PAIRS["B0"][0]), ("B0-FSA", PAIRS["B0"][1])):
        if not done(d):
            continue
        m = json.loads((d / "metrics.json").read_text())
        ck = P.load(d / "checkpoint_final.pt")
        rows.append({"run": label, "dir": rel(d), "sensitivity": ck["solver"]["sensitivity"],
                     "train_mse_clean_full_trajectory": m["train_mse"],
                     "test_mse_clean_full_trajectory": m["test_mse"],
                     **history_summary(d, "history.csv")})
    return rows


# ------------------------------------------------------------------------ Stage 1
def stage1_rows(D) -> list[dict]:
    rows = []
    T0, phi, _ = M.CONDS[0]
    ref = D.reference(T0, phi)
    for label, d in (("H_STAGE1 (direct autograd)", PAIRS["H_STAGE1"][0]),
                     ("H_STAGE1-FSA", PAIRS["H_STAGE1"][1])):
        if not done(d, stage1=True):
            continue
        s1 = P.load(d / "checkpoint_stage1.pt")
        model, inorm, solver = M.build(P.load(d / "checkpoint_final.pt"), D.m)
        obsT = ObservedTemperature(torch.as_tensor(D.t, dtype=torch.float32),
                                   torch.as_tensor(ref[:, -1], dtype=torch.float32).reshape(-1, 1, 1))
        kin = KineticDynamics(model.kinetic, obsT, input_normalizer=inorm)
        with torch.no_grad():
            y = integrate(kin, torch.as_tensor(ref[0, :D.m], dtype=torch.float32).unsqueeze(0),
                          torch.as_tensor(D.t, dtype=torch.float32), solver)[:, 0].numpy()
        refN = D.full_norm.normalize(torch.as_tensor(ref, dtype=torch.float32)).numpy()[:, :D.m]
        yN = D.full_norm.normalize(torch.as_tensor(np.c_[y, ref[:, -1]], dtype=torch.float32)).numpy()[:, :D.m]
        rows.append({"run": label, "dir": rel(d), "sensitivity": s1["solver"]["sensitivity"],
                     "stage1_checkpoint_sha256": hashlib.sha256(
                         (d / "checkpoint_stage1.pt").read_bytes()).hexdigest(),
                     "stage1_final_loss": float(s1["stage1_final_loss"]),
                     "open_loop_species_mse_1050_0.9_observedT": float(((yN - refN) ** 2).mean()),
                     **history_summary(d, "history_stage1.csv")})
    return rows


# ------------------------------------------------------------------------ Stage 2
def stage2_rows(D):
    rows, curves = [], {}
    for pair in ("H0", "Hnorm1"):
        for label, d in ((f"{pair} (direct autograd)", PAIRS[pair][0]), (f"{pair}-FSA", PAIRS[pair][1])):
            if not done(d):
                continue
            for budget, ck in ((500, d / "checkpoint_stage2_epoch_500.pt"), (10000, d / "checkpoint_final.pt")):
                if not ck.exists():
                    continue
                ev = M.evaluate_checkpoint(ck, D)
                curves[(label, budget)] = ev.pop("curves")
                coeffs = ev.pop("thermo_coeffs")
                per_state = ev.pop("per_state_train")
                row = {"run": label, "comparison": pair, "budget": budget, "checkpoint": rel(ck),
                       **{k: v for k, v in ev.items() if k != "arch"}}
                row.update({f"coeff_{s}": float(c) for s, c in zip(D.species, coeffs)})
                row.update({f"eq18_share_{s}": float(v / per_state.sum())
                            for s, v in zip(D.species + ["T"], per_state)})
                if budget == 10000:
                    h = M.history(d)
                    row.update({f"stability_{k}": v for k, v in M.stability(h).items()})
                    row.update(history_summary(d, "history_stage2.csv"))
                    stem = d.name
                    ign = (BASE_TABLES if "FSA" not in label else TABLES) / f"hydrogen_ignition_delay_{stem}.json"
                    grid = (BASE_GRID if "FSA" not in label else TABLES) / f"{stem}_generalization_441.json"
                    if ign.exists():
                        # Neutral fields only: the delay is argmax dT/dt for each of the
                        # paper's 30 conditions, read together with the temperature rises.
                        # No ignited/not count exists any more.
                        j = json.loads(ign.read_text())
                        row.update(
                            ignition_conditions=j["evaluated_conditions"],
                            ignition_median_abs_relative_error=j["median_abs_relative_delay_error"],
                            ignition_median_model_rise_K=j["median_model_temperature_rise_K"],
                            ignition_median_reference_rise_K=j["median_reference_temperature_rise_K"])
                    if grid.exists():
                        g = json.loads(grid.read_text())
                        row.update(grid441_mse_min=g["mse_min"], grid441_mse_median=g["mse_median"],
                                   grid441_mse_max=g["mse_max"], grid441_failed=g["failed"])
                    cfg = json.loads((d / "config.json").read_text())
                    row["stage1_checkpoint_sha256"] = cfg["stage1_from"]["stage1_checkpoint_sha256"]
                    row["sensitivity"] = cfg["sensitivity_backend"]
                rows.append(row)
    return rows, curves


def figures(D, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (T0, phi, role) in zip(axes, M.CONDS):
        ax.plot(D.t_dense * 1e3, D.reference_T_dense(T0, phi), "k-", lw=2, label="Cantera reference")
        for (label, budget), cv in curves.items():
            if budget != 10000:
                continue
            style = "-" if "FSA" in label else "--"
            ax.plot(D.t_dense * 1e3, cv[role], style, label=label)
        ax.set_title(f"{role} condition T0={T0:g} K, phi={phi:g}")
        ax.set_xlabel("t [ms]")
        ax.set_ylabel("T [K]")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "hydrogen_temperature_fsa_vs_direct_autograd.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (name, hist) in zip(axes, (("B0", "history.csv"), ("H_STAGE1", "history_stage1.csv"),
                                       ("H0", "history_stage2.csv"), ("Hnorm1", "history_stage2.csv"))):
        for lab, d in (("direct autograd", PAIRS[name][0]), ("FSA", PAIRS[name][1])):
            if (d / hist).exists():
                h = pd.read_csv(d / hist)
                ax.semilogy(h.epoch, h.total_loss, lw=0.8, label=lab)
        ax.set_title(f"{name}: training loss")
        ax.set_xlabel("epoch (= optimizer update)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_loss_fsa_vs_direct_autograd.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run the evaluation CLIs")
    a = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    if not a.skip_eval:
        run_evaluations(a.force)
    D = M.Data()
    bd, s1 = biodiesel_rows(), stage1_rows(D)
    s2, curves = stage2_rows(D)
    for name, rows in (("biodiesel_B0_fsa_comparison", bd), ("hydrogen_stage1_fsa_comparison", s1),
                       ("hydrogen_stage2_fsa_comparison", s2)):
        if rows:
            pd.DataFrame(rows).to_csv(TABLES / f"{name}.csv", index=False)
    payload = {"created": utc_now(), "git_commit": git_commit(), "code_state": P.code_state(),
               "biodiesel": bd, "hydrogen_stage1": s1, "hydrogen_stage2": s2,
               "interpretation_limits": [
                   "single seed and single configuration per comparison",
                   "no statement about noise robustness, seed robustness or full paper reproduction",
                   "end-to-end hydrogen differences are not attributable to Stage 1 or Stage 2 "
                   "without a controlled Stage-2-only ablation",
                   "Hnorm1 changes both the thermo direction and magnitude relative to H0",
                   "H0 is the primary hydrogen result; Hnorm1 is a labelled initialization comparison"]}
    (TABLES / "fsa_comparison.json").write_text(json.dumps(payload, indent=2, default=float))
    if curves:
        figures(D, curves)
    print(json.dumps({"biodiesel": bd, "stage1": s1}, indent=1, default=float)[:3000])
    for r in s2:
        print({k: r.get(k) for k in ("run", "budget", "train_mse", "test_mse", "train_peak_T_K",
                                     "train_delay_error_pct", "held_peak_T_K", "held_delay_error_pct",
                                     "gate_both", "ignition_median_model_rise_K",
                                     "thermo_norm")})


if __name__ == "__main__":
    main()

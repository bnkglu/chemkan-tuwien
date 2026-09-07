"""Validate completed biodiesel sweeps and rebuild Notebook 07 tables (no training)."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for folder in ("chemkan/src", "chemkan/scripts", "deeponet"):
    sys.path.insert(0, str(ROOT / folder))
from evaluate_biodiesel import evaluate_biodiesel
from evaluate_biodiesel_deeponet import evaluate as evaluate_don

TABLES = ROOT / "results/reproduction/tables"
FIGURES = ROOT / "results/reproduction/figures/biodiesel"
CK = ROOT / "results/reproduction/chemkan/biodiesel"
VERSION = "reference_final_trunk_relu"
DON = ROOT / "results/reproduction/baselines/deeponet/biodiesel" / VERSION
LEVELS = (0, 1, 2, 3, 5, 7, 10, 15)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_runs():
    path = ROOT / "chemkan/scripts/reproduction/biodiesel/_training.py"
    spec = importlib.util.spec_from_file_location("biodiesel_run_validation", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    audit = []
    for mode, filename in (("deeponet", "manifest_all_seed0.json"),
                           ("nmu2", "manifest_scaling_seed0.json")):
        root, jobs = runner.make_jobs(runner.build_parser().parse_args([mode]))
        manifest = json.loads((root / filename).read_text())
        assert manifest["dataset_sha256"] == hashlib.sha256((ROOT / manifest["dataset"]).read_bytes()).hexdigest()
        assert len(jobs) == len(manifest["jobs"])
        for job, item in zip(jobs, manifest["jobs"]):
            action, checkpoint = runner.plan_action(job)
            assert action in ("reuse", "skip"), (action, checkpoint)
            assert checkpoint == ROOT / item["checkpoint"]
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == item["checkpoint_sha256"]
            rows = read_csv(checkpoint.parent / "history.csv")
            budget = job["config"]["epochs"]
            assert [int(float(r["epoch"])) for r in rows[:budget]] == list(range(budget)), checkpoint
            segments, last = [], -1.0
            for row in rows[:budget]:
                elapsed = float(row["elapsed_seconds"])
                if elapsed < last:
                    segments.append(last)
                last = elapsed
            segments.append(last)
            audit.append(dict(model=job["model"], width=job["width"], noise_percent=job.get("noise"),
                              epochs=budget, parameters=job["config"]["parameter_count"],
                              n_mu=job["config"]["architecture"].get("n_mu", ""),
                              architecture_version=job["config"]["architecture"].get("architecture_version", ""),
                              reused=action == "reuse", training_seconds=sum(segments),
                              checkpoint=item["checkpoint"], checkpoint_sha256=item["checkpoint_sha256"],
                              verified=True))
    write_csv(TABLES / "biodiesel_completed_run_audit.csv", audit)


def archive_legacy_reports():
    path = TABLES / "biodiesel_fig5a_metrics.csv"
    if not path.exists() or not any(r["model"] == "DeepONet" and VERSION not in r["run_dir"]
                                    for r in read_csv(path)):
        return
    for folder, names in ((TABLES, ("biodiesel_fig5a_metrics.csv", "biodiesel_fig5b_overfit_assessment.json",
                                    "biodiesel_fig4_points.csv", "biodiesel_fig4_fits.csv", "reproduction_comparison.csv")),
                          (FIGURES, tuple(f"{stem}.{ext}" for stem in (
                              "fig04_biodiesel_neural_scaling", "fig05a_biodiesel_noise_robustness",
                              "fig05b_biodiesel_loss_dynamics", "fig06_biodiesel_15pct_profiles")
                              for ext in ("png", "pdf")))):
        dest = folder / "legacy_final_trunk_linear"
        dest.mkdir(exist_ok=True)
        for name in names:
            if (folder / name).exists() and not (dest / name).exists():
                shutil.copy2(folder / name, dest / name)
        (dest / "README.md").write_text(
            "# Archived legacy DeepONet reports\n\nThese reports used legacy_final_trunk_linear. "
            "The current reports use reference_final_trunk_relu. Original training checkpoints "
            "remain in baselines/deeponet/biodiesel/{noise,scaling}.\n")


def noise_metrics():
    rows = []
    for noise in LEVELS:
        for model in ("ChemKAN", "DeepONet"):
            directory = ((CK / "main/direct_autograd_seed0" if noise == 0 else CK / f"noise/noise{noise:02d}_seed0")
                         if model == "ChemKAN" else DON / f"noise/noise{noise:02d}_seed0")
            evaluator = evaluate_biodiesel if model == "ChemKAN" else evaluate_don
            train = evaluator(directory / "checkpoint_final.pt", "train", "cpu", noise_percent=noise)
            test = evaluator(directory / "checkpoint_final.pt", "test", "cpu", noise_percent=noise)
            if model == "DeepONet":
                assert test["model"].architecture_version == VERSION
            row = dict(model=model, noise_percent=noise, role="plotted", run_dir=str(directory.relative_to(ROOT)),
                       train_mse_noisy=train["mse"], test_mse_noisy=test["mse"], test_mse_clean=test["mse_clean"])
            for metric in ("train_mse_noisy", "test_mse_noisy", "test_mse_clean"):
                row[metric + "_timeavg"] = row[metric] / len(test["t"])
            row["architecture_version"] = VERSION if model == "DeepONet" else ""
            rows.append(row)
    directory = CK / "noise/clean_replay_seed0"
    tr = evaluate_biodiesel(directory / "checkpoint_final.pt", "train", "cpu", noise_percent=0)
    te = evaluate_biodiesel(directory / "checkpoint_final.pt", "test", "cpu", noise_percent=0)
    row = dict(model="ChemKAN", noise_percent=0, role="replay", run_dir=str(directory.relative_to(ROOT)),
               train_mse_noisy=tr["mse"], test_mse_noisy=te["mse"], test_mse_clean=te["mse_clean"])
    for metric in ("train_mse_noisy", "test_mse_noisy", "test_mse_clean"):
        row[metric + "_timeavg"] = row[metric] / len(te["t"])
    row["architecture_version"] = ""
    rows.append(row)
    write_csv(TABLES / "biodiesel_fig5a_metrics.csv", rows)
    return rows


def assess_overfitting(directory):
    rows = read_csv(directory / "history.csv")
    train = np.array([float(r["total_loss"]) for r in rows])
    test = np.array([float(r["test_mse_clean"]) for r in rows])
    epochs = np.array([int(float(r["epoch"])) for r in rows])
    i = int(test.argmin())
    smoothed = np.convolve(train, np.ones(201) / 201, mode="valid")
    index = int(np.clip(i - 100, 0, len(smoothed) - 1))
    rise = float(test[-1] / test[i] - 1)
    fall = float((smoothed[index] - smoothed[-1]) / smoothed[index])
    late = epochs >= 0.8 * epochs.max()
    return dict(min_epoch=int(epochs[i]), test_rise=rise, train_fall_after_min=fall,
                overfitting=bool(rise > .1 and fall > .05 and epochs[i] < .9 * len(rows)),
                train_late_span=float(np.ptp(train[late])), test_late_span=float(np.ptp(test[late])),
                source=str((directory / "history.csv").relative_to(ROOT)),
                criterion="test rise >10%, training fall >5%, minimum before 90% of budget; "
                          "training uses a centered 201-epoch mean, clamped to its available endpoints")


def refresh_comparison(metrics, assessment):
    rows = read_csv(TABLES / "reproduction_comparison.csv")
    values = {(r["model"], r["noise_percent"]): r for r in metrics if r["role"] == "plotted"}
    fits = {(r["model"], r["metric"]): r for r in read_csv(TABLES / "biodiesel_fig4_fits.csv")}
    for row in rows:
        claim = row["paper_result"]
        if row["figure"] == "Fig. 4" and "scaling slope" in claim:
            model = "DeepONet" if "DeepONet" in claim else "ChemKAN"
            fit = fits[model, "train_loss" if "training" in claim else "test_loss"]
            row["our_result"] = f"{float(fit['slope']):+.2f} (R2 {float(fit['r_squared']):.2f}, SE {float(fit['std_error']):.2f})"
            row["status"] = "evaluation completed; fitting scope differs"
            row["remaining_difference"] = ("Descriptive all-point regression; paper uses a pre-saturation subset. "
                                            "Current DeepONet uses reference_final_trunk_relu; legacy tables are archived.")
        elif row["figure"] == "Fig. 5A" and claim.startswith("DeepONet"):
            ratio = (values["DeepONet", 15]["test_mse_clean"] /
                     values["ChemKAN" if "/ ChemKAN" in claim else "DeepONet", 15 if "/ ChemKAN" in claim else 0]["test_mse_clean"])
            row["our_result"] = f"{ratio:.3f}x"
            row["status"] = "evaluation completed; numerical comparison reported"
            row["remaining_difference"] = ("Corrected reference_final_trunk_relu; final-checkpoint Eq. 22 losses "
                                            "on the unchanged test split. Paper activation placement is unspecified.")
        elif row["figure"] == "Fig. 5B":
            if claim.startswith("DeepONet"):
                pct = 7 if "7%" in claim else 15
                a = assessment[f"DeepONet_{pct}pct"]
                row["our_result"] = (f"minimum epoch {a['min_epoch']}; clean-test rise {a['test_rise']:.1%}; "
                                     f"post-minimum smoothed training fall {a['train_fall_after_min']:.1%}")
                row["status"] = "evaluation completed; " + ("meets our criterion" if a["overfitting"] else "does not meet our criterion")
                row["remaining_difference"] = "Corrected DeepONet. The 10%/5% thresholds, 201-epoch window and 90% cutoff are our diagnostic choices, not paper requirements."
            else:
                row["our_result"] = "ChemKAN panels meeting our criterion: " + str([n for n in (0,2,7,15) if assessment[f"ChemKAN_{n}pct"]["overfitting"]])
                row["remaining_difference"] = "Uses the unchanged ChemKAN runs and the stated diagnostic criterion; absence of this signature is not proof of no overfitting."
        elif row["figure"] == "Fig. 6":
            row["our_result"] = "ChemKAN and corrected DeepONet at Figure 3's condition: TG0=1.94, ROH0=1.43, T=334.8 K"
            row["status"] = "evaluation completed; plotted condition is a reproduction choice"
            row["remaining_difference"] = "Corrected reference_final_trunk_relu. This condition is unseen; its initial concentrations are consistent with the paper's plotted curves, but Figure 6's temperature is not independently established."
    write_csv(TABLES / "reproduction_comparison.csv", rows)


def main():
    verify_runs()
    archive_legacy_reports()
    script = Path(__file__).with_name("assemble_fig4_scaling.py")
    for mode in ("scaled", "2"):
        result = subprocess.run([sys.executable, str(script), "--n-mu", mode], cwd=ROOT,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    metrics = noise_metrics()
    assessment = {}
    for model in ("ChemKAN", "DeepONet"):
        for noise in (0, 2, 7, 15):
            directory = ((CK / "noise/clean_replay_seed0" if noise == 0 else CK / f"noise/noise{noise:02d}_seed0")
                         if model == "ChemKAN" else DON / f"noise/noise{noise:02d}_seed0")
            assessment[f"{model}_{noise}pct"] = assess_overfitting(directory)
    (TABLES / "biodiesel_fig5b_overfit_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
    refresh_comparison(metrics, assessment)
    print("Verified 19 points; refreshed corrected DeepONet noise/scaling tables and fixed-n_mu=2 Figure 4.")


if __name__ == "__main__":
    main()

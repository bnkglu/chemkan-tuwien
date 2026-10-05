"""Fig. 3 and Fig. 6 (biodiesel_v2 runs) with ChemKAN integrated at rtol 1e-6 / atol 1e-10.

Uses the unchanged figure code (fig03 ``author_trajectory_data`` + ``plot_figure``, fig06
``author_profile_data`` + ``plot_profiles``); only the tolerance that
``author_runs.chemkan_predict`` reads from each run's config is replaced, in memory, while
it runs. DeepONet predictions are unaffected (no ODE solve). Writes only into this
directory; the official plots in figures/plots/ are not touched. Run from the repo root:

    ~/uni_projects/chemkan-venv/bin/python \
        results/experiments/biodiesel/tolerance_sweep_diagnostic/make_tight_figures.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "chemkan/scripts/figures"))

import author_runs as ar  # noqa: E402
import fig03_biodiesel_trajectories as fig03  # noqa: E402
import fig06_biodiesel_profiles as fig06  # noqa: E402
from common import FIGURES_AUTHOR, save_figure, use_headless_backend  # noqa: E402

RTOL, ATOL = 1e-6, 1e-10
_orig_predict, _orig_config = ar.chemkan_predict, ar.config


def _tight_predict(run, *args, **kwargs):
    def config(r):
        c = copy.deepcopy(_orig_config(r))
        c["solver"].update(rtol=RTOL, atol=ATOL)
        return c
    ar.config = config
    try:
        return _orig_predict(run, *args, **kwargs)
    finally:
        ar.config = _orig_config


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    use_headless_backend()
    watched = sorted(FIGURES_AUTHOR.glob("fig0[36]_*")) + sorted(
        (ar.RUNS / "fig05").glob("chemkan_noise*_seed0/checkpoint_final.pt"))
    before = {str(p.relative_to(ROOT)): sha(p) for p in watched}
    ar.chemkan_predict = _tight_predict
    tol = f"ChemKAN evaluated at rtol {RTOL:g}, atol {ATOL:g} (training: rtol 1e-2, atol 1e-6)"

    condition, results = fig03.author_trajectory_data()
    fig = fig03.plot_figure(condition, results,
                            f"\nbiodiesel_v2 runs (test row 24); loss: {ar.CONVENTION_SHORT}; {tol}")
    for col, pct in enumerate(fig03.NOISE_LEVELS):
        r = results["levels"][pct]
        for row in range(len(condition["species"])):
            fig.axes[row * len(fig03.NOISE_LEVELS) + col].texts[0].set_text(
                f"Loss (clean): {r['loss_clean'][row]:.2e}\nLoss (obs): {r['loss_obs'][row]:.2e}")
    save_figure(fig, OUT / "fig03_trajectories_tight")

    data = fig06.author_profile_data()
    case = data["case"]
    fig = fig06.plot_profiles(
        case["t"], case["species"], data["truth"], data["noisy"],
        data["pred"]["chemkan"], data["pred"]["deeponet"],
        f"Fig. 6 - {fig06.NOISE_PERCENT}% noise, test row 24 (TG0={case['y0'][0]:.3f}, "
        f"ROH0={case['y0'][1]:.3f}, T={case['T']:.2f} K), biodiesel_v2 runs\n"
        f"noise-free {ar.CONVENTION_SHORT}: ChemKAN {data['mse']['chemkan']:.2e}, "
        f"{ar.DEEPONET_LABEL[ar.DEEPONET_VARIANT]} {data['mse']['deeponet']:.2e}\n{tol}")
    save_figure(fig, OUT / "fig06_profiles_15pct_tight")

    after = {k: sha(ROOT / k) for k in before}
    summary = {"rtol": RTOL, "atol": ATOL,
               "fig03_clean_loss_mean_per_level": {
                   str(p): float(results["levels"][p]["loss_clean"].mean())
                   for p in fig03.NOISE_LEVELS},
               "fig06_mse": data["mse"],
               "official_plots_and_checkpoints_unchanged": before == after,
               "watched_files": len(before)}
    (OUT / "tight_figures.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

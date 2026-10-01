"""Analysis of the 4-arm, 5-seed, 10k-epoch 0 % DeepONet runs (Fig. 5 reconstruction)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path("/Users/berke/Desktop/University/TU_Wien-MSC/2026S/Interdisciplinary_Project/development/"
           "chemkan-tuwien/results/experiments/biodiesel/deeponet_diagnostics/fig5_0pct_10k")
ARMS = {"arm1_refinit_lr1e-3": "reference-like DeepXDE initialization, lr1e-3",
        "arm2_refinit_lr2e-3": "reference-like DeepXDE initialization, lr2e-3 ablation",
        "arm3_torchinit_lr1e-3": "PyTorch-default initialization ablation, lr1e-3",
        "arm4_torchinit_lr2e-3": "PyTorch-default initialization ablation, lr2e-3"}
SEEDS = range(5)
MS = ["1000", "2500", "5000", "10000"]

# ---- pairing assertion -------------------------------------------------------------------
for a, b in (("arm1_refinit_lr1e-3", "arm2_refinit_lr2e-3"),
             ("arm3_torchinit_lr1e-3", "arm4_torchinit_lr2e-3")):
    for s in SEEDS:
        x = torch.load(OUT / a / f"seed{s}" / "checkpoint_init.pt", weights_only=False)["model_state"]
        y = torch.load(OUT / b / f"seed{s}" / "checkpoint_init.pt", weights_only=False)["model_state"]
        assert x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x), (a, b, s)
print("pairing OK: identical initial state_dicts for arm1/arm2 and arm3/arm4, every seed")

# ---- load ---------------------------------------------------------------------------------
H, M = {}, {}
for arm in ARMS:
    for s in SEEDS:
        d = OUT / arm / f"seed{s}"
        rows = list(csv.DictReader((d / "history.csv").open()))
        H[arm, s] = {k: np.array([float(r[k]) for r in rows])
                     for k in ("epoch", "train_mean_mse", "test_mean_mse")}
        M[arm, s] = json.loads((d / "metrics.json").read_text())

# ---- convergence plots --------------------------------------------------------------------
colors = dict(zip(ARMS, ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]))
ref = {"train": [(2500, 2e-4, 6e-4), (5000, 5e-5, 1.5e-4), (10000, 2e-5, 6e-5)],
       "test": [(2500, 7e-4, 1.5e-3), (5000, 2e-4, 6e-4), (10000, 2e-4, 6e-4)]}
with plt.rc_context({"figure.dpi": 110}):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, key, name in ((axs[0], "train_mean_mse", "train"), (axs[1], "test_mean_mse", "test")):
        for arm in ARMS:
            Y = np.vstack([H[arm, s][key] for s in SEEDS])
            ep = H[arm, 0]["epoch"]
            ax.semilogy(ep, np.median(Y, 0), color=colors[arm], label=ARMS[arm])
            ax.fill_between(ep, Y.min(0), Y.max(0), color=colors[arm], alpha=0.15, lw=0)
        for e, lo, hi in ref[name]:
            ax.plot([e, e], [lo, hi], color="k", lw=4, alpha=0.5)
        ax.plot([], [], color="k", lw=4, alpha=0.5, label="approximate visual read from Fig. 5B")
        ax.set(xlabel="epoch", title=f"median {name} MSE (normalized, time-averaged); band = min-max of 5 seeds")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("mean MSE")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "convergence_median_minmax.png")
    plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(18, 7), sharex=True, sharey=True)
    for j, arm in enumerate(ARMS):
        for i, key in enumerate(("train_mean_mse", "test_mean_mse")):
            for s in SEEDS:
                axs[i, j].semilogy(H[arm, s]["epoch"], H[arm, s][key], lw=0.7, label=f"seed {s}")
            axs[i, j].set_title(f"{arm}: {key.split('_')[0]}", fontsize=8)
            axs[i, j].grid(alpha=0.3, which="both")
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "convergence_individual_seeds.png")
    plt.close(fig)

    for arm in ARMS:
        fig, axs = plt.subplots(5, 6, figsize=(15, 11), sharex=True)
        for s in SEEDS:
            p = np.load(OUT / arm / f"seed{s}" / "predictions.npz")
            for k in range(6):
                a = axs[s, k]
                a.plot(p["t"], p["test_true"][3, :, k], "k-", lw=1)
                a.plot(p["t"], p["test_pred"][3, :, k], "r-", lw=1.2)
                if s == 0:
                    a.set_title(str(p["species"][k]))
                a.tick_params(labelsize=6)
            axs[s, 0].set_ylabel(f"seed {s}\ntest {M[arm, s]['final']['test']['mean_mse']:.1e}",
                                 fontsize=7)
        fig.suptitle(f"{ARMS[arm]}: Fig. 3 test condition after 10,000 epochs (black truth, red DeepONet)")
        fig.tight_layout()
        fig.savefig(OUT / f"composite_{arm}.png", dpi=75)
        plt.close(fig)

# ---- tables -------------------------------------------------------------------------------
def unstable(arm, s):
    """Spike count: epochs where train MSE jumps >10x above the running minimum after epoch 500."""
    y = H[arm, s]["train_mean_mse"]
    run_min = np.minimum.accumulate(y)
    return int(((y[500:] / run_min[500:]) > 10).sum()), bool(not np.isfinite(y).all())

print("\nper-seed")
print("arm | seed | train@1k | train@2.5k | train@5k | train@10k | test@1k | test@2.5k | test@5k | "
      "test@10k | best-train epoch | spikes>10x | max train/runmin")
for arm in ARMS:
    for s in SEEDS:
        ms = M[arm, s]["milestones"]
        y = H[arm, s]["train_mean_mse"]
        ratio = float((y[500:] / np.minimum.accumulate(y)[500:]).max())
        sp, nf = unstable(arm, s)
        print(f"{arm} | {s} | " + " | ".join(f"{ms[e]['train_mean_mse']:.2e}" for e in MS) + " | "
              + " | ".join(f"{ms[e]['test_mean_mse']:.2e}" for e in MS)
              + f" | {M[arm, s]['best_train_checkpoint']['epoch']} | {sp}{' NONFINITE' if nf else ''} | {ratio:.1f}")

print("\narm-level")
for arm in ARMS:
    tr = {e: np.array([M[arm, s]["milestones"][e]["train_mean_mse"] for s in SEEDS]) for e in MS}
    te = {e: np.array([M[arm, s]["milestones"][e]["test_mean_mse"] for s in SEEDS]) for e in MS}
    print(arm, "| train med " + " ".join(f"{np.median(tr[e]):.2e}" for e in MS),
          "| test med " + " ".join(f"{np.median(te[e]):.2e}" for e in MS),
          f"| best {tr['10000'].min():.2e} worst {tr['10000'].max():.2e} spread {tr['10000'].max() / tr['10000'].min():.1f}x"
          f" | test10k range {te['10000'].min():.2e}-{te['10000'].max():.2e}")

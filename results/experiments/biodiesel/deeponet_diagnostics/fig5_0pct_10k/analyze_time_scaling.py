"""HISTORICAL TIME-SCALING ABLATION. NOT PART OF THE CURRENT DEEPONET REPRODUCTION.

Paired raw-t vs t/30 (time-scaling ablation) analysis, PyTorch-default init, lr 2e-3."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ARMS = {"raw": "arm4_torchinit_lr2e-3", "t30": "arm5_torchinit_lr2e-3_t_over_30"}
LABEL = {"raw": "raw t (arm 4)", "t30": "t/30 time-scaling ablation"}
SEEDS = range(5)
MS = ["1000", "2500", "5000", "7500", "10000"]
BINS = [(0, 3), (3, 10), (10, 20), (20, 30)]

for s in SEEDS:
    x = torch.load(OUT / ARMS["raw"] / f"seed{s}" / "checkpoint_init.pt", weights_only=False)["model_state"]
    y = torch.load(OUT / ARMS["t30"] / f"seed{s}" / "checkpoint_init.pt", weights_only=False)["model_state"]
    assert x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x), s
print("pairing OK: raw-t and t/30 runs start from identical tensors for every seed")

H, M = {}, {}
for a, d in ARMS.items():
    for s in SEEDS:
        rows = list(csv.DictReader((OUT / d / f"seed{s}" / "history.csv").open()))
        H[a, s] = {k: np.array([float(r[k]) for r in rows]) for k in ("epoch", "train_mean_mse", "test_mean_mse")}
        M[a, s] = json.loads((OUT / d / f"seed{s}" / "metrics.json").read_text())


def bps(a, s, when, layer):
    return M[a, s]["initial" if when == "init" else "final"]["trunk"]["interior_breakpoints_s"][layer]


fmt = lambda L: "[" + ", ".join(f"{v:.1f}" for v in L) + "]"          # noqa: E731
print("\nper-seed paired")
print("seed | raw/t30 train@2.5k | raw/t30 train@5k | raw/t30 train@10k | raw/t30 test@10k | dlog10 train@10k")
dl = []
for s in SEEDS:
    g = lambda a, e, k="train_mean_mse": M[a, s]["milestones"][e][k]   # noqa: E731
    d = np.log10(g("t30", "10000")) - np.log10(g("raw", "10000"))
    dl.append(d)
    print(f"{s} | {g('raw','2500'):.2e} / {g('t30','2500'):.2e} | {g('raw','5000'):.2e} / {g('t30','5000'):.2e} | "
          f"{g('raw','10000'):.2e} / {g('t30','10000'):.2e} | {g('raw','10000','test_mean_mse'):.2e} / "
          f"{g('t30','10000','test_mean_mse'):.2e} | {d:+.2f}")
print(f"median dlog10 train@10k: {np.median(dl):+.2f}")

print("\nbreakpoints (physical s)")
for s in SEEDS:
    for a in ARMS:
        print(f"seed {s} {a:4s} L1 init {fmt(bps(a, s, 'init', 'layer1'))} final {fmt(bps(a, s, 'final', 'layer1'))} | "
              f"L2 init {fmt(bps(a, s, 'init', 'layer2'))} final {fmt(bps(a, s, 'final', 'layer2'))}")

print("\nbreakpoint distribution over 5 seeds (layer1 + layer2), counts per bin " + str(BINS))
for a in ARMS:
    for when in ("init", "final"):
        for layer in ("layer1", "layer2"):
            allp = [p for s in SEEDS for p in bps(a, s, when, layer)]
            print(f"{a:4s} {when:5s} {layer}: " + " ".join(f"{lo}-{hi}s:{sum(lo <= p < hi for p in allp)}" for lo, hi in BINS)
                  + f"  (total {len(allp)})")

print("\narm-level")
for a in ARMS:
    tr = {e: np.array([M[a, s]["milestones"][e]["train_mean_mse"] for s in SEEDS]) for e in MS}
    te = {e: np.array([M[a, s]["milestones"][e]["test_mean_mse"] for s in SEEDS]) for e in MS}
    print(f"{a}: train med " + " ".join(f"{np.median(tr[e]):.2e}" for e in MS)
          + " | test med " + " ".join(f"{np.median(te[e]):.2e}" for e in MS)
          + f" | best {tr['10000'].min():.2e} worst {tr['10000'].max():.2e} spread {tr['10000'].max() / tr['10000'].min():.1f}x"
          + " | eq18 train/test med @10k "
          + f"{np.median([M[a, s]['milestones']['10000']['train_eq18_obs'] for s in SEEDS]):.3g}/"
          + f"{np.median([M[a, s]['milestones']['10000']['test_eq18_obs'] for s in SEEDS]):.3g}")

col = {"raw": "tab:orange", "t30": "tab:green"}
with plt.rc_context({"figure.dpi": 110}):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, key in zip(axs, ("train_mean_mse", "test_mean_mse")):
        for a in ARMS:
            Y = np.vstack([H[a, s][key] for s in SEEDS])
            ep = H[a, 0]["epoch"]
            ax.semilogy(ep, np.median(Y, 0), color=col[a], label=LABEL[a])
            ax.fill_between(ep, Y.min(0), Y.max(0), color=col[a], alpha=0.18, lw=0)
        ax.set(xlabel="epoch", title=f"median {key.split('_')[0]} MSE; band = min-max of 5 seeds")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "time_scaling_convergence_median.png")
    plt.close(fig)

    fig, axs = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)
    for s in SEEDS:
        for i, key in enumerate(("train_mean_mse", "test_mean_mse")):
            for a in ARMS:
                axs[i, s].semilogy(H[a, s]["epoch"], H[a, s][key], color=col[a], lw=0.8, label=LABEL[a])
            axs[i, s].set_title(f"seed {s} {key.split('_')[0]}", fontsize=9)
            axs[i, s].grid(alpha=0.3, which="both")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "time_scaling_paired_seeds.png")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for i, when in enumerate(("init", "final")):
        for j, a in enumerate(ARMS):
            ax = axs[i, j]
            for layer, c in (("layer1", "tab:blue"), ("layer2", "tab:red")):
                allp = [p for s in SEEDS for p in bps(a, s, when, layer)]
                ax.hist(allp, bins=np.arange(0, 31, 1), color=c, alpha=0.6,
                        label=f"{layer} ({len(allp)})")
            ax.set(title=f"{LABEL[a]}, {'initialization' if when == 'init' else 'epoch 10k'}",
                   xlabel="breakpoint position (s)", ylabel="count over 5 seeds")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "time_scaling_breakpoints.png")
    plt.close(fig)

    fig, axs = plt.subplots(5, 6, figsize=(15, 11), sharex=True)
    for s in SEEDS:
        p = np.load(OUT / ARMS["t30"] / f"seed{s}" / "predictions.npz")
        for k in range(6):
            ax = axs[s, k]
            ax.plot(p["t"], p["test_true"][3, :, k], "k-", lw=1)
            ax.plot(p["t"], p["test_pred"][3, :, k], "g-", lw=1.2)
            if s == 0:
                ax.set_title(str(p["species"][k]))
            ax.tick_params(labelsize=6)
        axs[s, 0].set_ylabel(f"seed {s}\ntest {M['t30', s]['final']['test']['mean_mse']:.1e}", fontsize=7)
    fig.suptitle("t/30 time-scaling ablation (PyTorch-default init, lr 2e-3): Fig. 3 test condition, 10k epochs")
    fig.tight_layout()
    fig.savefig(OUT / "composite_arm5_torchinit_lr2e-3_t_over_30.png", dpi=75)
    plt.close(fig)

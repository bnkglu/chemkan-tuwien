"""HISTORICAL TIME-SCALING ABLATION. NOT PART OF THE CURRENT DEEPONET REPRODUCTION.

2x2 activation x time-input comparison (PyTorch-default init, Adam lr 2e-3, 10k epochs)."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "deeponet"))
import biodiesel_deeponet_repro as rp     # noqa: E402
from biodiesel_deeponet import build     # noqa: E402

ARMS = {"ReLU raw": ("arm4_torchinit_lr2e-3", "relu", 1.0),
        "ReLU t/30": ("arm5_torchinit_lr2e-3_t_over_30", "relu", 1 / 30),
        "tanh raw": ("arm6_tanh_torchinit_lr2e-3_raw_t", "tanh", 1.0),
        "tanh t/30": ("arm7_tanh_torchinit_lr2e-3_t_over_30", "tanh", 1 / 30)}
SEEDS = range(5)
MS = ["1000", "2500", "5000", "7500", "10000"]

for s in SEEDS:
    ref = torch.load(HERE / ARMS["ReLU raw"][0] / f"seed{s}" / "checkpoint_init.pt", weights_only=False)["model_state"]
    for name, (d, _, _) in ARMS.items():
        x = torch.load(HERE / d / f"seed{s}" / "checkpoint_init.pt", weights_only=False)["model_state"]
        assert x.keys() == ref.keys() and all(torch.equal(x[k], ref[k]) for k in x), (name, s)
print("pairing OK: all four arms start from identical tensors for every seed")

H, M = {}, {}
for name, (d, _, _) in ARMS.items():
    for s in SEEDS:
        rows = list(csv.DictReader((HERE / d / f"seed{s}" / "history.csv").open()))
        H[name, s] = {k: np.array([float(r[k]) for r in rows]) for k in ("epoch", "train_mean_mse", "test_mean_mse")}
        M[name, s] = json.loads((HERE / d / f"seed{s}" / "metrics.json").read_text())

print("\nper-seed (train / test mean MSE)")
for name in ARMS:
    for s in SEEDS:
        ms = M[name, s]["milestones"]
        print(f"{name:9s} seed {s} | train " + " ".join(f"{ms[e]['train_mean_mse']:.2e}" for e in MS)
              + " | test " + " ".join(f"{ms[e]['test_mean_mse']:.2e}" for e in MS)
              + f" | eq18 10k {ms['10000']['train_eq18_obs']:.3g}/{ms['10000']['test_eq18_obs']:.3g}")

print("\narm-level")
for name in ARMS:
    tr = {e: np.array([M[name, s]["milestones"][e]["train_mean_mse"] for s in SEEDS]) for e in MS}
    te = {e: np.array([M[name, s]["milestones"][e]["test_mean_mse"] for s in SEEDS]) for e in MS}
    print(f"{name:9s} train med " + " ".join(f"{np.median(tr[e]):.2e}" for e in MS)
          + " | test med " + " ".join(f"{np.median(te[e]):.2e}" for e in MS)
          + f" | best {tr['10000'].min():.2e} worst {tr['10000'].max():.2e} spread {tr['10000'].max() / tr['10000'].min():.1f}x")

print("\ntanh saturation (dense grid over 0-30 s); per layer: frac|z|>2/3/5, frac|a|>0.95/0.99, deriv mean/median, |z| q10/50/90/max")
for name in ("tanh raw", "tanh t/30"):
    for when in ("initial", "final"):
        for layer in ("layer1", "layer2"):
            vals = [M[name, s][when]["trunk"]["tanh_saturation"]["dense"][layer] for s in SEEDS]
            agg = lambda f: np.median([f(v) for v in vals])                          # noqa: E731
            print(f"{name:9s} {when:7s} {layer}: |z|>2 {agg(lambda v: v['frac_abs_z_gt']['2']):.2f} "
                  f">3 {agg(lambda v: v['frac_abs_z_gt']['3']):.2f} >5 {agg(lambda v: v['frac_abs_z_gt']['5']):.2f} | "
                  f"|a|>.95 {agg(lambda v: v['frac_abs_act_gt']['0.95']):.2f} >.99 {agg(lambda v: v['frac_abs_act_gt']['0.99']):.2f} | "
                  f"deriv mean {agg(lambda v: v['derivative_mean']):.3f} median {agg(lambda v: v['derivative_median']):.3f} | "
                  f"|z| q50 {agg(lambda v: v['abs_z_quantiles_10_50_90_max'][1]):.2f} q90 {agg(lambda v: v['abs_z_quantiles_10_50_90_max'][2]):.2f} "
                  f"max {agg(lambda v: v['abs_z_quantiles_10_50_90_max'][3]):.1f}   (medians over 5 seeds)")

col = dict(zip(ARMS, ["tab:orange", "tab:green", "tab:purple", "tab:blue"]))
td = torch.linspace(0, 30, 301)
with plt.rc_context({"figure.dpi": 110}):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, key in zip(axs, ("train_mean_mse", "test_mean_mse")):
        for name in ARMS:
            Y = np.vstack([H[name, s][key] for s in SEEDS])
            ep = H[name, 0]["epoch"]
            ax.semilogy(ep, np.median(Y, 0), color=col[name], label=name)
            ax.fill_between(ep, Y.min(0), Y.max(0), color=col[name], alpha=0.15, lw=0)
        ax.set(xlabel="epoch", title=f"median {key.split('_')[0]} MSE; band = min-max of 5 seeds")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "activation_2x2_convergence.png")
    plt.close(fig)

    fig, axs = plt.subplots(4, 5, figsize=(20, 12), sharex=True)
    for i, (name, (d, act, scale)) in enumerate(ARMS.items()):
        for s in SEEDS:
            m = rp.set_activation(build(8, seed=s), act)
            m.load_state_dict(torch.load(HERE / d / f"seed{s}" / "checkpoint_final.pt", weights_only=False)["model_state"])
            with torch.no_grad():
                z = m.trunk((td * scale).reshape(-1, 1)).numpy()
            axs[i, s].plot(td.numpy(), z, lw=1)
            axs[i, s].set_title(f"{name}, seed {s}", fontsize=8)
        axs[i, 0].set_ylabel("final trunk latent (8)")
    for a in axs[-1]:
        a.set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(HERE / "activation_2x2_trunk_basis.png", dpi=80)
    plt.close(fig)

    for name in ("tanh raw", "tanh t/30"):
        d = ARMS[name][0]
        fig, axs = plt.subplots(5, 6, figsize=(15, 11), sharex=True)
        for s in SEEDS:
            p = np.load(HERE / d / f"seed{s}" / "predictions.npz")
            for k in range(6):
                ax = axs[s, k]
                ax.plot(p["t"], p["test_true"][3, :, k], "k-", lw=1)
                ax.plot(p["t"], p["test_pred"][3, :, k], color=col[name], lw=1.2)
                if s == 0:
                    ax.set_title(str(p["species"][k]))
                ax.tick_params(labelsize=6)
            axs[s, 0].set_ylabel(f"seed {s}\ntest {M[name, s]['final']['test']['mean_mse']:.1e}", fontsize=7)
        fig.suptitle(f"{name} (activation ablation, PyTorch-default init, lr 2e-3): Fig. 3 test condition, 10k epochs")
        fig.tight_layout()
        fig.savefig(HERE / f"composite_{d}.png", dpi=75)
        plt.close(fig)

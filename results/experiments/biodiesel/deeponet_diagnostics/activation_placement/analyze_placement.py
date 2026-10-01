"""Activation-placement analysis: R1 four-way (A/B/C/D) and the R5 A-vs-B control.
A = branch-final linear / trunk-final ReLU (Lu-stated, DeepXDE reference; primary reconstruction)
B = both final linear; C = both final ReLU; D = branch-final ReLU / trunk-final linear
(B/C/D plausible, unconfirmed). Values at epoch e = model after e updates."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG / "relu_reachability"))
from common import at, branch_tensor, build, rp       # noqa: E402

EP = (500, 1000, 2500, 5000, 7500, 10000)
TH = (1e-3, 5e-4, 3e-4, 1e-4, 5e-5)
HINGE_EP = (0, 500, 1000, 2500, 5000, 10000)


def run_dir(cfg, P, s):
    if P == "A":
        if cfg == "R1":
            return DIAG / ("fig5_0pct_10k/arm1_refinit_lr1e-3" if s < 5 else "relu_reachability/stage4_tail/R1") / f"seed{s}"
        return DIAG / ("fig5_0pct_10k/arm4_torchinit_lr2e-3" if s < 5 else "relu_reachability/stage4_tail/R5") / f"seed{s}"
    return HERE / ("R1" if cfg == "R1" else "R5_control") / P / f"seed{s}"


def model(cfg, P, s, which):
    m = rp.set_placement(build(8, seed=s), P)
    m.load_state_dict(torch.load(run_dir(cfg, P, s) / which, weights_only=False)["model_state"])
    return m


def useful(m, P):
    """Placement-aware useful interior breakpoints (threshold 1e-3, normalized units)."""
    if P in "AC":
        u = rp.useful_breakpoints(m, branch_tensor(), 30.0)
        return u["n_useful"], u["useful_positions_s"]
    t = torch.linspace(0, 30, 3001).reshape(-1, 1)
    with torch.no_grad():
        a1 = torch.relu(m.trunk[0](t))
        coef = (m.head.weight.abs().unsqueeze(1) * m.branch(branch_tensor()).abs().unsqueeze(0)).amax(dim=(0, 1))
        W2 = m.trunk[2].weight
    w1, b1 = m.trunk[0].weight.detach().ravel(), m.trunk[0].bias.detach().ravel()
    pos = []
    for i in range(7):
        if w1[i] != 0 and 0 < float(-b1[i] / w1[i]) < 30:
            c = float(((W2[:, i].abs() * coef).max() * a1[:, i].abs().max()))
            if c > 1e-3:
                pos.append(float(-b1[i] / w1[i]))
    return len(pos), sorted(pos)


def hinges(cfg, P, s):
    """First-layer interior hinge counts at HINGE_EP (milestones where recorded, else init/final)."""
    r = run_dir(cfg, P, s)
    ms = json.loads((r / "metrics.json").read_text()).get("milestones", {})
    out = {}
    for e in HINGE_EP:
        if e == 10000:
            out[e] = len(rp.interior_breakpoints(model(cfg, P, s, "checkpoint_final.pt"), 30.0)["layer1"])
        elif e == 0:
            out[e] = len(rp.interior_breakpoints(model(cfg, P, s, "checkpoint_init.pt"), 30.0)["layer1"])
        else:
            v = ms.get(str(e), {}).get("layer1_hinges_s")
            out[e] = None if v is None else len(v)
    return out


def report(cfg, placements):
    print(f"\n################ {cfg}: placements {placements}")
    for s in range(10):                                            # pairing
        ref = torch.load(run_dir(cfg, "A", s) / "checkpoint_init.pt", weights_only=False)["model_state"]
        for P in placements:
            x = torch.load(run_dir(cfg, P, s) / "checkpoint_init.pt", weights_only=False)["model_state"]
            assert all(torch.equal(x[k], ref[k]) for k in ref), (cfg, P, s)
    print("pairing OK: identical initial affine tensors across placements, every seed")
    V = {P: {e: np.array([at(run_dir(cfg, P, s), e) for s in range(10)]) for e in EP} for P in placements}
    W = {P: {e: np.array([at(run_dir(cfg, P, s), e, "test_mean_mse") for s in range(10)]) for e in EP} for P in placements}
    Q = {P: {e: np.array([at(run_dir(cfg, P, s), e, "train_eq18_obs") for s in range(10)]) for e in (10000,)} for P in placements}
    for P in placements:
        print(f"\n== {cfg} {P}: {rp.PLACEMENTS[P]['label']}")
        print("  median train " + " ".join(f"{e}:{np.median(V[P][e]):.2e}" for e in EP))
        print("  median test  " + " ".join(f"{e}:{np.median(W[P][e]):.2e}" for e in EP))
        v = V[P][10000]
        print(f"  10k train median {np.median(v):.2e} min {v.min():.2e} max {v.max():.2e} | test median {np.median(W[P][10000]):.2e} "
              f"| Eq.18 train median {np.median(Q[P][10000]):.3g} | " + " ".join(f"<={t:g}:{int((v <= t).sum())}/10" for t in TH))
        for s in range(10):
            m = model(cfg, P, s, "checkpoint_final.pt")
            dead = rp.dead_dimensions(rp.trunk_activations(m, torch.linspace(0, 30, 301)))
            nu, pu = useful(m, P)
            h = hinges(cfg, P, s)
            print(f"   seed {s}: train {v[s]:.2e} test {W[P][10000][s]:.2e} | dead trunk {dead} | useful bp {nu} {[round(x, 1) for x in pu]} "
                  f"| L1 interior hinges " + " ".join(f"{e}:{h[e]}" for e in HINGE_EP))
        if P != "A":
            r = np.log10(V[P][10000] / V["A"][10000])
            print(f"  paired log10({P}/A) train@10k: " + " ".join(f"{x:+.2f}" for x in r)
                  + f" | median ratio {10 ** np.median(r):.2f} | {P} wins {int((r < 0).sum())}, A wins {int((r > 0).sum())}")
    fig, axs = plt.subplots(len(placements), 6, figsize=(15, 2.4 * len(placements)), sharex=True)
    for i, P in enumerate(placements):
        s = int(np.argsort(V[P][10000])[4])                        # median-loss seed (5th of 10)
        p = np.load(run_dir(cfg, P, s) / "predictions.npz")
        for k in range(6):
            a = axs[i, k]; a.plot(p["t"], p["test_true"][3, :, k], "k-", lw=1); a.plot(p["t"], p["test_pred"][3, :, k], "r-", lw=1.2)
            if i == 0: a.set_title(str(p["species"][k]))
            a.tick_params(labelsize=6)
        axs[i, 0].set_ylabel(f"{P} seed {s}\n(median)\ntrain {V[P][10000][s]:.1e}", fontsize=7)
    fig.suptitle(f"{cfg}: Fig. 3 test condition, median-loss seed per placement, 10k")
    fig.tight_layout(); fig.savefig(HERE / f"composite_{cfg}_median_seed.png", dpi=75); plt.close(fig)
    return V


if __name__ == "__main__":
    report("R5", ["A", "B"])
    if (HERE / "R1" / "D" / "seed9" / "metrics.json").exists():
        report("R1", ["A", "B", "C", "D"])

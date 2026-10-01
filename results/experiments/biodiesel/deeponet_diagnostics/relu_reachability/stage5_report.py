import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import DIAG, at, load_model, rp
from stage5_transmission import transmission

RR = DIAG / "relu_reachability"
knots = {e["K"]: e["knots_s"] for e in json.loads((RR / "stage2_knots" / "knots.json").read_text())["fits"]}
for arm in sorted(p.name for p in (RR / "stage5_planted").iterdir() if p.is_dir()):
    K = int(arm[1])
    print(f"\n== {arm} (planted {[round(k, 2) for k in knots[K]]} s; DIAGNOSTIC ONLY)")
    tr10 = []
    fig, axs = plt.subplots(5, 6, figsize=(15, 11), sharex=True)
    for s in range(5):
        run = RR / "stage5_planted" / arm / f"seed{s}"
        mi, mf = load_model(run / "checkpoint_init.pt", s), load_model(run / "checkpoint_final.pt", s)
        ti, tf = transmission(mi, knots[K]), transmission(mf, knots[K])
        w, b = mf.trunk[0].weight.detach().ravel().numpy(), mf.trunk[0].bias.detach().ravel().numpy()
        fates = []
        for i, k0 in enumerate(knots[K]):
            kf = -b[i] / w[i] if w[i] != 0 else np.nan
            alive = (np.maximum(w[i] * np.linspace(0, 30, 301) + b[i], 0) > 0).any()
            if not alive:
                fate = "disappeared (unit dead)"
            elif not 0 < kf < 30:
                fate = f"left interval ({kf:.1f})"
            else:
                fate = f"{'survived' if abs(kf - k0) < 0.5 else 'moved'} {k0:.1f}->{kf:.1f}"
            fates.append(fate)
        fin = [-b[i] / w[i] for i in range(K) if w[i] != 0 and 0 < -b[i] / w[i] < 30]
        merged = any(abs(x - y) < 0.5 for i, x in enumerate(fin) for y in fin[i + 1:])
        bp = rp.interior_breakpoints(mf, 30.0)
        dead = rp.dead_dimensions(rp.trunk_activations(mf, __import__("torch").linspace(0, 30, 301)))
        vals = {e: (at(run, e), at(run, e, "test_mean_mse")) for e in (1000, 2500, 5000, 10000)}
        tr10.append(vals[10000][0])
        print(f"seed {s}: train " + " ".join(f"{vals[e][0]:.2e}" for e in vals) + " | test " + " ".join(f"{vals[e][1]:.2e}" for e in vals)
              + f"\n   init: L1 active {ti['layer1_active']}/7 L2 active {ti['layer2_active']}/8, visible {[x['visible'] for x in ti['knots']]}"
              + f"\n   final: planted fates {fates}{' MERGED' if merged else ''} | still visible at planted positions {[x['visible'] for x in tf['knots']]}"
              + f"\n   final L1 kinks {[round(x, 1) for x in bp['layer1']]} L2 bp {[round(x, 1) for x in bp['layer2']]} | dead L1/L2 {dead['relu0']}/{dead['relu1']}")
        p = np.load(run / "predictions.npz")
        for k in range(6):
            a = axs[s, k]; a.plot(p["t"], p["test_true"][3, :, k], "k-", lw=1); a.plot(p["t"], p["test_pred"][3, :, k], "r-", lw=1.2)
            for kn in knots[K]:
                a.axvline(kn, color="grey", lw=0.5, ls=":")
            if s == 0: a.set_title(str(p["species"][k]))
            a.tick_params(labelsize=6)
        axs[s, 0].set_ylabel(f"seed {s}\ntrain {vals[10000][0]:.1e}", fontsize=7)
    fig.suptitle(f"{arm}: planted-kink ReLU (DIAGNOSTIC ONLY), Fig. 3 test condition, 10k epochs; dotted = planted knots")
    fig.tight_layout(); fig.savefig(RR / "stage5_planted" / f"composite_{arm}.png", dpi=75); plt.close(fig)
    a = np.array(tr10)
    print(f"{arm}: train@10k median {np.median(a):.2e} [{a.min():.2e}, {a.max():.2e}]; <=1e-4: {(a <= 1e-4).sum()}/5")

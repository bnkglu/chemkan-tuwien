import json
import numpy as np
from scipy.stats import spearmanr
from common import DIAG, at, history, load_model, trunk_diag

RR = DIAG / "relu_reachability"
SRC = {"R1": [DIAG / "fig5_0pct_10k" / "arm1_refinit_lr1e-3" / f"seed{s}" for s in range(5)]
             + [RR / "stage4_tail" / "R1" / f"seed{s}" for s in range(5, 30)],
       "R5": [DIAG / "fig5_0pct_10k" / "arm4_torchinit_lr2e-3" / f"seed{s}" for s in range(5)]
             + [RR / "stage4_tail" / "R5" / f"seed{s}" for s in range(5, 30)]}
TH = (3e-4, 1e-4, 5e-5)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


out = {}
for arm, runs in SRC.items():
    tr = np.array([at(r, 10000) for r in runs]); te = np.array([at(r, 10000, "test_mean_mse") for r in runs])
    q = lambda a: " ".join(f"{p}:{np.percentile(a, p):.2e}" for p in (0, 10, 25, 50, 75, 90, 100))  # noqa
    print(f"\n== {arm} (30 seeds, 10k)\ntrain {q(tr)}\ntest  {q(te)}")
    for th in TH:
        k = int((tr <= th).sum()); lo, hi = wilson(k, 30)
        print(f"  train <= {th:g}: {k}/30  (Wilson 95% [{lo:.2f}, {hi:.2f}])")
    dead, useful = [], []
    for s, r in enumerate(runs):
        d = trunk_diag(load_model(r / "checkpoint_final.pt", s))
        dead.append(d["dead1"] + d["dead2"]); useful.append(d["n_useful"])
        h = np.append(history(r)["train_mean_mse"], tr[s])
        cross = {th: (int(np.argmax(h <= th)) if (h <= th).any() else None) for th in TH}
        if tr[s] <= 3e-4 or any(v is not None for v in cross.values()):
            print(f"  seed {s}: train10k {tr[s]:.2e} first epoch <= 3e-4 / 1e-4 / 5e-5: "
                  f"{cross[3e-4]} / {cross[1e-4]} / {cross[5e-5]} | dead {dead[-1]} useful bp {useful[-1]}")
    r1 = spearmanr(dead, np.log10(tr)); r2 = spearmanr(useful, np.log10(tr))
    print(f"  Spearman(dead units, log train) = {r1.correlation:+.2f}; Spearman(useful bp, log train) = {r2.correlation:+.2f}")
    print("  per seed (train, test, dead1+dead2, useful bp): " + "; ".join(
        f"{s}:{tr[s]:.1e},{te[s]:.1e},{dead[s]},{useful[s]}" for s in range(30)))
    out[arm] = {"train": tr.tolist(), "test": te.tolist(), "dead": dead, "useful": useful}
(RR / "stage4_tail" / "stage4_values.json").write_text(json.dumps(out, indent=2))

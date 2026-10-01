import json
import numpy as np
import torch
from common import DIAG, at, branch_tensor, load_model, rp, trunk_diag

S1 = DIAG / "relu_reachability" / "stage1_branch"
MODES = {"A normalized": "normalized", "B raw": "raw", "C raw species + norm T": "raw_species_norm_T"}
for s in range(5):    # pairing: identical initial tensors across modes
    ref = torch.load(S1 / "normalized" / f"seed{s}" / "checkpoint_init.pt", weights_only=False)["model_state"]
    for m in MODES.values():
        x = torch.load(S1 / m / f"seed{s}" / "checkpoint_init.pt", weights_only=False)["model_state"]
        assert all(torch.equal(x[k], ref[k]) for k in ref)
print("pairing OK (identical initial tensors across branch modes, every seed)")
out = {}
for name, m in MODES.items():
    print(f"\n== {name}")
    tr = {e: [] for e in (500, 1000, 2500)}
    te = {e: [] for e in (500, 1000, 2500)}
    for s in range(5):
        run = S1 / m / f"seed{s}"
        for e in tr:
            tr[e].append(at(run, e)); te[e].append(at(run, e, "test_mean_mse"))
        bi = rp.branch_stats(load_model(run / "checkpoint_init.pt", s), branch_tensor(m))
        bf = rp.branch_stats(load_model(run / "checkpoint_final.pt", s), branch_tensor(m))
        tdi = trunk_diag(load_model(run / "checkpoint_init.pt", s), m)
        tdf = trunk_diag(load_model(run / "checkpoint_final.pt", s), m)
        fb = lambda b: " ".join(f"L{l[-1]}[{b[l]['pre_min']:.2g},{b[l]['pre_max']:.2g}] mean {b[l]['pre_mean']:.2g} sd {b[l]['pre_std']:.2g} zero {b[l]['frac_relu_zero']:.2f}" for l in ("layer1", "layer2"))  # noqa
        print(f"seed {s}: train {tr[500][-1]:.2e} {tr[1000][-1]:.2e} {tr[2500][-1]:.2e} | test {te[500][-1]:.2e} {te[1000][-1]:.2e} {te[2500][-1]:.2e}"
              f"\n   branch init  {fb(bi)}\n   branch 2.5k  {fb(bf)}"
              f"\n   trunk dead1/dead2 init {tdi['dead1']}/{tdi['dead2']} -> 2.5k {tdf['dead1']}/{tdf['dead2']} | useful bp 2.5k {tdf['n_useful']} at {[round(x, 1) for x in tdf['useful_s']]}")
    for e in tr:
        a, b = np.array(tr[e]), np.array(te[e])
        print(f"  epoch {e}: train median {np.median(a):.2e} [{a.min():.2e}, {a.max():.2e}] | test median {np.median(b):.2e} [{b.min():.2e}, {b.max():.2e}]")
    out[name] = {"train_2500": tr[2500], "test_2500": te[2500]}
ma = np.median(out["A normalized"]["train_2500"])
for k in ("B raw", "C raw species + norm T"):
    r = ma / np.median(out[k]["train_2500"])
    print(f"decision: median train@2.5k A / {k} = {r:.2f}  (switch only if >= 2)")

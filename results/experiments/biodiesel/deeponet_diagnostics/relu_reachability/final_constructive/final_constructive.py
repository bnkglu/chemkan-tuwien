"""FINAL constructive ReLU diagnostic (DIAGNOSTIC ONLY; no claim of author fidelity).

Stages, each adding one constraint (biodiesel_v2, 0 % noise, author-global normalized targets,
mean MSE over trajectory x time x species; Eq. 18 = mean over species, sum over time, mean over
trajectories is stored too):
  F1  8-latent factorization of the frozen spline basis (no neural network), float64.
  F2A real ReLU trunk (planted, FROZEN first layer; trainable 7->8 + ReLU) + head, with a FREE
      per-trajectory branch table B_free[20, 8] instead of the branch MLP.
  F2B the real 308-parameter network (branch MLP) with the planted first layer frozen.
  F3  F2B continued 5,000 epochs with the first layer unfrozen; the SAME Adam object, so the
      moments of the other layers are retained (the first layer's start when it first has a
      gradient).
Planted hinge j: phi_j(t) = a_j ReLU(t - k_j), a_j = 1 / RMS over the 30 training times of
ReLU(t - k_j); so w_j = a_j, b_j = -a_j k_j. Units K..6 keep their R1 initialization.
R1 = TF-truncated Glorot normal, zero bias (Aprime_tf_truncated_glorot_zero_bias), per seed.

    python final_constructive.py f1          # representational diagnostic
    python final_constructive.py neural      # F2A, F2B, F3 for K = 3, 4 and seeds 0..4
"""
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[5] / "deeponet"))
sys.path.insert(0, str(HERE.parent))
import biodiesel_deeponet_repro as rp            # noqa: E402
from biodiesel_deeponet import build, n_params   # noqa: E402
from stage5_transmission import transmission     # noqa: E402

torch.set_num_threads(1)
KNOTS = {e["K"]: e["knots_s"] for e in json.loads((HERE.parent / "stage2_knots" / "knots.json").read_text())["fits"]}
DATA = rp.load_author_data(noise_percent=0)
STATS = rp.author_global_stats(DATA)
T_NP = DATA["t"]
MODE = "author_global_normalized_states"


def tens(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


Y = {s: rp.to_u(DATA[f"{s}_obs"], STATS) for s in ("train", "test")}          # [N, 30, 6]
BR = {s: rp.branch_input_mode(DATA[f"{s}_branch_raw"], "normalized", MODE, STATS) for s in ("train", "test")}


def hinge_scale(k):
    return 1.0 / np.sqrt(np.mean(np.maximum(T_NP - k, 0.0) ** 2))


def eq18(pred, target):
    return float(((pred - target) ** 2).mean(dim=2).sum(dim=1).mean())


# ---------------------------------------------------------------- F1 (representational)
def f1():
    out = {}
    Yt = tens(Y["train"], torch.float64)
    for K in (3, 4):
        cols = [np.ones_like(T_NP), T_NP / np.sqrt(np.mean(T_NP ** 2))]
        cols += [hinge_scale(k) * np.maximum(T_NP - k, 0.0) for k in KNOTS[K]]
        phi = tens(np.column_stack(cols), torch.float64)                  # [30, K+2]
        res = []
        for r in range(10):
            g = torch.Generator().manual_seed(r)
            P = {"A": torch.randn(8, K + 2, generator=g, dtype=torch.float64) * 0.5,
                 "B": torch.randn(20, 8, generator=g, dtype=torch.float64) * 0.5,
                 "W": torch.randn(6, 8, generator=g, dtype=torch.float64) * 0.5,
                 "c": torch.zeros(6, dtype=torch.float64)}
            for v in P.values():
                v.requires_grad_(True)

            def loss():
                H = phi @ P["A"].T                                       # [30, 8]
                pred = torch.einsum("il,sl,tl->its", P["B"], P["W"], H) + P["c"]
                return ((pred - Yt) ** 2).mean()

            opt = torch.optim.Adam(P.values(), lr=1e-2)
            for _ in range(20000):
                opt.zero_grad(); L = loss(); L.backward(); opt.step()
            adam = float(loss())
            lb = torch.optim.LBFGS(P.values(), lr=1.0, max_iter=5000, max_eval=6250,
                                   history_size=100, line_search_fn="strong_wolfe",
                                   tolerance_grad=1e-14, tolerance_change=1e-16)

            def closure():
                lb.zero_grad(); L = loss(); L.backward(); return L
            lb.step(closure)
            final = float(loss())
            finite = all(torch.isfinite(v).all() for v in P.values()) and np.isfinite(final)
            res.append({"restart": r, "after_adam": adam, "after_lbfgs": final, "finite": bool(finite)})
            print(f"F1 K={K} restart {r}: Adam {adam:.3e} -> L-BFGS {final:.3e} finite {finite}", flush=True)
        f = np.array([x["after_lbfgs"] for x in res])
        out[K] = {"restarts": res, "best": float(f.min()), "median": float(np.median(f)), "worst": float(f.max())}
        print(f"F1 K={K}: best {f.min():.3e} median {np.median(f):.3e} worst {f.max():.3e}", flush=True)
    (HERE / "f1_results.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------- neural stages
def planted_r1(seed, K):
    m = rp.apply_init(build(8, seed=seed), "Aprime_tf_truncated_glorot_zero_bias", seed)
    assert n_params(m) == 308
    with torch.no_grad():
        for j, k in enumerate(KNOTS[K]):
            a = hinge_scale(k)
            m.trunk[0].weight[j, 0] = a
            m.trunk[0].bias[j] = -a * k
    return m


def hinge_fates(model, K, t_end=30.0):
    w = model.trunk[0].weight.detach().ravel().numpy()
    b = model.trunk[0].bias.detach().ravel().numpy()
    tt = np.linspace(0, t_end, 3001)
    fates = []
    for j, k0 in enumerate(KNOTS[K]):
        act = np.maximum(w[j] * tt + b[j], 0.0)
        kf = -b[j] / w[j] if w[j] != 0 else np.nan
        if not (act > 0).any():
            fate = "dies"
        elif 0 < kf < t_end:
            fate = "stays near" if abs(kf - k0) < 0.5 else "moves"
        elif (act > 0).all() or (act[1:] > 0).all():
            fate = "left [0,30]: linear over domain"
        else:
            fate = "left [0,30]"
        fates.append({"planted_s": float(k0), "now_s": float(kf), "fate": fate})
    return fates


def layer2_info(model):
    td = torch.linspace(0, 30, 301).reshape(-1, 1)
    with torch.no_grad():
        a2 = model.trunk(td)
    return {"layer2_dead": int((a2.amax(0) == 0).sum()),
            "trunk_coord_range": (a2.amax(0) - a2.amin(0)).numpy().round(4).tolist()}


def train(stage, K, seed, model, params_free, epochs, opt=None, b_free=None, log_hinges=False):
    """Full-batch Adam on train mean MSE. History row e = state after e updates."""
    tr_t, te_t = tens(T_NP), tens(T_NP)
    Ytr, Yte = tens(Y["train"]), tens(Y["test"])
    Btr, Bte = tens(BR["train"]), tens(BR["test"])

    def pred(split):
        tr_lat = model.trunk(tr_t.reshape(-1, 1))                           # [30, 8]
        if b_free is not None:
            if split == "test":
                return None
            lat = b_free
        else:
            lat = model.branch(Btr if split == "train" else Bte)
        return model.head(lat.unsqueeze(1) * tr_lat.unsqueeze(0))           # [N, 30, 6]

    if opt is None:
        opt = torch.optim.Adam(params_free, lr=2e-3)
    run = HERE / stage / f"K{K}" / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    hist, hinges = [], []
    t0 = time.perf_counter()
    for ep in range(epochs + 1):
        with torch.no_grad():
            p_tr, p_te = pred("train"), pred("test")
            row = {"epoch": ep, "train_mean_mse": float(((p_tr - Ytr) ** 2).mean()),
                   "train_eq18": eq18(p_tr, Ytr),
                   "test_mean_mse": float(((p_te - Yte) ** 2).mean()) if p_te is not None else "",
                   "test_eq18": eq18(p_te, Yte) if p_te is not None else ""}
        hist.append(row)
        if log_hinges and ep % 100 == 0:
            hinges.append({"epoch": ep, "hinges": hinge_fates(model, K)})
        if not np.isfinite(row["train_mean_mse"]):
            break
        if ep == epochs:
            break
        opt.zero_grad()
        loss = ((pred("train") - Ytr) ** 2).mean()
        loss.backward()
        opt.step()
    with (run / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    if hinges:
        (run / "hinge_track.json").write_text(json.dumps(hinges, indent=1))
    torch.save({"model_state": model.state_dict(), "b_free": b_free, "opt_state": opt.state_dict()},
               run / "checkpoint_final.pt")
    print(f"{stage} K={K} seed {seed}: train {hist[-1]['train_mean_mse']:.3e} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return opt, hist


def summarize(stage, K, seed, model, hist, frozen_ref=None, extra=None):
    at = lambda e, k: hist[e][k] if e < len(hist) else None                  # noqa: E731
    rec = {"train": {e: at(e, "train_mean_mse") for e in (1000, 2500, 5000, 10000) if e < len(hist)},
           "test": {e: at(e, "test_mean_mse") for e in (1000, 2500, 5000, 10000) if e < len(hist)},
           "final_train": hist[-1]["train_mean_mse"], "final_test": hist[-1]["test_mean_mse"],
           "final_train_eq18": hist[-1]["train_eq18"], "final_test_eq18": hist[-1]["test_eq18"],
           **layer2_info(model),
           "transmission": transmission(model, KNOTS[K])["knots"],
           "hinges": hinge_fates(model, K)}
    if frozen_ref is not None:
        rec["first_layer_unchanged"] = bool(torch.equal(model.trunk[0].weight, frozen_ref[0])
                                            and torch.equal(model.trunk[0].bias, frozen_ref[1]))
        assert rec["first_layer_unchanged"], f"{stage} K={K} seed {seed}: frozen layer changed"
    if extra:
        rec.update(extra)
    return rec


def neural():
    results = {}
    for K in (3, 4):
        for seed in range(5):
            # ---- F2A: free per-trajectory branch table
            m = planted_r1(seed, K)
            ref = (m.trunk[0].weight.detach().clone(), m.trunk[0].bias.detach().clone())
            for p in m.trunk[0].parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                b0 = m.branch(tens(BR["train"])).detach().clone()
            b_free = torch.nn.Parameter(b0)
            params = [b_free] + list(m.trunk[2].parameters()) + list(m.head.parameters())
            _, h = train("F2A", K, seed, m, params, 10000, b_free=b_free)
            results[f"F2A/K{K}/seed{seed}"] = summarize("F2A", K, seed, m, h, ref)

            # ---- F2B: real branch MLP, frozen planted first layer
            m = planted_r1(seed, K)
            ref = (m.trunk[0].weight.detach().clone(), m.trunk[0].bias.detach().clone())
            for p in m.trunk[0].parameters():
                p.requires_grad_(False)
            opt = torch.optim.Adam(m.parameters(), lr=2e-3)   # trunk[0] has no grad -> skipped
            opt, h = train("F2B", K, seed, m, None, 10000, opt=opt)
            results[f"F2B/K{K}/seed{seed}"] = summarize("F2B", K, seed, m, h, ref)
            assert all(p not in opt.state for p in m.trunk[0].parameters()), \
                "frozen layer acquired Adam state"
            before = hinge_fates(m, K)
            f2b_final = h[-1]["train_mean_mse"]

            # ---- F3: unfreeze, SAME optimizer (other layers' moments retained)
            n_state_before = len(opt.state)
            for p in m.trunk[0].parameters():
                p.requires_grad_(True)
            _, h3 = train("F3", K, seed, m, None, 5000, opt=opt, log_hinges=True)
            trs = [r["train_mean_mse"] for r in h3]
            results[f"F3/K{K}/seed{seed}"] = summarize(
                "F3", K, seed, m, h3, extra={
                    "f2b_final_train": f2b_final, "best_train_during_f3": float(np.min(trs)),
                    "best_epoch": int(np.argmin(trs)), "hinges_before_unfreeze": before,
                    "adam_states_before_f3": n_state_before, "adam_states_after_f3": len(opt.state)})
            (HERE / "neural_results.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    {"f1": f1, "neural": neural}[sys.argv[1]]()

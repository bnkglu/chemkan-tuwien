"""Stage 2: OPTIMISTIC shared piecewise-linear temporal basis with K interior knots.

Data: biodiesel_v2.npz (repository archive, not regenerated), 0 % noise, author-global
normalization from biodiesel_deeponet_repro. X_train [30, 120] = 30 sampled times x
(20 trajectories x 6 species), train_obs (== train_true at 0 %). Basis at the 30 sampled
times: [1, t, relu(t - k_1), ..., relu(t - k_K)] shared by all series; each series has its
own least-squares coefficients (more flexible than the DeepONet's branch x head coefficients).
Knots optimized by differential evolution (3 restarts) + Nelder-Mead polish, bounds (0, 30).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "deeponet"))
import biodiesel_deeponet_repro as rp   # noqa: E402

data = rp.load_author_data(noise_percent=0)
assert np.array_equal(data["train_obs"], data["train_true"])
stats = rp.author_global_stats(data)
t = data["t"]
to_X = lambda s: rp.to_u(s, stats).transpose(1, 0, 2).reshape(len(t), -1)   # noqa: E731
X, Xt = to_X(data["train_obs"]), to_X(data["test_true"])
assert X.shape == (30, 120)


def basis(knots):
    return np.column_stack([np.ones_like(t), t] + [np.maximum(t - k, 0.0) for k in knots])


def mse(knots, Y=X):
    A = basis(knots)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return float(((A @ coef - Y) ** 2).mean())


fits = []
for K in range(8):
    if K == 0:
        best = (mse([]), [])
    else:
        best = (np.inf, None)
        for rs in range(3):
            r = differential_evolution(mse, [(0.0, 30.0)] * K, seed=rs, popsize=40, maxiter=600,
                                       tol=1e-12, polish=False)
            p = minimize(mse, r.x, method="Nelder-Mead",
                         options={"xatol": 1e-8, "fatol": 1e-16, "maxiter": 20000})
            x = np.clip(p.x if p.fun < r.fun else r.x, 0, 30)
            if mse(x) < best[0]:
                best = (mse(x), sorted(x.tolist()))
    fits.append({"K": K, "knots_s": [round(k, 6) for k in best[1]], "train_mse": best[0],
                 "test_mse_exploratory": mse(best[1], Xt)})
    print(f"K={K}: train {best[0]:.3e}  test(exploratory) {fits[-1]['test_mse_exploratory']:.3e}  "
          f"knots {[round(k, 2) for k in best[1]]}")

Path(__file__).with_name("knots.json").write_text(json.dumps({
    "description": "OPTIMISTIC lower bound for a shared piecewise-linear temporal basis with K "
                   "interior knots (per-series free coefficients); not a DeepONet result",
    "data": "biodiesel_v2.npz, 0 % noise, author-global normalized, train_obs, 30 sampled times",
    "basis": "[1, t, relu(t - k_i)]", "metric": "mean MSE over time x trajectory x species",
    "test_metric": "exploratory: per-test-series least squares on the train knots (optimistic)",
    "fits": fits}, indent=2))

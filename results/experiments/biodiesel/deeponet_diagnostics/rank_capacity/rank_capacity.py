"""Optimistic latent-rank lower bound for the 8-latent DeepONet (no training).

Data: biodiesel_v2.npz, 0 % noise, the reproduction trainer's author-global normalization
(biodiesel_deeponet_repro.author_global_stats / to_u). Tensor used: train_obs (== train_true
at 0 %, asserted). X[:, traj*6 + species] = normalized trajectory, shape [30, 120].
MSE = mean over time x trajectory x species (the trainer's mean_mse convention).
The SVD gives arbitrary coefficients per series, so its error is an OPTIMISTIC lower bound on
what an 8-latent DeepONet (coefficients = branch(traj) * head(species)) could reach.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "deeponet"))
import biodiesel_deeponet_repro as rp   # noqa: E402

data = rp.load_author_data(noise_percent=0)
assert np.array_equal(data["train_obs"], data["train_true"])
stats = rp.author_global_stats(data)
to_X = lambda s: rp.to_u(s, stats).transpose(1, 0, 2).reshape(len(data["t"]), -1)  # noqa: E731
X, Xt = to_X(data["train_obs"]), to_X(data["test_true"])
assert X.shape == (30, 120) and Xt.shape == (30, 60)
mse = lambda a, b: float(((a - b) ** 2).mean())                                       # noqa: E731

mu, mut = X.mean(0, keepdims=True), Xt.mean(0, keepdims=True)
Uc, Sc, _ = np.linalg.svd(X - mu, full_matrices=False)
Uu, Su, _ = np.linalg.svd(X, full_matrices=False)
rows = []
for k in range(11):
    Pc, Pu = Uc[:, :k] @ Uc[:, :k].T, Uu[:, :k] @ Uu[:, :k].T
    rows.append({"k": k,
                 "centered_train": mse(mu + Pc @ (X - mu), X),
                 "centered_test_optimistic": mse(mut + Pc @ (Xt - mut), Xt),
                 "uncentered_train": mse(Pu @ X, X) if k else None,
                 "uncentered_test": mse(Pu @ Xt, Xt) if k else None})
out = {"tensor": "train_obs (== train_true at 0 % noise), author-global normalized; "
                 "test: test_true", "shape_train": list(X.shape), "shape_test": list(Xt.shape),
       "singular_values_centered": Sc.tolist(), "singular_values_uncentered": Su.tolist(),
       "rows": rows,
       "labels": {"centered_test_optimistic": "optimistic test projection using train temporal "
                  "basis (uses each test series' own mean; not a deployable procedure)"}}
Path(__file__).with_name("rank_capacity.json").write_text(json.dumps(out, indent=2))
print(" k | centered train | centered test (optimistic) | uncentered train | uncentered test")
for r in rows:
    f = lambda v: "      -" if v is None else f"{v:.2e}"                                  # noqa: E731
    print(f"{r['k']:2d} | {f(r['centered_train'])} | {f(r['centered_test_optimistic'])} | "
          f"{f(r['uncentered_train'])} | {f(r['uncentered_test'])}")

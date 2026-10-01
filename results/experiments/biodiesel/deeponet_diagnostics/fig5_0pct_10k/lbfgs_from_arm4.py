"""L-BFGS continuation of the raw-t ReLU arm-4 final checkpoints (PyTorch-default init,
Adam lr 2e-3, 10k epochs), on TRAIN mean MSE only. One optimizer.step(closure) per seed with
max_iter=500, max_eval=625, history_size=100, strong-Wolfe line search. No test data enters
the optimization; test metrics are computed before and after only.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "deeponet"))
import biodiesel_deeponet_repro as rp          # noqa: E402
from biodiesel_deeponet import build          # noqa: E402

MODE = "author_global_normalized_states"
OUT = HERE / "lbfgs_from_arm4"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)

data = rp.load_author_data(noise_percent=0)
stats = rp.author_global_stats(data)
T = lambda x: torch.as_tensor(x, dtype=torch.float32)                         # noqa: E731
t = T(data["t"])
split = {s: {"branch": T(rp.branch_input(data[f"{s}_branch_raw"], MODE, stats)),
             "obs": T(rp.encode_states(data[f"{s}_obs"], MODE, stats)),
             "true": T(rp.to_u(data[f"{s}_true"], stats))} for s in ("train", "test")}


def predict(model, s):
    return model(split[s]["branch"], t).permute(1, 0, 2)                     # [traj, time, sp]


def metrics(model, s):
    with torch.no_grad():
        p = predict(model, s)
        return {"mean_mse": float(rp.mean_mse(p, split[s]["obs"])),
                "eq18_obs": float(rp.eq18(p, split[s]["obs"])),
                "finite": bool(torch.isfinite(p).all())}


summary = {}
fig, axs = plt.subplots(5, 6, figsize=(15, 11), sharex=True)
for seed in range(5):
    ck = torch.load(HERE / "arm4_torchinit_lr2e-3" / f"seed{seed}" / "checkpoint_final.pt",
                    map_location="cpu", weights_only=False)
    model = build(8, seed=seed)
    model.load_state_dict(ck["model_state"])
    before = {s: metrics(model, s) for s in ("train", "test")}
    pred_before = rp.from_u(predict(model, "test").detach(), stats).numpy()

    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, max_eval=625,
                            history_size=100, line_search_fn="strong_wolfe")
    n_closure, losses = [0], []

    def closure():
        opt.zero_grad()
        loss = rp.mean_mse(predict(model, "train"), split["train"]["obs"])
        loss.backward()
        n_closure[0] += 1
        losses.append(float(loss.detach()))
        return loss

    opt.step(closure)                                     # ONE call, internal iterations
    st = opt.state[opt._params[0]]
    after = {s: metrics(model, s) for s in ("train", "test")}
    pred_after = rp.from_u(predict(model, "test").detach(), stats).numpy()
    n_iter, evals = int(st.get("n_iter", -1)), int(st.get("func_evals", -1))
    reason = ("max_iter reached" if n_iter >= 500 else "max_eval reached" if evals >= 625 else
              "stopped early: tolerance_grad/tolerance_change met or line search made no progress")
    summary[seed] = {
        "before": before, "after": after, "closure_evaluations": n_closure[0],
        "lbfgs_n_iter": n_iter, "lbfgs_func_evals": evals, "termination": reason,
        "all_closure_losses_finite": bool(np.isfinite(losses).all()),
        "final_loss_not_above_start": after["train"]["mean_mse"] <= before["train"]["mean_mse"],
        "trunk_after": {"interior_breakpoints_s": rp.interior_breakpoints(model, 30.0),
                        "dead_on_dense": rp.dead_dimensions(
                            rp.trunk_activations(model, torch.linspace(0, 30, 301)))}}
    torch.save({"model_state": model.state_dict(), "source": "arm4 final + L-BFGS",
                "lbfgs": summary[seed]}, OUT / f"seed{seed}_after_lbfgs.pt")
    for k in range(6):
        a = axs[seed, k]
        a.plot(data["t"], data["test_true"][3, :, k], "k-", lw=1)
        a.plot(data["t"], pred_before[3, :, k], "r-", lw=1, label="Adam 10k")
        a.plot(data["t"], pred_after[3, :, k], "b--", lw=1, label="+ L-BFGS")
        if seed == 0:
            a.set_title(data["species"][k])
        a.tick_params(labelsize=6)
    axs[seed, 0].set_ylabel(f"seed {seed}", fontsize=7)
    r = lambda L: [round(x, 1) for x in L]                                   # noqa: E731
    b = summary[seed]["trunk_after"]["interior_breakpoints_s"]
    print(f"seed {seed}: train {before['train']['mean_mse']:.3e} -> {after['train']['mean_mse']:.3e} | "
          f"test {before['test']['mean_mse']:.3e} -> {after['test']['mean_mse']:.3e} | "
          f"eq18 train {before['train']['eq18_obs']:.3g} -> {after['train']['eq18_obs']:.3g}, "
          f"test {before['test']['eq18_obs']:.3g} -> {after['test']['eq18_obs']:.3g} | closures {n_closure[0]} "
          f"n_iter {n_iter} | {reason} | finite {summary[seed]['all_closure_losses_finite']} | "
          f"L1 {r(b['layer1'])} L2 {r(b['layer2'])} dead {summary[seed]['trunk_after']['dead_on_dense']}")
axs[0, 0].legend(fontsize=6)
fig.suptitle("ReLU raw t (arm 4): Adam 10k (red) vs + L-BFGS (blue dashed); Fig. 3 test condition")
fig.tight_layout()
fig.savefig(OUT / "lbfgs_before_after_fig3.png", dpi=75)
(OUT / "lbfgs_summary.json").write_text(json.dumps(summary, indent=2))

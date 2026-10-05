"""READ-ONLY diagnostic: evaluation-time ODE tolerance sweep on the biodiesel_v2 Fig. 3 runs.

Same checkpoints (figures/fig05/chemkan_noise{00,05,10,15}_seed0/checkpoint_final.pt), same
data, same initial conditions, same model and normalization; ONLY the evaluation rtol/atol
change. No training, no backward, no checkpoint is written. Run from the repository root:

    ~/uni_projects/chemkan-venv/bin/python \
        results/experiments/biodiesel/tolerance_sweep_diagnostic/tolerance_sweep.py

Outputs (this directory only): sweep_fig3.csv, sweep_test.csv, fig3_tol_noise10.png,
fig3_tol_noise15.png, mse_vs_rtol.png, controls.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
for sub in ("chemkan/scripts/figures", "chemkan/scripts", "chemkan/scripts/author_repo_match"):
    sys.path.insert(0, str(ROOT / sub))

import author_runs as ar  # noqa: E402
from _author_match import NormalizedDynamics, build_core, julia_mse, rhs_divisor_from_config  # noqa: E402
from train_author_repo_match import load_canonical_data  # noqa: E402

from chemkan.solver import SolverConfig, integrate  # noqa: E402

torch.set_num_threads(1)               # keep the load on the concurrent hydrogen run small
NOISE = (0, 5, 10, 15)
TOLS = {"A": (1e-2, 1e-6), "B": (1e-4, 1e-8), "C": (1e-6, 1e-10), "D": (1e-8, 1e-10),
        "E": (1e-10, 1e-12)}
TEST_TOLS = ("A", "C", "D", "E")       # rtol 1e-2, 1e-6, 1e-8, 1e-10
REF = "E"
N_T = 30


class Counted(torch.nn.Module):
    """Counts right-hand-side evaluations without changing them."""

    def __init__(self, f):
        super().__init__()
        self.f, self.nfe = f, 0

    def forward(self, t, z):
        self.nfe += 1
        return self.f(t, z)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_state(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load(run):
    cfg = ar.config(run)
    core = build_core(cfg["architecture"]["hidden"]).double()
    core.load_state_dict(torch.load(run["dir"] / "checkpoint_final.pt", map_location="cpu",
                                    weights_only=False)["model_state"])
    core.eval()
    for p in core.parameters():
        p.requires_grad_(False)
    return cfg, core, NormalizedDynamics(core, rhs_divisor_from_config(cfg))


def solve(dyn, z0, t, tol):
    """Identical to author_runs.chemkan_predict except rtol/atol; returns (z, nfe, seconds)."""
    f = Counted(dyn)
    s = SolverConfig(method="tsit5", rtol=tol[0], atol=tol[1], sensitivity="direct_autograd")
    t0 = time.perf_counter()
    with torch.no_grad():
        z = integrate(f, z0, t, s)
    return z, f.nfe, time.perf_counter() - t0


def main():
    runs = ar.by_name()
    case = ar.fig3_case(NOISE)
    clean_obs = ar.dataset()["test_states"][ar.FIG3_TEST_INDEX]           # (30, 6) physical
    controls, fig3_rows, test_rows, dense = {}, [], [], {}

    for pct in NOISE:
        run = runs[f"fig05_chemkan_noise{pct:02d}"]
        ck = run["dir"] / "checkpoint_final.pt"
        file_sha_before = sha_file(ck)
        cfg, core, dyn = load(run)
        param_sha_before = sha_state(core.state_dict())
        src, sol = cfg["data_source"], cfg["solver"]
        ymin, ymax = np.array(src["species_min"]), np.array(src["species_max"])
        rng = ymax - ymin
        z0 = torch.tensor(np.concatenate([(case["y0"] - ymin) / rng,
                                          [(case["T"] - src["T_min"]) / (src["T_max"] - src["T_min"])]]),
                          dtype=torch.float64)[None]
        t_obs = torch.as_tensor(case["t"], dtype=torch.float64)
        t_dense = torch.as_tensor(case["t_dense"], dtype=torch.float64)

        # control 1: at the stored tolerance this route reproduces the figure route exactly
        stored = (sol["rtol"], sol["atol"])
        mine = solve(dyn, z0, t_obs, stored)[0][:, 0, :6].numpy() * rng + ymin
        fig_route = ar.chemkan_predict(run, case["y0"], case["T"], case["t"])
        ctrl = {"stored_tolerance": stored, "rhs_divisor": rhs_divisor_from_config(cfg),
                "fig3_route_max_abs_diff": float(np.abs(mine - fig_route).max())}

        # Fig. 3 condition: obs grid (metrics) and dense grid (trajectory differences)
        zd = {k: solve(dyn, z0, t_dense, tol) for k, tol in TOLS.items()}
        zo = {k: solve(dyn, z0, t_obs, tol) for k, tol in TOLS.items()}
        ref_d = zd[REF][0][:, 0, :6].numpy()
        noisy = case["observations"][pct]
        dense[pct] = {k: v[0][:, 0, :6].numpy() * rng + ymin for k, v in zd.items()}
        for k, (rt, at) in TOLS.items():
            pred_n = zo[k][0][:, 0, :6].numpy()                               # normalized
            clean_n, noisy_n = (clean_obs - ymin) / rng, (noisy - ymin) / rng
            mse_c = float(((pred_n - clean_n) ** 2).mean())
            mse_o = float(((pred_n - noisy_n) ** 2).mean())
            diff = np.abs(zd[k][0][:, 0, :6].numpy() - ref_d)
            fig3_rows.append({"noise_pct": pct, "tol": k, "rtol": rt, "atol": at,
                              "clean_mse_time_avg": mse_c, "clean_eq18": N_T * mse_c,
                              "noisy_mse_time_avg": mse_o, "noisy_eq18": N_T * mse_o,
                              "max_abs_diff_vs_E_norm": float(diff.max()),
                              "mean_abs_diff_vs_E_norm": float(diff.mean()),
                              "nfe_dense": zd[k][1], "seconds_dense": zd[k][2]})

        # all test trajectories, each in its own solve (as the trainer's final evaluation)
        noise_arg = int(run["args"][run["args"].index("--noise-percent") + 1])
        data = load_canonical_data(torch.device("cpu"), ar.DATA, noise_arg)
        te = data["test"]
        u0, tgt = data["u0"][te], data["target"][te]
        noisy_t = data.get("test_noisy_target")
        ref_pred = None
        for k in ("E",) + tuple(x for x in TEST_TOLS if x != "E"):
            t0, nfe, preds = time.perf_counter(), 0, []
            for i in range(u0.shape[0]):
                z, n, _ = solve(dyn, u0[i:i + 1], data["t"], TOLS[k])
                preds.append(z[:, 0, :6].T)
                nfe += n
            pred = torch.stack(preds)                                          # (B, 6, T)
            if k == "E":
                ref_pred = pred
            per = ((pred - tgt) ** 2).mean(dim=(1, 2)).numpy()
            row = {"noise_pct": pct, "tol": k, "rtol": TOLS[k][0], "atol": TOLS[k][1],
                   "n_test": len(per), "clean_mean": float(per.mean()),
                   "clean_median": float(np.median(per)), "clean_worst": float(per.max()),
                   "clean_mean_check_julia_mse": float(julia_mse(pred, tgt)),
                   "fig3_row_clean": float(per[ar.FIG3_TEST_INDEX]),
                   "max_abs_diff_vs_E_norm": float((pred - ref_pred).abs().max()),
                   "nfe_total": nfe, "seconds": time.perf_counter() - t0}
            if noisy_t is not None:
                row["noisy_mean"] = float(((pred - noisy_t) ** 2).mean())
            test_rows.append(row)

        # control 2: parameters and checkpoint file unchanged
        ctrl["param_sha_before"] = param_sha_before
        ctrl["param_sha_after"] = sha_state(core.state_dict())
        ctrl["checkpoint_sha_before"] = file_sha_before
        ctrl["checkpoint_sha_after"] = sha_file(ck)
        ctrl["unchanged"] = (ctrl["param_sha_before"] == ctrl["param_sha_after"]
                             and file_sha_before == ctrl["checkpoint_sha_after"])
        ctrl["checkpoint"] = str(ck.relative_to(ROOT))
        controls[pct] = ctrl

    for name, rows in (("sweep_fig3.csv", fig3_rows), ("sweep_test.csv", test_rows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    (OUT / "controls.json").write_text(json.dumps(controls, indent=2) + "\n")
    plots(case, dense, fig3_rows)


def plots(case, dense, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c", "D": "#1f77b4", "E": "#000000"}
    for pct in (10, 15):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(case["t_dense"], case["truth"][:, j], color="0.6", lw=3, label="clean truth")
            ax.scatter(case["t"], case["observations"][pct][:, j], s=10, color="0.4",
                       label=f"{pct}% noisy obs")
            for k, (rt, at) in TOLS.items():
                ax.plot(case["t_dense"], dense[pct][k][:, j], color=colors[k],
                        lw=1.6 if k in "AE" else 1.0, ls="--" if k == "E" else "-",
                        label=f"{k}: rtol {rt:g}, atol {at:g}")
            ax.set_title(case["species"][j])
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Fig. 3 condition, {pct}% model: same checkpoint, evaluation tolerance only")
        fig.tight_layout()
        fig.savefig(OUT / f"fig3_tol_noise{pct}.png", dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for pct in NOISE:
        r = [x for x in rows if x["noise_pct"] == pct]
        ax.loglog([x["rtol"] for x in r], [x["clean_mse_time_avg"] for x in r], "o-",
                  label=f"{pct}%")
    ax.set_xlabel("evaluation rtol")
    ax.set_ylabel("Fig. 3 clean MSE (time-averaged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mse_vs_rtol.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

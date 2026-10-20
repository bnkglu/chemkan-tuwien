r"""Diagnostic trainer for the source-faithful 308-parameter biodiesel DeepONet.

New reproduction path (``biodiesel_deeponet_repro.py``): raw time t in the trunk, author
data (``biodiesel_v2.npz``: t = linspace(0, 30, 30)), a state mode (raw_states or
author_global_normalized_states) and an explicit objective (mean_mse, eq18_obs,
eq22_true). Full-batch Adam, lr 1e-3, the unchanged 308-parameter model (ReLU, Glorot
normal, zero biases). The legacy trainer ``train_biodiesel_deeponet.py`` (tau = t/30,
train-only min-max) is kept only for the existing runs.

Every epoch logs, BEFORE the optimizer step (the state that produced the loss): the
optimized objective and, for train and test, mean_mse (trained representation), Eq. 18
against the observations and Eq. 22 against the clean truth (author-global u units).
The final metrics are those of the final checkpoint (after the last step).

    python train_biodiesel_deeponet_repro.py --state-mode raw_states --objective mean_mse \
        --epochs 2500 --run-dir ../results/experiments/biodiesel/deeponet_diagnostics/A_raw_mean_mse_seed0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "chemkan" / "scripts"))

from _run import RunManager                                             # noqa: E402

import biodiesel_deeponet_repro as rp                                   # noqa: E402
from biodiesel_deeponet import (BRANCH_IN, COMPARISON_WIDTH, PAPER_PARAMS,  # noqa: E402
                                architecture, build, n_params)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-mode", required=True, choices=rp.STATE_MODES)
    p.add_argument("--objective", default="mean_mse", choices=rp.OBJECTIVES,
                   help="mean_mse (default): divided by the number of time points, as the "
                        "authors' released Flux.mse; eq18_obs sums over time (30x)")
    p.add_argument("--noise-percent", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=1e-3, help="Lu/DeepXDE reference value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--branch-mode", choices=rp.BRANCH_MODES, default="follow_state_mode",
                   help="branch-input-only preprocessing ablation (targets unchanged)")
    p.add_argument("--plant-knots", type=Path, default=None,
                   help="DIAGNOSTIC ONLY: knots.json from the shared-knot fit")
    p.add_argument("--plant-k", type=int, default=None, help="which K entry of knots.json")
    p.add_argument("--layer2-bias", type=float, default=None,
                   help="ARTIFICIAL masking diagnostic: second-layer trunk bias value")
    p.add_argument("--placement", choices=rp.PLACEMENTS, default="A",
                   help="final-latent activation placement: A branch-final linear / trunk-final "
                        "activated (Lu-stated, DeepXDE reference; default), B both final linear, "
                        "C both final activated, D branch-final activated / trunk-final linear")
    p.add_argument("--width", type=int, default=COMPARISON_WIDTH,
                   help="branch/latent width w (8 = the paper's 308-parameter model)")
    p.add_argument("--trunk-hidden", type=int, default=None,
                   help="trunk hidden width q (default w - 1); Fig. 4 sizes use "
                        "biodiesel_deeponet.FIG4_ARCHITECTURES")
    p.add_argument("--report-train-only-norm", action="store_true",
                   help="also log, every epoch and at the end, time-averaged MSE in the "
                        "figure scripts' normalization (per-noise-level train-only min/max of "
                        "the archive, as the ChemKAN figure runs use)")
    p.add_argument("--activation", choices=rp.ACTIVATIONS, default="relu",
                   help="hidden activation (ACTIVATION ABLATION; the paper does not specify it)")
    p.add_argument("--init-from", type=Path, default=None,
                   help="checkpoint_init.pt of a paired run: its tensors must equal this "
                        "seed/scheme's initialization (asserted), then are used as-is")
    p.add_argument("--init", choices=list(rp.INIT_SCHEMES), default="A_xavier_normal_zero_bias",
                   help="initialization ablation arm (default: baseline A)")
    p.add_argument("--data-file", type=Path, default=rp.DEFAULT_DATA)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def _tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def main():
    args = build_parser().parse_args()
    if args.objective == "eq22_true" and args.noise_percent == 0:
        raise SystemExit("eq22_true equals eq18_obs at 0 % noise (observations == truth); "
                         "run eq18_obs instead.")
    oracle = args.objective == "eq22_true"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(1)

    data = rp.load_author_data(args.data_file, args.noise_percent)
    stats = rp.author_global_stats(data)
    mode = args.state_mode
    t = rp.raw_trunk_time(data, args.data_file)              # physical time (s), asserted raw
    t_end = float(t[-1])
    t_in = t                                                 # trunk input: raw t, no scaling
    assert torch.equal(t_in, t)

    def split_tensors(split):
        return {"branch": _tensor(rp.branch_input_mode(data[f"{split}_branch_raw"],
                                                      args.branch_mode, mode, stats)),
                "obs_rep": _tensor(rp.encode_states(data[f"{split}_obs"], mode, stats)),
                "obs_u": _tensor(rp.to_u(data[f"{split}_obs"], stats)),
                "true_u": _tensor(rp.to_u(data[f"{split}_true"], stats)),
                "true_phys": _tensor(data[f"{split}_true"]),
                "obs_phys": _tensor(data[f"{split}_obs"])}

    tr, te = split_tensors("train"), split_tensors("test")
    model = rp.apply_init(build(args.width, seed=args.seed, branch_in=BRANCH_IN,
                                trunk_hidden=args.trunk_hidden), args.init, args.seed)
    if args.width == COMPARISON_WIDTH and args.trunk_hidden in (None, COMPARISON_WIDTH - 1):
        assert n_params(model) == PAPER_PARAMS == 308, n_params(model)
    if args.init_from is not None:
        paired = torch.load(args.init_from, map_location="cpu", weights_only=False)["model_state"]
        own = model.state_dict()
        assert paired.keys() == own.keys() and all(torch.equal(paired[k], own[k]) for k in own), \
            f"{args.init_from} differs from the {args.init} seed-{args.seed} initialization"
        model.load_state_dict(paired)
    rp.set_activation(model, args.activation)
    rp.set_placement(model, args.placement)
    planting = None
    if args.plant_knots is not None:
        entry = next(e for e in json.loads(args.plant_knots.read_text())["fits"]
                     if e["K"] == args.plant_k)
        planting = rp.plant_knots(model, entry["knots_s"], float(data["t"][-1]), args.layer2_bias)
        planting["source"] = str(args.plant_knots)
    elif args.layer2_bias is not None:
        raise SystemExit("--layer2-bias is only allowed with --plant-knots (diagnostic)")
    is_tanh = args.activation == "tanh"
    objective = rp.objective_fn(args.objective)

    def forward(split):
        rep = model(split["branch"], t_in).permute(1, 0, 2)     # (T, B, 6) -> [traj, time, sp]
        phys = rp.decode_states(rep, mode, stats)
        u = rp.to_u(phys, stats) if mode == "raw_states" else rep
        return rep, phys, u

    def evaluate(split):
        with torch.no_grad():
            rep, phys, u = forward(split)
            return rp.all_metrics(rep, split["obs_rep"], u, split["obs_u"], split["true_u"]), phys

    fig_norm = None
    if args.report_train_only_norm:
        from _data import load_biodiesel
        fstats = load_biodiesel(split="train", noise_percent=args.noise_percent,
                                data_file=args.data_file)
        f_lo, f_hi = fstats["u_min"].double(), fstats["u_max"].double()
        fig_norm = {"u_min": f_lo.tolist(), "u_max": f_hi.tolist(),
                    "keys": fstats["normalization_keys"]}

        def fig_mse(phys, target):
            d = (phys.double() - target.double()) / (f_hi - f_lo)
            return float((d ** 2).mean())

        def fig_metrics():
            with torch.no_grad():
                _, p_tr, _ = forward(tr)
                _, p_te, _ = forward(te)
            return {"train_mse_fig": fig_mse(p_tr, tr["obs_phys"]),
                    "test_noisy_mse_fig": fig_mse(p_te, te["obs_phys"]),
                    "test_clean_mse_fig": fig_mse(p_te, te["true_phys"])}

    run = RunManager(args.run_dir, "biodiesel-deeponet-repro", overwrite=args.overwrite)
    run.start()
    arch = architecture(model)
    if is_tanh:
        arch["activation"] = ("tanh between branch layers and after every trunk layer "
                              "(none on branch output or head) -- ACTIVATION ABLATION")
    arch["activation_name"] = args.activation
    arch["placement"] = {"code": args.placement, **rp.PLACEMENTS[args.placement]}
    w_init, b_init, role = rp.INIT_SCHEMES[args.init]
    arch["init_scheme"] = args.init
    arch["weight_init"] = f"{w_init}, bias {b_init}"
    arch["init_role"] = role
    arch["weight_init_note"] = (
        "implemented with torch.nn.init.xavier_normal_; same Glorot/Xavier fan-in/fan-out "
        "variance rule as the DeepXDE reference; PyTorch samples an untruncated normal "
        "whereas the old TensorFlow Glorot-normal implementation uses truncated-normal "
        "sampling")
    config = {
        "model": "DeepONet-biodiesel", "path": "reproduction (biodiesel_deeponet_repro.py)",
        "purpose": ("ORACLE/DIAGNOSTIC: optimizes against the clean truth, not a paper "
                    "reproduction" if oracle else "diagnostic reproduction run"),
        "architecture": arch, "parameter_count": arch["parameter_count"],
        "seed": args.seed, "optimizer": "Adam (full batch)", "learning_rate": args.lr,
        "epochs": args.epochs, "noise_percent": args.noise_percent,
        "objective": args.objective, "dtype": "float32",
        "preprocessing": {
            "state_mode": mode,
            "trunk_input": "raw physical t in seconds (repository grid, no scaling)",
            "trunk_input_range_s": [float(t.min()), float(t.max())],
            "init_from": None if args.init_from is None else str(args.init_from),
            "time_grid": data["t"].tolist(),
            "branch_input": ["TG0", "ROH0", "T0"],
            "branch_mode": args.branch_mode,
            "planted_knots_DIAGNOSTIC_ONLY": planting,
            "branch_scaling": ("physical values" if mode == "raw_states" else
                               "TG0/ROH0 by their species' author-global min/max; T0 by "
                               "the min/max of all initial temperatures"),
            "targets": ("raw six-species states" if mode == "raw_states" else
                        "author-global normalized six-species states"),
            "author_global_stats": {"ymin": stats["ymin"].tolist(), "ymax": stats["ymax"].tolist(),
                                    "Tmin": stats["Tmin"], "Tmax": stats["Tmax"],
                                    "source": stats["source"]},
        },
        "metrics": {
            "mean_mse": "(pred - obs)^2 mean in the trained representation",
            "mean_mse_u": "(pred - obs)^2 mean in author-global u units (time-averaged Eq. 18)",
            "eq18_obs": "author-global u; mean over species, SUM over time, mean over trajectories",
            "eq22_true": "same reduction against the noise-free truth",
            "history_state": "before this epoch's optimizer step",
            "final_state": "final checkpoint (after the last step)",
        },
        "dataset_file": {"path": str(args.data_file),
                         "sha256": hashlib.sha256(Path(args.data_file).read_bytes()).hexdigest()},
        "noise": {"formula": "clean + randn * p * max_t(clean) per trajectory and species, "
                             "clipped at 0 (additive_species_max)",
                  "percent": args.noise_percent},
    }
    run.write_config(config)

    init_train, init_train_phys = evaluate(tr)
    init_test, init_test_phys = evaluate(te)
    t_dense = torch.linspace(float(t[0]), t_end, 301)        # physical seconds
    init_trunk = {"kinks": rp.first_layer_kinks(model).tolist(),
                  "dead_on_grid": rp.dead_dimensions(rp.trunk_activations(model, t_in)),
                  "dead_on_dense": rp.dead_dimensions(rp.trunk_activations(model, t_dense)),
                  "interior_breakpoints_s": rp.interior_breakpoints(model, t_end)}
    if is_tanh:
        init_trunk["tanh_saturation"] = {"grid": rp.tanh_saturation(model, t_in),
                                         "dense": rp.tanh_saturation(model, t_dense)}
    logging.info("configuration: branch inputs %s (%s), targets %s, trunk input raw physical t "
                 "%.4g..%.4g s (%d points), activation %s, placement %s, init %s, Adam lr %g",
                 args.branch_mode, "author-global normalized Mode A" if args.branch_mode in
                 ("normalized", "follow_state_mode") and mode != "raw_states" else "see config",
                 mode, float(t_in.min()), float(t_in.max()), t_in.numel(), args.activation,
                 args.placement, args.init, args.lr)
    logging.info("initial: train %s | test %s", init_train, init_test)
    torch.save({"model_state": {k: v.clone() for k, v in model.state_dict().items()},
                "init_scheme": args.init, "seed": args.seed},
               run.run_dir / "checkpoint_init.pt")      # for lr-pairing checks
    logging.info("initial prediction stats: %s", rp.prediction_stats(init_train_phys))

    fig_cols = (["train_mse_fig", "test_noisy_mse_fig", "test_clean_mse_fig"]
                if fig_norm is not None else [])
    cols = ["epoch", "objective"] + fig_cols + [f"{s}_{k}" for s in ("train", "test")
                                     for k in ("mean_mse", "mean_mse_u", "eq18_obs", "eq22_true")]
    history = run.history("history.csv", cols + ["elapsed_seconds"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    started = time.perf_counter()
    nonfinite_epoch = None
    best = {"loss": float("inf"), "epoch": None, "state": None}
    milestones = {}
    for epoch in range(args.epochs):
        opt.zero_grad()
        rep, _, u = forward(tr)
        loss = objective(rep, tr["obs_rep"], u, tr["obs_u"], tr["true_u"])
        m_tr, _ = evaluate(tr)
        m_te, _ = evaluate(te)
        row = {f"train_{k}": m_tr[k] for k in ("mean_mse", "mean_mse_u", "eq18_obs", "eq22_true")}
        row.update({f"test_{k}": m_te[k] for k in ("mean_mse", "mean_mse_u", "eq18_obs", "eq22_true")})
        row["objective"] = float(loss.detach())
        if fig_norm is not None:
            row.update(fig_metrics())
        history.on_epoch(epoch, float(loss.detach()), row, time.perf_counter() - started)
        if epoch in rp.MILESTONES:
            milestones[epoch] = {k: row[k] for k in rp.MILESTONE_KEYS}
            bp = rp.interior_breakpoints(model, t_end)
            milestones[epoch]["layer1_hinges_s"] = bp["layer1"]
            milestones[epoch]["layer2_breakpoints_s"] = bp["layer2"]
        if float(loss.detach()) < best["loss"]:          # pre-update state = this loss's state
            best = {"loss": float(loss.detach()), "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if not torch.isfinite(loss):
            nonfinite_epoch = epoch
            logging.warning("non-finite loss at epoch %d; stopping (recorded, not hidden)", epoch)
            break
        loss.backward()
        opt.step()
        if epoch % 250 == 0:
            logging.info("epoch %5d  %s %.4e | train eq18 %.4e test eq18 %.4e",
                         epoch, args.objective, float(loss.detach()),
                         m_tr["eq18_obs"], m_te["eq18_obs"])
    history.close()

    final_train, train_phys = evaluate(tr)
    final_test, test_phys = evaluate(te)
    acts_grid = rp.trunk_activations(model, t_in)
    acts_dense = rp.trunk_activations(model, t_dense)
    final_trunk = {"kinks": rp.first_layer_kinks(model).tolist(),
                   "dead_on_grid": rp.dead_dimensions(acts_grid),
                   "dead_on_dense": rp.dead_dimensions(acts_dense),
                   "interior_breakpoints_s": rp.interior_breakpoints(model, t_end)}
    if is_tanh:
        final_trunk["tanh_saturation"] = {"grid": rp.tanh_saturation(model, t_in),
                                          "dense": rp.tanh_saturation(model, t_dense)}
    milestones[args.epochs] = {"train_mean_mse": final_train["mean_mse"],
                               "test_mean_mse": final_test["mean_mse"],
                               "train_eq18_obs": final_train["eq18_obs"],
                               "test_eq18_obs": final_test["eq18_obs"]}
    if best["state"] is not None:
        torch.save({"model_state": best["state"], "epoch": best["epoch"],
                    "train_objective": best["loss"], "selection": "lowest training objective "
                    "(pre-update state of that epoch); never test-based"},
                   run.run_dir / "checkpoint_best_train.pt")
    metrics = {
        "learning_rate": args.lr, "init_scheme": args.init, "seed": args.seed,
        "milestones": {str(k): v for k, v in sorted(milestones.items())},
        "best_train_checkpoint": {"epoch": best["epoch"], "train_objective": best["loss"]},
        "state_mode": mode, "objective": args.objective, "noise_percent": args.noise_percent,
        "epochs_completed": args.epochs if nonfinite_epoch is None else nonfinite_epoch,
        "nonfinite_epoch": nonfinite_epoch, "n_time": int(t.numel()),
        "initial": {"train": init_train, "test": init_test,
                    "objective": init_train[args.objective],
                    "prediction_stats_train": rp.prediction_stats(init_train_phys),
                    "prediction_stats_test": rp.prediction_stats(init_test_phys),
                    "trunk": init_trunk},
        "figure_normalization": None if fig_norm is None else {**fig_norm, **fig_metrics()},
        "final": {"train": final_train, "test": final_test,
                  "objective": final_train[args.objective],
                  "linearity_index_test": rp.linearity_index(
                      test_phys.numpy(), data["test_true"], data["t"]),
                  "linearity_index_train": rp.linearity_index(
                      train_phys.numpy(), data["train_true"], data["t"]),
                  "prediction_stats_train": rp.prediction_stats(train_phys),
                  "prediction_stats_test": rp.prediction_stats(test_phys),
                  "trunk": final_trunk},
    }
    (run.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(run.run_dir / "predictions.npz", t=data["t"], species=np.array(data["species"]),
             train_pred=train_phys.numpy(), train_true=data["train_true"],
             train_obs=data["train_obs"], test_pred=test_phys.numpy(),
             test_true=data["test_true"], test_obs=data["test_obs"],
             init_test_pred=init_test_phys.numpy())
    np.savez(run.run_dir / "trunk_activations.npz", t=data["t"], t_dense=t_dense.numpy(),
             **{f"grid_{k}": v for k, v in acts_grid.items()},
             **{f"dense_{k}": v for k, v in acts_dense.items()})
    run.save_final({"model_state": model.state_dict(), "architecture": arch,
                    "preprocessing": config["preprocessing"], "objective": args.objective,
                    "noise_percent": args.noise_percent, "seed": args.seed})
    logging.info("final: train %s", final_train)
    logging.info("final: test %s", final_test)
    logging.info("final trunk: %s", final_trunk)
    run.finish(ok=nonfinite_epoch is None)
    plot_run(run.run_dir)


def plot_run(run_dir: Path):
    """Loss curves, the Fig. 3 test condition, representative trajectories, trunk units."""
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    rows = list(csv.DictReader((run_dir / "history.csv").open()))
    ep = np.array([int(r["epoch"]) for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])          # noqa: E731
    p = np.load(run_dir / "predictions.npz")
    ta = np.load(run_dir / "trunk_activations.npz")
    meta = json.loads((run_dir / "metrics.json").read_text())
    title = f"{meta['state_mode']} / {meta['objective']}"
    sp = list(p["species"])

    with plt.rc_context({"figure.dpi": 120}):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, ls in (("mean_mse_u", "-"), ("eq18_obs", "--")):
            ax[0].semilogy(ep, col(f"train_{k}"), ls, label=f"train {k}")
            ax[0].semilogy(ep, col(f"test_{k}"), ls, label=f"test {k}")
        ax[0].set(xlabel="epoch", ylabel="loss (author-global u)", title=title)
        ax[0].legend(fontsize=7)
        ax[1].semilogy(ep, col("objective"), label=f"optimized: {meta['objective']}")
        ax[1].semilogy(ep, col("train_mean_mse"), ":", label="train mean_mse (trained rep.)")
        ax[1].set(xlabel="epoch", title="optimized objective")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(run_dir / "loss_curves.png")
        plt.close(fig)

        def traj(split, idx, name):
            fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
            for s, a in enumerate(axs.ravel()):
                a.plot(p["t"], p[f"{split}_true"][idx, :, s], "k-", lw=1, label="truth")
                a.plot(p["t"], p[f"{split}_obs"][idx, :, s], "o", ms=3, mfc="none",
                       color="grey", label="observations")
                a.plot(p["t"], p[f"{split}_pred"][idx, :, s], "r-", label="DeepONet final")
                if split == "test":
                    a.plot(p["t"], p["init_test_pred"][idx, :, s], "b:", lw=0.8,
                           label="DeepONet init")
                a.set_title(sp[s])
            axs[0, 0].legend(fontsize=7)
            for a in axs[1]:
                a.set_xlabel("t (s)")
            fig.suptitle(f"{title}: {split} trajectory {idx}")
            fig.tight_layout()
            fig.savefig(run_dir / name)
            plt.close(fig)

        traj("test", rp.FIG3_TEST_INDEX, "fig3_test_trajectory.png")
        traj("train", 0, "train_trajectory_0.png")
        traj("test", 0, "test_trajectory_0.png")

        keys = sorted(k[len("dense_"):] for k in ta.files if k.startswith("dense_"))
        fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5))
        for a, k in zip(np.atleast_1d(axs), keys):
            a.plot(ta["t_dense"], ta[f"dense_{k}"])
            a.set(xlabel="t (s)", title=f"trunk {k} (final)")
        fig.tight_layout()
        fig.savefig(run_dir / "trunk_activations.png")
        plt.close(fig)


if __name__ == "__main__":
    main()

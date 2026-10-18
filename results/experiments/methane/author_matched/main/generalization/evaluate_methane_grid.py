r"""Methane generalization over a dense 21 x 21 condition grid (hydrogen Fig. 8A analogue).

Reference: ``chemkan/data/generated/methane_fine.npz``, written by
``data_gen/extensions/generate_methane.py`` with the same mechanism (GRI-Mech 3.0), pressure,
tolerances (rtol 1e-9, atol 1e-15), window (5 ms) and 100 observation times as
``methane.npz``; only the conditions differ: T0 = 1400-1650 K in 12.5 K steps x phi =
0.5-1.5 in 0.05 steps. "Seen" = the 35 training conditions of ``methane.npz``; the other
406 (405 new + the held-out 1550 K / phi 1.3) are unseen. The fine file's own split and
normalization are ignored.

Model: the selected Stage-2 snapshot (epoch 180,000, notebook 15), integrated from each
initial state alone (full trajectory) with its training solver (Tsit5, rtol 1e-6, atol
1e-8, float32), one condition per solve. Error: time-averaged MSE over all 53 states on
the 100 observation times, in the training normalization of the checkpoint. Ignition delay
(argmax dT/dt) on a 601-point grid, reference from the generator's 601-point diagnostic.

    ~/uni_projects/chemkan-venv/bin/python \
        results/experiments/methane/author_matched/main/generalization/evaluate_methane_grid.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
sys.path.insert(0, str(ROOT / "chemkan/scripts/author_repo_match"))
import plot_trajectory as pt  # noqa: E402
from _predictions import checkpoint_sha256  # noqa: E402

from chemkan.solver import integrate  # noqa: E402

RUN = ROOT / "results/experiments/methane/author_matched/main"
CKPT = RUN / "S2_seed0/archive/stage2_S2_seed0_epoch180000.pt"
FINE = ROOT / "chemkan/data/generated/methane_fine.npz"
COARSE = ROOT / "chemkan/data/generated/methane.npz"
DIAG_POINTS = 601


def main() -> int:
    torch.set_num_threads(4)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = pt.load_model(ckpt)                                   # also checks methane.npz sha256
    torch.set_default_dtype(model["dtype"])
    fine, coarse = np.load(FINE, allow_pickle=True), np.load(COARSE, allow_pickle=True)
    ics, states, t = fine["ics"], fine["states"], np.asarray(fine["t"], float)
    names = [str(s) for s in fine["species"]] + ["T"]
    if [str(s) for s in coarse["species"]] != names[:-1] or not np.allclose(coarse["t"], t):
        raise SystemExit("methane_fine.npz differs from methane.npz in species or time grid")

    def match(targets):
        return np.array([np.any(np.all(np.isclose(row, targets), axis=1)) for row in ics])
    seen, held = match(coarse["train_ics"]), match(coarse["test_ics"])
    # the 36 coarse conditions must reproduce methane.npz exactly (same generator settings)
    c_states = np.concatenate([coarse["train_states"], coarse["test_states"]])
    c_ics = np.concatenate([coarse["train_ics"], coarse["test_ics"]])
    diffs = [np.abs(states[np.flatnonzero(np.all(np.isclose(ics, ic), axis=1))[0]] - s).max()
             for ic, s in zip(c_ics, c_states)]

    span = model["u_max"] - model["u_min"]
    as_t = lambda a: torch.as_tensor(np.asarray(a), dtype=model["dtype"])  # noqa: E731
    t_diag = np.linspace(t[0], t[-1], DIAG_POINTS)
    mse, mse_T, delay_mod, rise_mod = [], [], [], []
    with torch.no_grad():
        for i in range(len(ics)):
            z0 = as_t((states[i, 0] - model["u_min"]) / span)[None]
            z = integrate(model["rhs"], z0, as_t(t), model["solver"])[:, 0].numpy().astype(float)
            target = (states[i] - model["u_min"]) / span
            err = (z - target) ** 2
            mse.append(err.mean())
            mse_T.append(err[:, -1].mean())
            zd = integrate(model["rhs"], z0, as_t(t_diag), model["solver"])[:, 0, -1].numpy().astype(float)
            Td = zd * span[-1] + model["u_min"][-1]
            delay_mod.append(t_diag[np.argmax(np.gradient(Td, t_diag))])
            rise_mod.append(Td[-1] - Td[0])
            if i % 50 == 0:
                print(f"{i}/{len(ics)}", flush=True)
    mse, mse_T = np.array(mse), np.array(mse_T)
    delay_ref = np.asarray(fine["ignition_delay"], float)
    delay_mod = np.array(delay_mod)
    delay_rel = (delay_mod - delay_ref) / delay_ref
    unseen = ~seen

    rows = [{"T0": float(a), "phi": float(b),
             "group": "training" if s else ("held-out" if h else "unseen"),
             "mse_time_avg": float(m), "mse_T_time_avg": float(mt),
             "delay_ref_ms": 1e3 * float(dr), "delay_model_ms": 1e3 * float(dm),
             "delay_rel_err": float(de), "rise_ref_K": float(rr), "rise_model_K": float(rm)}
            for (a, b), s, h, m, mt, dr, dm, de, rr, rm in zip(
                ics, seen, held, mse, mse_T, delay_ref, delay_mod, delay_rel,
                fine["temperature_rise"], rise_mod)]
    with open(HERE / "methane_grid_441.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def st(v):
        v = np.asarray(v)
        return {"median": float(np.median(v)), "mean": float(v.mean()), "min": float(v.min()),
                "max": float(v.max()), "n": int(v.size)}
    summary = {
        "checkpoint": str(CKPT.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256(CKPT),
        "checkpoint_epochs": model["epochs"],
        "fine_grid": {"file": str(FINE.relative_to(ROOT)), "sha256": checkpoint_sha256(FINE),
                      "T0_K": [float(ics[:, 0].min()), float(ics[:, 0].max()), 12.5],
                      "phi": [float(ics[:, 1].min()), float(ics[:, 1].max()), 0.05],
                      "conditions": int(len(ics)),
                      "coarse_conditions_reproduce_methane_npz_max_abs_diff": float(max(diffs))},
        "solver": {"method": "tsit5", "rtol": model["solver"].rtol, "atol": model["solver"].atol,
                   "dtype": str(model["dtype"])},
        "metric": "time-averaged MSE over all 53 states on the 100 observation times, checkpoint's "
                  "training min-max normalization; full trajectory from the initial state",
        "mse_time_avg": {"training_35": st(mse[seen]), "unseen_406": st(mse[unseen]),
                         "held_out_1550K_phi1.3": float(mse[held][0]), "all_441": st(mse)},
        "ignition_delay_abs_rel_err": {"training_35": st(np.abs(delay_rel[seen])),
                                       "unseen_406": st(np.abs(delay_rel[unseen])),
                                       "held_out": float(np.abs(delay_rel[held][0]))},
        "diagnostic_grid_points": DIAG_POINTS,
    }
    (HERE / "methane_grid_441.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(ics, mse, seen, held, summary)
    print(json.dumps(summary["mse_time_avg"], indent=1))
    print(json.dumps(summary["ignition_delay_abs_rel_err"], indent=1))
    print("coarse reproduce:", summary["fine_grid"]["coarse_conditions_reproduce_methane_npz_max_abs_diff"])
    return 0


def grid(ics, v):
    T0s, phis = np.unique(ics[:, 0]), np.unique(ics[:, 1])
    G = np.full((len(T0s), len(phis)), np.nan)
    for (a, b), x in zip(ics, v):
        G[np.searchsorted(T0s, a), np.searchsorted(phis, b)] = x
    return T0s, phis, G


PAPER_RANGE = 10e-4     # hydrogen Fig. 8A colour range (0-10 x 10^-4), for a common scale


def plot(ics, mse, seen, held, summary):
    """Fig. 8A-style maps of the time-averaged MSE over the 21 x 21 conditions: the hydrogen
    paper range (linear 0-10e-4), the full range (linear 0-max, as hydrogen's full-range map)
    and the full range on a log scale."""
    rc = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    T0s, phis, G = grid(ics, mse)
    dT, dp = T0s[1] - T0s[0], phis[1] - phis[0]
    ext = [phis[0] - dp / 2, phis[-1] + dp / 2, T0s[0] - dT / 2, T0s[-1] + dT / 2]
    s = summary["mse_time_avg"]
    above = int(np.sum(mse > PAPER_RANGE))
    variants = {
        "methane_generalization_441": (
            dict(cmap="Reds", vmin=0, vmax=PAPER_RANGE * 1e4), 1e4, r"MSE [$\times10^{-4}$]", "max",
            f"hydrogen Fig. 8A range 0-10 x 1e-4; {above}/{len(mse)} errors above colour limit"),
        "methane_generalization_441_full_range": (
            dict(cmap="Reds", vmin=0, vmax=float(np.nanmax(G))), 1.0, "MSE", "neither",
            f"full measured range 0-{np.nanmax(G):.3g} (linear, as hydrogen's full-range Fig. 8A)"),
        "methane_generalization_441_full_range_log": (
            dict(cmap="Reds", norm=matplotlib.colors.LogNorm(vmin=np.nanmin(G), vmax=np.nanmax(G))), 1.0,
            "time-averaged MSE (log scale)", "neither", "full range, log colour"),
    }
    with plt.rc_context(rc):
        for name, (kw, scale, label, extend, note) in variants.items():
            fig, ax = plt.subplots(figsize=(7.2, 5.6))
            im = ax.imshow(G * scale, origin="lower", aspect="auto", extent=ext, **kw)
            cb = fig.colorbar(im, ax=ax, extend=extend)
            cb.set_label(label)
            ax.plot(ics[seen, 1], ics[seen, 0], "x", color="navy", ms=6, label="35 training conditions")
            ax.plot(ics[held, 1], ics[held, 0], "o", color="teal", mec="k", ms=8, label="held-out condition")
            ax.set_xlabel("Equivalence ratio")
            ax.set_ylabel("Initial temperature [K]")
            ax.set_xticks(np.round(np.arange(0.5, 1.51, 0.2), 1))
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False, fontsize=8)
            ax.set_title(f"Methane, 21 x 21 conditions (406 unseen), Stage-2 epoch "
                         f"{summary['checkpoint_epochs']:,}\n{note}\nmedian MSE: training "
                         f"{s['training_35']['median']:.2e}, unseen {s['unseen_406']['median']:.2e}, "
                         f"held-out {s['held_out_1550K_phi1.3']:.2e}", fontsize=9)
            fig.tight_layout()
            for ext_ in ("png", "pdf"):
                fig.savefig(HERE / f"{name}.{ext_}", dpi=200)
            plt.close(fig)


def plot_only() -> int:
    """Redraw the maps from the saved CSV and JSON (no integration)."""
    with open(HERE / "methane_grid_441.csv") as f:
        rows = list(csv.DictReader(f))
    ics = np.array([[float(r["T0"]), float(r["phi"])] for r in rows])
    mse = np.array([float(r["mse_time_avg"]) for r in rows])
    seen = np.array([r["group"] == "training" for r in rows])
    held = np.array([r["group"] == "held-out" for r in rows])
    plot(ics, mse, seen, held, json.loads((HERE / "methane_grid_441.json").read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(plot_only() if "--plot-only" in sys.argv else main())

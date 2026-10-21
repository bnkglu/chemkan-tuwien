"""Methane extension: the main species of one condition in a single figure.

Not a figure of the ChemKAN paper (it has no methane case). CH4, O2, CO2, H2O and CO of the
selected methane Stage-2 model (epoch 180,000, ``results/methane/README.md``) against the
Cantera reference, by default at the held-out condition 1550 K, phi 1.3. Left: mass fractions
on a linear axis, including any negative predicted values; right: the same on a logarithmic
axis, which shows the decay of CH4 and O2 and the early rise of the products (it can show only
positive values: a prediction that falls below 1e-5, or below zero, leaves the panel). ChemKAN:
lines on a dense time grid; reference: markers at every second observation time.

Prediction and data loading are those of ``combustion/plot_trajectory.py``; nothing is
trained.

    python chemkan/scripts/figures/methane_species.py
    python chemkan/scripts/figures/methane_species.py --condition 1450,0.9
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import ROOT, add_repo_paths, load_checkpoint, save_figure, use_headless_backend

CHECKPOINT = ROOT / "results/methane/S2_seed0/archive/stage2_S2_seed0_epoch180000.pt"
FIGURES_METHANE = ROOT / "results/methane/figures"
SPECIES = ["CH4", "O2", "CO2", "H2O", "CO"]
COLOURS = dict(zip(SPECIES, ["tab:brown", "tab:blue", "tab:green", "tab:cyan", "tab:red"]))
LABELS = {"CH4": "CH$_4$", "O2": "O$_2$", "CO2": "CO$_2$", "H2O": "H$_2$O", "CO": "CO"}


def species_data(checkpoint=CHECKPOINT, condition=(1550.0, 1.3), points=601) -> dict:
    """Reference and ChemKAN mass fractions of ``SPECIES`` at one (T0 [K], phi) condition."""
    add_repo_paths()
    import plot_trajectory as pt

    model = pt.load_model(load_checkpoint(checkpoint))
    torch.set_default_dtype(model["dtype"])
    ref = pt.load_reference(model["data"])
    i = pt.find_condition(ref, *condition)
    span = model["u_max"] - model["u_min"]
    m = len(ref["species"])
    target_hat = (ref["states"][i] - model["u_min"]) / span
    at_obs = pt.predict(model, ref, i, ref["t"])
    dense_t = np.linspace(ref["t"][0], ref["t"][-1], points)
    dense = pt.predict(model, ref, i, dense_t) * span + model["u_min"]
    cols = [ref["species"].index(s) for s in SPECIES]
    return {"t": ref["t"], "reference": ref["states"][i][:, cols], "dense_t": dense_t,
            "prediction": dense[:, cols], "label": ref["labels"][i],
            "split": "held-out" if ref["is_test"][i] else "training",
            "mse_time_avg": float(((at_obs[:, :m + 1] - target_hat[:, :m + 1]) ** 2).mean()),
            "epochs": model["epochs"]}


def make_figure(output_path=FIGURES_METHANE / "methane_main_species_S2_epoch180000_T1550_phi1p3",
                checkpoint=CHECKPOINT, condition=(1550.0, 1.3)):
    d = species_data(checkpoint, condition)
    t_ms, dense_ms = d["t"] * 1e3, d["dense_t"] * 1e3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, log in zip(axes, (False, True)):
        for j, s in enumerate(SPECIES):
            c = COLOURS[s]
            ax.plot(dense_ms, d["prediction"][:, j], "-", lw=1.6, color=c)
            ax.plot(t_ms[::2], d["reference"][::2, j], "o", ms=3.5, mfc="none", mew=0.9, color=c)
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("mass fraction [–]")
        ax.set_xlim(0, t_ms[-1])
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 0.5)
            ax.set_title("logarithmic scale", fontsize=10)
        else:
            ax.axhline(0.0, color="0.5", lw=0.8)
            ax.set_ylim(bottom=min(0.0, 1.3 * d["prediction"].min()))   # negative predictions stay visible
            ax.set_title("linear scale", fontsize=10)
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], color=COLOURS[s], lw=2.5, label=LABELS[s]) for s in SPECIES]
    handles += [plt.Line2D([], [], color="0.2", lw=1.6, label="ChemKAN"),
                plt.Line2D([], [], color="0.2", ls="none", marker="o", mfc="none", label="Cantera reference")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"Methane, {d['label'].replace('phi', 'φ')} ({d['split']}): ChemKAN Stage 2, "
                 f"epoch {d['epochs']:,}  |  time-averaged MSE (normalized, 52 species + T) "
                 f"{d['mse_time_avg']:.3e}", y=1.09, fontsize=10.5)
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig, d


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--condition", default="1550,1.3", help="T0,phi")
    p.add_argument("--output", default=None, help="default: results/methane/figures/"
                   "methane_main_species_S2_epoch180000_T<T0>_phi<phi> (1.3 -> 1p3)")
    args = p.parse_args()
    T0, phi = map(float, args.condition.split(","))
    tag = f"T{T0:g}_phi{phi:g}".replace(".", "p")                 # no dot: save_figure sets the suffix
    output = args.output or FIGURES_METHANE / f"methane_main_species_S2_epoch180000_{tag}"
    _, d = make_figure(output, args.checkpoint, (T0, phi))
    print(f"{d['label']} ({d['split']}): time-averaged MSE {d['mse_time_avg']:.4e}")
    print(f"wrote {output}.pdf/.png")


if __name__ == "__main__":
    main()

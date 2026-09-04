r"""Temperature-trajectory comparison across thermo.linear initializations (DIAGNOSTIC).

This is **not** a reproduction of paper Fig. 7. It is an ablation/diagnostic figure that
happens to use the same two representative conditions the hydrogen diagnostics already
use (training 1050 K / phi 0.9 and held-out 1150 K / phi 1.3), so that the effect of the
``thermo.linear`` initialization on the *actual* temperature trajectory T(t) is visible
rather than inferred from a scalar loss.

The script only READS checkpoints. It never trains, never writes a checkpoint, and never
touches a run directory.

Model rebuilding, solver reconstruction and integration are reused from
``evaluate_hydrogen``; the ignition-delay definition is reused from
``hydrogen_thermo_intervention`` (time of maximum dT/dt, undefined below a 100 K rise).

Grids: every number in the CSV is computed on the dataset's own 50-point grid over
[0, 0.6 ms], which is the grid all existing hydrogen diagnostics (and the Stage-2 probe)
use. The plotted model curves are additionally integrated on a denser grid over the SAME
interval so that induction/ignition shape is faithful; the reference is only available at
its native 50 points and is drawn with markers.

The 50-point grid quantizes the ignition delay to ~12.2 us. The ``*_dense`` columns apply
the SAME ignition-delay definition to both the model and the GENUINE 20000-point Cantera
reference (hydrogen_temperature_20000.npz) on that same grid, so the comparison is
grid-consistent on both sides. Model_dense is never compared against reference_50, and no
interpolated reference is used. Neither column is a new ignition criterion.

    python3 chemkan/scripts/diagnostics/plot_thermo_initialization_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[1]                       # chemkan/scripts
_REPO = _SCRIPTS.parents[1]
for _p in (str(_SCRIPTS.parent / "src"), str(_SCRIPTS), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from _data import DATA_DIR, load_input_scaling, resolve_device        # noqa: E402
from evaluate_hydrogen import (build_chemkan, integrate_hydrogen,     # noqa: E402
                               solver_from_ckpt)
from hydrogen_thermo_intervention import ignition_delay               # noqa: E402

DIAG = _REPO / "results/reproduction/chemkan/hydrogen/diagnostics"
FIG_DIR = _REPO / "results/reproduction/chemkan/hydrogen/figures"
TAB_DIR = _REPO / "results/reproduction/chemkan/hydrogen/tables"

CONDITIONS = [(1050.0, 0.9, "training"), (1150.0, 1.3, "held-out")]

# key, run directory, legend label, initialization descriptor
RUNS = [
    ("random10k", "thermo_init_random_stage2_10000_seed0",
     "Random init, Stage-2 10k", "random (default)"),
    ("cantera500", "thermo_init_cantera_500_ref1050_phi05_seed0",
     "Cantera init, Stage-2 500", "cantera (ref 1050 K / phi 0.5)"),
    ("nm_dir0", "thermo_init_scaled_random_normmatched_dir0",
     "Norm-matched random dir0, 500", "scaled-random norm-matched, dir seed 0"),
    ("nm_dir1", "thermo_init_scaled_random_normmatched_dir1",
     "Norm-matched random dir1, 500", "scaled-random norm-matched, dir seed 1"),
    ("nm_dir2", "thermo_init_scaled_random_normmatched_dir2",
     "Norm-matched random dir2, 500", "scaled-random norm-matched, dir seed 2"),
    ("x1e4_dir0", "thermo_init_scaled_random_1e4_dir0",
     "Random x1e4 dir0, 500", "scaled-random x1e4, dir seed 0"),
    ("x1e5_dir0", "thermo_init_scaled_random_1e5_dir0",
     "Random x1e5 dir0, 500", "scaled-random x1e5, dir seed 0"),
]

MAIN_KEYS = ["random10k", "cantera500", "nm_dir0", "nm_dir1", "nm_dir2"]
SCALE_KEYS = ["random10k", "x1e4_dir0", "x1e5_dir0", "nm_dir0"]

STYLE = {
    "random10k":  dict(color="#7f7f7f", ls="-",  lw=1.6),
    "cantera500": dict(color="#1f77b4", ls="-",  lw=1.8),
    "nm_dir0":    dict(color="#2ca02c", ls="-",  lw=1.6),
    "nm_dir1":    dict(color="#ff7f0e", ls="-",  lw=1.6),
    "nm_dir2":    dict(color="#d62728", ls="-",  lw=1.6),
    "x1e4_dir0":  dict(color="#9467bd", ls="--", lw=1.6),
    "x1e5_dir0":  dict(color="#8c564b", ls="-.", lw=1.6),
}
REF_STYLE = dict(color="black", ls=":", lw=2.4, marker="o", ms=3.0, mfc="none")


def _finite(name, arr):
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        raise SystemExit(f"non-finite values in {name}")
    return a


def load_run(key, dirname, device):
    ckpt_path = DIAG / dirname / "checkpoint_final.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"[{key}] no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt, ckpt_path


def integrate_T(model, input_norm, solver, u0_state, t, device):
    """Integrate from the initial condition alone; return the temperature channel."""
    u0 = torch.as_tensor(u0_state, dtype=torch.float32, device=device).unsqueeze(0)
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = integrate_hydrogen(model, input_norm, solver, u0, tt, device)[:, 0].cpu().numpy()
    return pred[:, -1]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--dense-points", type=int, default=None,
                   help="ignored (kept for CLI compatibility): the dense grid is now the genuine "
                        "20000-point Cantera grid from hydrogen_temperature_20000.npz.")
    p.add_argument("--no-scale-figure", action="store_true")
    args = p.parse_args()

    dev = resolve_device(args.device)
    npz = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    t, ics, states = npz["t"], npz["ics"], npz["states"]
    species = [str(s) for s in npz["species"]]
    # GENUINE dense Cantera reference (hydrogen_temperature_20000.npz); the model is
    # integrated on exactly this grid so dense-vs-dense comparisons share one grid and
    # the reference is real Cantera output, never an interpolation of the 50-point data.
    _dense = np.load(DATA_DIR / "hydrogen_temperature_20000.npz", allow_pickle=True)
    t_dense = np.asarray(_dense["t"], dtype=float)
    def _ref_dense(T0, phi):
        for k_ic, k_T in (("train_ics", "train_T"), ("test_ics", "test_T")):
            ic = np.asarray(_dense[k_ic])
            hit = np.where((np.abs(ic[:, 0] - T0) < 1e-9) & (np.abs(ic[:, 1] - phi) < 1e-9))[0]
            if len(hit) == 1:
                return np.asarray(_dense[k_T])[:, int(hit[0]), 0].astype(float)
        raise SystemExit(f"({T0},{phi}) not in the dense Cantera cache")

    # ---- reference (Cantera) ------------------------------------------------
    ref = {}
    for T0, phi, role in CONDITIONS:
        hit = np.where((np.abs(ics[:, 0] - T0) < 1e-9) & (np.abs(ics[:, 1] - phi) < 1e-9))[0]
        if len(hit) != 1:
            raise SystemExit(f"condition (T0={T0:g}, phi={phi:g}) is not a dataset IC")
        i = int(hit[0])
        Tr = _finite(f"reference {T0:.0f}/{phi}", states[i][:, -1])
        Trd = _ref_dense(T0, phi)
        ref[(T0, phi)] = {
            "index": i, "state": states[i], "T": Tr, "T_dense": Trd,
            "peak": float(Tr.max()), "final": float(Tr[-1]), "initial": float(Tr[0]),
            "delay": ignition_delay(t, Tr),
            "delay_dense": ignition_delay(t_dense, Trd),
            "is_test": bool(npz["is_test"][i]), "role": role,
        }
        print(f"reference [{role}] {T0:.0f} K / phi {phi}: peak {Tr.max():7.1f} K, "
              f"delay {ref[(T0, phi)]['delay']}")

    # ---- models -------------------------------------------------------------
    rows, curves = [], {}
    for key, dirname, label, init_desc in RUNS:
        ckpt, ckpt_path = load_run(key, dirname, dev)
        model = build_chemkan(ckpt, len(species), dev)
        solver = solver_from_ckpt(ckpt)
        input_norm = load_input_scaling(ckpt, dev)
        cdata = ckpt.get("data", {})
        if cdata.get("species") is not None and list(cdata["species"]) != species:
            raise SystemExit(f"[{key}] checkpoint species order differs from the dataset")
        w = model.thermo.linear.weight.detach().cpu().numpy().ravel()
        print(f"\nloaded {key:11s} run_id={ckpt.get('run_id')}  "
              f"||thermo.linear||={np.linalg.norm(w):.4f}")

        for T0, phi, role in CONDITIONS:
            R = ref[(T0, phi)]
            Tc = _finite(f"{key} @{T0:.0f}/{phi} (coarse)",
                         integrate_T(model, input_norm, solver, R["state"][0], t, dev))
            Td = _finite(f"{key} @{T0:.0f}/{phi} (dense)",
                         integrate_T(model, input_norm, solver, R["state"][0], t_dense, dev))
            curves[(key, T0, phi)] = Td
            delay = ignition_delay(t, Tc)
            delay_d = ignition_delay(t_dense, Td)

            def _rel(d, ref_d):
                return ("" if (d is None or ref_d is None or ref_d == 0)
                        else f"{(d - ref_d) / ref_d:.4f}")

            rel = _rel(delay, R["delay"])
            rows.append({
                "run": dirname,
                "initialization": init_desc,
                "condition": role,
                "T0_K": f"{T0:.0f}",
                "phi": f"{phi:g}",
                "initial_T_K": f"{Tc[0]:.1f}",
                "peak_T_K": f"{Tc.max():.1f}",
                "reference_peak_T_K": f"{R['peak']:.1f}",
                "peak_error_K": f"{Tc.max() - R['peak']:.1f}",
                "ignition_delay_s": "" if delay is None else f"{delay:.6e}",
                "reference_ignition_delay_s": ("" if R["delay"] is None
                                               else f"{R['delay']:.6e}"),
                "ignition_delay_relative_error": rel,
                "ignition_delay_s_dense": "" if delay_d is None else f"{delay_d:.6e}",
                "ignition_delay_relative_error_dense": _rel(delay_d, R["delay_dense"]),
                "reference_ignition_delay_s_dense": ("" if R["delay_dense"] is None
                                                     else f"{R['delay_dense']:.6e}"),
                "min_T_K": f"{Tc.min():.1f}",
                "min_T_K_dense": f"{Td.min():.1f}",
                "final_T_K": f"{Tc[-1]:.1f}",
                "reference_final_T_K": f"{R['final']:.1f}",
                "thermo_linear_norm": f"{np.linalg.norm(w):.6e}",
                "checkpoint": str(ckpt_path.relative_to(_REPO)),
            })
            d_str = "none" if delay is None else f"{delay * 1e3:.4f} ms"
            print(f"  [{role:8s}] peak {Tc.max():7.1f} K (ref {R['peak']:7.1f}), "
                  f"final {Tc[-1]:7.1f} K, delay {d_str}")
            # plotting-grid consistency check (reported, not enforced)
            dd = ignition_delay(t_dense, Td)
            if (dd is None) != (delay is None):
                print(f"    note: ignition detection differs between the 50-point grid "
                      f"({delay}) and the {args.dense_points}-point plotting grid ({dd})")

    # ---- CSV ----------------------------------------------------------------
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TAB_DIR / "hydrogen_thermo_initialization_trajectory_comparison.csv"
    with csv_path.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    expected = len(RUNS) * len(CONDITIONS)
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} rows, wrote {len(rows)}")
    print(f"\nsaved {csv_path}  ({len(rows)} rows)")

    # ---- figures ------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    def make_figure(keys, stem, title, subtitle):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=False)
        handles = None
        for ax, (T0, phi, role) in zip(axes, CONDITIONS):
            R = ref[(T0, phi)]
            ax.plot(t * 1e3, R["T"], label="Reference (Cantera, 50 pts)", **REF_STYLE)
            for key in keys:
                label = next(l for k, _, l, _ in RUNS if k == key)
                ax.plot(t_dense * 1e3, curves[(key, T0, phi)], label=label, **STYLE[key])
            ax.set_title(f"$T_0$ = {T0:.0f} K, $\\phi$ = {phi:g}   ({role})", fontsize=11)
            ax.set_xlabel("time [ms]")
            ax.set_ylabel("temperature [K]")
            ax.set_xlim(t[0] * 1e3, t[-1] * 1e3)
            ax.grid(alpha=0.25, lw=0.6)
            handles = ax.get_legend_handles_labels()
        fig.suptitle(title, fontsize=13, y=0.995)
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=9, color="#444444")
        fig.legend(*handles, fontsize=9, loc="lower center",
                   ncol=min(len(keys) + 1, 6), frameon=False,
                   bbox_to_anchor=(0.5, 0.045))
        fig.text(0.995, 0.006, "DIAGNOSTIC / ABLATION -- not a paper Fig. 7 reproduction",
                 ha="right", fontsize=8.5, color="#b00000", style="italic")
        fig.tight_layout(rect=(0, 0.11, 1, 0.90))
        for ext in ("png", "pdf"):
            out = FIG_DIR / f"{stem}.{ext}"
            fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
            written.append(out)
        plt.close(fig)

    make_figure(
        MAIN_KEYS,
        "hydrogen_DIAGNOSTIC_thermo_initialization_temperature_comparison",
        "Hydrogen thermo-initialization diagnostic: temperature trajectories",
        "Effect of the thermo.linear initialization (magnitude and direction) on T(t); "
        "identical Stage-1 kinetic model, seed 0.")

    if not args.no_scale_figure:
        make_figure(
            SCALE_KEYS,
            "hydrogen_DIAGNOSTIC_thermo_initialization_scale_response",
            "Hydrogen thermo-initialization diagnostic: scale response",
            "Same random direction (dir seed 0) at increasing initialization magnitude.")

    for out in written:
        if not out.exists() or out.stat().st_size == 0:
            raise SystemExit(f"figure not written: {out}")
        print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

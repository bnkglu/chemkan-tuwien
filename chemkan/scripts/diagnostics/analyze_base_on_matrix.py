r"""Base-activation architecture sensitivity: N=5/base-OFF vs N=4/base-ON (DIAGNOSTIC).

Recomputes every metric from checkpoints / probe CSVs / history CSVs using the repository
evaluation functions. Nothing is trained; nothing is modified. Importable from Notebook 09
(``compute_matrix``, ``figure_*``) and runnable as a CLI that writes the CSV/figure
artifacts under ``results/reproduction/chemkan/hydrogen/{tables,figures}``.

Terminology (kept strict throughout):
    base activation ON  -- PAPER-EXPLICIT structural feature of Eq. 11
    hydrogen N=4        -- INFERRED from Eq. 11 + the reported 344-parameter count
    N=5 / base OFF      -- the historical INFERRED count-matching interpretation
Neither N is stated in the paper.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[1]
_REPO = _SCRIPTS.parents[1]
for _p in (str(_SCRIPTS.parent / "src"), str(_SCRIPTS), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _data import DATA_DIR, load_hydrogen, load_input_scaling                # noqa: E402
from evaluate_hydrogen import (build_chemkan, integrate_hydrogen,            # noqa: E402
                               solver_from_ckpt)
from hydrogen_thermo_intervention import ignition_delay                       # noqa: E402
from chemkan.losses import trajectory_mse                                     # noqa: E402
from chemkan.normalization import MinMaxNormalizer                            # noqa: E402

H2 = _REPO / "results/reproduction/chemkan/hydrogen"
DIAG = H2 / "diagnostics"
BASE_ON = DIAG / "base_on_n4"
FIGURES, TABLES = H2 / "figures", H2 / "tables"

CONDS = [(1050.0, 0.9, "training"), (1150.0, 1.3, "held-out")]
PEAK_TOL_K, DELAY_TOL_REL = 150.0, 0.25          # representative-condition gate (notebook 08)

# arm key -> (label, base-OFF 500 run, base-OFF 10k run, base-ON run [snapshot@500 + final])
ARMS = {
    "random":   ("default random",          None,
                 "thermo_init_random_stage2_10000_seed0",           "random_stage2_10000_seed0"),
    "cantera":  ("Cantera init",            "thermo_init_cantera_500_ref1050_phi05_seed0",
                 "thermo_init_cantera_stage2_10000_seed0",          "cantera_stage2_10000_seed0"),
    "x1e5":     ("random x1e5 dir0",        "thermo_init_scaled_random_1e5_dir0",
                 "thermo_init_scaled_random_1e5_dir0_stage2_10000", "scaled_random_1e5_dir0_stage2_10000"),
    "nm_dir0":  ("norm-matched dir0",       "thermo_init_scaled_random_normmatched_dir0",
                 "thermo_init_scaled_random_normmatched_dir0_stage2_10000", "normmatched_dir0_stage2_10000"),
    "nm_dir1":  ("norm-matched dir1",       "thermo_init_scaled_random_normmatched_dir1",
                 "thermo_init_scaled_random_normmatched_dir1_stage2_10000", "normmatched_dir1_stage2_10000"),
    "nm_dir2":  ("norm-matched dir2",       "thermo_init_scaled_random_normmatched_dir2",
                 "thermo_init_scaled_random_normmatched_dir2_stage2_10000", "normmatched_dir2_stage2_10000"),
}
COLORS = {"random": "#7f7f7f", "cantera": "#1f77b4", "x1e5": "#8c564b",
          "nm_dir0": "#2ca02c", "nm_dir1": "#ff7f0e", "nm_dir2": "#d62728"}


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        npz = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
        self.t, self.ics, self.states = npz["t"], npz["ics"], npz["states"]
        self.species = [str(s) for s in npz["species"]]
        self.m = len(self.species)
        self.full_norm = MinMaxNormalizer(torch.as_tensor(npz["u_min"]),
                                          torch.as_tensor(npz["u_max"]))
        dense = np.load(DATA_DIR / "hydrogen_temperature_20000.npz", allow_pickle=True)
        self.t_dense = np.asarray(dense["t"], float)
        self._dense = dense
        self.train = load_hydrogen(split="train")
        self.test = load_hydrogen(split="test")
        self.fn = MinMaxNormalizer(self.train["u_min"], self.train["u_max"])

    def reference(self, T0, phi):
        hit = np.where((np.abs(self.ics[:, 0] - T0) < 1e-9) & (np.abs(self.ics[:, 1] - phi) < 1e-9))[0]
        assert len(hit) == 1
        return self.states[int(hit[0])]

    def reference_T_dense(self, T0, phi):
        """GENUINE 20000-point Cantera T(t); never an interpolation of the 50-point data."""
        for k_ic, k_T in (("train_ics", "train_T"), ("test_ics", "test_T")):
            ic = np.asarray(self._dense[k_ic])
            hit = np.where((np.abs(ic[:, 0] - T0) < 1e-9) & (np.abs(ic[:, 1] - phi) < 1e-9))[0]
            if len(hit) == 1:
                return np.asarray(self._dense[k_T])[:, int(hit[0]), 0].astype(float)
        raise AssertionError(f"({T0},{phi}) not in dense cache")


# ---------------------------------------------------------------------- one model
def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build(ck, m):
    return build_chemkan(ck, m, "cpu"), load_input_scaling(ck, "cpu"), solver_from_ckpt(ck)


def split_mse(model, inorm, solver, split_data, fn):
    """Eq. 18 normalized trajectory MSE over a whole split, plus per-state contributions."""
    with torch.no_grad():
        pred = integrate_hydrogen(model, inorm, solver, split_data["full_TBm1"][0],
                                  split_data["t"])
    truth = split_data["full_TBm1"]
    mse = float(trajectory_mse(fn.normalize(pred), fn.normalize(truth)))
    dN = (fn.normalize(pred) - fn.normalize(truth)).numpy()
    per_state = (dN ** 2).sum(0).mean(0)                      # sum over time, mean over cases
    P = pred.numpy()
    return dict(mse=mse, per_state=per_state,
                negative_species_frac=float((P[..., :-1] < 0).mean()),
                T_min=float(P[..., -1].min()), T_max=float(P[..., -1].max()))


def condition_metrics(model, inorm, solver, D: Data, T0, phi):
    ref = D.reference(T0, phi)
    u0 = torch.as_tensor(ref[0], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        Td = integrate_hydrogen(model, inorm, solver, u0,
                                torch.as_tensor(D.t_dense, dtype=torch.float32))[:, 0, -1].numpy()
    rdn = D.reference_T_dense(T0, phi)
    d_model, d_ref = ignition_delay(D.t_dense, Td), ignition_delay(D.t_dense, rdn)
    err = np.nan if (d_model is None or not d_ref) else 100.0 * (d_model - d_ref) / d_ref
    gate = (abs(Td.max() - rdn.max()) <= PEAK_TOL_K and np.isfinite(err)
            and abs(err) / 100.0 <= DELAY_TOL_REL)
    return dict(peak_T_K=float(Td.max()), ref_peak_T_K=float(rdn.max()),
                peak_error_K=float(Td.max() - rdn.max()),
                T_min_K=float(Td.min()), cools_below_T0=bool(Td.min() < T0 - 1.0),
                delay_dense_ms=np.nan if d_model is None else d_model * 1e3,
                ref_delay_dense_ms=d_ref * 1e3, delay_error_pct=err, gate=bool(gate),
                T_dense=Td)


def evaluate_checkpoint(path, D: Data):
    ck = load_ckpt(path)
    model, inorm, solver = build(ck, D.m)
    tr = split_mse(model, inorm, solver, D.train, D.fn)
    te = split_mse(model, inorm, solver, D.test, D.fn)
    w = model.thermo.linear.weight.detach().numpy().ravel().copy()
    out = dict(train_mse=tr["mse"], test_mse=te["mse"],
               train_dominant_state=(D.species + ["T"])[int(np.argmax(tr["per_state"]))],
               train_dominant_share=float(tr["per_state"].max() / tr["per_state"].sum()),
               train_T_share=float(tr["per_state"][-1] / tr["per_state"].sum()),
               train_negative_species_frac=tr["negative_species_frac"],
               train_T_min_K=tr["T_min"], train_T_max_K=tr["T_max"],
               thermo_norm=float(np.linalg.norm(w)), thermo_coeffs=w,
               per_state_train=tr["per_state"],
               arch=ck["architecture"], n_params=sum(p.numel() for p in model.parameters()))
    curves = {}
    for T0, phi, role in CONDS:
        cm = condition_metrics(model, inorm, solver, D, T0, phi)
        curves[role] = cm.pop("T_dense")
        pre = "train" if role == "training" else "held"
        out.update({f"{pre}_{k}": v for k, v in cm.items()})
    out["gate_both"] = bool(out["train_gate"] and out["held_gate"])
    out["curves"] = curves
    return out


# ------------------------------------------------------------------ run artifacts
def history(run_dir):
    p = Path(run_dir) / "history_stage2.csv"
    return pd.read_csv(p) if p.exists() else None


def probe(run_dir):
    p = Path(run_dir) / "stage2_probe.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["ignites"] = df["ignites"].astype(str).str.lower().eq("true")
    return df


def run_wall_hours(run_dir):
    log = Path(run_dir) / "run.log"
    if not log.exists():
        return np.nan
    ts = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+", log.read_text(), re.M)
    if len(ts) < 2:
        return np.nan
    f = "%Y-%m-%d %H:%M:%S"
    return (_dt.datetime.strptime(ts[-1], f) - _dt.datetime.strptime(ts[0], f)).total_seconds() / 3600


def stability(hist):
    """Final vs minimum Stage-2 loss: a run whose final loss is >2x its own minimum diverged."""
    if hist is None:
        return dict(min_loss=np.nan, min_epoch=np.nan, final_loss=np.nan, final_over_min=np.nan,
                    stable=None)
    mn = float(hist.total_loss.min()); fi = float(hist.total_loss.iloc[-1])
    return dict(min_loss=mn, min_epoch=int(hist.loc[hist.total_loss.idxmin(), "epoch"]),
                final_loss=fi, final_over_min=fi / mn, stable=bool(fi / mn < 2.0))


def sec_per_epoch(hist):
    e = hist.elapsed_seconds.to_numpy(float)
    return np.diff(e, prepend=0.0)


# --------------------------------------------------------------------- the matrix
def compute_matrix(D: Data | None = None, verbose=True):
    """Every (interpretation, arm, budget) cell, recomputed. Returns (df, curves, probes)."""
    D = D or Data()
    rows, curves, probes, hists = [], {}, {}, {}
    for key, (label, off500, off10k, on) in ARMS.items():
        cells = [("base-OFF (N=5)", 500,   DIAG / off500 / "checkpoint_final.pt" if off500 else None, DIAG / off500 if off500 else None),
                 ("base-OFF (N=5)", 10000, DIAG / off10k / "checkpoint_final.pt", DIAG / off10k),
                 ("base-ON (N=4)",  500,   BASE_ON / on / "checkpoint_stage2_epoch_500.pt", BASE_ON / on),
                 ("base-ON (N=4)",  10000, BASE_ON / on / "checkpoint_final.pt", BASE_ON / on)]
        for interp, budget, ckpt, run_dir in cells:
            if ckpt is None or not ckpt.exists():
                rows.append(dict(arm=key, label=label, interpretation=interp, budget=budget,
                                 available=False))
                continue
            if verbose:
                print(f"  evaluating {interp:15s} {label:20s} @{budget:>5} ...", flush=True)
            ev = evaluate_checkpoint(ckpt, D)
            curves[(interp, key, budget)] = ev.pop("curves")
            coeffs = ev.pop("thermo_coeffs"); per_state = ev.pop("per_state_train")
            h = history(run_dir); pr = probe(run_dir)
            hists[(interp, key)] = h; probes[(interp, key)] = pr
            st = stability(h) if budget == 10000 else {}
            row = dict(arm=key, label=label, interpretation=interp, budget=budget,
                       available=True, checkpoint=str(ckpt.relative_to(_REPO)), **ev)
            if h is not None and budget == 10000:
                spe = sec_per_epoch(h)
                row.update(wall_hours=run_wall_hours(run_dir),
                           median_s_per_epoch=float(np.median(spe)),
                           median_nfe=float(h["nfe"].median()) if "nfe" in h else np.nan,
                           # exact per-epoch NFE stays in history_stage2.csv; each run is
                           # summarized by median NFE/epoch AND total solver work.
                           total_nfe=float(h["nfe"].sum()) if "nfe" in h else np.nan,
                           **{f"stability_{k}": v for k, v in st.items()})
            if pr is not None:
                r500 = pr[pr.epoch == 500]
                row["probe_loss_at_500"] = float(r500.stage2_loss.iloc[0]) if len(r500) else np.nan
                row["probe_loss_at_10000"] = float(pr[pr.epoch == 10000].stage2_loss.iloc[0]) \
                    if (pr.epoch == 10000).any() else np.nan
            for s, v in zip(D.species, coeffs):
                row[f"coeff_{s}"] = float(v)
            for s, v in zip(D.species + ["T"], per_state):
                row[f"eq18_{s}"] = float(v)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, curves, probes, hists, D


def stage1_comparison(D: Data | None = None):
    """Stage-1 (species-only, observed T) comparison of the two shared kinetic cores."""
    from chemkan.dynamics import KineticDynamics
    from chemkan.solver import integrate
    from chemkan.temperature import ObservedTemperature
    D = D or Data()
    rows = []
    for interp, run_dir in (("base-OFF (N=5)", DIAG / "stage1_seed0"),
                            ("base-ON (N=4)", BASE_ON / "stage1_seed0")):
        ck = load_ckpt(run_dir / "checkpoint_stage1.pt")
        model, inorm, solver = build(load_ckpt(run_dir / "checkpoint_final.pt"), D.m)
        h1 = pd.read_csv(run_dir / "history_stage1.csv")
        # species-only integration with the OBSERVED temperature, over the whole train split
        obsT = ObservedTemperature(D.train["t"], D.train["T_obs_TB1"]) if "T_obs_TB1" in D.train \
            else None
        row = dict(interpretation=interp, architecture=json.dumps(ck["architecture"]),
                   n_params=sum(p.numel() for p in model.parameters()),
                   stage1_final_loss=float(ck["stage1_final_loss"]),
                   stage1_min_loss=float(h1.total_loss.min()),
                   stage1_wall_hours=run_wall_hours(run_dir),
                   stage1_median_nfe=float(h1["nfe"].median()) if "nfe" in h1 else np.nan,
                   stage1_total_nfe=float(h1["nfe"].sum()) if "nfe" in h1 else np.nan,
                   stage1_sha256=__import__("hashlib").sha256(
                       (run_dir / "checkpoint_stage1.pt").read_bytes()).hexdigest()[:12])
        # open-loop species MSE at the representative training condition
        T0, phi, _ = CONDS[0]
        ref = D.reference(T0, phi)
        obsT = ObservedTemperature(torch.as_tensor(D.t, dtype=torch.float32),
                                   torch.as_tensor(ref[:, -1], dtype=torch.float32).reshape(-1, 1, 1))
        kin = KineticDynamics(model.kinetic, obsT, input_normalizer=inorm)
        with torch.no_grad():
            s1 = integrate(kin, torch.as_tensor(ref[0, :D.m], dtype=torch.float32).unsqueeze(0),
                           torch.as_tensor(D.t, dtype=torch.float32), solver)[:, 0].numpy()
        sub = D.full_norm.subset(slice(0, D.m)) if hasattr(D.full_norm, "subset") else None
        refN = D.full_norm.normalize(torch.as_tensor(ref, dtype=torch.float32)).numpy()[:, :D.m]
        s1N = D.full_norm.normalize(torch.as_tensor(np.c_[s1, ref[:, -1]], dtype=torch.float32)).numpy()[:, :D.m]
        row["open_loop_species_mse_1050_0.9"] = float(((s1N - refN) ** 2).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("interpretation")


# ----------------------------------------------------------------------- figures
def _save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    return FIGURES / f"{stem}.png"


def figure_A_trajectories(curves, D, interp="base-ON (N=4)", budget=10000, stem=None):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (T0, phi, role) in zip(axs, CONDS):
        ax.plot(D.t_dense * 1e3, D.reference_T_dense(T0, phi), "k:", lw=3, label="ground truth (Cantera, 20k pts)", zorder=10)
        for key, (label, *_ ) in ARMS.items():
            c = curves.get((interp, key, budget))
            if c is not None:
                ax.plot(D.t_dense * 1e3, c[role], color=COLORS[key], lw=1.7, label=f"{label}")
        ax.axhline(T0, color="#999", lw=.8)
        ax.set_title(f"$T_0$ = {T0:.0f} K, φ = {phi:g}  ({role})", fontsize=11)
        ax.set_xlabel("time [ms]"); ax.set_ylabel("temperature [K]"); ax.set_xlim(0, D.t_dense[-1] * 1e3)
    h = axs[0].get_legend_handles_labels()
    fig.suptitle(f"DIAGNOSTIC — thermo initialization comparison, {interp}, {budget} Stage-2 epochs",
                 fontsize=12, y=1.0)
    fig.legend(*h, fontsize=8.5, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.text(.995, -.06, "NOT a paper Fig. 7 reproduction", ha="right", fontsize=8.5, color="#b00000", style="italic")
    fig.tight_layout(rect=(0, .09, 1, .97))
    return _save(fig, stem or f"hydrogen_DIAGNOSTIC_base_on_n4_temperature_comparison_{budget}"), fig


def figure_A2_off_vs_on(curves, D, budget=10000, stem=None):
    """Fig.-7-style T(t): both interpretations, both representative conditions, one figure.

    Rows = interpretation (N=5/base-OFF, N=4/base-ON); columns = condition. Same six arms
    and the same colours in every panel, so a row-to-row read is a like-for-like
    architecture comparison. DIAGNOSTIC, not a reproduction of paper Fig. 7.
    """
    import matplotlib.pyplot as plt
    interps = ["base-OFF (N=5)", "base-ON (N=4)"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True)
    for r, interp in enumerate(interps):
        for c, (T0, phi, role) in enumerate(CONDS):
            ax = axs[r, c]
            ax.plot(D.t_dense * 1e3, D.reference_T_dense(T0, phi), "k:", lw=3,
                    label="ground truth (Cantera)", zorder=10)
            for key, (label, *_ ) in ARMS.items():
                cv = curves.get((interp, key, budget))
                if cv is not None:
                    ax.plot(D.t_dense * 1e3, cv[role], color=COLORS[key], lw=1.7, label=label)
            ax.axhline(T0, color="#999", lw=.8)
            ax.set_title(f"{interp}   —   $T_0$={T0:.0f} K, φ={phi:g} ({role})", fontsize=10.5)
            ax.set_ylabel("temperature [K]")
            if r == 1:
                ax.set_xlabel("time [ms]")
            ax.set_xlim(0, D.t_dense[-1] * 1e3)
    h = axs[0, 0].get_legend_handles_labels()
    fig.suptitle(f"DIAGNOSTIC — temperature at {budget} Stage-2 epochs: "
                 f"N=5/base-OFF (top) vs N=4/base-ON (bottom)", fontsize=13, y=1.0)
    fig.legend(*h, fontsize=9, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.015))
    fig.text(.995, -.05, "NOT a paper Fig. 7 reproduction", ha="right", fontsize=8.5,
             color="#b00000", style="italic")
    fig.tight_layout(rect=(0, .07, 1, .975))
    return _save(fig, stem or f"hydrogen_DIAGNOSTIC_base_off_vs_on_temperature_{budget}"), fig


def figure_B_persistence(curves, D, role="training", stem=None):
    import matplotlib.pyplot as plt
    T0, phi = next((c[0], c[1]) for c in CONDS if c[2] == role)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.2))
    for ax, (key, (label, *_ )) in zip(axs.ravel(), ARMS.items()):
        ax.plot(D.t_dense * 1e3, D.reference_T_dense(T0, phi), "k:", lw=2.4, label="ground truth")
        for budget, col, lw in ((500, "#c0a0e0", 1.6), (10000, "#5b2d8e", 1.9)):
            c = curves.get(("base-ON (N=4)", key, budget))
            if c is not None:
                ax.plot(D.t_dense * 1e3, c[role], color=col, lw=lw, label=f"epoch {budget}")
        ax.axhline(T0, color="#999", lw=.8); ax.set_title(label, fontsize=10)
        ax.set_xlabel("time [ms]"); ax.set_ylabel("T [K]"); ax.legend(fontsize=8)
    fig.suptitle(f"base-ON (N=4): 500 vs 10 000 epochs — $T_0$={T0:.0f} K, φ={phi:g} ({role})  [DIAGNOSTIC]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return _save(fig, stem or f"hydrogen_DIAGNOSTIC_base_on_n4_500_vs_10k_{role.replace('-', '')}"), fig


def figure_C_weights(probes, D, stem=None):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    cmap = plt.get_cmap("tab10")
    cols = [f"coeff_{s}" for s in D.species]
    for ax, (key, (label, *_ )) in zip(axs.ravel(), ARMS.items()):
        pr = probes.get(("base-ON (N=4)", key))
        if pr is None: continue
        th = pr[cols].to_numpy(float)
        for k, s in enumerate(D.species):
            ax.plot(pr.epoch, th[:, k], "-o", ms=2.5, lw=1.3, color=cmap(k % 10), label=s)
        ax.set_xscale("symlog", linthresh=1)
        if np.abs(th).max() > 1e3: ax.set_yscale("symlog", linthresh=max(1.0, np.abs(th).max() * 1e-4))
        ax.set_title(f"{label}   (init ‖θ‖={np.linalg.norm(th[0]):.3g})", fontsize=9.5)
        ax.set_xlabel("epoch"); ax.set_ylabel(r"$\theta_{thermo,k}$ [K]")
    axs.ravel()[0].legend(fontsize=7, ncol=3)
    fig.suptitle("base-ON (N=4): the nine Eq. 14 coefficients over Stage-2 training [DIAGNOSTIC]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return _save(fig, stem or "hydrogen_DIAGNOSTIC_base_on_n4_thermo_weights"), fig


def figure_DE_norm_loss(probes, hists, stem=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for key, (label, *_ ) in ARMS.items():
        for interp, ls in (("base-ON (N=4)", "-"), ("base-OFF (N=5)", "--")):
            pr = probes.get((interp, key))
            if pr is None: continue
            ax[0].plot(pr.epoch, pr.thermo_linear_norm, ls, marker="o", ms=2.5, color=COLORS[key],
                       label=f"{label} [{interp[:8]}]")
            ax[1].plot(pr.epoch, pr.stage2_loss, ls, marker="o", ms=2.5, color=COLORS[key],
                       label=f"{label} [{interp[:8]}]")
    for a in ax: a.set_xscale("symlog", linthresh=1); a.set_yscale("log"); a.set_xlabel("Stage-2 epoch")
    ax[0].set_ylabel(r"$\|\theta_{thermo}\|_2$"); ax[0].set_title("D. thermo norm (solid = base-ON, dashed = base-OFF)")
    ax[1].set_ylabel("Stage-2 loss"); ax[1].set_title("E. Stage-2 loss at probe epochs")
    ax[0].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    return _save(fig, stem or "hydrogen_DIAGNOSTIC_base_on_n4_norm_and_loss"), fig


def figure_FG_runtime_nfe(hists, stem=None):
    import matplotlib.pyplot as plt
    edges = [0, 100, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for key, (label, *_ ) in ARMS.items():
        for interp, ls in (("base-ON (N=4)", "-"), ("base-OFF (N=5)", "--")):
            h = hists.get((interp, key))
            if h is None: continue
            spe = sec_per_epoch(h); ep = h.epoch.to_numpy()
            mid = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
            med = [np.median(spe[(ep >= a) & (ep < b)]) for a, b in zip(edges[:-1], edges[1:])]
            ax[0].plot(mid, med, ls, marker="o", ms=3, color=COLORS[key], label=f"{label} [{interp[:8]}]")
            if "nfe" in h:
                nf = h.nfe.to_numpy(float)
                mn = [np.median(nf[(ep >= a) & (ep < b)]) for a, b in zip(edges[:-1], edges[1:])]
                ax[1].plot(mid, mn, "-o", ms=3, color=COLORS[key], label=label)
                ax[2].scatter(nf[1:], spe[1:], s=3, alpha=.25, color=COLORS[key], label=label)
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch block"); ax[0].set_ylabel("median s / epoch")
    ax[0].set_title("F. runtime (solid = base-ON, dashed = base-OFF)"); ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_xlabel("epoch block"); ax[1].set_ylabel("median NFE / epoch"); ax[1].set_title("G. solver work (base-ON only; base-OFF predates NFE logging)")
    ax[1].legend(fontsize=7)
    ax[2].set_xlabel("NFE per epoch"); ax[2].set_ylabel("s / epoch"); ax[2].set_title("G'. wall time vs NFE, per epoch")
    fig.tight_layout()
    return _save(fig, stem or "hydrogen_DIAGNOSTIC_base_on_n4_runtime_nfe"), fig


# ---------------------------------------------------------------------------- CLI
def main():
    import matplotlib; matplotlib.use("Agg")
    print("=== base-ON (N=4) vs base-OFF (N=5): recomputing every cell ===")
    df, curves, probes, hists, D = compute_matrix()
    TABLES.mkdir(parents=True, exist_ok=True)
    keep = [c for c in df.columns if not c.startswith("eq18_")]
    df[keep].to_csv(TABLES / "hydrogen_base_on_n4_matrix.csv", index=False)
    df.to_csv(TABLES / "hydrogen_base_on_n4_matrix_full.csv", index=False)
    s1 = stage1_comparison(D); s1.to_csv(TABLES / "hydrogen_base_on_n4_stage1_comparison.csv")
    for f in (figure_A_trajectories(curves, D), figure_A2_off_vs_on(curves, D, 10000),
              figure_A2_off_vs_on(curves, D, 500), figure_B_persistence(curves, D, "training"),
              figure_B_persistence(curves, D, "held-out"), figure_C_weights(probes, D),
              figure_DE_norm_loss(probes, hists), figure_FG_runtime_nfe(hists)):
        print("saved", f[0].relative_to(_REPO))
    print("saved", (TABLES / "hydrogen_base_on_n4_matrix.csv").relative_to(_REPO))
    print("saved", (TABLES / "hydrogen_base_on_n4_stage1_comparison.csv").relative_to(_REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""Interpretability analysis of the final biodiesel ChemKAN (7b noise-free run). Read-only.

Model (checkpoint config): z = [TG, ROH, DG, MG, GL, RCO2R, T] min-max normalized with the
training statistics; hidden h = AddKAN(z) (4 nodes, sum over the 7 inputs, no input tanh);
output_i = phi_i1(h1) * phi_i2(h2) + phi_i3(h3) + phi_i4(h4) (LeanKAN, n_mu = 2; each
phi applies tanh to its input first); every phi is a sum of three Gaussians
exp(-(x - c)^2) at c = -1, 0, 1. Species rates in physical units:
dy_i/dt = (y_max_i - y_min_i) * output_i / 50   [mol L^-1 s^-1].

Reference: the known kinetics of the data generator (released-code orders),
r1 = k1 TG ROH, r2 = k2 DG ROH, r3 = k3 MG ROH, k = exp(lnA - Ea / (R T)).

Outputs (this folder): fig1_edges_layer1, fig2_edges_layer2, fig3_rate_parity,
fig4_perturbations, fig5_arrhenius (.png and .pdf), summary.json. Nothing is trained and
no parameter is changed (parameter sha256 checked before and after).

    ~/uni_projects/chemkan-venv/bin/python results/experiments/biodiesel/interpretability/interpretability.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for sub in ("chemkan/scripts", "chemkan/scripts/author_repo_match", "chemkan/scripts/figures"):
    sys.path.insert(0, str(ROOT / sub))
import author_runs  # noqa: E402
from _author_match import build_core, rhs_divisor_from_config  # noqa: E402

RUN = ROOT / "results/experiments/biodiesel/author_repo_match/figures/fig05/chemkan_noise00_seed0"
DATA = ROOT / "chemkan/data/generated/biodiesel_v2.npz"
SPECIES = ["TG", "ROH", "DG", "MG", "GL", "RCO2R"]
INPUTS = SPECIES + ["T"]
LABEL = {"RCO2R": "R′CO$_2$R"}
ORDER = "released_code"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
RC = {"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
      "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
      "axes.titlesize": 8.5, "legend.frameon": False}
# Representative states: (split, trajectory index within split, time index on the 30-point grid)
STATES = [("train", None, 1), ("train", None, 5), ("train", None, 20), ("test", 3, 8)]
PERTURB_FACTOR = (0.8, 1.2)         # TG and ROH multiplied over this range, clipped to the data
N_PERT = 41
NEAR = 3.0                          # nearest-training-state distance (median spacings) drawn bold


def sha(t: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(t):
        h.update(k.encode())
        h.update(t[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load():
    ckpt = torch.load(RUN / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    core = build_core(cfg["architecture"]["hidden"]).double()
    core.load_state_dict(ckpt["model_state"])
    core.eval()
    src = cfg["data_source"]
    u_min = np.array(src["species_min"] + [src["T_min"]])
    u_max = np.array(src["species_max"] + [src["T_max"]])
    return ckpt, cfg, core, u_min, u_max, rhs_divisor_from_config(cfg)


class Model:
    """ChemKAN pieces evaluated with the checkpoint's own modules (no re-implementation)."""

    def __init__(self, core, u_min, u_max, divisor):
        self.core, self.u_min, self.span, self.div = core, u_min, u_max - u_min, divisor

    def z(self, u):                                     # physical (N, 7) -> normalized
        return (np.asarray(u, float) - self.u_min) / self.span

    @torch.no_grad()
    def parts(self, u):
        z = torch.as_tensor(self.z(u))
        e1 = self.core.add.edges(z)                     # (N, 4, 7) layer-1 edge values
        h = e1.sum(-1)                                  # (N, 4) hidden nodes
        e2 = self.core.lean.edges(h)                    # (N, 6, 4) layer-2 edge values
        out = self.core(z)                              # (N, 6)
        return z.numpy(), e1.numpy(), h.numpy(), e2.numpy(), out.numpy()

    def rates(self, u):                                 # physical mol/L/s, (N, 6)
        return self.parts(u)[-1] * self.span[:6] / self.div


def true_rates(u):
    gb = author_runs._generator()
    u = np.atleast_2d(u)
    return np.array([gb.rhs(0.0, row[:6], gb.rate_constants(row[6], ORDER)) for row in u])


def data_states():
    d = np.load(DATA)
    out = {}
    for split in ("train", "test"):
        y, T = d[f"{split}_states"], d[f"{split}_T"].astype(float)       # (n, 30, 6), (n,)
        u = np.concatenate([y, np.broadcast_to(T[:, None, None], y.shape[:2] + (1,))], -1)
        out[split] = u                                                   # (n, 30, 7)
    out["t"] = d["t"]
    return out


def contributions(e1, e2):
    """Spread (std over the data) of every edge output and its share within the node.

    Layer 1, node o: std(phi_oj(z_j)) / sum_j std(phi_oj). Layer 2, output i: the
    multiplicative term phi_i1*phi_i2 and the additive edges phi_i3, phi_i4 compared the
    same way (std of each term / sum of the three stds).
    """
    s1 = e1.std(0)                                                       # (4, 7)
    prod = e2[:, :, 0] * e2[:, :, 1]
    terms = np.stack([prod, e2[:, :, 2], e2[:, :, 3]], -1)               # (N, 6, 3)
    s2 = terms.std(0)
    return {"layer1_edge_std": s1, "layer1_share": s1 / s1.sum(1, keepdims=True),
            "layer2_term_std": s2, "layer2_share": s2 / s2.sum(1, keepdims=True),
            "layer2_edge_std": e2.std(0)}


def fig_edges(model, train_u, test_u, contrib):
    z_tr, e1_tr, h_tr, _, _ = model.parts(train_u)
    z_te, _, h_te, _, _ = model.parts(test_u)
    W1 = model.core.add.edges
    W2 = model.core.lean.edges
    # layer 1
    with plt.rc_context(RC):
        fig, axes = plt.subplots(4, 7, figsize=(13, 6.6), sharex="col")
        x = np.linspace(-0.1, 1.1, 241)
        for j in range(7):
            grid = torch.zeros(len(x), 7, dtype=torch.float64)
            grid[:, j] = torch.as_tensor(x)
            with torch.no_grad():
                vals = W1(grid).numpy()[:, :, j]                          # (N, 4)
            lo, hi = z_tr[:, j].min(), z_tr[:, j].max()
            inside = (x >= lo) & (x <= hi)
            for o in range(4):
                ax = axes[o, j]
                ax.plot(x, vals[:, o], color=MUTED, lw=0.8, ls=":")
                ax.plot(x[inside], vals[inside, o], color=COLORS[0], lw=1.6)
                ax.plot(z_tr[:, j], np.full(len(z_tr), 0.03), "|", color=INK, ms=4, alpha=0.15,
                        transform=ax.get_xaxis_transform())
                ax.set_title(f"{LABEL.get(INPUTS[j], INPUTS[j])} -> h{o + 1}  ({100 * contrib['layer1_share'][o, j]:.0f} %)",
                             loc="left")
                ax.grid(True, color=GRID, lw=0.5)
                if o == 3:
                    ax.set_xlabel(f"normalized {LABEL.get(INPUTS[j], INPUTS[j])}")
                if j == 0:
                    ax.set_ylabel(f"edge value\n(into hidden node h{o + 1})")
        fig.suptitle("Layer 1 (additive KAN, 7 -> 4): edge functions phi(z_j). Solid: range seen in the "
                     "20 training trajectories (ticks = training states); dotted: outside it.\n"
                     "(%) = edge's share of the node's spread over the training states.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, "fig1_edges_layer1")
    # layer 2
    with plt.rc_context(RC):
        fig, axes = plt.subplots(6, 4, figsize=(10.5, 10), sharex="col")
        for j in range(4):
            lo, hi = h_tr[:, j].min(), h_tr[:, j].max()
            pad = 0.15 * (hi - lo)
            x = np.linspace(lo - pad, hi + pad, 241)
            grid = torch.zeros(len(x), 4, dtype=torch.float64)
            grid[:, j] = torch.as_tensor(x)
            with torch.no_grad():
                vals = W2(grid).numpy()[:, :, j]                          # (N, 6)
            inside = (x >= lo) & (x <= hi)
            for i in range(6):
                ax = axes[i, j]
                ax.plot(x, vals[:, i], color=MUTED, lw=0.8, ls=":")
                ax.plot(x[inside], vals[inside, i], color=COLORS[1] if j < 2 else COLORS[0], lw=1.6)
                ax.plot(h_tr[:, j], np.full(len(h_tr), 0.03), "|", color=INK, ms=4, alpha=0.15,
                        transform=ax.get_xaxis_transform())
                if j < 2:
                    share = contrib["layer2_share"][i, 0]
                    kind = f"x ({100 * share:.0f} %)"
                else:
                    share = contrib["layer2_share"][i, j - 1]
                    kind = f"+ ({100 * share:.0f} %)"
                ax.set_title(f"h{j + 1} -> d{LABEL.get(SPECIES[i], SPECIES[i])}/dt  {kind}", loc="left")
                ax.grid(True, color=GRID, lw=0.5)
                if i == 5:
                    ax.set_xlabel(f"h{j + 1}")
                if j == 0:
                    ax.set_ylabel("edge value")
        fig.suptitle("Layer 2 (LeanKAN, 4 -> 6): out_i = phi_i1(h1) * phi_i2(h2) + phi_i3(h3) + phi_i4(h4). "
                     "Orange: multiplied edges; blue: added edges.\nSolid: range of h over the training "
                     "states (ticks at the bottom = training states); each edge applies tanh to h first. (%) = term's share of the "
                     "output's spread; x: the h1 and h2 edges form one product term.",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, "fig2_edges_layer2")


def parity(model, states, split_names):
    out = {}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
        for i, ax in enumerate(axes.flat):
            for k, split in enumerate(split_names):
                u = states[split].reshape(-1, 7)
                r_true, r_kan = true_rates(u)[:, i], model.rates(u)[:, i]
                ax.plot(r_true, r_kan, "o", ms=2.5, alpha=0.6, color=COLORS[k], mec="none",
                        label=f"{split} states" if i == 0 else None)
                rms = np.sqrt(np.mean(r_true ** 2))
                out.setdefault(SPECIES[i], {})[split] = {
                    "rmse_mol_per_L_s": float(np.sqrt(np.mean((r_kan - r_true) ** 2))),
                    "relative_rmse": float(np.sqrt(np.mean((r_kan - r_true) ** 2)) / rms),
                    "r2": float(1 - np.sum((r_kan - r_true) ** 2) / np.sum((r_true - r_true.mean()) ** 2)),
                    "sign_agreement": float(np.mean(np.sign(r_kan) == np.sign(r_true))),
                    "n_states": int(len(u))}
            lim = np.array(ax.get_xlim() + ax.get_ylim())
            lo, hi = lim.min(), lim.max()
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
            ax.set_title(f"d[{LABEL.get(SPECIES[i], SPECIES[i])}]/dt  rel. RMSE "
                         f"{out[SPECIES[i]]['train']['relative_rmse']:.2f} train, "
                         f"{out[SPECIES[i]]['test']['relative_rmse']:.2f} test", loc="left")
            ax.set_xlabel("known kinetics [mol L$^{-1}$ s$^{-1}$]")
            ax.set_ylabel("ChemKAN [mol L$^{-1}$ s$^{-1}$]")
            ax.grid(True, color=GRID, lw=0.5)
        axes.flat[0].legend(loc="upper left")
        fig.suptitle("Species production rates at every state of the clean reference trajectories "
                     "(30 points each): full ChemKAN vs known Arrhenius kinetics", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, "fig3_rate_parity")
    return out


def pick_states(states):
    T_train = states["train"][:, 0, 6]
    mid = int(np.argsort(T_train)[len(T_train) // 2])                    # median-T trajectory
    chosen = []
    for split, idx, ti in STATES:
        traj = mid if idx is None else idx
        u = states[split][traj, ti].copy()
        chosen.append({"split": split, "trajectory": int(traj), "time_index": int(ti),
                       "t_s": float(states["t"][ti]), "u": u})
    return chosen


def nn_distance(model, pts, train_u):
    """Distance (normalized 7-D) to the nearest training state, in units of the median
    nearest-neighbour spacing among training states."""
    zt = model.z(train_u)
    d_tt = np.sqrt(((zt[:, None] - zt[None]) ** 2).sum(-1))
    np.fill_diagonal(d_tt, np.inf)
    scale = np.median(d_tt.min(1))
    zp = model.z(pts)
    d = np.sqrt(((zp[:, None] - zt[None]) ** 2).sum(-1)).min(1)
    return d / scale, float(scale)


def perturbations(model, chosen, train_u):
    T_lo, T_hi = train_u[:, 6].min(), train_u[:, 6].max()
    res, curves = [], {}
    for s_i, s in enumerate(chosen):
        u0 = s["u"]
        sweeps = {"T": np.linspace(T_lo, T_hi, N_PERT)}
        for name, col in (("TG", 0), ("ROH", 1)):
            lo = max(u0[col] * PERTURB_FACTOR[0], train_u[:, col].min())
            hi = min(u0[col] * PERTURB_FACTOR[1], train_u[:, col].max())
            sweeps[name] = np.linspace(lo, hi, N_PERT)
        entry = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        entry["sweeps"] = {}
        for var, vals in sweeps.items():
            col = INPUTS.index(var)
            U = np.repeat(u0[None], N_PERT, 0)
            U[:, col] = vals
            rk, rt = model.rates(U), true_rates(U)
            dist, scale = nn_distance(model, U, train_u)
            curves[(s_i, var)] = (vals, rk, rt, dist)
            dev = {}
            for i, sp in enumerate(SPECIES):
                denom = np.sqrt(np.mean(rt[:, i] ** 2))
                dev[sp] = {"relative_l2_deviation": float(np.sqrt(np.mean((rk[:, i] - rt[:, i]) ** 2)) / denom)
                           if denom > 0 else None,
                           "true_change_over_sweep": float(rt[-1, i] - rt[0, i]),
                           "chemkan_change_over_sweep": float(rk[-1, i] - rk[0, i]),
                           "same_direction_of_change": (None if abs(rt[-1, i] - rt[0, i]) < 1e-12 else
                                                        bool(np.sign(rt[-1, i] - rt[0, i])
                                                             == np.sign(rk[-1, i] - rk[0, i])))}
            entry["sweeps"][var] = {"range": [float(vals[0]), float(vals[-1])], "points": N_PERT,
                                    "nearest_training_state_distance_max": float(dist.max()),
                                    "nearest_training_state_distance_median": float(np.median(dist)),
                                    "distance_unit": f"median training nearest-neighbour spacing ({scale:.3g}, normalized 7-D)",
                                    "deviation": dev}
            # local sensitivities of the TG consumption rate: apparent Ea and reaction orders
            # local sensitivities of the TG consumption rate: apparent Ea and reaction orders,
            # fitted over the whole sweep and over its part near the training states only
            rate_k, rate_t = -rk[:, 0], -rt[:, 0]
            near = dist <= NEAR
            x = 1.0 / vals if var == "T" else np.log(vals)
            scale = -author_runs._generator().R_KCAL if var == "T" else 1.0

            def fit(r, m):
                if m.sum() < 5 or np.any(r[m] <= 0):
                    return None
                return float(np.polyfit(x[m], np.log(r[m]), 1)[0] * scale)
            name = ("apparent_Ea_TG_consumption_kcal_per_mol" if var == "T"
                    else "apparent_order_TG_consumption")
            entry["sweeps"][var][name] = {
                "whole_sweep": {"chemkan": fit(rate_k, near | ~near), "known": fit(rate_t, near | ~near)},
                "near_training_states": {"chemkan": fit(rate_k, near), "known": fit(rate_t, near),
                                         "points": int(near.sum()),
                                         "range": [float(vals[near].min()), float(vals[near].max())]
                                         if near.any() else None}}
        res.append(entry)
    return res, curves


def fig_perturbations(chosen, curves):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 6, figsize=(14, 7.2))
        for r, var in enumerate(("T", "TG", "ROH")):
            for i, sp in enumerate(SPECIES):
                ax = axes[r, i]
                for s_i, s in enumerate(chosen):
                    vals, rk, rt, dist = curves[(s_i, var)]
                    near = dist <= NEAR
                    ax.plot(vals, rt[:, i], color=COLORS[s_i], lw=1.0, ls="--")
                    ax.plot(vals, np.where(near, rk[:, i], np.nan), color=COLORS[s_i], lw=1.8)
                    ax.plot(vals, rk[:, i], color=COLORS[s_i], lw=0.9, alpha=0.5)
                    ax.plot(s["u"][INPUTS.index(var)], np.interp(s["u"][INPUTS.index(var)], vals, rk[:, i]),
                            "o", ms=3.5, color=COLORS[s_i])
                ax.grid(True, color=GRID, lw=0.5)
                if r == 0:
                    ax.set_title(f"d[{LABEL.get(sp, sp)}]/dt", loc="left")
                if i == 0:
                    ax.set_ylabel(f"vary {var}\nrate [mol L$^{{-1}}$ s$^{{-1}}$]")
                ax.set_xlabel("T [K]" if var == "T" else f"[{var}] [mol/L]")
        handles = [plt.Line2D([], [], color=COLORS[k], lw=1.6,
                              label=f"{s['split']} traj. {s['trajectory']}, t = {s['t_s']:.1f} s, "
                                    f"T = {s['u'][6]:.1f} K") for k, s in enumerate(chosen)]
        handles += [plt.Line2D([], [], color=MUTED, lw=1.8, label=f"ChemKAN, within {NEAR:g} spacings of a training state"),
                    plt.Line2D([], [], color=MUTED, lw=0.9, alpha=0.5, label="ChemKAN, farther from the training states"),
                    plt.Line2D([], [], color=MUTED, lw=1.0, ls="--", label="known Arrhenius kinetics")]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle("Local perturbations around reference states (dots): one input varied, all others fixed. "
                     "T over the training range; [TG], [ROH] x0.8-1.2, clipped to the training range.",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        save(fig, "fig4_perturbations")


def fig_arrhenius(chosen, curves):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for s_i, s in enumerate(chosen):
            vals, rk, rt, dist = curves[(s_i, "T")]
            near = dist <= NEAR
            ax.plot(1000 / vals, -rt[:, 0], color=COLORS[s_i], lw=1.0, ls="--")
            ax.plot(1000 / vals, -rk[:, 0], color=COLORS[s_i], lw=0.9, alpha=0.5)
            ax.plot(1000 / vals, np.where(near, -rk[:, 0], np.nan), color=COLORS[s_i], lw=2.0,
                    label=f"{s['split']} t = {s['t_s']:.1f} s")
        ax.set_yscale("log")
        ax.set_xlabel("1000 / T [1/K]")
        ax.set_ylabel("-d[TG]/dt [mol L$^{-1}$ s$^{-1}$]")
        ax.grid(True, color=GRID, lw=0.5)
        ax.legend(fontsize=7, title=f"ChemKAN (bold: within {NEAR:g} spacings of\na training state), dashed: known",
                  title_fontsize=6.5, loc="lower left")
        ax.set_title("TG consumption vs temperature (Arrhenius plot)", loc="left")
        fig.tight_layout()
        save(fig, "fig5_arrhenius")


def hidden_correlations(model, train_u):
    """Descriptive only: |Pearson r| between hidden-node values and the three true reaction
    rates over the training states. Not evidence that a node 'is' a reaction."""
    _, _, h, _, _ = model.parts(train_u)
    gb = author_runs._generator()
    r = np.array([[gb.rate_constants(u[6], ORDER)[0] * u[0] * u[1],
                   gb.rate_constants(u[6], ORDER)[1] * u[2] * u[1],
                   gb.rate_constants(u[6], ORDER)[2] * u[3] * u[1]] for u in train_u])
    c = np.corrcoef(np.concatenate([h, r], 1).T)[:4, 4:]
    return {f"h{j + 1}": {f"r{k + 1}": float(c[j, k]) for k in range(3)} for j in range(4)}


def initial_state_checks(model, states):
    """Structure the known kinetics imposes exactly: at t = 0 (DG = MG = GL = RCO2R = 0) MG and
    GL are not produced; a species at zero concentration is never consumed."""
    out = {}
    for split in ("train", "test"):
        u0 = states[split][:, 0]
        rk = model.rates(u0)
        ug = states[split].reshape(-1, 7)
        rg = model.rates(ug)
        zero = ug[:, :6] <= 1e-12
        out[split] = {
            "t0_MG_rate_chemkan_median_max": [float(np.median(rk[:, 3])), float(rk[:, 3].max())],
            "t0_GL_rate_chemkan_median_max": [float(np.median(rk[:, 4])), float(rk[:, 4].max())],
            "t0_MG_GL_rate_known": 0.0,
            "zero_concentration_species_with_negative_chemkan_rate": int((zero & (rg < 0)).sum()),
            "zero_concentration_species_total": int(zero.sum())}
    return out


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def main() -> int:
    ckpt, cfg, core, u_min, u_max, div = load()
    before = sha(ckpt["model_state"])
    model = Model(core, u_min, u_max, div)
    states = data_states()
    train_u, test_u = states["train"].reshape(-1, 7), states["test"].reshape(-1, 7)
    # the analysis uses the model exactly as integrated in the 7b runs
    z, e1, h, e2, out = model.parts(train_u)
    contrib = contributions(e1, e2)
    fig_edges(model, train_u, test_u, contrib)
    par = parity(model, states, ["train", "test"])
    chosen = pick_states(states)
    pert, curves = perturbations(model, chosen, train_u)
    fig_perturbations(chosen, curves)
    fig_arrhenius(chosen, curves)
    after = sha({k: v for k, v in core.state_dict().items()})
    summary = {
        "checkpoint": str((RUN / "checkpoint_final.pt").relative_to(ROOT)),
        "checkpoint_parameters_sha256": before,
        "parameters_unchanged": before == after,
        "model": {"layers": cfg["architecture"]["layers"], "n_mu": cfg["architecture"]["n_mu"],
                  "rbf": cfg["architecture"]["rbf"], "dynamics": cfg["dynamics"], "rhs_divisor": div,
                  "parameters": cfg["architecture"]["parameter_count"]},
        "normalization": {"inputs": INPUTS, "u_min": u_min.tolist(), "u_max": u_max.tolist(),
                          "source": cfg["data_source"]["normalization"]},
        "data": {"file": str(DATA.relative_to(ROOT)), "sha256": author_runs.DATA_SHA256,
                 "clean_train_states": int(len(train_u)), "clean_test_states": int(len(test_u))},
        "reference_kinetics": {"orders": ORDER, **{k: v.tolist() for k, v in
                                                   author_runs._generator().KINETICS[ORDER].items()}},
        "input_ranges_seen_in_training": {
            "normalized_z": {n: [float(z[:, j].min()), float(z[:, j].max())] for j, n in enumerate(INPUTS)},
            "hidden_h": {f"h{j + 1}": [float(h[:, j].min()), float(h[:, j].max())] for j in range(4)}},
        "edge_contributions": {
            "definition": "std over the 600 training states of each edge output (layer 1) or of each "
                          "layer-2 term (product phi_i1*phi_i2, phi_i3, phi_i4), and its share of the sum "
                          "within the node",
            "layer1_share": {f"h{o + 1}": {INPUTS[j]: float(contrib["layer1_share"][o, j]) for j in range(7)}
                             for o in range(4)},
            "layer2_share": {SPECIES[i]: {"h1*h2": float(contrib["layer2_share"][i, 0]),
                                          "h3": float(contrib["layer2_share"][i, 1]),
                                          "h4": float(contrib["layer2_share"][i, 2])} for i in range(6)}},
        "rate_parity_all_reference_states": par,
        "initial_state_structure_checks": initial_state_checks(model, states),
        "perturbations": pert,
        "hidden_node_vs_true_reaction_rate_correlation_descriptive": hidden_correlations(model, train_u),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"parameters_unchanged": before == after,
                      "parity_relative_rmse": {k: {s: round(v[s]["relative_rmse"], 3) for s in v} for k, v in par.items()},
                      "apparent_Ea": [p["sweeps"]["T"]["apparent_Ea_TG_consumption_kcal_per_mol"] for p in pert],
                      "orders_TG": [p["sweeps"]["TG"]["apparent_order_TG_consumption"] for p in pert],
                      "orders_ROH": [p["sweeps"]["ROH"]["apparent_order_TG_consumption"] for p in pert]},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

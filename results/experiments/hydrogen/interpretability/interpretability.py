r"""Interpretability of the author-matched Stage-2 ChemKAN (hydrogen). Read-only.

The same script lives in ``results/experiments/{hydrogen,methane}/interpretability/``; the
system is taken from the folder it is in.

Model (each checkpoint's config): z = [Y_1..Y_m, T] min-max normalized with the training
statistics;
    hidden   h_o = sum_j phi_oj(z_j)                       (additive KAN, m+1 -> 3, no tanh)
    species  r_i = phi_i1(h1) * phi_i2(h2) * phi_i3(h3)    (LeanKAN with n_mu = 3: a pure
                                                            product, tanh inside each edge)
    temp.    dT_hat = sum_i w_i r_i + KAN_cor(z)           (linear + additive KAN, m+1 -> 1)
    physical du_k/dt = (u_max - u_min)_k * output_k / t_ref.
Every phi is a sum of five Gaussians on the grid -1..1.

Reference: Cantera's instantaneous rates of the data generator's reactor (adiabatic,
constant pressure) at the same state: dY_k/dt = wdot_k W_k / rho,
dT/dt = -sum_k h_k wdot_k / (rho c_p) (h_k partial molar enthalpy).

Outputs (this folder): fig1 layer-1 edges, fig2 layer-2 edges, fig3
temperature structure (thermo coefficients and KAN_cor edges), fig4 rate parity, fig5
perturbations, fig6 Arrhenius plot at an induction state; summary.json. Nothing is trained;
parameter sha256 checked before and after.

    ~/uni_projects/chemkan-venv/bin/python results/experiments/hydrogen/interpretability/interpretability.py
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
sys.path.insert(0, str(ROOT / "chemkan/scripts/author_repo_match"))
sys.path.insert(0, str(ROOT / "chemkan/scripts"))
import plot_trajectory as pt  # noqa: E402
import train_author_matched_hydrogen as amh  # noqa: E402

SYSTEMS = {
    "hydrogen": {"checkpoint": "results/experiments/hydrogen/author_matched/main/S2_seed0_200k/stage2_final.pt",
                 "fuel": "H2", "panels": ["H2", "O2", "H2O", "OH"], "state_condition": (1100.0, 1.1),
                 "species_shown": None, "top_inputs": None},
    "methane": {"checkpoint": "results/experiments/methane/author_matched/main/S2_seed0/archive/"
                              "stage2_S2_seed0_epoch180000.pt",
                "fuel": "CH4", "panels": ["CH4", "O2", "CO", "H2O", "OH"], "state_condition": (1550.0, 1.1),
                "species_shown": ["CH4", "O2", "H2O", "CO2", "CO", "H2", "OH", "CH2O", "HO2"],
                "top_inputs": 8},
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
RC = {"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
      "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
      "axes.titlesize": 8.5, "legend.frameon": False}
N_PERT, NEAR = 41, 3.0
T_PERT_K = 60.0               # temperature varied by +-60 K around the state
FRACTION_FACTOR = (0.8, 1.2)  # fuel / O2 mass fraction multiplied over this range (N2 absorbs the change)


def sha(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)


class Model:
    def __init__(self, ckpt):
        self.cfg = ckpt["config"]
        info = pt.load_model(ckpt)                      # configures the system, checks the data sha256
        self.data = info["data"]
        net = amh.build_model().double()
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        self.net = net
        self.u_min, self.u_max = info["u_min"], info["u_max"]
        self.span = self.u_max - self.u_min
        self.t_ref = self.cfg["t_ref"]["value_s"]

    def z(self, u):
        return (np.asarray(u, float) - self.u_min) / self.span

    @torch.no_grad()
    def parts(self, u):
        z = torch.as_tensor(self.z(u))
        k = self.net.kinetic
        e1 = k.add.edges(z)                             # (N, 3, m+1)
        h = e1.sum(-1)                                  # (N, 3)
        e2 = k.lean.edges(h)                            # (N, m, 3)
        r = k.lean(h)                                   # (N, m) = product over the 3 edges
        ec = self.net.thermo.correction.edges(z)        # (N, 1, m+1)
        lin = self.net.thermo.linear(r)                 # (N, 1)
        out = self.net(z)                               # (N, m+1)
        return dict(z=z.numpy(), e1=e1.numpy(), h=h.numpy(), e2=e2.numpy(), r=r.numpy(),
                    ec=ec.numpy()[:, 0], lin=lin.numpy()[:, 0], out=out.numpy())

    def rates(self, u):                                 # physical du/dt (N, m+1)
        return self.parts(u)["out"] * self.span / self.t_ref


class Reference:
    def __init__(self, data_path, species):
        import cantera as ct
        d = np.load(data_path, allow_pickle=True)
        self.gas = ct.Solution(str(d["mechanism"]))
        self.P = float(d["pressure"])
        self.keep = np.array([self.gas.species_index(s) for s in species])
        self.ct = ct

    def rates(self, u):
        g, out = self.gas, []
        for row in np.atleast_2d(u):
            Y = np.zeros(g.n_species)
            Y[self.keep] = np.clip(row[:-1], 0.0, None)
            g.TPY = row[-1], self.P, Y
            w = g.net_production_rates
            dY = w * g.molecular_weights / g.density
            dT = -np.dot(g.partial_molar_enthalpies, w) / (g.density * g.cp_mass)
            out.append(np.append(dY[self.keep], dT))
        return np.array(out)

    def finite_difference_check(self, u, dt=1e-9):
        """Rates from a short reactor step, to confirm the formula above (relative error)."""
        g, errs = self.gas, []
        for row in u:
            Y = np.zeros(g.n_species)
            Y[self.keep] = row[:-1]
            g.TPY = row[-1], self.P, Y
            r = self.ct.IdealGasConstPressureReactor(g)
            net = self.ct.ReactorNet([r])
            net.rtol, net.atol = 1e-12, 1e-20
            y0 = np.append(r.Y[self.keep], r.T)
            net.advance(dt)
            fd = (np.append(r.Y[self.keep], r.T) - y0) / dt
            an = self.rates(row[None])[0]
            errs.append(np.abs(fd - an).max() / np.abs(an).max())
        return float(np.max(errs))

    def thermo_coefficients(self, u):
        """-h_k / c_p [K per unit mass fraction] at each state (mass-specific enthalpy)."""
        g, out = self.gas, []
        for row in np.atleast_2d(u):
            Y = np.zeros(g.n_species)
            Y[self.keep] = np.clip(row[:-1], 0.0, None)
            g.TPY = row[-1], self.P, Y
            h_mass = g.partial_molar_enthalpies / g.molecular_weights
            out.append(-h_mass[self.keep] / g.cp_mass)
        return np.array(out)

    def element_rate_imbalance(self, rates_Y):
        """|sum_k a_ek dY_k/dt / W_k| / sum_k |...| per state and element: (N, E), element names."""
        g = self.gas
        W = g.molecular_weights[self.keep]
        elems = [e for e in g.element_names if any(g.n_atoms(g.species_names[k], e) for k in self.keep)]
        a = np.array([[g.n_atoms(g.species_names[k], e) for k in self.keep] for e in elems])
        terms = rates_Y[:, None, :] / W * a[None]                         # (N, E, m)
        num, den = np.abs(terms.sum(-1)), np.abs(terms).sum(-1)
        return np.where(den > 0, num / np.where(den > 0, den, 1), 0), elems


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(rb) > 0 else None


def within2(a, b):
    m = np.abs(b) > 0.01 * np.abs(b).max() if np.abs(b).max() > 0 else np.zeros_like(b, bool)
    if not m.any():
        return None
    ratio = a[m] / b[m]
    return float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))


def domain_distance(model, pts, train_u):
    zt = model.z(train_u)
    sub = zt[:: max(1, len(zt) // 1500)]
    d_tt = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
    np.fill_diagonal(d_tt, np.inf)
    scale = float(np.median(d_tt.min(1)))
    zp = model.z(pts)
    d = np.sqrt(((zp[:, None] - zt[None]) ** 2).sum(-1)).min(1)
    return d / scale, scale


def edge_figures(system, model, names, parts, out, top):
    m = len(names) - 1
    z, h = parts["z"], parts["h"]
    s1 = parts["e1"].std(0)
    share1 = s1 / s1.sum(1, keepdims=True)                              # (3, m+1)
    logf = np.log(np.abs(parts["e2"]) + 1e-300)
    s2 = logf.std(0)                                                    # (m, 3)
    share2 = s2 / s2.sum(1, keepdims=True)
    sign_change = [[bool(np.any(parts["e2"][:, i, j] > 0) and np.any(parts["e2"][:, i, j] < 0))
                    for j in range(3)] for i in range(m)]
    # layer 1
    cols = list(range(m + 1)) if top is None else None
    with plt.rc_context(RC):
        ncol = (m + 1) if top is None else top
        fig, axes = plt.subplots(3, ncol, figsize=(1.85 * ncol + 1, 6.4), squeeze=False)
        x = np.linspace(-0.1, 1.1, 241)
        for o in range(3):
            order = cols if cols is not None else list(np.argsort(share1[o])[::-1][:top])
            for c, j in enumerate(order):
                grid = torch.zeros(len(x), m + 1, dtype=torch.float64)
                grid[:, j] = torch.as_tensor(x)
                with torch.no_grad():
                    v = model.net.kinetic.add.edges(grid).numpy()[:, o, j]
                lo, hi = z[:, j].min(), z[:, j].max()
                inside = (x >= lo) & (x <= hi)
                ax = axes[o, c]
                ax.plot(x, v, color=MUTED, lw=0.8, ls=":")
                ax.plot(x[inside], v[inside], color=COLORS[0], lw=1.6)
                ax.plot(z[:, j], np.full(len(z), 0.03), "|", color=INK, ms=4, alpha=0.08,
                        transform=ax.get_xaxis_transform())
                ax.set_title(f"{names[j]} -> h{o + 1} ({100 * share1[o, j]:.0f} %)", loc="left")
                ax.set_xlabel(f"normalized {names[j]}", fontsize=7.5)
                ax.grid(True, color=GRID, lw=0.5)
            axes[o, 0].set_ylabel(f"edge value into h{o + 1}")
        what = "all inputs" if top is None else f"the {top} inputs with the largest share, per node"
        fig.suptitle(f"{system}, layer 1 (additive KAN, {m + 1} -> 3): edge functions, {what}. Solid: range "
                     "in the training states (ticks); dotted: outside.\n(%) = edge's share of the hidden "
                     "node's spread over the training states.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save(fig, out, "fig1_edges_layer1")
    # layer 2
    shown = SYSTEMS[system]["species_shown"] or names[:-1]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(shown), 3, figsize=(9.5, 1.75 * len(shown) + 1.2), squeeze=False)
        for j in range(3):
            lo, hi = h[:, j].min(), h[:, j].max()
            pad = 0.15 * (hi - lo)
            x = np.linspace(lo - pad, hi + pad, 241)
            grid = torch.zeros(len(x), 3, dtype=torch.float64)
            grid[:, j] = torch.as_tensor(x)
            with torch.no_grad():
                vals = model.net.kinetic.lean.edges(grid).numpy()[:, :, j]
            inside = (x >= lo) & (x <= hi)
            for r_, sp in enumerate(shown):
                i = names.index(sp)
                ax = axes[r_, j]
                ax.plot(x, vals[:, i], color=MUTED, lw=0.8, ls=":")
                ax.plot(x[inside], vals[inside, i], color=COLORS[1], lw=1.6)
                ax.plot(h[:, j], np.full(len(h), 0.03), "|", color=INK, ms=4, alpha=0.08,
                        transform=ax.get_xaxis_transform())
                ax.axhline(0, color=MUTED, lw=0.5)
                ax.set_title(f"h{j + 1} -> d{sp}/dt  x ({100 * share2[i, j]:.0f} %)", loc="left")
                ax.grid(True, color=GRID, lw=0.5)
                if r_ == len(shown) - 1:
                    ax.set_xlabel(f"h{j + 1}")
        fig.suptitle(f"{system}, layer 2 (LeanKAN with n_mu = 3): rate_i = phi_i1(h1) x phi_i2(h2) x phi_i3(h3). "
                     "(%) = factor's share of the spread of log|rate| over the training states.",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, out, "fig2_edges_layer2")
    return {"layer1_share": {f"h{o + 1}": {names[j]: float(share1[o, j]) for j in range(m + 1)} for o in range(3)},
            "layer2_log_share": {names[i]: {f"h{j + 1}": float(share2[i, j]) for j in range(3)} for i in range(m)},
            "layer2_factor_changes_sign_over_data": {names[i]: sign_change[i] for i in range(m)},
            "hidden_range": {f"h{j + 1}": [float(h[:, j].min()), float(h[:, j].max())] for j in range(3)}}


def thermo_figure(system, model, ref, names, parts, train_u, out, top):
    m = len(names) - 1
    w = model.net.thermo.linear.weight.detach().numpy()[0]
    implied = w * model.span[-1] / model.span[:m]                      # K per unit mass fraction
    true = ref.thermo_coefficients(train_u[:: max(1, len(train_u) // 400)])
    t_med, t_lo, t_hi = np.median(true, 0), true.min(0), true.max(0)
    # share of dT_hat/dt from the linear term and from KAN_cor over the training states
    lin, cor = parts["lin"], parts["ec"].sum(-1)
    s_lin, s_cor = lin.std(), cor.std()
    sc = parts["ec"].std(0)
    share_c = sc / sc.sum()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12, 4.4))
        ax = fig.add_subplot(1, 2, 1)
        order = np.argsort(t_med)
        idx = order if m <= 12 else order[np.r_[0:6, -6:0]]
        xs = np.arange(len(idx))
        ax.errorbar(xs, t_med[idx], yerr=[t_med[idx] - t_lo[idx], t_hi[idx] - t_med[idx]], fmt="s",
                    color=COLORS[1], ms=5, label="-h_i/c_p from Cantera (median, range over training states)")
        ax.plot(xs, implied[idx], "o", color=COLORS[0], ms=6, label="ChemKAN w_i x span_T / span_i")
        ax.set_xticks(xs, [names[i] for i in idx], rotation=60, fontsize=7.5)
        ax.set_yscale("symlog", linthresh=1e3)
        ax.set_ylabel("dT/dt per unit dY_i/dt [K]")
        ax.grid(True, color=GRID, lw=0.5)
        ax.legend(fontsize=7, loc="best")
        ax.set_title("thermo linear coefficients (Eq. 14) vs physical -h_i/c_p"
                     + ("" if m <= 12 else ": 6 lowest and 6 highest"), loc="left")
        ax2 = fig.add_subplot(1, 2, 2)
        x = np.linspace(-0.1, 1.1, 241)
        best = list(np.argsort(share_c)[::-1][: (top or 5) if m > 12 else 5])
        z = parts["z"]
        for k, j in enumerate(best[:5]):
            grid = torch.zeros(len(x), m + 1, dtype=torch.float64)
            grid[:, j] = torch.as_tensor(x)
            with torch.no_grad():
                v = model.net.thermo.correction.edges(grid).numpy()[:, 0, j]
            inside = (x >= z[:, j].min()) & (x <= z[:, j].max())
            ax2.plot(x[inside], v[inside], color=COLORS[k % 4] if k < 4 else MUTED, lw=1.6,
                     label=f"{names[j]} ({100 * share_c[j]:.0f} %)")
        ax2.set_xlabel("normalized input")
        ax2.set_ylabel("KAN_cor edge value")
        ax2.grid(True, color=GRID, lw=0.5)
        ax2.legend(fontsize=7)
        ax2.set_title(f"KAN_cor (Eq. 17), 5 largest edges; spread of dT_hat: linear {s_lin:.2g}, "
                      f"KAN_cor {s_cor:.2g}", loc="left")
        fig.tight_layout()
        save(fig, out, "fig3_temperature_structure")
    rel = np.abs(implied - t_med) / np.abs(t_med)
    return {"implied_coefficient_K": {names[i]: float(implied[i]) for i in range(m)},
            "cantera_minus_h_over_cp_median_K": {names[i]: float(t_med[i]) for i in range(m)},
            "cantera_minus_h_over_cp_range_K": {names[i]: [float(t_lo[i]), float(t_hi[i])] for i in range(m)},
            "within_cantera_range": {names[i]: bool(t_lo[i] <= implied[i] <= t_hi[i]) for i in range(m)},
            "relative_deviation_from_median": {names[i]: float(rel[i]) for i in range(m)},
            "sign_agreement": float(np.mean(np.sign(implied) == np.sign(t_med))),
            "spread_dT_hat_linear_term": float(s_lin), "spread_dT_hat_KAN_cor": float(s_cor),
            "KAN_cor_edge_share": {names[j]: float(share_c[j]) for j in range(m + 1)}}


def parity(system, model, ref, names, states, is_test, out):
    shown = (SYSTEMS[system]["species_shown"] or names[:-1])[:9]
    cols = [names.index(s) for s in shown] + [len(names) - 1]
    res = {}
    u_all = states.reshape(-1, states.shape[-1])
    split = np.repeat(is_test, states.shape[1])
    rk, rt = model.rates(u_all), ref.rates(u_all)
    for c in range(len(names)):
        for sname, mask in (("train", ~split), ("test", split)):
            a, b = rk[mask, c], rt[mask, c]
            rms = np.sqrt(np.mean(b ** 2))
            res.setdefault(names[c], {})[sname] = {
                "relative_rmse": float(np.sqrt(np.mean((a - b) ** 2)) / rms) if rms > 0 else None,
                "r2": (float(1 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2))
                       if np.sum((b - b.mean()) ** 2) > 0 else None),
                "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                "spearman": spearman(a, b),
                "within_factor_2": within2(a, b),
                "rms_reference": float(rms),
                "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
                "n_states": int(mask.sum())}
    with plt.rc_context(RC):
        n = len(cols)
        ncol = 5
        nrow = -(-n // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow), squeeze=False)
        for ax, c in zip(axes.flat, cols):
            for k, (sname, mask) in enumerate((("train", ~split), ("test", split))):
                ax.plot(rt[mask, c], rk[mask, c], "o", ms=2.2, alpha=0.5, mec="none", color=COLORS[k],
                        label=f"{sname} states" if c == cols[0] else None)
            lo = min(rt[:, c].min(), rk[:, c].min())
            hi = max(rt[:, c].max(), rk[:, c].max())
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
            lin = 1e-3 * max(abs(lo), abs(hi))
            ax.set_xscale("symlog", linthresh=lin)
            ax.set_yscale("symlog", linthresh=lin)
            unit = "K/s" if c == len(names) - 1 else "1/s"
            nm = "T" if c == len(names) - 1 else names[c]
            q = res[names[c]]
            txt = (f"Spearman {q['train']['spearman']:.2f}, x2 {100 * q['train']['within_factor_2']:.0f} %, "
                   f"rel. RMSE {q['train']['relative_rmse']:.2g}"
                   if q["train"]["relative_rmse"] is not None else
                   f"inert in Cantera; ChemKAN RMS {q['train']['rmse']:.2g}")
            ax.set_title(f"d{nm}/dt [{unit}]  {txt}", loc="left", fontsize=7.5)
            ax.grid(True, color=GRID, lw=0.5)
            ax.tick_params(labelsize=6.5)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.supxlabel("Cantera rate at the reference state", fontsize=8.5)
        fig.supylabel("ChemKAN rate", fontsize=8.5)
        fig.suptitle(f"{system}: instantaneous rates at every state of the reference trajectories (symlog axes). "
                     "Training states: rank correlation, share within a factor 2 (states with |rate| > 1 % of the "
                     "species' max), relative RMSE", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save(fig, out, "fig4_rate_parity")
    imb_k, elems = ref.element_rate_imbalance(rk[:, :-1])
    imb_t, _ = ref.element_rate_imbalance(rt[:, :-1])
    return res, {"per_element": {e: {"chemkan_median": float(np.median(imb_k[:, k])),
                                      "chemkan_p90": float(np.percentile(imb_k[:, k], 90)),
                                      "cantera_median": float(np.median(imb_t[:, k]))}
                                  for k, e in enumerate(elems)},
                 "definition": "per state and element: |sum_k a_ek dY_k/dt / W_k| / sum_k |a_ek dY_k/dt / W_k| "
                               "(0 = conserved; 1 = all contributions of one sign, e.g. an element held only by "
                               "an inert species that ChemKAN gives a nonzero rate)"}


def time_aligned(system, model, ref, rd, names):
    """ChemKAN's rate on its OWN predicted trajectory vs Cantera's rate on the reference
    trajectory at the same time (all 36 conditions, 100 times): does the learned field
    reproduce the time evolution even where it differs from the rate law at the same state?"""
    info = pt.load_model(torch.load(ROOT / SYSTEMS[system]["checkpoint"], map_location="cpu",
                                    weights_only=False))
    own, refs = [], []
    for i in range(len(rd["ics"])):
        u = pt.predict(info, rd, i, rd["t"]) * model.span + model.u_min
        own.append(model.rates(u))
        refs.append(ref.rates(np.concatenate([rd["states"][i][:, :-1], rd["temperature"][:, i:i + 1]], 1)))
    a, b = np.concatenate(own), np.concatenate(refs)
    out = {}
    for c in range(len(names)):
        out[names[c]] = {"spearman": spearman(a[:, c], b[:, c]), "within_factor_2": within2(a[:, c], b[:, c]),
                         "relative_rmse": (float(np.sqrt(np.mean((a[:, c] - b[:, c]) ** 2))
                                                 / np.sqrt(np.mean(b[:, c] ** 2)))
                                           if np.any(b[:, c]) else None)}
    return out


def pick_states(system, ref_data, names):
    ics, states, is_test = ref_data["ics"], ref_data["states"], ref_data["is_test"]
    t = ref_data["t"]
    T0, phi = SYSTEMS[system]["state_condition"]
    ci = int(np.flatnonzero(np.isclose(ics[:, 0], T0) & np.isclose(ics[:, 1], phi))[0])
    hi = int(np.flatnonzero(is_test)[0])

    def ign(i):
        return int(np.argmax(np.gradient(states[i, :, -1], t)))
    chosen = [(ci, max(1, ign(ci) // 2), "induction"), (ci, ign(ci), "ignition"),
              (ci, len(t) - 1, "post-ignition"), (hi, ign(hi), "ignition, held-out")]
    return [{"condition": int(i), "T0": float(ics[i, 0]), "phi": float(ics[i, 1]),
             "split": "test" if is_test[i] else "train", "phase": phase, "time_index": ti,
             "t_s": float(t[ti]), "u": states[i, ti].copy()} for i, ti, phase in chosen]


def perturb(system, model, ref, names, chosen, train_u, out):
    fuel = SYSTEMS[system]["fuel"]
    jf, jo, jn, jT = names.index(fuel), names.index("O2"), names.index("N2"), len(names) - 1
    panels = SYSTEMS[system]["panels"]
    curves, res = {}, []
    for s_i, s in enumerate(chosen):
        u0 = s["u"]
        entry = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        entry["u"] = {n: float(v) for n, v in zip(names, u0)}
        entry["sweeps"] = {}
        sweeps = {"T": (jT, np.linspace(max(u0[jT] - T_PERT_K, train_u[:, jT].min()),
                                         min(u0[jT] + T_PERT_K, train_u[:, jT].max()), N_PERT))}
        for var, j in ((fuel, jf), ("O2", jo)):
            lo = max(u0[j] * FRACTION_FACTOR[0], train_u[:, j].min())
            hi = min(u0[j] * FRACTION_FACTOR[1], train_u[:, j].max())
            sweeps[var] = (j, np.linspace(lo, hi, N_PERT))
        for var, (j, vals) in sweeps.items():
            U = np.repeat(u0[None], N_PERT, 0)
            U[:, j] = vals
            if var != "T":
                U[:, jn] = u0[jn] - (vals - u0[j])                       # N2 keeps sum(Y) fixed
            rk, rt = model.rates(U), ref.rates(U)
            dist, scale = domain_distance(model, U, train_u)
            curves[(s_i, var)] = (vals, rk, rt, dist)
            dev = {}
            for c in [names.index(p) for p in panels] + [jT]:
                denom = np.sqrt(np.mean(rt[:, c] ** 2))
                near = dist <= NEAR
                dev[names[c]] = {
                    "relative_l2_deviation": (float(np.sqrt(np.mean((rk[:, c] - rt[:, c]) ** 2)) / denom)
                                              if denom > 0 else None),
                    "relative_l2_deviation_near_data": (float(np.sqrt(np.mean((rk[near, c] - rt[near, c]) ** 2))
                                                              / np.sqrt(np.mean(rt[near, c] ** 2)))
                                                        if near.sum() >= 3 else None),
                    "same_direction_of_change": (None if abs(rt[-1, c] - rt[0, c]) < 1e-12 * max(1, abs(rt[0, c]))
                                                 else bool(np.sign(rt[-1, c] - rt[0, c]) == np.sign(rk[-1, c] - rk[0, c])))}
            entry["sweeps"][var] = {"range": [float(vals[0]), float(vals[-1])], "points": N_PERT,
                                    "nearest_training_state_distance_median": float(np.median(dist)),
                                    "nearest_training_state_distance_max": float(dist.max()),
                                    "points_within_near": int((dist <= NEAR).sum()),
                                    "distance_unit": f"median nearest-neighbour spacing of training states ({scale:.3g})",
                                    "deviation": dev}
        res.append(entry)
    with plt.rc_context(RC):
        cols = [names.index(p) for p in panels] + [jT]
        fig, axes = plt.subplots(3, len(cols), figsize=(2.9 * len(cols), 7.4), squeeze=False)
        for r_, var in enumerate(("T", fuel, "O2")):
            for c_, c in enumerate(cols):
                ax = axes[r_, c_]
                for s_i, s in enumerate(chosen):
                    vals, rk, rt, dist = curves[(s_i, var)]
                    near = dist <= NEAR
                    scale = np.abs(rt[:, c]).max() or 1.0
                    ax.plot(vals, rt[:, c] / scale, color=COLORS[s_i], lw=1.0, ls="--")
                    ax.plot(vals, rk[:, c] / scale, color=COLORS[s_i], lw=0.8, alpha=0.45)
                    ax.plot(vals, np.where(near, rk[:, c] / scale, np.nan), color=COLORS[s_i], lw=1.8)
                ax.set_yscale("symlog", linthresh=1.0)
                ax.grid(True, color=GRID, lw=0.5)
                nm = "T" if c == jT else names[c]
                if r_ == 0:
                    ax.set_title(f"d{nm}/dt", loc="left")
                if c_ == 0:
                    ax.set_ylabel(f"vary {var}\nrate / max |Cantera rate|")
                ax.set_xlabel("T [K]" if var == "T" else f"Y {var}", fontsize=7.5)
        handles = [plt.Line2D([], [], color=COLORS[k], lw=1.8,
                              label=f"{s['phase']}: {s['T0']:g} K, phi {s['phi']:g}, t = {1e3 * s['t_s']:.3g} ms")
                   for k, s in enumerate(chosen)]
        handles += [plt.Line2D([], [], color=MUTED, lw=1.8, label=f"ChemKAN, within {NEAR:g} spacings of a training state"),
                    plt.Line2D([], [], color=MUTED, lw=0.8, alpha=0.45, label="ChemKAN, farther away"),
                    plt.Line2D([], [], color=MUTED, lw=1.0, ls="--", label="Cantera")]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5)
        fig.suptitle(f"{system}: one input varied around four reference states (others fixed; N2 absorbs fuel/O2 "
                     f"changes). Each curve scaled by its max |Cantera rate| (symlog axis: |values| > 1 = ChemKAN beyond the Cantera range).", fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        save(fig, out, "fig5_perturbations")
    return res, curves


def arrhenius(system, chosen, curves, names, out):
    fuel = SYSTEMS[system]["fuel"]
    jf = names.index(fuel)
    R_KCAL = 1.987204259e-3
    s_i = 0                                                             # induction state
    vals, rk, rt, dist = curves[(s_i, "T")]
    near = dist <= NEAR

    def fit(r, m):
        if m.sum() < 5 or np.any(r[m] <= 0):
            return None
        return float(-np.polyfit(1 / vals[m], np.log(r[m]), 1)[0] * R_KCAL)
    whole, nearm = np.ones_like(near), near
    res = {"state": chosen[s_i]["phase"], "fuel": fuel,
           "whole_sweep": {"chemkan": fit(-rk[:, jf], whole), "cantera": fit(-rt[:, jf], whole)},
           "near_training_states": {"chemkan": fit(-rk[:, jf], nearm), "cantera": fit(-rt[:, jf], nearm),
                                    "points": int(near.sum())},
           "range_K": [float(vals[0]), float(vals[-1])]}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.8, 3.7))
        ax.plot(1000 / vals, -rt[:, jf], color=COLORS[0], lw=1.0, ls="--", label="Cantera")
        ax.plot(1000 / vals, -rk[:, jf], color=COLORS[0], lw=0.8, alpha=0.45)
        ax.plot(1000 / vals, np.where(near, -rk[:, jf], np.nan), color=COLORS[0], lw=2.0,
                label=f"ChemKAN (bold: within {NEAR:g} spacings)")
        ax.set_yscale("log")
        ax.set_xlabel("1000 / T [1/K]")
        ax.set_ylabel(f"-d Y_{fuel}/dt [1/s]")
        ax.grid(True, color=GRID, lw=0.5)
        ax.legend(fontsize=7)
        s = chosen[s_i]
        ax.set_title(f"{system}: {fuel} consumption at the induction state\n({s['T0']:g} K, phi {s['phi']:g}, "
                     f"t = {1e3 * s['t_s']:.3g} ms), T varied, composition fixed", loc="left")
        fig.tight_layout()
        save(fig, out, "fig6_arrhenius_induction")
    return res


def hidden_correlations(parts, ref_rates, names):
    """Descriptive only: Pearson r of each hidden node with T, the fuel, OH and dT/dt (Cantera)."""
    h = parts["h"]
    cols = {"T": parts["z"][:, -1], "dT/dt (Cantera)": ref_rates[:, -1]}
    for sp in ("H2", "CH4", "OH", "H2O"):
        if sp in names:
            cols[sp] = parts["z"][:, names.index(sp)]
    return {f"h{j + 1}": {k: float(np.corrcoef(h[:, j], v)[0, 1]) for k, v in cols.items()} for j in range(3)}


def main(system: str) -> int:
    spec = SYSTEMS[system]
    ckpt_path = ROOT / spec["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    before = sha(ckpt["model_state"])
    model = Model(ckpt)
    rd = pt.load_reference(model.data)
    names = rd["names"]
    states = np.concatenate([rd["states"][..., :-1], rd["temperature"].T[..., None]], -1)
    train_u = states[~rd["is_test"]].reshape(-1, len(names))
    out = HERE
    ref = Reference(model.data, names[:-1])
    parts = model.parts(train_u)
    edges = edge_figures(system, model, names, parts, out, spec["top_inputs"])
    thermo = thermo_figure(system, model, ref, names, parts, train_u, out, spec["top_inputs"])
    par, elem = parity(system, model, ref, names, states, rd["is_test"], out)
    chosen = pick_states(system, rd, names)
    pert, curves = perturb(system, model, ref, names, chosen, train_u, out)
    arr = arrhenius(system, chosen, curves, names, out)
    fd = ref.finite_difference_check(np.array([c["u"] for c in chosen]))
    after = sha({k: v.to(ckpt["model_state"][k].dtype) for k, v in model.net.state_dict().items()})
    after = after if all(torch.equal(v.to(ckpt["model_state"][k].dtype), ckpt["model_state"][k])
                         for k, v in model.net.state_dict().items()) else "changed"
    summary = {
        "system": system, "checkpoint": spec["checkpoint"],
        "checkpoint_epochs": ckpt.get("phase_epochs_completed"),
        "checkpoint_parameters_sha256": before,
        "parameters_unchanged": before == after,
        "model": {"architecture": model.cfg["architecture"], "t_ref_s": model.t_ref},
        "normalization": {"states": names, "u_min": model.u_min.tolist(), "u_max": model.u_max.tolist()},
        "reference": {"mechanism": str(np.load(model.data, allow_pickle=True)["mechanism"]),
                      "pressure_pa": ref.P, "rates": "Cantera net production rates, constant-pressure "
                                                    "adiabatic energy equation",
                      "finite_difference_check_max_rel_error": fd},
        "data": {"file": str(Path(model.data).relative_to(ROOT)), "train_states": int(len(train_u)),
                 "test_states": int(rd["is_test"].sum() * states.shape[1])},
        "edges": edges,
        "temperature_structure": thermo,
        "rate_parity_all_reference_states": par,
        "rate_along_own_trajectory_vs_reference_same_time": time_aligned(system, model, ref, rd, names),
        "element_conservation_of_rates": elem,
        "perturbations": {"T_range_K": T_PERT_K, "fraction_factor": FRACTION_FACTOR,
                          "near_threshold_spacings": NEAR, "states": pert},
        "arrhenius_induction": arr,
        "hidden_node_correlations_descriptive": hidden_correlations(parts, ref.rates(train_u), names),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({"parameters_unchanged": before == after, "fd_check": fd,
                      "parity_train_spearman_within2_relrmse": {
                          k: [v["train"]["spearman"] and round(v["train"]["spearman"], 3),
                              v["train"]["within_factor_2"] and round(v["train"]["within_factor_2"], 3),
                              v["train"]["relative_rmse"] and round(v["train"]["relative_rmse"], 3)]
                          for k, v in par.items()},
                      "element_imbalance": elem, "arrhenius": arr,
                      "thermo_sign_agreement": thermo["sign_agreement"],
                      "thermo_within_range": sum(thermo["within_cantera_range"].values()),
                      "spread_lin_cor": [thermo["spread_dT_hat_linear_term"], thermo["spread_dT_hat_KAN_cor"]]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(HERE.parent.name))          # the system is this folder's parent (hydrogen)

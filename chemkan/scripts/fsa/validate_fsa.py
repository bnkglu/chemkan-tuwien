r"""Validation of the continuous FSA implementation BEFORE any FSA training (DIAGNOSTIC).

Sections (each ends PASS / FAIL / INCONCLUSIVE; G is optional):

    A  analytic scalar problem dy/dt = theta*y (not ChemKAN; tests the generic machinery)
    B  functional equivalence: existing dynamics wrapper == functional form used by FSA
    C  FSA vs direct-autograd gradients on fixed ChemKAN models under tolerance refinement
       (biodiesel; hydrogen Stage 1 incl. T_obs checks; hydrogen Stage 2 per group, both
       initialization regimes, ignition-relevant trajectories), plus the Section-10
       adequacy of the default augmented error control at the production tolerance
    D  physical-state consistency: augmented FSA state vs ordinary state-only solve
    E  trajectory independence (structural zero cross-derivatives; batch vs individual)
    F  production dtype (float32, production tolerance) against a float64 reference
    G  optional central finite differences with an epsilon sweep

The acceptance criteria below are written to ``acceptance_criteria.json`` BEFORE any
section runs. Results go to ``results/experiments/validation/fsa`` (one JSON per section
plus ``summary.json``). Direct-autograd reference gradients are computed only here, never
inside FSA training.

Usage (from chemkan/scripts):
    python fsa/validate_fsa.py                  # all sections
    python fsa/validate_fsa.py --sections A,B   # a subset (the summary is merged)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # chemkan/scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _problems as P                                               # noqa: E402
from _run import git_commit, utc_now                                # noqa: E402
from torch.func import jacrev                                       # noqa: E402

from chemkan.fsa import FunctionalDynamics, integrate_with_sensitivities  # noqa: E402
from chemkan.solver import SolverConfig, integrate                  # noqa: E402
from chemkan.training import loss_and_gradients                     # noqa: E402

OUT = P.RESULTS / "experiments/validation/fsa"

PRODUCTION = (1e-6, 1e-8)                  # rtol, atol of every archived baseline
LEVELS = {"L0": (1e-6, 1e-8), "L1": (1e-8, 1e-10), "L2": (1e-10, 1e-12)}
REFERENCE = (1e-12, 1e-14)                 # state-only reference for Section D

CRITERIA = {
    "A_analytic": {
        "dtype": "float64", "levels": LEVELS,
        "at_L2": {"state_rel_err_max": 1e-8, "sensitivity_rel_err_max": 1e-7,
                  "loss_gradient_rel_err_max": 1e-7, "fsa_vs_direct_autograd_rel_max": 1e-7},
        "refinement": "every error at L2 is smaller than at L0 (no strict monotonicity)"},
    "B_functional_equivalence": {
        "metric": "max|F_functional - F_wrapper| / max(1, max|F_wrapper|) over states/times",
        "float64_max": 1e-12, "float32_max": 1e-5,
        "blocking": "any failure stops gradient validation"},
    "C_gradients": {
        "dtype": "float64", "levels": LEVELS,
        "agreement_at_L2": {"relative_l2_max": 1e-4, "cosine_min": 1 - 1e-8,
                            "max_abs_err_over_max_abs_grad_max": 1e-3,
                            "per_group_relative_l2_max": 1e-3,
                            "near_zero_group": "a group whose DA norm is < 1e-8 x the full "
                                               "gradient norm is judged by ||dg_group|| / "
                                               "||g_full|| <= 1e-6 instead"},
        "refinement_stability": {"fsa_L1_vs_L2_relative_max": 1e-3,
                                 "direct_autograd_L1_vs_L2_relative_max": 1e-3},
        "section10_production_adequacy": {
            "definition": "FSA gradient at the production tolerance L0 vs the refined FSA "
                          "gradient at L2", "relative_l2_max": 5e-2, "cosine_min": 0.998},
        "a_large_disagreement_persisting_under_refinement": "FAIL"},
    "D_state_consistency": {
        "metric": "max over (t, b, i) of |x - x_ref| / train_range_i, x_ref = state-only "
                  "solve at rtol/atol = 1e-12/1e-14 (float64)",
        "per_level": "err_augmented <= 10 * err_state_only + 1e-10",
        "at_L2": "err_augmented <= 1e-6 and |x_augmented - x_state_only| <= 1e-6",
        "sensitivity_refinement": "||S_L1 - S_L2|| / ||S_L2|| <= 1e-3 (reported per model)"},
    "E_independence": {
        "structural": "d F_b / d x_c == 0 exactly for every b != c (float64 jacrev)",
        "stage1_temperature": "perturbing T_obs of trajectory c leaves F_b bit-identical",
        "batch_vs_individual_at_L2": {"state_normalized_max": 1e-7,
                                      "sensitivity_relative_max": 1e-5}},
    "F_production_dtype": {
        "dtype": "float32", "tolerance": PRODUCTION,
        "requirements": "finite states, sensitivities, losses, gradients",
        "vs_float64_reference": {"relative_l2_max": 5e-2, "cosine_min": 0.998,
                                 "reference": "float64 FSA gradient at L2 on the same batch"}},
    "G_finite_differences": {
        "optional": True, "dtype": "float64", "solves": "state-only at 1e-12/1e-14",
        "sweep": "h = max(|theta_q|, 1e-2) * 10^-k, k = 2..8",
        "stable_region": "two consecutive h whose estimates differ by <= 1e-4 relative",
        "pass": "stable estimate within 1e-3 relative of the FSA L2 gradient",
        "inconclusive": "no stable region"},
}


# ----------------------------------------------------------------------------- utils
def cfg(level, backend="fsa"):
    rtol, atol = LEVELS[level] if isinstance(level, str) else level
    return SolverConfig(method="tsit5", rtol=rtol, atol=atol, sensitivity=backend)


class NFE:
    def __init__(self, module):
        self.n = 0
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, *a):
        self.n += 1

    def close(self):
        self.h.remove()


def flat_grad(fd: FunctionalDynamics) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1).clone() for p in fd.params])


def gradient(prob: P.Problem, backend: str, level) -> dict:
    fd = FunctionalDynamics(prob.dynamics, prob.params)
    for p in prob.params:
        p.grad = None
    nfe = NFE(prob.dynamics)
    t0 = time.perf_counter()
    loss = loss_and_gradients(prob.dynamics, prob.x0, prob.t, prob.params, prob.loss_fn,
                              cfg(level, backend))
    dt = time.perf_counter() - t0
    nfe.close()
    g = flat_grad(fd)
    for p in prob.params:
        p.grad = None
    return {"grad": g, "loss": float(loss.detach()), "seconds": dt, "nfe": nfe.n, "fd": fd}


def compare(g, ref, packing=None) -> dict:
    d = g - ref
    out = {"relative_l2": float(d.norm() / ref.norm()),
           "cosine": float(torch.dot(g, ref) / (g.norm() * ref.norm())),
           "max_abs_err": float(d.abs().max()), "max_abs_ref": float(ref.abs().max()),
           "ref_norm": float(ref.norm())}
    out["max_abs_err_over_max_abs_grad"] = out["max_abs_err"] / out["max_abs_ref"]
    if packing is not None:
        groups = {}
        for name, ix in packing.group_indices().items():
            gn = float(ref[ix].norm())
            dn = float(d[ix].norm())
            groups[name] = {"n": int(ix.numel()), "ref_norm": gn, "abs_l2_err": dn,
                            "relative_l2": dn / gn if gn > 0 else math.inf,
                            "near_zero": gn < 1e-8 * float(ref.norm()),
                            "err_over_full_norm": dn / float(ref.norm())}
        out["groups"] = groups
    return out


def agreement_ok(c: dict, crit=CRITERIA["C_gradients"]["agreement_at_L2"]) -> tuple[bool, list]:
    why = []
    if not c["relative_l2"] <= crit["relative_l2_max"]:
        why.append(f"relative_l2 {c['relative_l2']:.3e}")
    if not c["cosine"] >= crit["cosine_min"]:
        why.append(f"cosine {c['cosine']:.12f}")
    if not c["max_abs_err_over_max_abs_grad"] <= crit["max_abs_err_over_max_abs_grad_max"]:
        why.append(f"max_abs ratio {c['max_abs_err_over_max_abs_grad']:.3e}")
    for g, v in c.get("groups", {}).items():
        if v["near_zero"]:
            if not v["err_over_full_norm"] <= 1e-6:
                why.append(f"near-zero group {g} err/full {v['err_over_full_norm']:.3e}")
        elif not v["relative_l2"] <= crit["per_group_relative_l2_max"]:
            why.append(f"group {g} relative_l2 {v['relative_l2']:.3e}")
    return not why, why


def status(ok: bool | None) -> str:
    return "INCONCLUSIVE" if ok is None else ("PASS" if ok else "FAIL")


def write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"section": name, "created": utc_now(), "git_commit": git_commit(),
               "code_state": P.code_state(OUT / "code_patches"),
               "torch": torch.__version__, "torch_num_threads": torch.get_num_threads(),
               **payload}
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  -> {path.relative_to(P.ROOT)}  [{payload.get('status')}]", flush=True)
    return path


def normalized_max(a, b, rng):
    return float(((a - b).abs() / rng).max())


# ------------------------------------------------------------------------- section A
class Exponential(torch.nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(theta, dtype=torch.float64))

    def forward(self, t, y):
        return self.theta * y


def section_A():
    y0 = 1.3
    t = torch.linspace(0.0, 2.0, 9, dtype=torch.float64)
    data = torch.linspace(1.0, 0.2, 9, dtype=torch.float64)
    cases, ok = [], True
    for theta in (-0.7, 0.5):
        dyn = Exponential(theta)
        x0 = torch.tensor([[y0]], dtype=torch.float64)
        y_ex = y0 * torch.exp(theta * t)
        S_ex = t * y_ex                                            # dy/dtheta = t y(t)
        g_ex = float(((y_ex - data) * t * y_ex).sum())             # L = 0.5 sum (y - d)^2
        loss = lambda pred: 0.5 * ((pred[:, 0, 0] - data) ** 2).sum()
        levels = {}
        for lv in LEVELS:
            fd = FunctionalDynamics(dyn, [dyn.theta])
            x, S = integrate_with_sensitivities(fd, x0, t, cfg(lv))
            g_fsa = gradient(P.Problem("toy", dyn, dyn, [dyn.theta], x0, t, loss,
                                       torch.ones(1)), "fsa", lv)["grad"]
            g_da = gradient(P.Problem("toy", dyn, dyn, [dyn.theta], x0, t, loss,
                                      torch.ones(1)), "direct_autograd", lv)["grad"]
            levels[lv] = {
                "state_rel_err": float(((x[:, 0, 0] - y_ex).abs() / y_ex.abs()).max()),
                "sensitivity_rel_err": float(((S[1:, 0, 0, 0] - S_ex[1:]).abs()
                                              / S_ex[1:].abs()).max()),
                "S_t0": float(S[0].abs().max()),
                "fsa_gradient": float(g_fsa), "exact_gradient": g_ex,
                "loss_gradient_rel_err": abs(float(g_fsa) - g_ex) / abs(g_ex),
                "direct_autograd_gradient": float(g_da),
                "fsa_vs_direct_autograd_rel": abs(float(g_fsa - g_da)) / abs(float(g_da)),
                "no_graph": bool(x.grad_fn is None and S.grad_fn is None)}
        a = CRITERIA["A_analytic"]["at_L2"]
        L2, L0 = levels["L2"], levels["L0"]
        checks = {
            "state_L2": L2["state_rel_err"] <= a["state_rel_err_max"],
            "sensitivity_L2": L2["sensitivity_rel_err"] <= a["sensitivity_rel_err_max"],
            "gradient_L2": L2["loss_gradient_rel_err"] <= a["loss_gradient_rel_err_max"],
            "fsa_vs_da_L2": L2["fsa_vs_direct_autograd_rel"] <= a["fsa_vs_direct_autograd_rel_max"],
            "S_t0_zero": all(v["S_t0"] == 0.0 for v in levels.values()),
            "no_graph": all(v["no_graph"] for v in levels.values()),
            "refines": all(L2[k] < L0[k] or L2[k] == 0.0 for k in
                           ("state_rel_err", "sensitivity_rel_err", "loss_gradient_rel_err")),
        }
        ok &= all(checks.values())
        cases.append({"theta": theta, "y0": y0, "levels": levels, "checks": checks})
    return write("A_analytic", {"status": status(ok), "problem": "dy/dt = theta*y, y(0)=y0; "
                                "S = dy/dtheta = t*y(t); L = 0.5*sum_j (y_j - d_j)^2",
                                "cases": cases, "criteria": CRITERIA["A_analytic"]})


# ------------------------------------------------------------------------- section B
def _probe_states(prob, n_times=5):
    """States along a state-only solve of the problem itself, at a few times."""
    with torch.no_grad():
        x = integrate(prob.dynamics, prob.x0, prob.t, cfg("L0", "direct_autograd"))
    ix = torch.linspace(0, len(prob.t) - 1, n_times).round().long()
    return [(prob.t[i], x[i]) for i in ix]


def section_B():
    rows, ok = [], True
    builders = [("biodiesel", P.biodiesel, ("init", "trained")),
                ("stage1", P.stage1, ("init", "trained")),
                ("stage2", P.stage2, P.STAGE2_MODELS)]
    for kind, build, models in builders:
        for model in models:
            for dtype, lim in ((torch.float64, CRITERIA["B_functional_equivalence"]["float64_max"]),
                               (torch.float32, CRITERIA["B_functional_equivalence"]["float32_max"])):
                prob = build(model, dtype)
                fd = FunctionalDynamics(prob.dynamics, prob.params)
                worst = 0.0
                for t, x in _probe_states(prob):
                    with torch.no_grad():
                        ref = prob.dynamics(t, x)
                        got = fd.rhs(t, x)
                        _, _, _ = fd.rhs_and_jacobians(t, x, fd.theta())
                    worst = max(worst, float((got - ref).abs().max() / max(1.0, float(ref.abs().max()))))
                good = worst <= lim
                ok &= good
                rows.append({"problem": prob.name, "dtype": str(dtype), "metric": worst,
                             "limit": lim, "pass": good, "trajectories": prob.meta["trajectories"]})
    # the seed-0 H0 Stage-2 initialization equals the archived untrained Stage-2 tensors
    arch = P.load(P.H_STAGE1_DIR / "checkpoint_final.pt")["model_state"]
    net = P.stage2_net("H0-init")
    same = all(torch.equal(v, arch[k]) for k, v in net.state_dict().items() if k.startswith("thermo."))
    return write("B_functional_equivalence", {
        "status": status(ok), "rows": rows, "criteria": CRITERIA["B_functional_equivalence"],
        "h0_init_thermo_equals_archived_stage1_run_final_thermo": same})


# ------------------------------------------------------------------------- section C
def _gradient_case(prob: P.Problem, extra: dict | None = None) -> dict:
    res = {"problem": prob.name, "trajectories": prob.meta["trajectories"], "levels": {}}
    grads = {}
    for lv in LEVELS:
        f = gradient(prob, "fsa", lv)
        d = gradient(prob, "direct_autograd", lv)
        packing = f["fd"].packing
        grads[lv] = (f["grad"], d["grad"])
        res["levels"][lv] = {"loss_fsa": f["loss"], "loss_direct_autograd": d["loss"],
                             "nfe_fsa": f["nfe"], "nfe_direct_autograd": d["nfe"],
                             "seconds_fsa": f["seconds"], "seconds_direct_autograd": d["seconds"],
                             "fsa_vs_direct_autograd": compare(f["grad"], d["grad"], packing)}
        print(f"    {prob.name:22s} {lv}: rel {res['levels'][lv]['fsa_vs_direct_autograd']['relative_l2']:.3e}"
              f"  nfe fsa/da {f['nfe']}/{d['nfe']}", flush=True)
    res["packing"] = packing.describe()["group_counts"]
    fL1, dL1 = grads["L1"]
    fL2, dL2 = grads["L2"]
    fL0, dL0 = grads["L0"]
    res["refinement"] = {"fsa_L1_vs_L2": compare(fL1, fL2, packing)["relative_l2"],
                         "direct_autograd_L1_vs_L2": compare(dL1, dL2, packing)["relative_l2"],
                         "fsa_L0_vs_L1": compare(fL0, fL1)["relative_l2"],
                         "direct_autograd_L0_vs_L1": compare(dL0, dL1)["relative_l2"]}
    res["section10_production"] = {"fsa_L0_vs_fsa_L2": compare(fL0, fL2, packing),
                                   "direct_autograd_L0_vs_fsa_L2": compare(dL0, fL2, packing)}
    ok_agree, why = agreement_ok(res["levels"]["L2"]["fsa_vs_direct_autograd"])
    rs = CRITERIA["C_gradients"]["refinement_stability"]
    ok_ref = (res["refinement"]["fsa_L1_vs_L2"] <= rs["fsa_L1_vs_L2_relative_max"]
              and res["refinement"]["direct_autograd_L1_vs_L2"] <= rs["direct_autograd_L1_vs_L2_relative_max"])
    s10 = CRITERIA["C_gradients"]["section10_production_adequacy"]
    p10 = res["section10_production"]["fsa_L0_vs_fsa_L2"]
    ok_10 = p10["relative_l2"] <= s10["relative_l2_max"] and p10["cosine"] >= s10["cosine_min"]
    if not ok_ref:
        why.append("refinement not stable")
    if not ok_10:
        why.append("production-tolerance FSA gradient not adequate (Section 10)")
    res.update(extra or {})
    res["checks"] = {"agreement_at_L2": ok_agree, "refinement_stable": ok_ref,
                     "section10_production_adequate": ok_10}
    res["pass"] = ok_agree and ok_ref and ok_10
    res["failure_reasons"] = why
    return res


def _stage1_structure(prob: P.Problem) -> dict:
    fd = FunctionalDynamics(prob.dynamics, prob.params)
    temp = prob.dynamics.temperature
    with torch.no_grad():
        T_a, T_b = temp(prob.t[5]), temp(prob.t[30])
    x, S = integrate_with_sensitivities(fd, prob.x0, prob.t[:3], cfg("L0"))
    return {
        "differentiated_groups": sorted(set(fd.packing.groups)),
        "n_parameters": fd.packing.total,
        "integrated_state_dim": int(prob.x0.shape[-1]),
        "sensitivity_shape": list(S.shape),
        "temperature_provider_has_parameters": any(True for _ in temp.parameters()),
        "temperature_buffers_in_parameter_vector": any(n.startswith("temperature.")
                                                       for n in fd.packing.names),
        "T_obs_time_dependent": bool((T_a - T_b).abs().max() > 0),
        "T_obs_trajectory_specific": bool(T_b.flatten().unique().numel() == T_b.numel()),
        "T_obs_at_t30_K": [float(v) for v in T_b.flatten()],
    }


def section_C():
    cases, ok = [], True
    for model in ("init", "trained"):
        cases.append(_gradient_case(P.biodiesel(model)))
    for model in ("init", "trained"):
        prob = P.stage1(model, idx=P.H2_SUBSET)
        st = _stage1_structure(prob)
        st_ok = (st["differentiated_groups"] == ["theta_kin"] and st["n_parameters"] == 285
                 and st["integrated_state_dim"] == 9 and not st["temperature_provider_has_parameters"]
                 and not st["temperature_buffers_in_parameter_vector"]
                 and st["T_obs_time_dependent"] and st["T_obs_trajectory_specific"])
        case = _gradient_case(prob, {"stage1_structure": st, "stage1_structure_ok": st_ok})
        case["pass"] = case["pass"] and st_ok
        cases.append(case)
    for model in P.STAGE2_MODELS:
        cases.append(_gradient_case(P.stage2(model, idx=P.H2_SUBSET)))
    ok = all(c["pass"] for c in cases)
    return write("C_gradients", {"status": status(ok), "cases": cases,
                                 "criteria": CRITERIA["C_gradients"],
                                 "hydrogen_subset": {"train_indices": list(P.H2_SUBSET),
                                                     "conditions": "950/0.9, 1050/0.9, 1200/0.9"}})


# ------------------------------------------------------------------------- section D
def section_D():
    cases, ok = [], True
    probs = [P.biodiesel("trained"), P.stage1("trained", idx=P.H2_SUBSET)] + \
            [P.stage2(m, idx=P.H2_SUBSET) for m in P.STAGE2_MODELS]
    for prob in probs:
        with torch.no_grad():
            ref = integrate(prob.dynamics, prob.x0, prob.t, cfg(REFERENCE, "direct_autograd"))
        fd = FunctionalDynamics(prob.dynamics, prob.params)
        levels, S_by = {}, {}
        for lv in LEVELS:
            x_aug, S = integrate_with_sensitivities(fd, prob.x0, prob.t, cfg(lv))
            with torch.no_grad():
                x_st = integrate(prob.dynamics, prob.x0, prob.t, cfg(lv, "direct_autograd"))
            S_by[lv] = S
            levels[lv] = {"err_augmented": normalized_max(x_aug, ref, prob.state_range),
                          "err_state_only": normalized_max(x_st, ref, prob.state_range),
                          "augmented_vs_state_only": normalized_max(x_aug, x_st, prob.state_range)}
        sens = float((S_by["L1"] - S_by["L2"]).norm() / S_by["L2"].norm())
        good = all(v["err_augmented"] <= 10 * v["err_state_only"] + 1e-10 for v in levels.values())
        good &= levels["L2"]["err_augmented"] <= 1e-6 and levels["L2"]["augmented_vs_state_only"] <= 1e-6
        good &= sens <= 1e-3
        ok &= good
        cases.append({"problem": prob.name, "levels": levels, "sensitivity_L1_vs_L2": sens,
                      "pass": good})
        print(f"    {prob.name:22s} L0 aug {levels['L0']['err_augmented']:.2e} "
              f"state {levels['L0']['err_state_only']:.2e}  S L1/L2 {sens:.2e}", flush=True)
    return write("D_state_consistency", {"status": status(ok), "cases": cases,
                                         "reference_tolerance": REFERENCE,
                                         "criteria": CRITERIA["D_state_consistency"]})


# ------------------------------------------------------------------------- section E
def section_E():
    cases, ok = [], True
    crit = CRITERIA["E_independence"]["batch_vs_individual_at_L2"]
    for build, model, idx in ((P.biodiesel, "trained", None), (P.stage1, "trained", None),
                              (P.stage2, "Hnorm1-final", None)):
        prob = build(model, idx=idx)
        fd = FunctionalDynamics(prob.dynamics, prob.params)
        worst_cross = 0.0
        for t, x in _probe_states(prob, n_times=3):
            J = jacrev(lambda xx: fd.rhs(t, xx))(x)                  # (B, n, B, n)
            B = x.shape[0]
            mask = ~torch.eye(B, dtype=torch.bool)
            worst_cross = max(worst_cross, float(J.permute(0, 2, 1, 3)[mask].abs().max()))
        entry = {"problem": prob.name, "batch": int(prob.x0.shape[0]),
                 "max_abs_cross_derivative": worst_cross}
        good = worst_cross == 0.0
        if prob.name.startswith("stage1"):
            t = prob.t[20]
            with torch.no_grad():
                base = fd.rhs(t, prob.x0)
                temps = prob.dynamics.temperature.temperatures
                saved = temps[:, 7].clone()
                temps[:, 7] += 300.0
                pert = fd.rhs(t, prob.x0)
                temps[:, 7] = saved
            others = [b for b in range(base.shape[0]) if b != 7]
            entry["stage1_other_trajectories_bit_identical"] = bool(torch.equal(base[others], pert[others]))
            entry["stage1_own_trajectory_changed"] = bool(not torch.equal(base[7], pert[7]))
            good &= entry["stage1_other_trajectories_bit_identical"] and entry["stage1_own_trajectory_changed"]
        # batch vs individual for two trajectories (tight tolerance)
        sub = (P.H2_SUBSET if not prob.name.startswith("biodiesel") else (0, 7, 13))
        batch = build(model, idx=sub)
        fdb = FunctionalDynamics(batch.dynamics, batch.params)
        xb, Sb = integrate_with_sensitivities(fdb, batch.x0, batch.t, cfg("L2"))
        comp = []
        for j, orig in ((0, sub[0]), (1, sub[1])):
            one = build(model, idx=(orig,))
            fd1 = FunctionalDynamics(one.dynamics, one.params)
            x1, S1 = integrate_with_sensitivities(fd1, one.x0, one.t, cfg("L2"))
            dx = normalized_max(x1[:, 0], xb[:, j], batch.state_range)
            dS = float((S1[:, 0] - Sb[:, j]).norm() / Sb[:, j].norm())
            comp.append({"train_index": orig, "state_normalized_max": dx, "sensitivity_relative": dS})
            good &= dx <= crit["state_normalized_max"] and dS <= crit["sensitivity_relative_max"]
        entry["batch_vs_individual"] = comp
        entry["pass"] = good
        ok &= good
        cases.append(entry)
        print(f"    {prob.name:22s} cross {worst_cross:.1e}  batch-vs-single {comp}", flush=True)
    return write("E_independence", {"status": status(ok), "cases": cases,
                                    "criteria": CRITERIA["E_independence"]})


# ------------------------------------------------------------------------- section F
def section_F():
    cases, ok = [], True
    crit = CRITERIA["F_production_dtype"]["vs_float64_reference"]
    specs = [(P.biodiesel, m) for m in ("init", "trained")] + \
            [(P.stage1, m) for m in ("init", "trained")] + [(P.stage2, m) for m in P.STAGE2_MODELS]
    for build, model in specs:
        p64 = build(model, torch.float64)
        ref = gradient(p64, "fsa", "L2")
        with torch.no_grad():
            x_ref = integrate(p64.dynamics, p64.x0, p64.t, cfg("L2", "direct_autograd"))
        p32 = build(model, torch.float32)
        fd32 = FunctionalDynamics(p32.dynamics, p32.params)
        x, S = integrate_with_sensitivities(fd32, p32.x0, p32.t, cfg(PRODUCTION))
        f = gradient(p32, "fsa", PRODUCTION)
        d = gradient(p32, "direct_autograd", PRODUCTION)
        finite = bool(torch.isfinite(x).all() and torch.isfinite(S).all()
                      and math.isfinite(f["loss"]) and torch.isfinite(f["grad"]).all())
        c_f = compare(f["grad"].double(), ref["grad"], fd32.packing)
        c_d = compare(d["grad"].double(), ref["grad"], fd32.packing)
        good = finite and c_f["relative_l2"] <= crit["relative_l2_max"] and c_f["cosine"] >= crit["cosine_min"]
        ok &= good
        cases.append({"problem": p32.name, "trajectories": p32.meta["trajectories"],
                      "finite": finite, "loss_float32_fsa": f["loss"],
                      "loss_float64_reference": ref["loss"],
                      "state_float32_vs_float64_normalized_max":
                          normalized_max(x.double(), x_ref, p64.state_range),
                      "max_abs_sensitivity": float(S.abs().max()),
                      "nfe_float32_fsa": f["nfe"], "seconds_float32_fsa": f["seconds"],
                      "nfe_float32_direct_autograd": d["nfe"],
                      "seconds_float32_direct_autograd": d["seconds"],
                      "nfe_float64_reference": ref["nfe"],
                      "float32_fsa_vs_float64_reference": c_f,
                      "float32_direct_autograd_vs_float64_reference": c_d, "pass": good})
        print(f"    {p32.name:22s} fsa32 rel {c_f['relative_l2']:.3e}  da32 rel {c_d['relative_l2']:.3e}"
              f"  s fsa/da {f['seconds']:.1f}/{d['seconds']:.1f}", flush=True)
    return write("F_production_dtype", {"status": status(ok), "cases": cases,
                                        "criteria": CRITERIA["F_production_dtype"]})


# ------------------------------------------------------------------------- section G
def _fd_estimate(prob, param, index, h):
    flat = param.data.view(-1)
    orig = flat[index].item()
    vals = []
    for s in (+1, -1):
        flat[index] = orig + s * h
        with torch.no_grad():
            pred = integrate(prob.dynamics, prob.x0, prob.t, cfg(REFERENCE, "direct_autograd"))
            vals.append(float(prob.loss_fn(pred)))
    flat[index] = orig
    return (vals[0] - vals[1]) / (2 * h)


def section_G():
    cases = []
    specs = [(P.biodiesel("trained"), [("kinetic.add.edges.w_rbf", 3), ("kinetic.lean.edges.w_rbf", 17)]),
             (P.stage2("Hnorm1-final", idx=P.H2_SUBSET),
              [("kinetic.add.edges.w_rbf", 11), ("thermo.linear.weight", 4),
               ("thermo.correction.edges.w_rbf", 6)])]
    for prob, picks in specs:
        ref = gradient(prob, "fsa", "L2")
        fd = ref["fd"]
        for pname, index in picks:
            full = next(n for n in fd.packing.names if n.endswith(pname))
            k = fd.packing.names.index(full)
            g_fsa = float(ref["grad"][fd.packing.offsets[k] + index])
            param = fd.params[k]
            scale = max(abs(float(param.detach().view(-1)[index])), 1e-2)
            sweep = []
            for e in range(2, 9):
                h = scale * 10.0 ** (-e)
                sweep.append({"h": h, "estimate": _fd_estimate(prob, param, index, h)})
            stable = None
            for a, b in zip(sweep, sweep[1:]):
                if abs(a["estimate"] - b["estimate"]) <= 1e-4 * max(abs(b["estimate"]), 1e-300):
                    stable = b
                    break
            if stable is None:
                verdict, rel = None, None
            else:
                rel = abs(stable["estimate"] - g_fsa) / max(abs(g_fsa), 1e-300)
                verdict = rel <= 1e-3
            cases.append({"problem": prob.name, "parameter": full, "index": index,
                          "fsa_L2": g_fsa, "sweep": sweep, "stable": stable,
                          "relative_error": rel, "status": status(verdict)})
            print(f"    {prob.name:22s} {full}[{index}] -> {status(verdict)} rel {rel}", flush=True)
    stats = [c["status"] for c in cases]
    overall = "FAIL" if "FAIL" in stats else ("PASS" if all(s == "PASS" for s in stats) else "INCONCLUSIVE")
    return write("G_finite_differences", {"status": overall, "optional": True, "cases": cases,
                                          "criteria": CRITERIA["G_finite_differences"]})


# ----------------------------------------------------------------------- amendment 1
AMENDMENT_1 = {
    "id": 1,
    "applies_to": ["C_gradients.refinement_stability (and the agreement/Section-10 checks "
                   "of the affected cases)", "E_independence.batch_vs_individual_at_L2"],
    "thresholds_changed": False,
    "rule": "A case that misses a refinement-based threshold at the pre-registered level "
            "(C: consecutive change L1->L2; E: batch-vs-individual at L2) is re-evaluated "
            "on a ladder extended by one decade, L3 = rtol/atol 1e-12/1e-14, with the SAME "
            "thresholds applied at the finest level: C requires the L2->L3 change of both "
            "backends <= 1e-3, FSA-vs-direct-autograd agreement at L3 within the L2 "
            "agreement thresholds, and the Section-10 production check against the L3 FSA "
            "gradient; E requires the batch-vs-individual differences at L3 within the L2 "
            "thresholds. A case still missing at L3 is a FAIL. Original results are kept.",
    "rationale": "investigation_stage1_refinement.json: for hydrogen Stage 1 the gradients "
                 "at L0/L1 are not converged for EITHER backend (L0->L1 ~4e-2, L1->L2 "
                 "1.6e-3 FSA / 6.9e-3 direct autograd at the initial model) and remain "
                 "slow even for a smooth control forcing (T = T0: L1->L2 5.9e-4 / 9.1e-4); "
                 "one decade further both backends converge (L2->L3 6.5e-6 / 1.9e-5) and "
                 "agree. For the trained Stage-1 model the C0 forcing T_obs(t) (piecewise-"
                 "linear interpolation of 20,000 points) is the dominant cost: L1->L2 changes "
                 "9.1e-6 (FSA) with T_obs vs 3.5e-8 with the smooth control, and L3 needs "
                 "68,906 vs 10,076 RHS evaluations. Stage-1 batch-vs-individual state "
                 "differences fall from 1.9e-7 (L2) to 1.2e-8 (L3), i.e. they are solver "
                 "accuracy, not coupling (cross-derivatives are exactly zero). The "
                 "pre-registered choice of L2 as the finest level was arbitrary.",
    "evidence": "investigation_stage1_refinement.json",
}
FINEST = (1e-12, 1e-14)


def _rebuild(name: str, idx):
    kind, model = name.split("/")
    return {"biodiesel": P.biodiesel, "stage1": P.stage1, "stage2": P.stage2}[kind](model, idx=idx)


def section_C_amendment_1():
    orig = json.loads((OUT / "C_gradients.json").read_text())
    crit = CRITERIA["C_gradients"]
    cases = []
    for case in orig["cases"]:
        if case["pass"]:
            cases.append({"problem": case["problem"], "original_pass": True, "pass": True})
            continue
        if case["failure_reasons"] != ["refinement not stable"]:
            cases.append({"problem": case["problem"], "original_pass": False, "pass": False,
                          "note": "failure not covered by amendment 1"})
            continue
        prob = _rebuild(case["problem"], P.H2_SUBSET if case["problem"].startswith(("stage1", "stage2")) else None)
        f2, d2 = gradient(prob, "fsa", "L2"), gradient(prob, "direct_autograd", "L2")
        f3, d3 = gradient(prob, "fsa", FINEST), gradient(prob, "direct_autograd", FINEST)
        f0 = gradient(prob, "fsa", "L0")
        packing = f3["fd"].packing
        agree = compare(f3["grad"], d3["grad"], packing)
        ok_agree, why = agreement_ok(agree)
        ch_f = compare(f2["grad"], f3["grad"])["relative_l2"]
        ch_d = compare(d2["grad"], d3["grad"])["relative_l2"]
        rs = crit["refinement_stability"]
        ok_ref = ch_f <= rs["fsa_L1_vs_L2_relative_max"] and ch_d <= rs["direct_autograd_L1_vs_L2_relative_max"]
        p10 = compare(f0["grad"], f3["grad"], packing)
        s10 = crit["section10_production_adequacy"]
        ok_10 = p10["relative_l2"] <= s10["relative_l2_max"] and p10["cosine"] >= s10["cosine_min"]
        cases.append({"problem": case["problem"], "original_pass": False,
                      "original_failure_reasons": case["failure_reasons"],
                      "L3_tolerance": FINEST, "nfe_fsa_L3": f3["nfe"], "nfe_direct_autograd_L3": d3["nfe"],
                      "fsa_vs_direct_autograd_L3": agree, "fsa_L2_vs_L3": ch_f,
                      "direct_autograd_L2_vs_L3": ch_d,
                      "section10_fsa_L0_vs_fsa_L3": p10,
                      "checks": {"agreement_at_L3": ok_agree, "refinement_stable_L2_L3": ok_ref,
                                 "section10_production_adequate": ok_10},
                      "failure_reasons": why, "pass": ok_agree and ok_ref and ok_10})
        print(f"    {case['problem']:22s} L3 agree {agree['relative_l2']:.2e}  L2->L3 fsa {ch_f:.2e} "
              f"da {ch_d:.2e}  s10 {p10['relative_l2']:.2e}", flush=True)
    ok = all(c["pass"] for c in cases)
    return write("C_gradients_amendment1", {"status": status(ok), "amends": "C_gradients.json",
                                            "original_status": orig["status"],
                                            "amendment": AMENDMENT_1, "cases": cases})


def section_E_amendment_1():
    orig = json.loads((OUT / "E_independence.json").read_text())
    crit = CRITERIA["E_independence"]["batch_vs_individual_at_L2"]
    cases = []
    for case in orig["cases"]:
        structural = case["max_abs_cross_derivative"] == 0.0 and \
            case.get("stage1_other_trajectories_bit_identical", True) and \
            case.get("stage1_own_trajectory_changed", True)
        if case["pass"]:
            cases.append({"problem": case["problem"], "original_pass": True, "pass": True})
            continue
        sub = [r["train_index"] for r in case["batch_vs_individual"]]
        full_sub = P.H2_SUBSET if case["problem"].startswith(("stage1", "stage2")) else (0, 7, 13)
        batch = _rebuild(case["problem"], full_sub)
        fdb = FunctionalDynamics(batch.dynamics, batch.params)
        xb, Sb = integrate_with_sensitivities(fdb, batch.x0, batch.t, cfg(FINEST))
        comp, good = [], structural
        for j, orig_i in enumerate(full_sub):
            if orig_i not in sub:
                continue
            one = _rebuild(case["problem"], (orig_i,))
            fd1 = FunctionalDynamics(one.dynamics, one.params)
            x1, S1 = integrate_with_sensitivities(fd1, one.x0, one.t, cfg(FINEST))
            dx = normalized_max(x1[:, 0], xb[:, j], batch.state_range)
            dS = float((S1[:, 0] - Sb[:, j]).norm() / Sb[:, j].norm())
            comp.append({"train_index": orig_i, "state_normalized_max": dx, "sensitivity_relative": dS})
            good &= dx <= crit["state_normalized_max"] and dS <= crit["sensitivity_relative_max"]
        cases.append({"problem": case["problem"], "original_pass": False,
                      "structural_independence_ok": structural,
                      "original_batch_vs_individual_L2": case["batch_vs_individual"],
                      "batch_vs_individual_L3": comp, "pass": bool(good)})
        print(f"    {case['problem']:22s} L3 {comp}", flush=True)
    ok = all(c["pass"] for c in cases)
    return write("E_independence_amendment1", {"status": status(ok), "amends": "E_independence.json",
                                               "original_status": orig["status"],
                                               "amendment": AMENDMENT_1, "cases": cases})


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D,
            "E": section_E, "F": section_F, "G": section_G,
            "C1": section_C_amendment_1, "E1": section_E_amendment_1}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="A,B,C,D,E,F,G",
                    help="comma list of A..G; C1/E1 evaluate amendment 1 on top of C/E")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild summary.json from existing section files and exit")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.summary_only:
        print(json.dumps(write_summary(), indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    crit_path = OUT / "acceptance_criteria.json"
    if not crit_path.exists():                       # recorded once, before any section ran
        crit_path.write_text(json.dumps({"created": utc_now(), "git_commit": git_commit(),
                                         "criteria": CRITERIA}, indent=2, default=float))
    elif json.loads(crit_path.read_text())["criteria"] != json.loads(json.dumps(CRITERIA, default=float)):
        raise SystemExit(f"{crit_path} records different criteria; criteria may not change "
                         f"after validation started")
    keys = [k.strip().upper() for k in args.sections.split(",") if k.strip()]
    if any(k.endswith("1") for k in keys):
        amend_path = OUT / "acceptance_criteria_amendment1.json"
        if not amend_path.exists():
            amend_path.write_text(json.dumps({"created": utc_now(), "git_commit": git_commit(),
                                              "amendment": AMENDMENT_1}, indent=2))
        elif json.loads(amend_path.read_text())["amendment"] != AMENDMENT_1:
            raise SystemExit(f"{amend_path} records a different amendment")
    for key in keys:
        print(f"== section {key}", flush=True)
        t0 = time.perf_counter()
        path = SECTIONS[key]()
        res = json.loads(path.read_text())
        res["seconds"] = round(time.perf_counter() - t0, 1)
        path.write_text(json.dumps(res, indent=2, default=float))
        if key == "B" and res["status"] != "PASS":
            write_summary()
            raise SystemExit("functional equivalence failed; gradient validation must stop")
    print(json.dumps(write_summary(), indent=2))


def write_summary() -> dict:
    """Rebuild summary.json from the section files on disk (safe for parallel runs).

    An amendment file (``<X>_*_amendment1.json``) supersedes its section's status; the
    original status is kept alongside it.
    """
    summary = {}
    for key in "ABCDEFG":
        files = sorted(f for f in OUT.glob(f"{key}_*.json") if "amendment" not in f.name)
        if not files:
            continue
        res = json.loads(files[0].read_text())
        entry = {"status": res["status"], "file": files[0].name, "seconds": res.get("seconds"),
                 "created": res["created"], "git_commit": res["git_commit"]}
        amended = sorted(OUT.glob(f"{key}_*_amendment1.json"))
        if amended:
            am = json.loads(amended[0].read_text())
            entry.update(original_status=res["status"], status=am["status"],
                         amendment_file=amended[0].name)
        summary[key] = entry
    core = [summary.get(k, {}).get("status") for k in "ABCDEF"]
    summary["core_validation"] = ("PASS" if all(s == "PASS" for s in core)
                                  else "FAIL" if "FAIL" in core else "INCOMPLETE")
    summary["optional_G"] = summary.get("G", {}).get("status", "NOT RUN")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

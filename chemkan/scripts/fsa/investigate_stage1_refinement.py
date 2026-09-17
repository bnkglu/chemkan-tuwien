r"""Investigation of the Section-C/E Stage-1 refinement result (DIAGNOSTIC).

Pre-registered validation found, for hydrogen Stage 1 only, gradients still changing by
>1e-3 between the L1 and L2 tolerance levels (both backends; stage1/init) and a
batch-vs-individual state difference of 1.9e-7 at L2 (criterion 1e-7).

Hypothesis: the Stage-1 forcing T_obs(t) is a piecewise-LINEAR interpolant of 20,000
Cantera points, i.e. only C0 in time with a slope jump every ~3e-8 s. Tsit5's order and
its local error estimate assume a smooth right-hand side, so convergence under tolerance
refinement is slower and less regular than for a smooth problem -- for direct autograd and
FSA alike, since both integrate the same forcing.

Test: extend the ladder by one decade (L3 = 1e-12/1e-14) on the real problem, and repeat
the whole ladder on a SMOOTH CONTROL: the same model and trajectories with T held at each
trajectory's initial temperature (a constant, C-infinity forcing). Nothing here changes
training; the control is never used to train.

Writes ``results/experiments/fsa/validation/investigation_stage1_refinement.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _problems as P                                               # noqa: E402
import validate_fsa as V                                            # noqa: E402
from _run import git_commit, utc_now                                # noqa: E402

from chemkan.dynamics import KineticDynamics                        # noqa: E402
from chemkan.fsa import FunctionalDynamics, integrate_with_sensitivities  # noqa: E402
from chemkan.temperature import ConstantTemperature                 # noqa: E402

LADDER = {**V.LEVELS, "L3": (1e-12, 1e-14)}


def smooth_control(prob: P.Problem) -> P.Problem:
    T0 = prob.dynamics.temperature.temperatures[0]                   # (B, 1) at t0
    dyn = KineticDynamics(prob.net.kinetic, ConstantTemperature(T0.clone()).to(T0.dtype),
                          input_normalizer=prob.dynamics.input_normalizer)
    return P.Problem(prob.name + " [smooth control: T = T0]", prob.net, dyn, prob.params,
                     prob.x0, prob.t, prob.loss_fn, prob.state_range, prob.meta)


def ladder(prob: P.Problem) -> dict:
    g = {}
    out = {"problem": prob.name, "levels": {}}
    for lv in LADDER:
        f = V.gradient(prob, "fsa", LADDER[lv])
        d = V.gradient(prob, "direct_autograd", LADDER[lv])
        g[lv] = (f["grad"], d["grad"])
        out["levels"][lv] = {"tolerance": LADDER[lv], "nfe_fsa": f["nfe"], "nfe_direct_autograd": d["nfe"],
                             "fsa_vs_direct_autograd": V.compare(f["grad"], d["grad"])["relative_l2"]}
    names = list(LADDER)
    out["consecutive_change"] = {
        f"{a}->{b}": {"fsa": V.compare(g[a][0], g[b][0])["relative_l2"],
                      "direct_autograd": V.compare(g[a][1], g[b][1])["relative_l2"]}
        for a, b in zip(names, names[1:])}
    out["vs_finest_fsa"] = {lv: {"fsa": V.compare(g[lv][0], g["L3"][0])["relative_l2"],
                                 "direct_autograd": V.compare(g[lv][1], g["L3"][0])["relative_l2"]}
                            for lv in names}
    print(json.dumps({k: out[k] for k in ("problem", "consecutive_change")}, indent=1), flush=True)
    return out


def batch_vs_individual(model="trained", levels=("L2", "L3")) -> list:
    rows = []
    for lv in levels:
        batch = P.stage1(model, idx=P.H2_SUBSET)
        fdb = FunctionalDynamics(batch.dynamics, batch.params)
        xb, Sb = integrate_with_sensitivities(fdb, batch.x0, batch.t, V.cfg(LADDER[lv]))
        for j, orig in enumerate(P.H2_SUBSET):
            one = P.stage1(model, idx=(orig,))
            fd1 = FunctionalDynamics(one.dynamics, one.params)
            x1, S1 = integrate_with_sensitivities(fd1, one.x0, one.t, V.cfg(LADDER[lv]))
            rows.append({"level": lv, "train_index": orig,
                         "state_normalized_max": V.normalized_max(x1[:, 0], xb[:, j], batch.state_range),
                         "sensitivity_relative": float((S1[:, 0] - Sb[:, j]).norm() / Sb[:, j].norm())})
    return rows


def main():
    res = {"created": utc_now(), "git_commit": git_commit(), "code_state": P.code_state(),
           "hypothesis": __doc__.split("Hypothesis:")[1].split("Test:")[0].strip(),
           "ladder": LADDER, "cases": []}
    for model in ("init", "trained"):
        prob = P.stage1(model, idx=P.H2_SUBSET)
        res["cases"].append(ladder(prob))
        res["cases"].append(ladder(smooth_control(P.stage1(model, idx=P.H2_SUBSET))))
    res["batch_vs_individual_stage1_trained"] = batch_vs_individual()
    out = V.OUT / "investigation_stage1_refinement.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"-> {out.relative_to(P.ROOT)}")


if __name__ == "__main__":
    main()

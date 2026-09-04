r"""Thermodynamic coefficient-intervention diagnostic (hydrogen).

DIAGNOSTIC ONLY -- not a paper reproduction, and it NEVER writes to the checkpoint or
the primary run directory.

For a trained checkpoint it compares:

    BASELINE      -- the checkpoint exactly as trained
    INTERVENTION  -- an identical model in which ONLY ``thermo.linear.weight`` is
                     replaced by Cantera-derived physical coefficients (-h_k/cp),
                     leaving the kinetic core, thermo.correction, normalizer, solver
                     and tolerances untouched

integrated from the initial condition alone at the Fig. 7 conditions. Because -h_k/cp is
state dependent, the intervention is repeated for several reference states (initial /
pre-ignition / ignition / post-ignition) to show whether the effect is robust.

    python3 chemkan/scripts/diagnostics/hydrogen_thermo_intervention.py \
        --run-dir results/reproduction/chemkan/hydrogen/main/base_off_direct_autograd_seed0
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[1]                       # chemkan/scripts
for _p in (str(_SCRIPTS.parent / "src"), str(_SCRIPTS), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _data import DATA_DIR, load_input_scaling, resolve_device       # noqa: E402
from _thermo_coeffs import MECH, coefficients_at_states              # noqa: E402
from evaluate_hydrogen import (build_chemkan, integrate_hydrogen,    # noqa: E402
                               solver_from_ckpt)

from chemkan.normalization import MinMaxNormalizer                   # noqa: E402

CONDITIONS = [(1050.0, 0.9, "training"), (1150.0, 1.3, "held-out")]


def ignition_delay(t, T, rise_threshold: float = 100.0):
    """Time of maximum dT/dt, or None if the trajectory never rises enough."""
    T = np.asarray(T, dtype=float)
    t = np.asarray(t, dtype=float)
    if float(T.max() - T[0]) < rise_threshold:
        return None
    return float(t[int(np.argmax(np.gradient(T, t)))])


def apply_thermo_coefficients(model, coeffs):
    """Return a DEEP COPY of ``model`` with only ``thermo.linear.weight`` replaced."""
    clone = copy.deepcopy(model)
    with torch.no_grad():
        w = clone.thermo.linear.weight
        w.copy_(torch.as_tensor(np.asarray(coeffs).reshape(w.shape), dtype=w.dtype))
    return clone


def changed_parameters(model_a, model_b, atol: float = 0.0) -> list[str]:
    """Names of parameters that differ between two models (provenance check)."""
    sa, sb = model_a.state_dict(), model_b.state_dict()
    assert sa.keys() == sb.keys(), "state_dict keys differ"
    return [k for k in sa if not torch.allclose(sa[k].float(), sb[k].float(), atol=atol)]


def evaluate_case(model, input_norm, solver, full_norm, t, ref, device="cpu") -> dict:
    """Integrate from the initial condition alone; report temperature + MSE metrics."""
    u0 = torch.as_tensor(ref[0], dtype=torch.float32, device=device).unsqueeze(0)
    tt = torch.as_tensor(t, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = integrate_hydrogen(model, input_norm, solver, u0, tt, device)[:, 0].cpu().numpy()
    Tp = pred[:, -1]
    dN = (full_norm.normalize(torch.as_tensor(pred, dtype=torch.float32))
          - full_norm.normalize(torch.as_tensor(ref, dtype=torch.float32))).numpy()
    per_state = (dN ** 2).sum(0)                                 # Eq. 18 per state
    delay = ignition_delay(t, Tp)
    return {
        "initial_T": float(Tp[0]), "peak_T": float(Tp.max()), "final_T": float(Tp[-1]),
        "T_rise": float(Tp.max() - Tp[0]),
        "ignition_delay_s": delay, "ignites": delay is not None,
        "trajectory_MSE": float(per_state.mean()), "temperature_MSE": float(per_state[-1]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="run directory holding checkpoint_final.pt")
    p.add_argument("--out-csv", default=None,
                   help="default: results/reproduction/tables/hydrogen_thermo_intervention.csv")
    p.add_argument("--mech", default=MECH)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = p.parse_args()

    dev = resolve_device(args.device)
    ckpt_path = Path(args.run_dir) / "checkpoint_final.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    npz = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    t, ics, states = npz["t"], npz["ics"], npz["states"]
    species = [str(s) for s in npz["species"]]
    full_norm = MinMaxNormalizer(torch.as_tensor(npz["u_min"], dtype=torch.float32),
                                 torch.as_tensor(npz["u_max"], dtype=torch.float32))

    baseline = build_chemkan(ckpt, len(species), dev)
    solver = solver_from_ckpt(ckpt)
    input_norm = load_input_scaling(ckpt, dev)
    learned = baseline.thermo.linear.weight.detach().cpu().numpy().ravel()

    print(f"checkpoint : {ckpt_path}")
    print(f"run_id     : {ckpt.get('run_id')}")
    print(f"species    : {species}")
    print("learned thermo.linear:", np.round(learned, 3).tolist())
    print()

    rows = []
    for T0, phi, role in CONDITIONS:
        i = int(np.argmin(np.abs(ics[:, 0] - T0) + np.abs(ics[:, 1] - phi)))
        ref = states[i]
        base = evaluate_case(baseline, input_norm, solver, full_norm, t, ref, dev)
        rows.append(dict(coefficient_reference_state="none (baseline, as trained)",
                         condition=f"{T0:.0f}/{phi}", role=role, T_ref_K="", cp_mass="",
                         **base))
        print(f"[{role}] BASELINE: peak {base['peak_T']:.0f} K, rise {base['T_rise']:.0f} K, "
              f"ignites={base['ignites']}, traj MSE {base['trajectory_MSE']:.3f}")

        for name, info in coefficients_at_states(t, ref, species=species, mech=args.mech).items():
            model_i = apply_thermo_coefficients(baseline, info["coeffs"])
            changed = changed_parameters(baseline, model_i)
            assert changed == ["thermo.linear.weight"], f"unexpected changes: {changed}"
            r = evaluate_case(model_i, input_norm, solver, full_norm, t, ref, dev)
            rows.append(dict(coefficient_reference_state=name, condition=f"{T0:.0f}/{phi}",
                             role=role, T_ref_K=round(info["T"], 1),
                             cp_mass=round(info["cp_mass"], 1), **r))
            print(f"[{role}] coeffs@{name:14s} (T={info['T']:6.0f} K): peak {r['peak_T']:6.0f} K, "
                  f"rise {r['T_rise']:6.0f} K, ignites={str(r['ignites']):5s}, "
                  f"traj MSE {r['trajectory_MSE']:.3f}")
        print()

    # the baseline model must be untouched by the interventions
    after = build_chemkan(ckpt, len(species), dev)
    assert changed_parameters(baseline, after) == [], "baseline model was mutated"
    print("verified: baseline model unchanged; only thermo.linear.weight was replaced.")

    out = Path(args.out_csv) if args.out_csv else (
        _SCRIPTS.parents[1] / "results/reproduction/tables/hydrogen_thermo_intervention.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

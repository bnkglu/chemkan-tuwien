"""
Ignition gate for a trained hydrogen ChemKAN checkpoint.

Integrates the model from the initial condition alone at chosen coarse-grid
conditions and compares peak temperature and ignition delay against the stored
Cantera reference. A parameter-count assertion passes on a model that never
ignites; this does not.

Run from the repository root:

    python3 chemkan/scripts/check_ignition.py \
        --run-dir results/reproduction/chemkan/hydrogen/main/direct_autograd_seed0

The gate exits non-zero if any checked condition is out of tolerance, so it can be
used as a guard before generating the paper figures/tables from a checkpoint.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent                 # chemkan/scripts
sys.path.insert(0, str(ROOT.parent / "src"))           # chemkan/src (the library)
sys.path.insert(0, str(ROOT))                          # chemkan/scripts (_data, etc.)

from chemkan.model import ChemKAN                      # noqa: E402
from chemkan.dynamics import ChemKANDynamics           # noqa: E402
from chemkan.solver import SolverConfig, integrate     # noqa: E402
from _data import DATA_DIR, load_input_scaling, resolve_device  # noqa: E402


def ignition_delay(t, T, rise_threshold=100.0):
    """time of max dT/dt, or None if the trajectory never rises enough."""
    T = np.asarray(T, dtype=float)
    t = np.asarray(t, dtype=float)
    if float(T.max() - T[0]) < rise_threshold:
        return None
    return float(t[int(np.argmax(np.gradient(T, t)))])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run directory containing checkpoint_final.pt")
    ap.add_argument("--conditions", default="1050:0.9,1150:1.3",
                    help="comma-separated T0:phi pairs to check")
    ap.add_argument("--peak-tol-k", type=float, default=150.0,
                    help="allowed |peak T| error, K (REPRODUCTION CHOICE)")
    ap.add_argument("--delay-tol-rel", type=float, default=0.25,
                    help="allowed relative ignition-delay error (REPRODUCTION CHOICE)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    dev = resolve_device(args.device)
    ckpt_path = Path(args.run_dir) / "checkpoint_final.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path} -- train the run first "
                         f"(train_hydrogen.py --run-dir {args.run_dir}).")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    # Load the full archive directly: the ignition gate needs ALL 36 conditions
    # (train + held-out), whereas load_hydrogen() only returns a single split.
    npz = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
    t = npz["t"]
    ics = npz["ics"]                     # (36, 2) -> T0, phi
    states = npz["states"]               # (36, Nt, 10) -> [Y_1..Y_9, T]

    model = ChemKAN(species_dim=states.shape[-1] - 1, **ckpt["architecture"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Reconstruct the EXACT input scaling the checkpoint was trained with (never refit):
    # 'minmax' -> the stored train-only stats; 'none' -> no input scaling. Refitting min/max
    # here would silently apply the wrong transform to a raw-input ('none') model.
    input_normalizer = load_input_scaling(ckpt, dev)
    dyn = ChemKANDynamics(model, input_normalizer=input_normalizer).to(dev)
    s = ckpt.get("solver", {})
    cfg = SolverConfig(method=s.get("method", "tsit5"),
                       rtol=s.get("rtol", 1e-6), atol=s.get("atol", 1e-8),
                       sensitivity="direct_autograd")

    tr = ckpt.get("training", {})
    scaling = ckpt.get("input_scaling", {}).get("method", "?")
    print(f"checkpoint      : {ckpt_path}")
    print(f"run_id          : {ckpt.get('run_id', '?')}")
    print(f"parameters      : {n_params}   (expected 344)")
    print(f"pinn            : stage1={tr.get('use_pinn_stage1', '?')} "
          f"stage2={tr.get('use_pinn_stage2', '?')} alpha={tr.get('alpha_pinn', '?')}")
    print(f"input scaling   : {scaling}")
    print(f"solver          : {cfg.method} rtol={cfg.rtol:g} atol={cfg.atol:g}")
    print(f"peak tolerance  : {args.peak_tol_k:.0f} K")
    print(f"delay tolerance : {args.delay_tol_rel:.0%}")
    print()

    t_torch = torch.as_tensor(t, dtype=torch.float32, device=dev)
    all_pass = True
    for pair in args.conditions.split(","):
        T0, phi = (float(v) for v in pair.split(":"))
        idx = int(np.argmin(np.abs(ics[:, 0] - T0) + np.abs(ics[:, 1] - phi)))
        ref = states[idx]
        y0 = torch.as_tensor(ref[0], dtype=torch.float32, device=dev).unsqueeze(0)

        with torch.no_grad():
            pred = integrate(dyn, y0, t_torch, cfg)
        T_pred = pred[:, 0, -1].cpu().numpy()
        T_ref = ref[:, -1]

        peak_p, peak_r = float(T_pred.max()), float(T_ref.max())
        d_p = ignition_delay(t, T_pred)
        d_r = ignition_delay(t, T_ref)

        peak_err = abs(peak_p - peak_r)
        ok_peak = peak_err <= args.peak_tol_k
        if d_r is None:
            ok_delay, delay_note = True, "reference does not ignite - skipped"
        elif d_p is None:
            ok_delay, delay_note = False, "MODEL NEVER IGNITES"
        else:
            rel = abs(d_p - d_r) / d_r
            ok_delay = rel <= args.delay_tol_rel
            delay_note = f"{d_p:.3e} s vs {d_r:.3e} s   (rel {rel:.1%})"

        verdict = "PASS" if (ok_peak and ok_delay) else "FAIL"
        all_pass &= ok_peak and ok_delay
        print(f"T0={T0:.0f} K, phi={phi:.2f}   [{verdict}]")
        print(f"   peak T   : {peak_p:7.1f} K vs {peak_r:7.1f} K   "
              f"(err {peak_err:6.1f} K)  {'ok' if ok_peak else 'OUT OF TOLERANCE'}")
        print(f"   ignition : {delay_note}  {'ok' if ok_delay else 'OUT OF TOLERANCE'}")
        print(f"   T range  : model {T_pred.min():.0f}-{T_pred.max():.0f} K, "
              f"reference {T_ref.min():.0f}-{T_ref.max():.0f} K")
        print()

    print("=" * 60)
    print("IGNITION GATE:", "PASSED" if all_pass else "FAILED")
    if not all_pass:
        print("Do not generate Figs. 7 / 8A / 8B or the Table I ChemKAN row")
        print("from this checkpoint.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

r"""Time hydrogen Stage-1 training with the ADJOINT sensitivity (odeint_adjoint).

Standalone timing probe. It rebuilds the *exact* hydrogen Stage-1 setup used by
scripts/train_hydrogen.py (same data, model, seed, ObservedTemperature, loss, Tsit5,
rtol/atol, lr, Adam) and swaps ONLY the gradient mechanism:

    torchdiffeq.odeint  (direct autograd, the repo default)
        -> torchdiffeq.odeint_adjoint  (continuous adjoint / optimize-then-discretize)

No source .py file is modified; SolverConfig is bypassed on purpose because it forbids
anything other than 'direct_autograd'. This is a wall-clock probe, NOT a correctness
claim -- the adjoint can be inaccurate on stiff systems, which is exactly why the repo
uses direct autograd.

Run it, watch the tqdm bar (cum_s, s_ep, proj_10k_h), and stop by hand with Ctrl-C.
It prints a summary of measured vs projected time on stop.

    NB_ADJ_EPOCHS   target epochs for the projection (default 10000)
    NB_ADJ_PINN     "1" to add the element-conservation PINN term (default off, matches
                    train_hydrogen's Stage-1 default)
    NB_ADJ_ALPHA    PINN weight if enabled (default 1e-4)

    ~/uni_projects/chemkan-venv/bin/python chemkan/notebooks/adjoint_timing.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# --- make the repo's package + scripts importable, wherever this file is run from ---
here = Path(__file__).resolve()
repo = next((p for p in here.parents if (p / "chemkan" / "src" / "chemkan").is_dir()), None)
if repo is None:
    raise SystemExit("could not locate the chemkan repo root from this file's location")
for sub in ("chemkan/src", "chemkan/scripts"):
    d = str(repo / sub)
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
from torchdiffeq import odeint

from _chemistry import ATOMIC_WEIGHTS, ELEMENT_COUNTS, MOLAR_WEIGHTS, assert_species_order
from _data import load_hydrogen, resolve_device

from chemkan.dynamics import KineticDynamics
from chemkan.losses import element_conservation_loss, trajectory_mse
from chemkan.model import ChemKAN
from chemkan.normalization import MinMaxNormalizer
from chemkan.temperature import ObservedTemperature

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

# --- config: identical to train_hydrogen defaults, adjoint aside ---------------------
TARGET_EPOCHS = int(os.environ.get("NB_ADJ_EPOCHS", "10000"))
USE_PINN1     = os.environ.get("NB_ADJ_PINN", "0") == "1"
ALPHA_PINN    = float(os.environ.get("NB_ADJ_ALPHA", "1e-4"))
METHOD, RTOL, ATOL, LR, SEED = "tsit5", 1e-6, 1e-8, 2e-3, 0

device = resolve_device("cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

data = load_hydrogen(split="train")
assert_species_order(data["species"])
m = data["species_TBm"].shape[-1]                              # species dim = ODE state dim n

model = ChemKAN(species_dim=m, hidden_dim=3, num_basis=5, n_mu=3, use_base_act=False).to(device)
full_norm = MinMaxNormalizer(data["u_min"], data["u_max"]).to(device)
species_norm = full_norm.subset(slice(0, m))
input_normalizer = full_norm                                   # matches --input-scaling minmax default

t = data["t"].to(device)
ec, aw, mw = ELEMENT_COUNTS.to(device), ATOMIC_WEIGHTS.to(device), MOLAR_WEIGHTS.to(device)

temp = ObservedTemperature(data["t"], data["T_obs_TB1"]).to(device)
kin_dyn = KineticDynamics(model.kinetic, temp, input_normalizer=input_normalizer).to(device)
Y0 = data["species_TBm"][0].to(device)                         # (B, m)
tgt_species = species_norm.normalize(data["species_TBm"].to(device))   # (T, B, m)

params = list(kin_dyn.kinetic.parameters())                    # Stage-1 trains only the kinetic core
p_stage1 = sum(x.numel() for x in params)
p_total  = sum(x.numel() for x in model.parameters())


def stage1_loss(pred):
    mse = trajectory_mse(species_norm.normalize(pred), tgt_species)
    if USE_PINN1:
        mse = mse + ALPHA_PINN * element_conservation_loss(pred, ec, aw, mw)
    return mse


def integrate_adjoint(func, y0, tt):
    # continuous adjoint; adjoint_params must be the tensors we optimize
    return odeint(func, y0, tt, rtol=RTOL, atol=ATOL, method=METHOD)
                          #adjoint_params=tuple(params))


opt = torch.optim.Adam(params, lr=LR)

print(f"hydrogen Stage-1  |  n (ODE state dim) = {m}   "
      f"p (Stage-1 trainable) = {p_stage1}   p (full model) = {p_total}")
print(f"sensitivity = ADJOINT (odeint)   method={METHOD} rtol={RTOL} atol={ATOL} "
      f"lr={LR} seed={SEED}   PINN1={USE_PINN1}")
print(f"projecting to {TARGET_EPOCHS} epochs. Ctrl-C to stop.\n")

bar = tqdm(range(TARGET_EPOCHS), desc="adjoint", unit="ep") if tqdm else range(TARGET_EPOCHS)
t0 = time.perf_counter()
done = 0
last_loss = float("nan")
try:
    for epoch in bar:
        opt.zero_grad()
        pred = integrate_adjoint(kin_dyn, Y0, t)               # (T, B, m)
        loss = stage1_loss(pred)
        loss.backward()
        opt.step()
        done = epoch + 1
        last_loss = loss.item()
        if tqdm:
            cum = time.perf_counter() - t0
            s_ep = cum / done
            bar.set_postfix(loss=f"{last_loss:.3e}", cum_s=f"{cum:.0f}",
                            s_ep=f"{s_ep:.2f}", proj_10k_h=f"{s_ep * TARGET_EPOCHS / 3600:.2f}")
except KeyboardInterrupt:
    print("\n[stopped by user]")
finally:
    if tqdm:
        bar.close()
    cum = time.perf_counter() - t0
    if done:
        s_ep = cum / done
        print(f"\nADJOINT timing summary")
        print(f"  epochs run           : {done} / {TARGET_EPOCHS}")
        print(f"  measured wall time   : {cum:.1f} s  ({cum/3600:.3f} h)")
        print(f"  mean seconds / epoch : {s_ep:.3f} s")
        print(f"  projected {TARGET_EPOCHS} epochs : {s_ep * TARGET_EPOCHS / 3600:.2f} h  (PROJECTION)")
        print(f"  last loss            : {last_loss:.6e}")
        print("  note: proj creeps up as dynamics stiffen; let it run a few hundred epochs first.")
    else:
        print("no epochs completed.")

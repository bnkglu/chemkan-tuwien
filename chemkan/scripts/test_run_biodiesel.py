"""Quick end-to-end test run: biodiesel kinetic stage (isothermal, Stage 1 only).

This script wires together ALL the library modules so you can verify the full
pipeline works.  It trains for just 200 epochs (a few seconds on CPU) and plots
the predicted vs. ground-truth trajectories for a single case.

Usage:
    python scripts/test_run_biodiesel.py
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import logging

# ── Make `import chemkan` work when running from scripts/ ────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from chemkan.model import KineticCore
from chemkan.dynamics import KineticDynamics
from chemkan.temperature import ConstantTemperature
from chemkan.normalization import MinMaxNormalizer
from chemkan.losses import trajectory_mse
from chemkan.solver import SolverConfig, integrate
from chemkan.training import train_kinetic_stage

# ── Local data loader ───────────────────────────────────────────────────
from _data import load_biodiesel

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
data = load_biodiesel("train")
t       = data["t"]              # (T=30,)
Y0      = data["Y0"][:2]         # Take only first 2 trajectories
Y_true  = data["species_TBm"][:, :2, :]
T_const = data["T_const"][:2]
species = data["species"]       # ['TG', 'ROH', 'DG', 'MG', 'GL', 'RCO2R']

m = Y0.shape[-1]  # 6 species

print(f"Loaded biodiesel data: {Y0.shape[0]} trajectories, {t.shape[0]} timesteps, {m} species")
print(f"Species: {species}")
print(f"Temperature range: {T_const.min():.0f} K – {T_const.max():.0f} K")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  BUILD NORMALIZER (for Eq. 18 MSE on [0,1]-normalized states)
# ═══════════════════════════════════════════════════════════════════════════
# The normalizer maps physical mass fractions to [0,1] using TRAIN-only statistics.
# We need a "full-state" normalizer that covers [Y, T] (m+1 columns).
# For the biodiesel (isothermal) experiment, we build it from species min/max
# and append the temperature range manually.
species_norm = MinMaxNormalizer(data["u_min"], data["u_max"])  # species-only (m,)
full_u_min = torch.cat([data["u_min"], T_const.min().unsqueeze(0)])
full_u_max = torch.cat([data["u_max"], T_const.max().unsqueeze(0)])
full_normalizer = MinMaxNormalizer(full_u_min, full_u_max)  # (m+1,)

# ═══════════════════════════════════════════════════════════════════════════
# 3.  BUILD MODEL  (Paper: hidden=13, num_basis=4, n_mu=3 for biodiesel)
# ═══════════════════════════════════════════════════════════════════════════
torch.manual_seed(42)

kinetic_core = KineticCore(
    species_dim=m,       # 6 species
    hidden_dim=4,       # paper value
    num_basis=4,         # paper value
    n_mu=2,              # paper value for LeanKANLayer
    use_base_act=False,  # matches paper parameter count (156)
)

temperature_provider = ConstantTemperature(T_const.unsqueeze(-1))  # (B, 1)

dynamics = KineticDynamics(
    kinetic_core=kinetic_core,
    temperature=temperature_provider,
    input_normalizer=full_normalizer,
)

n_params = sum(p.numel() for p in kinetic_core.parameters())
print(f"\nKineticCore parameters: {n_params}")

# --- Manual check of initial rates ---
with torch.no_grad():
    u_initial = full_normalizer.normalize(torch.cat([Y0, T_const.unsqueeze(-1)], dim=-1))
    dydt_initial = kinetic_core(u_initial)
print(f"\nManual forward pass rates (dY/dt) at t=0 for the 2 trajectories:")
print(dydt_initial)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  DEFINE LOSS (normalized MSE, Eq. 18 without PINN)
# ═══════════════════════════════════════════════════════════════════════════
def loss_fn(pred_Y_phys: torch.Tensor) -> torch.Tensor:
    """pred is (T, B, m) in physical coordinates; normalize then MSE."""
    pred_norm = species_norm.normalize(pred_Y_phys)
    target_norm = species_norm.normalize(Y_true)
    return trajectory_mse(pred_norm, target_norm)

# ═══════════════════════════════════════════════════════════════════════════
# 5.  TRAIN (200 epochs — just a quick test, paper uses 30 000)
# ═══════════════════════════════════════════════════════════════════════════
solver = SolverConfig(
    method="tsit5",
    rtol=1e-5,
    atol=1e-7,
    sensitivity="direct_autograd",
)

print("\n--- Training (200 epochs) ---")
final_loss = train_kinetic_stage(
    dynamics, Y0, t, loss_fn,
    epochs=2,
    lr=2e-3,
    solver=solver,
    log_every=1,
)
print(f"Final loss: {final_loss:.6e}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  PREDICT & PLOT (single trajectory, case 0)
# ═══════════════════════════════════════════════════════════════════════════
with torch.no_grad():
    pred = integrate(dynamics, Y0, t, solver)  # (T, B, m)

case_idx = 0
pred_case = pred[:, case_idx, :].numpy()
true_case = Y_true[:, case_idx, :].numpy()

print(f"\nFinal predicted mass fractions for case 0 at last timestep:")
for i, name in enumerate(species):
    print(f"  {name}: True={true_case[-1, i]:.4f}, Pred={pred_case[-1, i]:.4f}")

print("\nTest run completed successfully!")

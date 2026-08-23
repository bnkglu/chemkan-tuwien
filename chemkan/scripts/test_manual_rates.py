import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path

# Make `import chemkan` work when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from chemkan.model import KineticCore
from chemkan.normalization import MinMaxNormalizer
from _data import load_biodiesel

# 1. LOAD DATA
data = load_biodiesel("train")
Y0      = data["Y0"][:2]         # (2, 6)
Y_true  = data["species_TBm"][:, :2, :] # (30, 2, 6)
T_const = data["T_const"][:2]    # (2,)
t_steps = data["t"]              # (30,)

# 2. BUILD NORMALIZER
full_u_min = torch.cat([data["u_min"], data["T_const"].min().unsqueeze(0)])
full_u_max = torch.cat([data["u_max"], data["T_const"].max().unsqueeze(0)])
full_normalizer = MinMaxNormalizer(full_u_min, full_u_max)

# 3. BUILD MODEL
torch.manual_seed(42)
kinetic_core = KineticCore(
    species_dim=6,
    hidden_dim=4,
    num_basis=4,
    n_mu=2,
    use_base_act=False,
)

# 4. MANUAL FORWARD EULER TRAINING LOOP
optimizer = torch.optim.Adam(kinetic_core.parameters(), lr=1e-2, foreach=False)
loss_fn = nn.MSELoss()

epochs = 10
print("\nStarting Manual Forward Euler Training (10 Epochs) without torchdiffeq...")
print("Initial parameters:", sum(p.numel() for p in kinetic_core.parameters()))

for epoch in range(epochs):
    optimizer.zero_grad()
    
    Y_pred = [Y0]
    Y_curr = Y0
    
    # Forward Euler Integration
    for i in range(1, len(t_steps)):
        dt = t_steps[i] - t_steps[i-1]
        
        # Prepare input [Y, T]
        u_curr = full_normalizer.normalize(torch.cat([Y_curr, T_const.unsqueeze(-1)], dim=-1))
        
        # Predict rates (dY/dt)
        dY_dt = kinetic_core(u_curr)
        
        # Euler step: Y_next = Y_curr + dY/dt * dt
        Y_next = Y_curr + dY_dt * dt
        
        Y_pred.append(Y_next)
        Y_curr = Y_next
        
    Y_pred_stack = torch.stack(Y_pred, dim=0) # (30, 2, 6)
    
    # Compute Loss
    loss = loss_fn(Y_pred_stack, Y_true)
    
    # Backprop
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1:2d}/{epochs} - MSE Loss: {loss.item():.6f}")

print("\nManual training completed successfully!")

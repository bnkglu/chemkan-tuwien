r"""Smallest viable biodiesel DeepONet smoke test (ChemKAN Sec. III A 3 comparison).

This is a SMOKE TEST, not a reproduction run: it verifies that a biodiesel-shaped
DeepONet can be built, counted, run forward, scored with the SAME loss convention as
the ChemKAN runs, and trained for a few hundred epochs. It launches no scaling run and
writes no artifacts.

Eq. 4:  u(u0, t) = MLP_opt[ MLP_br(u0) (combine) MLP_tr(t) ]

``MLP_opt`` is ChemKAN's own addition -- neither upstream DeepONet implementation in
this vendored repo can emit a 6-species vector (both end in a scalar
``sum(branch*trunk) + b``). See ``deeponet_param_audit.py`` for that analysis and for
the cross-check that this layer-count convention agrees with ``dde.nn.FNN`` and
``seq2seq/learner/nn/FNN``.

**The literal paper-described architecture is the reconstruction used here**: branch
[7, 8, 8, 8], trunk [1, 7, 8], Hadamard combine, MLP_opt Linear(8, 6). It totals **340**
parameters against the **308** the paper reports -- a documented, unexplained discrepancy.
Architectures are NOT selected to match the reported count: alternative width/combine
choices that happen to total exactly 308 exist, but choosing one on that basis would fit
the architecture to a single number rather than to the paper's description. They are
listed for reference only and are not the default.

Usage
-----
    python biodiesel_deeponet_smoke.py                 # all checks + 200-epoch run
    python biodiesel_deeponet_smoke.py --epochs 0      # checks only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "chemkan" / "scripts"))

from _data import load_biodiesel                      # noqa: E402

from chemkan.losses import trajectory_mse             # noqa: E402
from chemkan.normalization import MinMaxNormalizer    # noqa: E402

PAPER_PARAMS = 308          # count stated in the ChemKAN paper for this DeepONet

# THE reconstruction: the architecture as the paper describes it. Branch input 7 =
# 6 species + T; trunk input 1 = t; MLP_opt maps the latent to the 6 species.
LITERAL = ((7, 8, 8, 8), (1, 7, 8), "hadamard")             # -> 340 params

# Count-matching variants, REPORTED FOR REFERENCE ONLY -- never selected as the model.
# Each reaches exactly 308 by departing from the described widths/combine in a different
# way, which is why the count alone cannot identify the intended architecture.
COUNT_MATCHING = {
    "T dropped from branch, concat": ((6, 8, 8),    (1, 7, 8), "concat"),
    "branch width 7 not 8":          ((7, 7, 7, 8), (1, 7, 8), "hadamard"),
}


def mlp(dims) -> nn.Sequential:
    """Biased Linear stack with ReLU between layers, no output activation."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BiodieselDeepONet(nn.Module):
    """u0:(B, branch_in), t:(T,) -> (T, B, 6) species."""

    def __init__(self, branch_dims, trunk_dims, combine="hadamard", out_dim=6):
        super().__init__()
        if branch_dims[-1] != trunk_dims[-1]:
            raise ValueError("branch and trunk must share the latent width")
        self.branch, self.trunk, self.combine = mlp(branch_dims), mlp(trunk_dims), combine
        latent = branch_dims[-1]
        self.opt = nn.Linear(latent if combine == "hadamard" else 2 * latent, out_dim)
        self.branch_in = branch_dims[0]

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = self.branch(u0)                       # (B, latent)
        tr = self.trunk(t.reshape(-1, 1))         # (T, latent)
        b, tr = b.unsqueeze(0), tr.unsqueeze(1)   # (1,B,L) (T,1,L)
        if self.combine == "hadamard":
            z = b * tr
        else:
            z = torch.cat([b.expand(tr.shape[0], -1, -1),
                           tr.expand(-1, b.shape[1], -1)], dim=-1)
        return self.opt(z)                        # (T, B, out_dim)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_inputs(data, branch_in: int):
    """Branch input is [Y0, T] (7); the 6-dim candidate_a branch drops T."""
    u0 = torch.cat([data["Y0"], data["T_const"].reshape(-1, 1)], dim=-1)   # (B, 7)
    return u0[:, :branch_in]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=200, help="0 = checks only")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    train = load_biodiesel(split="train")
    test = load_biodiesel(split="test")
    # The trunk consumes the DATASET's own time vector -- never a re-derived grid.
    t, truth = train["t"], train["species_TBm"]                 # (T,), (T, B, m)
    T_steps, B, m = truth.shape
    print(f"data: t {tuple(t.shape)}  species {tuple(truth.shape)}  "
          f"dt={float(t[1] - t[0]):.5f}s  t[0]={float(t[0]):.4f}  t[-1]={float(t[-1]):.4f}")
    print(f"      species={train['species']}")
    if T_steps != 30:
        print(f"      WARNING: expected the canonical 30-point biodiesel grid, got {T_steps}")

    # --- 1-4. dimensions and parameter count of the literal reconstruction ---------
    bd, td, comb = LITERAL
    model = BiodieselDeepONet(bd, td, comb, out_dim=m)
    n = n_params(model)
    print(f"\nliteral paper-described architecture (THE reconstruction used here):")
    print(f"  branch  FNN{bd}  trunk FNN{td}  combine={comb}  MLP_opt Linear({bd[-1]},{m})")
    print(f"  parameters = {n}   paper reports {PAPER_PARAMS}   "
          f"DISCREPANCY {n - PAPER_PARAMS:+d}  (documented, unexplained)")
    print("\n  count-matching variants (reference only, NOT used as the model):")
    for label, (b2, t2, c2) in COUNT_MATCHING.items():
        print(f"    {label:32s} branch{str(b2):14s} {c2:9s} -> "
              f"{n_params(BiodieselDeepONet(b2, t2, c2, out_dim=m))}")
    u0 = build_inputs(train, model.branch_in)
    assert u0.shape == (B, bd[0]), f"branch input {tuple(u0.shape)} != {(B, bd[0])}"
    assert td[0] == 1, "trunk input must be the scalar time coordinate"

    # --- 5. forward pass -----------------------------------------------------------
    pred = model(u0, t)
    assert pred.shape == truth.shape, f"output {tuple(pred.shape)} != {tuple(truth.shape)}"
    print(f"\nforward OK: branch {tuple(u0.shape)} + trunk ({T_steps}, 1) -> {tuple(pred.shape)}")

    # --- 6. loss, in the SAME convention as the ChemKAN runs ------------------------
    norm = MinMaxNormalizer(train["u_min"], train["u_max"])     # train-only stats (Eq. 18)
    def loss_of(pred_, truth_):
        return trajectory_mse(norm.normalize(pred_), norm.normalize(truth_))
    print(f"loss OK: normalized trajectory MSE at init = {loss_of(pred, truth).item():.6e}")

    if args.epochs == 0:
        return

    # --- 7. one short training run -------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = loss_of(model(u0, t), truth)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0:
            print(f"  epoch {epoch:5d}  train loss {loss.item():.6e}")
    with torch.no_grad():
        tr_mse = loss_of(model(u0, t), truth).item()
        te_mse = loss_of(model(build_inputs(test, model.branch_in), test["t"]),
                         test["species_TBm"]).item()
    print(f"\nliteral reconstruction: {args.epochs} epochs, {n_params(model)} params")
    print(f"  literal Eq. 18   train {tr_mse:.6e}  test {te_mse:.6e}")
    print(f"  time-averaged    train {tr_mse / T_steps:.6e}  test {te_mse / T_steps:.6e}")
    print("\nSMOKE TEST ONLY -- not a reproduction result. Nothing is written to disk.")


if __name__ == "__main__":
    main()

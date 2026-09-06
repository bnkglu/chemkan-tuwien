r"""Biodiesel DeepONet: the model, its width family, and its parameter count.

Single source of truth for the DeepONet baseline used in ChemKAN Figs. 4, 5 and 6. It is
a RECONSTRUCTION: the ChemKAN paper describes the architecture in prose (Sec. III A 3)
but tabulates neither every layer nor the optimizer, so the choices below are labelled by
where they come from.

Eq. 4:  u(u0, t) = MLP_opt[ MLP_br(u0) (Hadamard) MLP_tr(t) ]

PAPER-DESCRIBED
    7-dimensional branch input (6 initial concentrations + T); three branch layers of
    width 8; trunk 1 -> 7 -> 8; a final mapping from the 8-dimensional latent to the six
    species. That literal reading is ``COMPARISON_WIDTH`` and totals **340** trainable
    parameters against the **308** the paper reports -- a documented, unexplained
    discrepancy. Architectures are not selected to match 308; see
    ``biodiesel_deeponet_smoke.py`` for count-matching variants kept for reference only.

REPRODUCTION CHOICE (width family, Fig. 4)
    branch [7, w, w, w], trunk [1, w - 1, w], head Linear(w, 6). At w = 8 this reproduces
    the paper-described architecture exactly. The paper does not state a Fig. 4 width rule.

REPRODUCTION CHOICE (input/output scaling)
    The paper states no scaling for its DeepONet. Fed raw, the branch input mixes
    concentrations in [0, 2] with T ~ 334 K and the trunk sees t up to 30 s, which at
    Glorot initialization puts the initial Eq. 18 loss around 5e7 -- an artifact of
    conditioning, not of the architecture. So, matching what the ChemKAN kinetic core
    already receives:

        branch input : min-max normalized [Y0, T] using the SAME train-only statistics
        trunk input  : tau = t / t_end in [0, 1]
        output       : normalized species u_hat, compared directly with the normalized
                       targets, and denormalized for physical plots

    ``prepare_inputs`` is the single implementation of this transform, shared by the
    trainer and the evaluator, and its statistics are stored in the checkpoint so
    evaluation reconstructs it rather than refitting it.

REFERENCE-DERIVED REPRODUCTION CHOICES (from the bundled example
``deeponet/src/deeponet_dataset.py``, which builds ``dde.maps.OpNN(..., "relu",
"Glorot normal", use_bias=True, stacked=False)`` and trains with ``adam, lr=1e-3``)
    * biased Linear layers throughout, including the six-output head;
    * ReLU BETWEEN layers only -- no activation on the last branch layer, the last trunk
      layer, or the head;
    * Glorot-normal (Xavier normal) weight initialization, zero bias initialization,
      applied identically to branch, trunk and head;
    * Adam, lr = 1e-3.
    These are the reference example's conventions, not ChemKAN-paper facts. They are held
    fixed across every width and noise level.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PAPER_PARAMS = 308          # count stated in the ChemKAN paper for this DeepONet
COMPARISON_WIDTH = 8        # w reproducing the paper-described branch/trunk widths
OUT_DIM = 6                 # TG, ROH, DG, MG, GL, R'CO2R
BRANCH_IN = 7               # 6 initial concentrations + T


def dims_for(width: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(branch_dims, trunk_dims) for the reconstructed width family."""
    if width < 2:
        raise ValueError(f"width must be >= 2; got {width}")
    return (BRANCH_IN, width, width, width), (1, width - 1, width)


def mlp(dims) -> nn.Sequential:
    """Biased Linear stack, ReLU between layers, no output activation."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BiodieselDeepONet(nn.Module):
    """u0:(B, 7), t:(T,) -> (T, B, 6) species."""

    def __init__(self, width: int = COMPARISON_WIDTH, out_dim: int = OUT_DIM):
        super().__init__()
        branch_dims, trunk_dims = dims_for(width)
        if branch_dims[-1] != trunk_dims[-1]:
            raise ValueError("branch and trunk must share the latent width")
        self.width = width
        self.branch_dims, self.trunk_dims = branch_dims, trunk_dims
        self.branch, self.trunk = mlp(branch_dims), mlp(trunk_dims)
        self.head = nn.Linear(width, out_dim, bias=True)     # ChemKAN's MLP_opt

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = self.branch(u0).unsqueeze(0)                     # (1, B, w)
        tr = self.trunk(t.reshape(-1, 1)).unsqueeze(1)       # (T, 1, w)
        return self.head(b * tr)                             # (T, B, out_dim)


def glorot_normal_(model: nn.Module) -> nn.Module:
    """Glorot-normal weights, zero biases, on every Linear (the reference convention)."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def build(width: int = COMPARISON_WIDTH, seed: int | None = None) -> BiodieselDeepONet:
    """Deterministically construct and initialize the model."""
    if seed is not None:
        torch.manual_seed(seed)
    return glorot_normal_(BiodieselDeepONet(width))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture(model: BiodieselDeepONet) -> dict:
    """Checkpoint/config record of what was actually built."""
    return {
        "family": "reconstructed DeepONet: branch [7,w,w,w], trunk [1,w-1,w], "
                  "Hadamard combine, head Linear(w,6)",
        "width": model.width,
        "branch_dims": list(model.branch_dims),
        "trunk_dims": list(model.trunk_dims),
        "combine": "hadamard",
        "out_dim": model.head.out_features,
        "activation": "relu between layers only (none on branch/trunk output or head)",
        "bias": True,
        "weight_init": "glorot_normal (xavier_normal_), zero bias",
        "parameter_count": n_params(model),
    }


def raw_branch_input(data: dict) -> torch.Tensor:
    """Physical branch input [Y0, T] -> (B, 7), from a ``_data.load_biodiesel`` dict."""
    return torch.cat([data["Y0"], data["T_const"].reshape(-1, 1)], dim=-1)


def prepare_inputs(data: dict, full_normalizer, t_end: float):
    """(branch_input, tau) under the documented scaling choice.

    ``full_normalizer`` is the (m+1,) train-only min-max normalizer over [Y1..Ym, T];
    ``t_end`` is the training time window. Both come from the training split and are
    stored in the checkpoint -- never refitted on test data.
    """
    branch = full_normalizer.normalize(raw_branch_input(data).to(full_normalizer.u_min.device))
    tau = data["t"].to(branch.device) / t_end
    return branch, tau


if __name__ == "__main__":                                   # quick count audit
    for w in (2, 3, 4, 6, 8, 10, 12, 18):
        m = build(w)
        marker = "   <- paper-described widths" if w == COMPARISON_WIDTH else ""
        print(f"w={w:3d}  branch{list(m.branch_dims)}  trunk{list(m.trunk_dims)}  "
              f"params={n_params(m):5d}{marker}")
    print(f"\nat w={COMPARISON_WIDTH}: {n_params(build(COMPARISON_WIDTH))} parameters, "
          f"paper reports {PAPER_PARAMS} (documented, unexplained discrepancy)")

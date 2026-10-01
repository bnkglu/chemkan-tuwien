r"""Biodiesel DeepONet: the model, its width family, and its parameter count.

Single source of truth for the DeepONet baseline used in ChemKAN Figs. 4, 5 and 6. It is
a RECONSTRUCTION: the ChemKAN paper describes the architecture in prose (Sec. III A 3)
but tabulates neither every layer nor the optimizer, so the choices below are labelled by
where they come from.

Eq. 4:  u(u0, t) = MLP_opt[ MLP_br(u0) (Hadamard) MLP_tr(t) ]

PAPER-DESCRIBED + AUTHORS' CLARIFICATION (email B3)
    Branch input [TG0, ROH0, T0] (``BRANCH_IN = 3``): the four always-zero initial species
    are dropped. Three branch layers of width 8; trunk 1 -> 7 -> 8; a final mapping from the
    8-dimensional latent to the six species: 3*8+8 + 8*8+8 + 8*8+8 = 176 (branch) +
    1*7+7 + 7*8+8 = 78 (trunk) + 8*6+6 = 54 (head) = **308** parameters at
    ``COMPARISON_WIDTH``, the count the paper reports.
    LEGACY: every run before the clarification used the full 7-dimensional state
    [Y0 (6), T] (``LEGACY_BRANCH_IN = 7``, 340 parameters at w = 8). Legacy checkpoints
    record ``branch_dims[0] = 7`` and are rebuilt with it; they are never reinterpreted.

FIG. 4 SIZES: EXPLICIT ARCHITECTURES (w, q, p)
    branch [3, w, w, p], trunk [1, q, p], head Linear(p, 6); here p = w throughout.
    count = w*w + w*p + q*p + 5w + 2q + 8p + 6. ``FIG4_ARCHITECTURES`` lists the sweep:
    78 (3,3,3), 156 (5,5,5), 249 (7,6,7), 308 (8,7,8), 384 (9,9,9), 456 (10,10,10).
    Confirmed: the 308 architecture (email B3); 456 is the largest DeepONet and exactly two
    are larger than 308 (paper Sec. III A 2). Reconstructed (our choice): every other size
    and architecture. The trunk hidden width q defaults to w - 1.

LEGACY PREPROCESSING (input/output scaling) -- existing runs and checkpoints only.
    New reproduction runs use ``biodiesel_deeponet_repro.py`` (raw time t in the trunk,
    raw or author-global normalized states) and must not call ``prepare_inputs``.
    The upstream repository has no biodiesel case and therefore does not specify
    preprocessing for this 7-dimensional chemical input. DeepXDE 0.11.2 OpNN.build()
    supports a trunk _input_transform and an _output_transform (see source below),
    so transforms are supported reference behaviour. Our particular transforms are:

        branch input : min-max normalized [Y0, T] using the SAME train-only statistics
        trunk input  : tau = t / t_end in [0, 1]
        output       : normalized species u_hat, compared directly with the normalized
                       targets, and denormalized for physical plots

    ``prepare_inputs`` is the single implementation of this transform, shared by the
    trainer and the evaluator, and its statistics are stored in the checkpoint so
    evaluation reconstructs it rather than refitting it.
    ChemKAN uses paper-specified tanh normalization at each layer input (Sec. II C 1);
    Sec. III A 2-3 frames the experiments as a fair comparison. We retain DeepONet's
    scaling to avoid introducing a conditioning disadvantage while ChemKAN keeps its
    own normalization. These are distinct transforms, not a claim of identical scaling.
    Eq. 18's normalized loss is a separate convention from output parameterization.

REFERENCE-DERIVED REPRODUCTION CHOICES (from the bundled example
``deeponet/src/deeponet_dataset.py``, which builds ``dde.maps.OpNN(..., "relu",
"Glorot normal", use_bias=True, stacked=False)`` and trains with ``adam, lr=1e-3``)
    * biased Linear layers throughout, including the six-output head;
    * ReLU between branch layers, with a LINEAR final branch layer; ReLU after EVERY
      trunk layer, including the last; the six-output head remains LINEAR.
      DeepXDE 0.11.2 deepxde/maps/opnn.py, OpNN.build(), stacked=False, uses
      ``range(1, len(self.layer_size_loc))`` for the trunk, applying activation_trunk
      inside that loop, including the final layer:
      https://github.com/lululxvi/deepxde/blob/v0.11.2/deepxde/maps/opnn.py#L116-L158
    * Glorot-normal (Xavier normal) weight initialization, zero bias initialization,
      applied identically to branch, trunk and head;
    * Adam, lr = 1e-3.
    These are the reference example's conventions, not ChemKAN-paper facts. They are held
    fixed across every width and noise level.

ARCHITECTURE VERSIONS
    ``legacy_final_trunk_linear`` preserves the original missing final trunk ReLU.
    ``reference_final_trunk_relu`` fixes only that activation and is the default for
    new models. Checkpoint/config architecture records without a version are legacy;
    existing weights must never be reinterpreted as the corrected architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PAPER_PARAMS = 308          # count stated in the ChemKAN paper for this DeepONet
COMPARISON_WIDTH = 8        # w reproducing the paper-described branch/trunk widths
OUT_DIM = 6                 # TG, ROH, DG, MG, GL, R'CO2R
BRANCH_IN = 3               # [TG0, ROH0, T0] (authors' clarification, email B3)
LEGACY_BRANCH_IN = 7        # [Y0 (6), T]: every run before the clarification
BRANCH_INPUTS = {3: ("TG0", "ROH0", "T0"),
                 7: ("TG0", "ROH0", "DG0", "MG0", "GL0", "RCO2R0", "T0")}
BRANCH_COLUMNS = {3: [0, 1, 6], 7: [0, 1, 2, 3, 4, 5, 6]}   # columns of [Y0 (6), T]
# Fig. 4 DeepONet sweep: parameter count -> (w, q, p). Only 308 is confirmed (email B3).
FIG4_ARCHITECTURES = {78: (3, 3, 3), 156: (5, 5, 5), 249: (7, 6, 7), 308: (8, 7, 8),
                      384: (9, 9, 9), 456: (10, 10, 10)}


def count_parameters(w: int, q: int, p: int, branch_in: int = BRANCH_IN) -> int:
    """Branch [branch_in, w, w, p] + trunk [1, q, p] + head p -> 6, all biased."""
    return ((branch_in + 1) * w + (w + 1) * w + (w + 1) * p   # branch
            + 2 * q + (q + 1) * p                             # trunk
            + (p + 1) * OUT_DIM)                              # head
LEGACY_ARCHITECTURE_VERSION = "legacy_final_trunk_linear"
REFERENCE_ARCHITECTURE_VERSION = "reference_final_trunk_relu"
ARCHITECTURE_VERSIONS = (LEGACY_ARCHITECTURE_VERSION, REFERENCE_ARCHITECTURE_VERSION)


def dims_for(width: int, branch_in: int = BRANCH_IN,
             trunk_hidden: int | None = None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(branch_dims, trunk_dims): branch [branch_in, w, w, w], trunk [1, q, w], q = w - 1
    unless ``trunk_hidden`` is given."""
    if width < 2:
        raise ValueError(f"width must be >= 2; got {width}")
    if branch_in not in BRANCH_INPUTS:
        raise ValueError(f"branch_in must be one of {sorted(BRANCH_INPUTS)}; got {branch_in}")
    q = width - 1 if trunk_hidden is None else trunk_hidden
    if q < 1:
        raise ValueError(f"trunk_hidden must be >= 1; got {q}")
    return (branch_in, width, width, width), (1, q, width)


def mlp(dims, *, activate_output: bool = False) -> nn.Sequential:
    """Biased Linear stack, with an optional ReLU on its final layer."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
        if i < len(dims) - 2 or activate_output:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BiodieselDeepONet(nn.Module):
    """u0:(B, branch_in), t:(T,) -> (T, B, 6) species."""

    def __init__(self, width: int = COMPARISON_WIDTH, out_dim: int = OUT_DIM,
                 architecture_version: str = REFERENCE_ARCHITECTURE_VERSION,
                 branch_in: int = BRANCH_IN, trunk_hidden: int | None = None):
        super().__init__()
        if architecture_version not in ARCHITECTURE_VERSIONS:
            raise ValueError(f"unknown DeepONet architecture_version: {architecture_version!r}")
        branch_dims, trunk_dims = dims_for(width, branch_in, trunk_hidden)
        if branch_dims[-1] != trunk_dims[-1]:
            raise ValueError("branch and trunk must share the latent width")
        self.width = width
        self.architecture_version = architecture_version
        self.branch_dims, self.trunk_dims = branch_dims, trunk_dims
        self.branch = mlp(branch_dims)
        self.trunk = mlp(trunk_dims, activate_output=(
            architecture_version == REFERENCE_ARCHITECTURE_VERSION))
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


def build(width: int = COMPARISON_WIDTH, seed: int | None = None,
          architecture_version: str = REFERENCE_ARCHITECTURE_VERSION,
          branch_in: int = BRANCH_IN, trunk_hidden: int | None = None) -> BiodieselDeepONet:
    """Deterministically construct and initialize the model."""
    if seed is not None:
        torch.manual_seed(seed)
    model = glorot_normal_(BiodieselDeepONet(width, architecture_version=architecture_version,
                                             branch_in=branch_in, trunk_hidden=trunk_hidden))
    w, q, p = width, model.trunk_dims[1], width
    assert n_params(model) == count_parameters(w, q, p, branch_in), n_params(model)
    if branch_in == BRANCH_IN and (w, q, p) in FIG4_ARCHITECTURES.values():
        expected = next(n for n, a in FIG4_ARCHITECTURES.items() if a == (w, q, p))
        assert n_params(model) == expected, (n_params(model), expected)
    if width == COMPARISON_WIDTH and branch_in == BRANCH_IN and q == COMPARISON_WIDTH - 1:
        assert n_params(model) == PAPER_PARAMS, n_params(model)
    return model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture(model: BiodieselDeepONet) -> dict:
    """Checkpoint/config record of what was actually built."""
    return {
        "architecture_version": model.architecture_version,
        "family": f"DeepONet: branch [{model.branch_dims[0]},w,w,p], trunk [1,q,p], "
                  "Hadamard combine, head Linear(p,6)",
        "w": model.width, "q": model.trunk_dims[1], "p": model.trunk_dims[-1],
        "branch_input": list(BRANCH_INPUTS[model.branch_dims[0]]),
        "width": model.width,
        "branch_dims": list(model.branch_dims),
        "trunk_dims": list(model.trunk_dims),
        "combine": "hadamard",
        "out_dim": model.head.out_features,
        "activation": ("relu between layers only (none on branch/trunk output or head)"
                       if model.architecture_version == LEGACY_ARCHITECTURE_VERSION else
                       "relu between branch layers and after every trunk layer "
                       "(none on branch output or head)"),
        "bias": True,
        "weight_init": "glorot_normal (xavier_normal_), zero bias",
        "parameter_count": n_params(model),
    }


def raw_branch_input(data: dict, branch_in: int = BRANCH_IN) -> torch.Tensor:
    """Physical branch input from a ``_data.load_biodiesel`` dict: [TG0, ROH0, T] -> (B, 3)
    (default), or the legacy full [Y0, T] -> (B, 7)."""
    full = torch.cat([data["Y0"], data["T_const"].reshape(-1, 1)], dim=-1)
    return full[:, BRANCH_COLUMNS[branch_in]]


def prepare_inputs(data: dict, full_normalizer, t_end: float, branch_in: int = BRANCH_IN):
    """LEGACY: (branch_input, tau = t / t_end) for the existing runs and checkpoints.
    Not used by the reproduction path (``biodiesel_deeponet_repro.py``).

    ``full_normalizer`` is the (m+1,) train-only min-max normalizer over [Y1..Ym, T]; the
    branch columns are selected after normalizing, so each keeps its own statistics.
    ``t_end`` is the training time window. Both come from the training split and are
    stored in the checkpoint -- never refitted on test data.
    """
    full = torch.cat([data["Y0"], data["T_const"].reshape(-1, 1)], dim=-1)
    branch = full_normalizer.normalize(full.to(full_normalizer.u_min.device))
    branch = branch[:, BRANCH_COLUMNS[branch_in]]
    tau = data["t"].to(branch.device) / t_end
    return branch, tau


if __name__ == "__main__":                                   # quick count audit
    for n, (w, q, p) in FIG4_ARCHITECTURES.items():
        m = build(w, trunk_hidden=q)
        print(f"Fig. 4 {n:4d}: (w, q, p) = ({w}, {q}, {p}) -> {n_params(m)} parameters")
    for w in (2, 3, 4, 6, 8, 10, 12, 18):
        m = build(w)
        marker = "   <- paper-described widths" if w == COMPARISON_WIDTH else ""
        print(f"w={w:3d}  branch{list(m.branch_dims)}  trunk{list(m.trunk_dims)}  "
              f"params={n_params(m):5d}{marker}")
    print(f"\nat w={COMPARISON_WIDTH}: {n_params(build(COMPARISON_WIDTH))} parameters "
          f"(branch [TG0, ROH0, T0]); paper reports {PAPER_PARAMS}. Legacy 7-input: "
          f"{n_params(build(COMPARISON_WIDTH, branch_in=LEGACY_BRANCH_IN))}")

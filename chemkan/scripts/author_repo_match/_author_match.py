r"""Shared pieces of the authors-repo-matched biodiesel experiment.

Source of every setting here: the authors' RELEASED biodiesel example
(DENG-MIT/ChemKAN ``ChemKAN_biodiesel_example.jl``, commit d7aa5ab). This is a separate
controlled experiment, NOT the canonical paper reproduction, and it changes no library
default. The library's ``KineticCore`` is reused; only three released-example settings are
applied on top of it, all local to this module:

* first KAN layer without the input ``tanh`` (Julia ``KDense(...; normalizer=false)``);
  the second layer keeps its ``tanh`` (Julia ``normalizer=tanh_fast``);
* Julia's RBF ``exp(-((x - c)/h_J)^2)`` with ``h_J = 2/(G-1) = 1``. The library Gaussian is
  ``exp(-r^2 / (2 h^2))``, so ``h = h_J / sqrt(2)`` gives the identical function;
* dynamics in NORMALIZED coordinates ``z = [Y_hat, T_hat]`` (7 states, ``dT_hat/dt = 0``)
  with the learned right-hand side divided by 50: ``dz/dt = [KAN(z), 0] / 50``.

Julia layout of the flat parameter vector (ComponentArray): ``p[1:84]`` is layer-1 ``C``
(4 x 21), ``p[85:156]`` is layer-2 ``C`` (6 x 12), each column-major. Inside ``KDense`` the
basis rows are ordered basis-fastest, so ``C[o, i*G + g]`` (0-based) is the coefficient of
output ``o``, input ``i``, basis ``g``, i.e. the library's ``w_rbf[o, i, g]``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from chemkan.kan._common import gaussian
from chemkan.kan.rbf import RBFEdgeFunctions
from chemkan.model import KineticCore

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "results/experiments/biodiesel/author_repo_match"
JULIA_REF = EXP / "julia_reference"

SPECIES = ["TG", "ROH", "DG", "MG", "GL", "RCO2R"]
N_SPECIES, HIDDEN, NUM_BASIS, N_MU = 6, 4, 3, 2      # KDense(7,4,3) -> KDense(4,6,3), mult 2
RHS_DIVISOR = 50.0                                    # du .= vcat(kan1(u), 0) ./ 50
JULIA_RBF_H = 2.0 / (NUM_BASIS - 1)                  # Julia `denominator` = 1.0
LIBRARY_H = JULIA_RBF_H / math.sqrt(2.0)             # exp(-r^2/(2h^2)) == exp(-(r/h_J)^2)
N_TRAIN, N_TEST = 20, 10


class NoInputTanhRBFEdges(RBFEdgeFunctions):
    """``RBFEdgeFunctions`` without the input ``tanh`` (Julia ``normalizer=false``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in) -> (B, out, in)
        if self.w_base is not None:
            raise ValueError("the released example has no base activation")
        psi = gaussian(x, self.centers, self.h)
        return torch.einsum("bik,oik->boi", psi, self.w_rbf)


def build_core(hidden: int = HIDDEN) -> KineticCore:
    """7 -> H -> 6 kinetic core configured like the released example (default H = 4: 156
    parameters). n_mu = ceil(H/2) (LeanKAN rule, email B4), grid 3: 39 * H parameters."""
    n_mu = math.ceil(hidden / 2)
    core = KineticCore(species_dim=N_SPECIES, hidden_dim=hidden, num_basis=NUM_BASIS,
                       n_mu=n_mu, use_base_act=False)
    core.add.edges = NoInputTanhRBFEdges(N_SPECIES + 1, hidden, NUM_BASIS, use_base_act=False)
    for edges in (core.add.edges, core.lean.edges):
        edges.h = LIBRARY_H
        assert torch.equal(edges.centers, torch.tensor([-1.0, 0.0, 1.0]))
    assert sum(p.numel() for p in core.parameters()) == 39 * hidden
    return core


def julia_to_w_rbf(p: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat Julia parameters (156,) -> (layer-1 w_rbf (4,7,3), layer-2 w_rbf (6,4,3))."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if p.size != 156:
        raise ValueError(f"expected 156 parameters, got {p.size}")
    c1 = p[:84].reshape(HIDDEN, NUM_BASIS * (N_SPECIES + 1), order="F")
    c2 = p[84:].reshape(N_SPECIES, NUM_BASIS * HIDDEN, order="F")
    return (torch.from_numpy(c1.reshape(HIDDEN, N_SPECIES + 1, NUM_BASIS).copy()),
            torch.from_numpy(c2.reshape(N_SPECIES, HIDDEN, NUM_BASIS).copy()))


def load_julia_params(core: KineticCore, p: np.ndarray) -> None:
    w1, w2 = julia_to_w_rbf(p)
    with torch.no_grad():
        core.add.edges.w_rbf.copy_(w1)
        core.lean.edges.w_rbf.copy_(w2)


def glorot_uniform_edges_(edges_list, generator: torch.Generator) -> None:
    """Lux ``glorot_uniform`` on each ``C`` (out x G*in): U(-a, a), a = sqrt(6/(G*in + out)).

    Not ``torch.nn.init.xavier_uniform_``: on the library's (out, in, G) tensor that uses
    fan_out = out*G, whereas Julia's 2-D ``C`` has fan_out = out. Draws in list order.
    """
    with torch.no_grad():
        for edges in edges_list:
            out, n_in, g = edges.w_rbf.shape
            a = math.sqrt(6.0 / (g * n_in + out))
            u = torch.rand(edges.w_rbf.shape, generator=generator, dtype=edges.w_rbf.dtype)
            edges.w_rbf.copy_((2.0 * u - 1.0) * a)


def glorot_uniform_(core: KineticCore, generator: torch.Generator) -> None:
    """``glorot_uniform_edges_`` on a kinetic core's two layers (add, then lean)."""
    glorot_uniform_edges_((core.add.edges, core.lean.edges), generator)


class NormalizedDynamics(nn.Module):
    """``dz/dt = [KAN(z), 0] / 50`` on normalized ``z = [Y_hat (6), T_hat]`` (B, 7)."""

    def __init__(self, core: KineticCore):
        super().__init__()
        self.core = core

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        rates = self.core(z)
        return torch.cat([rates, torch.zeros_like(z[:, -1:])], dim=-1) / RHS_DIVISOR


def julia_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Released-example loss: Flux ``mse`` per trajectory (mean over species AND times),
    then mean over trajectories.  pred/target: (B, 6, T)."""
    return ((pred - target) ** 2).mean(dim=(1, 2)).mean()


def _flat_to_EST(a: np.ndarray, n_exp: int) -> np.ndarray:
    """Exported (E*S, T), row (e-1)*S + s  ->  (E, S, T)."""
    return a.reshape(n_exp, N_SPECIES, -1)


def load_julia_reference(path: Path = JULIA_REF) -> dict:
    """The exported released-example data, normalization and trained-checkpoint probes.

    Arrays that are Float32 in Julia are printed in shortest Float32 round-trip form, so
    they are parsed as float32 FIRST and then widened (parsing ``1.7911348`` straight to
    float64 would be off by up to half a Float32 ulp).
    """
    txt = lambda name: np.loadtxt(path / name)
    f32 = lambda name: np.loadtxt(path / name, dtype=np.float32).astype(np.float64)
    u0_norm = txt("u0_norm.txt")
    n_exp = u0_norm.shape[0]
    return {
        "p": f32("p.txt"),
        "u0_raw": txt("u0_raw.txt"),
        "u0_norm": u0_norm,
        "ode_data": _flat_to_EST(f32("ode_data.txt"), n_exp),
        "normdata": _flat_to_EST(f32("normdata.txt"), n_exp),
        "pred_final_p": _flat_to_EST(txt("pred_final_p.txt"), n_exp),
        "ymin": f32("ymin.txt"), "ymax": f32("ymax.txt"),
        "T_minmax": txt("T_minmax.txt"),
        "t": f32("tsteps.txt"),
        "states": txt("states.txt"), "layer1": txt("layer1.txt"),
        "kan_out": txt("kan_out.txt"), "rhs": txt("rhs.txt"),
        "list_loss_train": txt("list_loss_train.txt"),
        "list_loss_val": txt("list_loss_val.txt"),
        "list_loss_val_noisefree": txt("list_loss_val_noisefree.txt"),
    }

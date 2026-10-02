r"""Author-matched hydrogen ChemKAN (standalone; canonical ``train_hydrogen.py`` untouched).

Carries the conventions of the authors' RELEASED biodiesel example (DENG-MIT/ChemKAN d7aa5ab)
and the authors' email clarifications (items cited as in ``docs/authors_materials.md``) over
to the two-stage hydrogen case. No hydrogen code was released, so every carried-over choice
is labelled with its source in ``config.json["provenance"]``.

Model (344 parameters)
    KAN_kin   AddKAN 10 -> 3 (no input tanh) then LeanKAN 3 -> 9, n_mu = 3 (input tanh)
    thermo    Linear 9 -> 1, no bias, all weights 1e-5 (email H6)
    KAN_cor   AddKAN 10 -> 1 (no input tanh: its inputs are already min-max normalized)
    RBF       exp(-((x - c)/h)^2), c = linspace(-1, 1, 5), h = 2/(G - 1) = 0.5 (released code)
    init      Glorot uniform on every KAN coefficient block, bound sqrt(6/(G*in + out))

State and dynamics (normalized coordinates, training-set min/max of ``hydrogen.npz``)
    Stage 1   dY_hat/dt = KAN_kin([Y_hat, T_hat_obs(t)]) / t_ref
              T_hat_obs = dense Cantera temperature interpolant, normalized the same way
    Stage 2   du_hat/dt = [r, Linear(r) + KAN_cor(u_hat)] / t_ref,  r = KAN_kin(u_hat)
    1/t_ref is applied once, to the complete right-hand side; observation times stay in s.

Loss (Eq. 18 as stated for hydrogen, email H8)
    MSE mean over states, SUM over time, mean over trajectories (``trajectory_mse``)
    + alpha_pinn * element conservation on de-normalized species (summed over time),
    alpha_pinn = 1e-4 in both stages. MSE, PINN and MSE / N_t are logged separately.

    python train_author_matched_hydrogen.py stage1 --run-dir <dir> [--epochs 100000] [--resume]
    python train_author_matched_hydrogen.py stage2 --run-dir <dir> --stage1-archive <pt>
        [--warmup-epochs 0] [--epochs 100000] [--resume]

Stage 1 writes ``stage1_working.pt`` (resume), ``stage1_final.pt`` and a permanent archive
``archive/stage1_<run_id>_epoch<N>.pt`` that is never overwritten; extending Stage 1
(``--resume --epochs M``) writes a new archive. Any number of Stage-2 branches can start from
one archive: each copies the kinetic weights, freshly initializes thermo.linear and KAN_cor,
and writes to its own directory; the archive file is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))                                  # chemkan/scripts
from _author_match import NoInputTanhRBFEdges, glorot_uniform_edges_  # noqa: E402
from _chemistry import (ATOMIC_WEIGHTS, ELEMENT_COUNTS, MOLAR_WEIGHTS,  # noqa: E402
                        assert_species_order)
from _data import DATA_DIR, load_hydrogen, load_hydrogen_temperature  # noqa: E402
from _predictions import checkpoint_sha256  # noqa: E402
from _run import _atomic_torch_save, check_resume_config, git_commit, utc_now  # noqa: E402

from chemkan.losses import element_conservation_loss, trajectory_mse  # noqa: E402
from chemkan.model import ChemKAN  # noqa: E402
from chemkan.normalization import MinMaxNormalizer  # noqa: E402
from chemkan.solver import SolverConfig  # noqa: E402
from chemkan.temperature import ObservedTemperature  # noqa: E402
from chemkan.training import loss_and_gradients  # noqa: E402

logger = logging.getLogger("author_matched_hydrogen")

N_SPECIES, HIDDEN, NUM_BASIS, N_MU = 9, 3, 5, 3
N_PARAMS = 344
RELEASED_RBF_H = 2.0 / (NUM_BASIS - 1)                  # released code: h = grid spacing = 0.5
LIBRARY_H = RELEASED_RBF_H / math.sqrt(2.0)             # exp(-r^2/(2h^2)) == exp(-(r/h_rel)^2)
THERMO_LINEAR_INIT = 1e-5                               # email H6
ALPHA_PINN = 1e-4                                       # paper Sec. II C 4
LR = 2e-3                                               # paper Sec. II C 5; email H9
RTOL, ATOL = 1e-6, 1e-8                                 # current train_hydrogen.py defaults
T_REF_DEFAULT = 6e-4                                    # hydrogen time window [s] (inference)
DENSE_T_POINTS = 20000
HISTORY_COLUMNS = ["epoch", "loss", "mse", "pinn", "mse_time_avg", "elapsed_seconds"]
KINETIC_COLUMNS = ["kin_out_max_raw", "kin_out_max_scaled"]      # all phases
SPECIES_NAMES = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2"]
THERMO_COLUMNS = [f"w_{s}" for s in SPECIES_NAMES] + ["w_norm"]   # warm-up and Stage 2
MAX_COMBINED = {"kin_out_max_raw", "kin_out_max_scaled"}          # max, not sum, over units
SYSTEM = "hydrogen"           # set by configure_system(); hydrogen is the paper's case
SYSTEMS = ("hydrogen", "methane")
METHANE_T_REF = 5e-3          # methane time window [s] (same rule as hydrogen's 6e-4 s)


_HYDROGEN_DEFAULTS = (N_SPECIES, N_PARAMS, list(SPECIES_NAMES), list(THERMO_COLUMNS),
                      T_REF_DEFAULT, ELEMENT_COUNTS, ATOMIC_WEIGHTS, MOLAR_WEIGHTS)


def configure_system(name: str) -> None:
    """Select the chemical system. ``hydrogen`` (default) keeps every constant above.
    ``methane`` (optional extension, not in the paper): methane.npz / methane_temperature_20000.npz
    (GRI-Mech 3.0, 52 species), the element table from Cantera, the same architecture settings
    and loss; the parameter count follows from the 52-species width."""
    global SYSTEM, N_SPECIES, N_PARAMS, SPECIES_NAMES, THERMO_COLUMNS, T_REF_DEFAULT
    global ELEMENT_COUNTS, ATOMIC_WEIGHTS, MOLAR_WEIGHTS
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}")
    SYSTEM = name
    if name == "hydrogen":
        (N_SPECIES, N_PARAMS, SPECIES_NAMES, THERMO_COLUMNS, T_REF_DEFAULT, ELEMENT_COUNTS,
         ATOMIC_WEIGHTS, MOLAR_WEIGHTS) = _HYDROGEN_DEFAULTS
        return
    import numpy as np
    import cantera as ct
    d = np.load(DATA_DIR / "methane.npz")
    SPECIES_NAMES = [str(x) for x in d["species"]]
    N_SPECIES = len(SPECIES_NAMES)
    gas = ct.Solution(str(d["mechanism"]))
    elements = [e for e in gas.element_names
                if any(gas.n_atoms(sp, e) for sp in SPECIES_NAMES)]
    ELEMENT_COUNTS = torch.tensor([[gas.n_atoms(sp, e) for sp in SPECIES_NAMES]
                                   for e in elements], dtype=torch.float32)
    ATOMIC_WEIGHTS = torch.tensor([gas.atomic_weight(e) for e in elements], dtype=torch.float32)
    MOLAR_WEIGHTS = torch.tensor([gas.molecular_weights[gas.species_index(sp)]
                                  for sp in SPECIES_NAMES], dtype=torch.float32)
    THERMO_COLUMNS = [f"w_{sp}" for sp in SPECIES_NAMES] + ["w_norm"]
    T_REF_DEFAULT = METHANE_T_REF
    N_PARAMS = sum(p.numel() for p in ChemKAN(species_dim=N_SPECIES, hidden_dim=HIDDEN,
                                              num_basis=NUM_BASIS, n_mu=N_MU,
                                              use_base_act=False).parameters())

PROVENANCE = {
    "released_biodiesel_code": [
        "RBF exp(-((x-c)/h)^2) with h = 2/(G-1) (src/utils.jl, kdense_mult_add.jl)",
        "first KAN layer without input tanh, second with tanh (normalizer=false / tanh_fast)",
        "Glorot-uniform KAN coefficients, bound sqrt(6/(G*in+out)) (kdense_mult_add.jl:36)",
        "integration in min-max normalized coordinates",
        "time scaling: divide the right-hand side by the total time window (script line 146-147)",
    ],
    "email": [
        "G2 inputs min-max normalized to [0, 1]",
        "H2 no base activation, grid size 5",
        "H6 thermo linear weights initialized to 1e-5",
        "H7 optional warm-up of the linear thermo coefficients with the kinetic core frozen",
        "H8 loss not divided by the number of time steps (Eq. 18 sum over time)",
        "H9 1e5-2e5 epochs per stage; higher learning rates diverged",
    ],
    "paper": ["hidden width 3, n_mu = 3 (Sec. III B)", "alpha_pinn = 1e-4 (Sec. II C 4)",
              "Adam lr 2e-3, Tsit5 (Sec. II C 5)"],
    "our_inference": [
        "t_ref = hydrogen time window (6e-4 s): the released rule 'divide by the total time "
        "window' carried over; NOT an author statement for hydrogen",
        "KAN_cor without input tanh (its inputs are already normalized)",
        "thermo.linear acts on normalized rates (dY_hat/dt -> dT_hat/dt)",
        "solver tolerances from the current train_hydrogen.py, not the example's rtol 1e-2",
        "data generated with Cantera, not Arrhenius.jl CVODE_BDF (email H3)",
    ],
}


# --------------------------------------------------------------------------- model

def build_model() -> ChemKAN:
    """The 344-parameter hydrogen ChemKAN with the released-code KAN conventions."""
    model = ChemKAN(species_dim=N_SPECIES, hidden_dim=HIDDEN, num_basis=NUM_BASIS,
                    n_mu=N_MU, use_base_act=False)
    model.kinetic.add.edges = NoInputTanhRBFEdges(N_SPECIES + 1, HIDDEN, NUM_BASIS,
                                                  use_base_act=False)
    model.thermo.correction.edges = NoInputTanhRBFEdges(N_SPECIES + 1, 1, NUM_BASIS,
                                                        use_base_act=False)
    for edges in kan_edges(model):
        edges.h = LIBRARY_H
        assert torch.equal(edges.centers, torch.linspace(-1.0, 1.0, NUM_BASIS))
    assert model.thermo.linear.bias is None
    n = sum(p.numel() for p in model.parameters())
    assert n == N_PARAMS, f"expected {N_PARAMS} parameters, got {n}"
    return model


def kan_edges(model: ChemKAN) -> tuple:
    return (model.kinetic.add.edges, model.kinetic.lean.edges, model.thermo.correction.edges)


def init_kinetic_(model: ChemKAN, generator: torch.Generator) -> None:
    glorot_uniform_edges_((model.kinetic.add.edges, model.kinetic.lean.edges), generator)


def init_thermo_(model: ChemKAN, generator: torch.Generator) -> None:
    """thermo.linear = 1e-5 everywhere (email H6); KAN_cor Glorot."""
    with torch.no_grad():
        model.thermo.linear.weight.fill_(THERMO_LINEAR_INIT)
    glorot_uniform_edges_((model.thermo.correction.edges,), generator)


class Stage1Dynamics(nn.Module):
    """dY_hat/dt = KAN_kin([Y_hat, T_hat_obs(t)]) / t_ref."""

    def __init__(self, kinetic: nn.Module, temperature_hat: ObservedTemperature, t_ref: float):
        super().__init__()
        self.kinetic = kinetic
        self.temperature = temperature_hat
        self.t_ref = float(t_ref)

    def forward(self, t: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self.kinetic(torch.cat([y_hat, self.temperature(t)], dim=-1)) / self.t_ref


class Stage2Dynamics(nn.Module):
    """du_hat/dt = [r, Linear(r) + KAN_cor(u_hat)] / t_ref, r = KAN_kin(u_hat)."""

    def __init__(self, model: ChemKAN, t_ref: float):
        super().__init__()
        self.model = model
        self.t_ref = float(t_ref)

    def forward(self, t: torch.Tensor, u_hat: torch.Tensor) -> torch.Tensor:
        return self.model(u_hat) / self.t_ref


# --------------------------------------------------------------------------- data / loss

def _load_methane() -> tuple[dict, dict]:
    """methane.npz (train split) and its dense temperature, in the hydrogen loaders' layout,
    with the same consistency checks (time range, initial-condition order, batch size)."""
    import numpy as np
    d = np.load(DATA_DIR / "methane.npz")
    dn = np.load(DATA_DIR / f"methane_temperature_{DENSE_T_POINTS}.npz")
    full = torch.as_tensor(np.transpose(d["train_states"], (1, 0, 2)), dtype=torch.float32)
    data = {"t": torch.as_tensor(d["t"], dtype=torch.float32), "full_TBm1": full,
            "u_min": torch.as_tensor(d["u_min"], dtype=torch.float32),
            "u_max": torch.as_tensor(d["u_max"], dtype=torch.float32),
            "species": [str(x) for x in d["species"]]}
    t_dense = torch.as_tensor(dn["t"], dtype=torch.float32)
    T_dense = torch.as_tensor(dn["train_T"], dtype=torch.float32)
    if not (np.isclose(dn["t"][0], d["t"][0]) and np.isclose(dn["t"][-1], d["t"][-1])):
        raise ValueError("methane dense temperature: time range differs from methane.npz")
    if not np.allclose(dn["train_ics"], d["train_ics"]) or T_dense.shape[1] != full.shape[1]:
        raise ValueError("methane dense temperature: initial conditions differ from methane.npz")
    if not torch.all(t_dense[1:] > t_dense[:-1]) or not torch.isfinite(T_dense).all():
        raise ValueError("methane dense temperature: bad time grid or non-finite values")
    return data, {"t_dense": t_dense, "T_dense_TB1": T_dense}


def load_problem(dtype: torch.dtype) -> dict:
    """Training split of ``<system>.npz`` + the dense Stage-1 temperature, normalized."""
    if SYSTEM == "hydrogen":
        data = load_hydrogen(split="train")
        assert_species_order(data["species"])
        dense = load_hydrogen_temperature(split="train", n_points=DENSE_T_POINTS)
    else:
        data, dense = _load_methane()
        if data["species"] != SPECIES_NAMES:
            raise ValueError("methane species order differs from the configured element table")
    norm = MinMaxNormalizer(data["u_min"].to(dtype), data["u_max"].to(dtype))
    m = N_SPECIES
    t_path = DATA_DIR / f"{SYSTEM}_temperature_{DENSE_T_POINTS}.npz"
    return {
        "t": data["t"].to(dtype),
        "u_hat": norm.normalize(data["full_TBm1"].to(dtype)),                 # (T, B, m+1)
        "t_dense": dense["t_dense"].to(dtype),
        "T_dense_hat": norm.subset(slice(m, m + 1)).normalize(dense["T_dense_TB1"].to(dtype)),
        "norm": norm,
        "dataset": {"path": f"chemkan/data/generated/{SYSTEM}.npz", "split": "train",
                    "sha256": checkpoint_sha256(DATA_DIR / f"{SYSTEM}.npz"),
                    "n_points": int(data["t"].shape[0]),
                    "n_conditions": int(data["full_TBm1"].shape[1])},
        "dense_temperature": {"path": f"chemkan/data/generated/{t_path.name}",
                              "sha256": checkpoint_sha256(t_path), "n_points": DENSE_T_POINTS},
        "species": list(data["species"]),
    }


def make_loss_fn(target_hat: torch.Tensor, norm: MinMaxNormalizer, kinetic_probe=None,
                 t_ref: float = 1.0, scale: float = 1.0):
    """Eq. 18 (sum over time) + alpha * PINN on de-normalized species, times ``scale``.

    ``scale`` = 1/B for one trajectory of a per-trajectory epoch, so the summed losses and
    gradients equal the batched mean over trajectories. ``kinetic_probe(pred_hat)`` returns
    KAN_kin at the observation points (raw, before 1/t_ref); its max |.| is logged.
    """
    n_t = target_hat.shape[0]
    species_norm = norm.subset(slice(0, N_SPECIES))
    dt = target_hat.dtype
    ec, aw, mw = (x.to(dt) for x in (ELEMENT_COUNTS, ATOMIC_WEIGHTS, MOLAR_WEIGHTS))

    def loss_fn(pred_hat: torch.Tensor):
        mse = trajectory_mse(pred_hat, target_hat)
        y_phys = species_norm.denormalize(pred_hat[..., :N_SPECIES])
        pinn = element_conservation_loss(y_phys, ec, aw, mw)
        total = scale * (mse + ALPHA_PINN * pinn)
        comp = {"mse": scale * mse.detach(), "pinn": scale * pinn.detach(),
                "mse_time_avg": scale * mse.detach() / n_t}
        if kinetic_probe is not None:
            with torch.no_grad():
                kmax = kinetic_probe(pred_hat.detach()).abs().max()
            comp["kin_out_max_raw"] = kmax
            comp["kin_out_max_scaled"] = kmax / t_ref
        return total, comp
    return loss_fn


# --------------------------------------------------------------------------- config

def base_config(args, problem: dict) -> dict:
    norm = problem["norm"]
    return {
        "experiment": f"{SYSTEM}/author_matched",
        "seed": args.seed, "dtype": args.dtype, "sensitivity": args.sensitivity,
        "dataset": problem["dataset"], "dense_temperature": problem["dense_temperature"],
        "normalization": {"method": f"min-max, training-set statistics from {SYSTEM}.npz",
                          "columns": problem["species"] + ["T"],
                          "u_min": [float(v) for v in norm.u_min],
                          "u_max": [float(v) for v in norm.u_max]},
        "state_space": {"stage1": f"normalized species Y_hat ({N_SPECIES}); T_hat from the dense "
                                  "observed interpolant, normalized with the same statistics",
                        "stage2": f"normalized species and temperature u_hat ({N_SPECIES + 1})"},
        "t_ref": {"value_s": args.t_ref,
                  "rule": "du_hat/dt = RHS / t_ref, applied once to the complete RHS"},
        "architecture": {"hidden_dim": HIDDEN, "num_basis": NUM_BASIS, "n_mu": N_MU,
                         "use_base_act": False, "parameter_count": N_PARAMS,
                         "rbf": "exp(-((x-c)/h)^2), c = linspace(-1, 1, 5), h = 0.5 "
                                "(library gaussian with h = 0.5/sqrt(2))",
                         "input_tanh": {"kinetic_layer1": False, "kinetic_layer2": True,
                                        "correction": False}},
        "initialization": {"kan": "Glorot uniform, bound sqrt(6/(G*in+out)), torch.Generator "
                                  "seeded with 'seed'",
                           "thermo_linear": THERMO_LINEAR_INIT, "thermo_linear_bias": None},
        "loss": {"mse": "Eq. 18: mean over states, sum over time, mean over trajectories",
                 "pinn": "element conservation on de-normalized species, summed over time",
                 "alpha_pinn": ALPHA_PINN, "logged": ["mse", "pinn", "mse_time_avg"]},
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8},
        "solver": {"method": "tsit5", "rtol": args.rtol, "atol": args.atol,
                   "library": "torchdiffeq",
                   "solve_mode": args.solve_mode,
                   "solve_mode_note": "per_trajectory: one solve per trajectory, each loss "
                                      "weighted 1/B so loss and gradient equal the batched "
                                      "Eq. 18 mean over trajectories"},
        "history_extra": {"all_phases": "max |KAN_kin| over batch/species at the observation "
                                        "points, raw and after 1/t_ref",
                          "warmup_and_stage2": f"the {N_SPECIES} thermo.linear weights and "
                                               "their norm"},
        "provenance": PROVENANCE,
        **_overrides(args),
        **({"system": {"name": SYSTEM, "note": "optional extension, not in the paper; "
                       "element table from Cantera " + _mechanism_name()}}
           if SYSTEM != "hydrogen" else {}),
        **({"compile": "torch.compile on the ODE right-hand side (not bit-identical to eager)"}
           if getattr(args, "compile", False) else {}),
        "git_commit": git_commit(), "created": utc_now(),
        "python": platform.python_version(), "torch": torch.__version__,
    }


def _mechanism_name() -> str:
    import numpy as np
    return str(np.load(DATA_DIR / f"{SYSTEM}.npz")["mechanism"])


def _maybe_compile(units: list, args) -> list:
    """torch.compile the ODE right-hand side of every unit (hydrogen only)."""
    if not getattr(args, "compile", False):
        return units
    return [(torch.compile(func), y0, loss_fn) for func, y0, loss_fn in units]


def _overrides(args) -> dict:
    """A ``deviations`` entry only when lr or tolerances differ from the defaults, so the
    configs of default runs stay unchanged (and resumable)."""
    dev = {}
    if args.lr != LR:
        dev["lr"] = (f"{args.lr:g} instead of the paper's {LR:g}; 1e-2 is the released "
                     "biodiesel example's Flux.Adam(1f-2)")
    if (args.rtol, args.atol) != (RTOL, ATOL):
        dev["solver_tolerances"] = (f"rtol {args.rtol:g}, atol {args.atol:g} instead of "
                                    f"{RTOL:g}, {ATOL:g}; rtol 1e-2 / atol 1e-6 is the released "
                                    "biodiesel example (ODEProblem reltol=1e-2, solve abstol=1e-6)")
    return {"deviations": dev} if dev else {}


IMPORT_KEYS = ("dataset", "state_space", "architecture", "normalization", "t_ref")


def check_stage1_import(archive_config: dict, current: dict) -> None:
    """Refuse a Stage-1 archive whose dataset, state space, architecture, normalization or
    t_ref differ from the current run."""
    for key in IMPORT_KEYS:
        a, c = archive_config.get(key), current.get(key)
        if key == "dataset":
            a = {k: v for k, v in (a or {}).items() if k != "path"}
            c = {k: v for k, v in (c or {}).items() if k != "path"}
        if a != c:
            raise SystemExit(f"Stage-1 archive rejected: '{key}' differs\n"
                             f"    archive = {a!r}\n    current = {c!r}")


# --------------------------------------------------------------------------- loop

def _history(path: Path, keep_through: int, columns: list[str]) -> csv.DictWriter:
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= keep_through]
    fh = open(path, "w", newline="")  # noqa: SIM115  (kept open for the phase)
    w = csv.DictWriter(fh, fieldnames=columns)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    fh.flush()
    w.file = fh
    return w


def epoch_gradients(units, t, params, solver) -> dict:
    """One gradient evaluation over all units (func, y0, loss_fn); gradients accumulate in
    ``.grad``. Loss parts are summed over units (each already weighted), maxima maxed."""
    vals: dict[str, float] = {}
    for func, y0, loss_fn in units:
        loss, comp = loss_and_gradients(func, y0, t, params, loss_fn, solver)
        for k, v in {"loss": loss.detach(), **comp}.items():
            v = float(v)
            vals[k] = max(vals.get(k, v), v) if k in MAX_COMBINED else vals.get(k, 0.0) + v
    return vals


def thermo_weights(model: ChemKAN) -> dict:
    w = model.thermo.linear.weight.detach().reshape(-1)
    return {**{f"w_{s}": float(v) for s, v in zip(SPECIES_NAMES, w)}, "w_norm": float(w.norm())}


def run_phase(name, units, t, params, opt, solver, *, start, total, history,
              on_checkpoint, checkpoint_every, prior_elapsed, stop_after=None,
              extra=None) -> tuple[int, float]:
    """Epochs start+1..total: loss (and ``extra()``, e.g. thermo weights) at the epoch's
    starting parameters, then one Adam step. Returns (last completed epoch, elapsed s)."""
    t0 = time.perf_counter()
    epoch = start
    for epoch in range(start + 1, total + 1):
        opt.zero_grad(set_to_none=True)
        vals = epoch_gradients(units, t, params, solver)
        if not all(math.isfinite(v) for v in vals.values()):
            raise FloatingPointError(f"{name}: non-finite loss at epoch {epoch}: {vals}")
        if extra is not None:
            vals.update(extra())
        opt.step()
        elapsed = prior_elapsed + time.perf_counter() - t0
        history.writerow({"epoch": epoch, **vals, "elapsed_seconds": round(elapsed, 4)})
        history.file.flush()
        if epoch % 100 == 0 or epoch == start + 1:
            logger.info("%s epoch %d  loss %.4e  mse %.4e  pinn %.4e  (%.0f s)", name, epoch,
                        vals["loss"], vals["mse"], vals["pinn"], elapsed)
        if stop_after is not None and epoch - start >= stop_after:
            on_checkpoint(epoch, elapsed)                    # as if killed right after a save
            return epoch, elapsed
        if checkpoint_every and epoch % checkpoint_every == 0 and epoch < total:
            on_checkpoint(epoch, elapsed)
    return epoch, prior_elapsed + time.perf_counter() - t0


def stage1_units(model: ChemKAN, problem: dict, t_ref: float, solve_mode: str) -> list:
    """(func, y0, loss_fn) per solve. Batched: one unit; per_trajectory: one per trajectory,
    each with its own slice of the observed temperature and loss weight 1/B."""
    target = problem["u_hat"][..., :N_SPECIES]
    B = target.shape[1]
    groups = [slice(0, B)] if solve_mode == "batched" else [slice(i, i + 1) for i in range(B)]
    units = []
    for g in groups:
        temp = ObservedTemperature(problem["t_dense"], problem["T_dense_hat"][:, g])
        T_obs = torch.stack([temp(tj) for tj in problem["t"]])            # (T, b, 1)
        probe = (lambda y, T_obs=T_obs: kinetic_on_trajectory(model, torch.cat([y, T_obs], -1)))
        loss_fn = make_loss_fn(target[:, g], problem["norm"], probe, t_ref,
                               scale=(g.stop - g.start) / B)
        units.append((Stage1Dynamics(model.kinetic, temp, t_ref), target[0, g], loss_fn))
    return units


def stage2_units(model: ChemKAN, problem: dict, t_ref: float, solve_mode: str) -> list:
    target = problem["u_hat"]
    B = target.shape[1]
    groups = [slice(0, B)] if solve_mode == "batched" else [slice(i, i + 1) for i in range(B)]
    func = Stage2Dynamics(model, t_ref)
    probe = (lambda u: kinetic_on_trajectory(model, u))
    return [(func, target[0, g],
             make_loss_fn(target[:, g], problem["norm"], probe, t_ref,
                          scale=(g.stop - g.start) / B)) for g in groups]


def kinetic_on_trajectory(model: ChemKAN, u_hat: torch.Tensor) -> torch.Tensor:
    """KAN_kin (raw, before 1/t_ref) at every (time, trajectory): (T, B, m+1) -> (T, B, m)."""
    return model.kinetic(u_hat.reshape(-1, u_hat.shape[-1])).reshape(*u_hat.shape[:-1], -1)


def _adam(params, lr: float = LR) -> torch.optim.Adam:
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def _solver(sensitivity: str, rtol: float = RTOL, atol: float = ATOL) -> SolverConfig:
    return SolverConfig(method="tsit5", rtol=rtol, atol=atol, sensitivity=sensitivity)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


# --------------------------------------------------------------------------- stage 1

def run_stage1(args, problem: dict, run_dir: Path, *, stop_after: int | None = None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    working, final = run_dir / "stage1_working.pt", run_dir / "stage1_final.pt"
    run_id = run_dir.name
    cfg = {**base_config(args, problem), "stage": 1, "run_id": run_id,
           "epochs": {"stage1": args.epochs}}
    model = build_model()
    params = list(model.kinetic.parameters())
    opt = _adam(params, args.lr)
    if args.resume:
        state = torch.load(working, weights_only=False)
        check_resume_config(state["config"], cfg)
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["rng_state"])
        start, prior = state["epochs_completed"], state["elapsed_seconds"]
        if args.epochs < start:
            raise SystemExit(f"--epochs {args.epochs} < completed {start}")
        logger.info("stage 1 resumed at epoch %d", start)
    else:
        if any(f.name != "run.log" for f in run_dir.iterdir()):
            raise SystemExit(f"{run_dir} is not empty; pass --resume or choose another dir")
        gen = torch.Generator().manual_seed(args.seed)
        init_kinetic_(model, gen)
        init_thermo_(model, gen)          # unused in Stage 1; fixed by the seed, not global RNG
        start, prior = 0, 0.0
    _write_json(run_dir / "config.json", cfg)

    units = _maybe_compile(stage1_units(model, problem, args.t_ref, args.solve_mode), args)

    def save(path, epoch, elapsed):
        _atomic_torch_save({"stage": 1, "run_id": run_id, "config": cfg,
                            "model_state": model.state_dict(),
                            "optimizer_state": opt.state_dict(),
                            "rng_state": torch.get_rng_state(),
                            "epochs_completed": epoch, "elapsed_seconds": elapsed}, path)

    def write_archive(epoch, elapsed) -> Path:
        archive = run_dir / "archive" / f"stage1_{run_id}_epoch{epoch}.pt"
        registry_path = run_dir / "stage1_archives.json"
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
        if archive.exists():
            logger.info("archive %s already exists; left unchanged", archive.name)
        else:
            archive.parent.mkdir(exist_ok=True)
            save(archive, epoch, elapsed)
            registry.append({"file": f"archive/{archive.name}", "epoch": epoch,
                             "sha256": checkpoint_sha256(archive), "created": utc_now()})
            _write_json(registry_path, registry)
            logger.info("stage 1 archive %s  sha256 %s", archive.name, registry[-1]["sha256"])
        return archive

    def on_checkpoint(e, s):
        save(working, e, s)
        if args.archive_every and e % args.archive_every == 0:
            write_archive(e, s)

    history = _history(run_dir / "stage1_history.csv", start, HISTORY_COLUMNS + KINETIC_COLUMNS)
    epoch, elapsed = run_phase("stage1", units, problem["t"], params, opt,
                               _solver(args.sensitivity, args.rtol, args.atol), start=start,
                               total=args.epochs, history=history,
                               checkpoint_every=args.checkpoint_every,
                               on_checkpoint=on_checkpoint,
                               prior_elapsed=prior, stop_after=stop_after)
    history.file.close()
    result = {"epochs_completed": epoch, "archive": None}
    if epoch < args.epochs:
        return result
    save(working, epoch, elapsed)
    save(final, epoch, elapsed)
    result["archive"] = write_archive(epoch, elapsed)
    return result


# --------------------------------------------------------------------------- stage 2

def run_stage2(args, problem: dict, run_dir: Path, *, stop_after: int | None = None) -> dict:
    """Optional warm-up (thermo.linear only), then full Stage 2 with a fresh Adam state."""
    run_dir.mkdir(parents=True, exist_ok=True)
    working, final = run_dir / "stage2_working.pt", run_dir / "stage2_final.pt"
    run_id = run_dir.name
    cfg = {**base_config(args, problem), "stage": 2, "run_id": run_id,
           "epochs": {"stage2": args.epochs},
           "warmup": {"epochs": args.warmup_epochs,
                      "trainable": "thermo.linear only; kinetic core and KAN_cor frozen "
                                   "(requires_grad False, still in the trajectory graph)",
                      "optimizer": "fresh Adam for warm-up, fresh Adam again for Stage 2"}}
    model = build_model()
    if args.resume:
        state = torch.load(working, weights_only=False)
        cfg["stage1_archive"] = state["config"]["stage1_archive"]
        check_resume_config(state["config"], cfg)
        model.load_state_dict(state["model_state"])
        torch.set_rng_state(state["rng_state"])
        phase, done = state["phase"], state["phase_epochs_completed"]
        warm_done = state["warmup_epochs_completed"]
        prior = state["elapsed_seconds"]
        opt_state = state["optimizer_state"]
        logger.info("stage 2 resumed in %s at epoch %d", phase, done)
    else:
        if any(f.name != "run.log" for f in run_dir.iterdir()):
            raise SystemExit(f"{run_dir} is not empty; pass --resume or choose another dir")
        archive_path = Path(args.stage1_archive)
        archive_sha = checkpoint_sha256(archive_path)
        arch = torch.load(archive_path, weights_only=False)
        check_stage1_import(arch["config"], cfg)
        kin = {k[len("kinetic."):]: v.clone() for k, v in arch["model_state"].items()
               if k.startswith("kinetic.")}
        model.kinetic.load_state_dict(kin)                    # own evolving copy
        init_thermo_(model, torch.Generator().manual_seed(args.seed))
        cfg["stage1_archive"] = {"path": str(archive_path), "sha256": archive_sha,
                                 "run_id": arch["run_id"],
                                 "epochs_completed": arch["epochs_completed"]}
        phase, done, warm_done, prior, opt_state = "warmup", 0, 0, 0.0, None
    _write_json(run_dir / "config.json", cfg)

    units = _maybe_compile(stage2_units(model, problem, args.t_ref, args.solve_mode), args)
    columns = HISTORY_COLUMNS + KINETIC_COLUMNS + THERMO_COLUMNS
    solver = _solver(args.sensitivity, args.rtol, args.atol)
    frozen = list(model.kinetic.parameters()) + list(model.thermo.correction.parameters())

    def save(path, ph, epoch, warm, elapsed, opt):
        _atomic_torch_save({"stage": 2, "run_id": run_id, "config": cfg,
                            "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                            "rng_state": torch.get_rng_state(), "phase": ph,
                            "phase_epochs_completed": epoch, "warmup_epochs_completed": warm,
                            "elapsed_seconds": elapsed}, path)

    elapsed = prior
    if phase == "warmup" and args.warmup_epochs > 0 and done < args.warmup_epochs:
        for p in frozen:
            p.requires_grad_(False)
        params = [model.thermo.linear.weight]
        opt = _adam(params, args.lr)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        history = _history(run_dir / "warmup_history.csv", done, columns)
        warm_start = done
        done, elapsed = run_phase(
            "warmup", units, problem["t"], params, opt, solver,
            start=done, total=args.warmup_epochs, history=history,
            checkpoint_every=args.checkpoint_every, prior_elapsed=elapsed,
            on_checkpoint=lambda e, s: save(working, "warmup", e, e, s, opt),
            stop_after=stop_after, extra=lambda: thermo_weights(model))
        history.file.close()
        warm_done = done
        if done < args.warmup_epochs:
            return {"phase": "warmup", "epochs_completed": done}
        for p in frozen:
            p.requires_grad_(True)
        phase, done, opt_state = "stage2", 0, None
        stop_after = None if stop_after is None else stop_after - (done - warm_start)
        if stop_after == 0:                                   # stopped exactly at the boundary
            save(working, "warmup", done, done, elapsed, opt)
            return {"phase": "warmup", "epochs_completed": done}

    if phase == "warmup":                                     # no (remaining) warm-up
        phase, done, opt_state = "stage2", 0, None
    params = list(model.parameters())
    opt = _adam(params, args.lr)                                       # fresh Adam state
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    history = _history(run_dir / "stage2_history.csv", done, columns)

    def on_stage2_checkpoint(e, s, opt):
        """Working checkpoint; every --archive-every epochs also a permanent archive
        ``archive/stage2_<run_id>_epoch<N>.pt`` (never overwritten)."""
        save(working, "stage2", e, warm_done, s, opt)
        if args.archive_every and e % args.archive_every == 0:
            archive = run_dir / "archive" / f"stage2_{run_id}_epoch{e}.pt"
            if not archive.exists():
                archive.parent.mkdir(exist_ok=True)
                save(archive, "stage2", e, warm_done, s, opt)
                logger.info("stage 2 archive %s  sha256 %s", archive.name,
                            checkpoint_sha256(archive))

    done, elapsed = run_phase(
        "stage2", units, problem["t"], params, opt, solver,
        start=done, total=args.epochs, history=history, checkpoint_every=args.checkpoint_every,
        prior_elapsed=elapsed,
        on_checkpoint=lambda e, s: on_stage2_checkpoint(e, s, opt),
        stop_after=stop_after, extra=lambda: thermo_weights(model))
    history.file.close()
    save(working, "stage2", done, warm_done, elapsed, opt)
    if done >= args.epochs:
        save(final, "stage2", done, warm_done, elapsed, opt)
    return {"phase": "stage2", "epochs_completed": done}


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("stage1", "stage2"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--epochs", type=int, default=100000)
        p.add_argument("--seed", type=int, default=0,
                       help="stage1: kinetic Glorot draw; stage2: KAN_cor Glorot draw")
        p.add_argument("--system", choices=SYSTEMS, default="hydrogen",
                       help="hydrogen (default, the paper's case) or methane (optional "
                            "extension: methane.npz, GRI-Mech 3.0, 52 species)")
        p.add_argument("--t-ref", type=float, default=None,
                       help="RHS divisor in seconds (default: the system's time window, "
                            "6e-4 s for hydrogen, 5e-3 s for methane)")
        p.add_argument("--compile", action="store_true",
                       help="torch.compile the ODE right-hand side (hydrogen only; ~1.8x faster "
                            "per epoch in our test, not bit-identical to eager)")
        p.add_argument("--sensitivity", choices=["direct_autograd", "fsa"],
                       default="direct_autograd")
        p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
        p.add_argument("--solve-mode", choices=["batched", "per_trajectory"], default="batched",
                       help="one solve for all trajectories (default) or one per trajectory")
        p.add_argument("--lr", type=float, default=LR,
                       help="Adam learning rate (default: the paper's 2e-3; the released "
                            "biodiesel example uses 1e-2)")
        p.add_argument("--rtol", type=float, default=RTOL)
        p.add_argument("--atol", type=float, default=ATOL,
                       help="solver tolerances (default 1e-6 / 1e-8; the released biodiesel "
                            "example uses rtol 1e-2, atol 1e-6)")
        p.add_argument("--checkpoint-every", type=int, default=500)
        p.add_argument("--archive-every", type=int, default=5000,
                       help="also write a permanent archive every N epochs (a multiple "
                            "of --checkpoint-every; 0 = stage1: only at the end, stage2: none)")
        p.add_argument("--threads", type=int, default=None)
        p.add_argument("--resume", action="store_true")
        if name == "stage2":
            p.add_argument("--stage1-archive", type=Path,
                           help="Stage-1 archive to branch from (required unless --resume)")
            p.add_argument("--warmup-epochs", type=int, default=0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "stage2" and not args.resume and args.stage1_archive is None:
        raise SystemExit("stage2 needs --stage1-archive (or --resume)")
    if args.archive_every and (
            not args.checkpoint_every or args.archive_every % args.checkpoint_every):
        raise SystemExit("--archive-every must be a multiple of --checkpoint-every")
    if args.compile and args.system != "hydrogen":
        raise SystemExit("--compile is only enabled for hydrogen")
    configure_system(args.system)
    if args.t_ref is None:
        args.t_ref = T_REF_DEFAULT
    torch.set_default_dtype(getattr(torch, args.dtype))
    if args.threads:
        torch.set_num_threads(args.threads)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.run_dir / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    problem = load_problem(getattr(torch, args.dtype))
    if args.stage == "stage1":
        run_stage1(args, problem, args.run_dir)
    else:
        run_stage2(args, problem, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

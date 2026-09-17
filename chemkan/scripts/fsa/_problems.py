"""Fixed ChemKAN gradient problems for FSA validation and regression checks.

Each problem mirrors what a training script builds for one gradient evaluation: the same
data, train-only normalizers, temperature provider, dynamics wrapper, optimized
parameters and loss. Model tensors come from the archived baselines (or from the seed-0
construction the baselines started from), so validation runs on the actual problems the
scientific runs train, never on a stand-in model.

Baselines (repository-relative):
    B0        results/reproduction/chemkan/biodiesel/main/direct_autograd_seed0
    H_STAGE1  results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4/stage1_seed0
    H0        results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4/random_stage2_10000_seed0
    Hnorm1    results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4/normmatched_dir1_stage2_10000
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import torch

from _chemistry import ATOMIC_WEIGHTS, ELEMENT_COUNTS, MOLAR_WEIGHTS
from _data import load_biodiesel, load_hydrogen, load_hydrogen_temperature
from chemkan.dynamics import ChemKANDynamics, KineticDynamics
from chemkan.losses import chemkan_loss, element_conservation_loss, trajectory_mse
from chemkan.model import ChemKAN, KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.temperature import ConstantTemperature, ObservedTemperature

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
B0_DIR = RESULTS / "reproduction/chemkan/biodiesel/main/direct_autograd_seed0"
BASE_ON = RESULTS / "reproduction/chemkan/hydrogen/diagnostics/base_on_n4"
H_STAGE1_DIR = BASE_ON / "stage1_seed0"
H0_DIR = BASE_ON / "random_stage2_10000_seed0"
HNORM1_DIR = BASE_ON / "normmatched_dir1_stage2_10000"

# Hydrogen trajectories for tight-tolerance comparisons (train-split indices):
#   2 = 950 K / phi 0.9 (no ignition in the 0.6 ms window)
#  14 = 1050 K / phi 0.9 (representative condition; ignites at ~0.15 ms)
#  31 = 1200 K / phi 0.9 (fast ignition, ~0.05 ms)
H2_SUBSET = (2, 14, 31)
ALPHA_PINN = 1e-4                     # the archived hydrogen runs' alpha_pinn


def _tree_sha256(files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def training_code_files() -> list[Path]:
    """The files covered by ``training_code_sha256`` (see ``code_state``)."""
    tools_dir = ROOT / "chemkan/scripts/fsa"
    return sorted(f for d in ("chemkan/src", "chemkan/scripts") for f in (ROOT / d).rglob("*")
                  if f.suffix in (".py", ".sh") and "__pycache__" not in f.parts
                  and tools_dir not in f.parents)


def training_code_manifest() -> dict:
    """Per-file sha256 of the training-code files, keyed by repository-relative path."""
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in training_code_files()}


def code_state(save_patch_to: Path | None = None) -> dict:
    """Content identity of the code that produced an FSA artifact.

    ``training_code_sha256`` hashes every .py/.sh file of the library (chemkan/src) and of
    chemkan/scripts EXCEPT the FSA validation/driver tools in chemkan/scripts/fsa, which
    get their own ``fsa_tools_sha256``. The working tree may be uncommitted, so HEAD
    alone would misidentify the code; the uncommitted diff is also saved as a patch.
    """
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    src = [f for d in ("chemkan/src", "chemkan/scripts") for f in (ROOT / d).rglob("*")
           if f.suffix in (".py", ".sh") and "__pycache__" not in f.parts]
    tools_dir = ROOT / "chemkan/scripts/fsa"
    tools = sorted(f for f in src if tools_dir in f.parents)
    training = sorted(f for f in src if tools_dir not in f.parents)
    out = {"git_head": git("rev-parse", "HEAD").strip(),
           "training_code_sha256": _tree_sha256(training),
           "fsa_tools_sha256": _tree_sha256(tools),
           "uncommitted_changes": bool(git("status", "--porcelain", "--", "chemkan/src",
                                           "chemkan/scripts").strip())}
    if save_patch_to is not None and out["uncommitted_changes"]:
        diff = git("diff", "HEAD", "--", "chemkan/src", "chemkan/scripts")
        untracked = git("ls-files", "--others", "--exclude-standard", "--", "chemkan/src",
                        "chemkan/scripts").split()
        blob = diff + "".join(f"\n=== untracked {f}\n" + (ROOT / f).read_text()
                              for f in sorted(untracked) if f.endswith((".py", ".sh")))
        save_patch_to.mkdir(parents=True, exist_ok=True)
        name = f"training_{out['training_code_sha256'][:12]}_tools_{out['fsa_tools_sha256'][:12]}.patch"
        (save_patch_to / name).write_text(blob)
        out["patch"] = name
    return out


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class Problem:
    """One gradient evaluation: ``loss_fn(integrate(dynamics, x0, t))`` w.r.t. ``params``."""

    name: str
    net: torch.nn.Module
    dynamics: torch.nn.Module
    params: list
    x0: torch.Tensor
    t: torch.Tensor
    loss_fn: object
    state_range: torch.Tensor            # train min-max range of the integrated state
    meta: dict = field(default_factory=dict)


def _select(x, idx, axis):
    return x if idx is None else x.index_select(axis, torch.as_tensor(idx))


# ----------------------------------------------------------------------- biodiesel
def biodiesel(model: str, dtype=torch.float64, idx=None) -> Problem:
    """B0: clean data, full-batch full-trajectory MSE; ``model`` = 'init' or 'trained'."""
    data = load_biodiesel(split="train")
    torch.manual_seed(0)                                    # B0 seed, default init
    core = KineticCore(species_dim=6, hidden_dim=4, num_basis=3, n_mu=2, use_base_act=False)
    if model == "trained":
        core.load_state_dict(load(B0_DIR / "checkpoint_final.pt")["model_state"])
    elif model != "init":
        raise ValueError(model)
    core = core.to(dtype)
    T_const = data["T_const"]
    u_min = torch.cat([data["u_min"], T_const.min().reshape(1)])
    u_max = torch.cat([data["u_max"], T_const.max().reshape(1)])
    full_norm = MinMaxNormalizer(u_min, u_max).to(dtype)
    loss_norm = full_norm.subset(slice(0, 6))
    temp = ConstantTemperature(_select(T_const, idx, 0)).to(dtype)
    dyn = KineticDynamics(core, temp, input_normalizer=full_norm)
    target = loss_norm.normalize(_select(data["targets_TBm"], idx, 1).to(dtype))

    def loss_fn(pred):
        return trajectory_mse(loss_norm.normalize(pred), target)

    return Problem(f"biodiesel/{model}", core, dyn, list(core.parameters()),
                   _select(data["Y0"], idx, 0).to(dtype), data["t"].to(dtype), loss_fn,
                   loss_norm.range.clone(), {"trajectories": "all 20" if idx is None else list(idx)})


# ----------------------------------------------------------------------- hydrogen
def _hydrogen_model(kinetic: str) -> ChemKAN:
    torch.manual_seed(0)                                    # H_STAGE1 / H0 / Hnorm1 seed
    model = ChemKAN(species_dim=9, hidden_dim=3, num_basis=4, n_mu=3, use_base_act=True)
    if kinetic == "stage1-trained":
        model.kinetic.load_state_dict(load(H_STAGE1_DIR / "checkpoint_stage1.pt")["kinetic_state"])
    elif kinetic != "init":
        raise ValueError(kinetic)
    return model


def _hydrogen_tensors(dtype):
    data = load_hydrogen(split="train")
    full_norm = MinMaxNormalizer(data["u_min"], data["u_max"]).to(dtype)
    chem = (ELEMENT_COUNTS.to(dtype), ATOMIC_WEIGHTS.to(dtype), MOLAR_WEIGHTS.to(dtype))
    return data, full_norm, chem


def stage1(model: str, dtype=torch.float64, idx=None) -> Problem:
    """H_STAGE1: species only, dense Cantera T_obs(t), MSE + alpha*PINN.

    ``model`` = 'init' (seed-0 kinetic core, the Stage-1 starting point) or 'trained'
    (the archived direct-autograd Stage-1 kinetic core).
    """
    data, full_norm, (ec, aw, mw) = _hydrogen_tensors(dtype)
    net = _hydrogen_model("stage1-trained" if model == "trained" else "init").to(dtype)
    dense = load_hydrogen_temperature(split="train", n_points=20000)
    temp = ObservedTemperature(dense["t_dense"], _select(dense["T_dense_TB1"], idx, 1)).to(dtype)
    species_norm = full_norm.subset(slice(0, 9))
    dyn = KineticDynamics(net.kinetic, temp, input_normalizer=full_norm)
    species = _select(data["species_TBm"], idx, 1).to(dtype)
    target = species_norm.normalize(species)

    def loss_fn(pred):
        return (trajectory_mse(species_norm.normalize(pred), target)
                + ALPHA_PINN * element_conservation_loss(pred, ec, aw, mw))

    return Problem(f"stage1/{model}", net, dyn, list(net.kinetic.parameters()), species[0],
                   data["t"].to(dtype), loss_fn, species_norm.range.clone(),
                   {"trajectories": "all 35" if idx is None else list(idx)})


STAGE2_MODELS = ("H0-init", "Hnorm1-init", "H0-final", "Hnorm1-final")


def stage2_net(model: str) -> ChemKAN:
    """Float32 Stage-2 model tensors for one of ``STAGE2_MODELS``.

    Inits: seed-0 construction + archived Stage-1 kinetic core; H0 keeps the seed-0
    thermo draw, Hnorm1 replaces thermo.linear by its recorded norm-matched vector.
    Finals: the archived direct-autograd checkpoints.
    """
    if model in ("H0-final", "Hnorm1-final"):
        net = _hydrogen_model("init")
        d = H0_DIR if model == "H0-final" else HNORM1_DIR
        net.load_state_dict(load(d / "checkpoint_final.pt")["model_state"])
        return net
    net = _hydrogen_model("stage1-trained")
    if model == "Hnorm1-init":
        vec = json.loads((HNORM1_DIR / "config.json").read_text())["thermo_init"]["resulting_vector"]
        with torch.no_grad():
            net.thermo.linear.weight.copy_(torch.tensor(vec, dtype=torch.float32).reshape(1, 9))
    elif model != "H0-init":
        raise ValueError(model)
    return net


def stage2(model: str, dtype=torch.float64, idx=None) -> Problem:
    """Full [Y, T] with the complete ChemKAN, MSE + alpha*PINN, all parameters."""
    data, full_norm, (ec, aw, mw) = _hydrogen_tensors(dtype)
    net = stage2_net(model).to(dtype)
    dyn = ChemKANDynamics(net, input_normalizer=full_norm)
    full = _select(data["full_TBm1"], idx, 1).to(dtype)
    tgt = full_norm.normalize(full)

    def loss_fn(pred):
        return chemkan_loss(full_norm.normalize(pred), tgt, use_pinn=True, alpha_pinn=ALPHA_PINN,
                            Y_phys=pred[..., :9], element_counts=ec, atomic_weights=aw,
                            molar_weights=mw)

    return Problem(f"stage2/{model}", net, dyn, list(net.parameters()), full[0],
                   data["t"].to(dtype), loss_fn, full_norm.range.clone(),
                   {"trajectories": "all 35" if idx is None else list(idx)})

r"""ChemKAN in one file: train the biodiesel kinetic model, top to bottom.

A small, editable example of the whole ChemKAN training workflow on the isothermal
biodiesel system (6 species, constant temperature per trajectory):

    1. load the data            normalized initial states u0 = [Y_hat, T_hat] and targets
    2. settings                 everything below in SETTINGS, nothing hidden in a config
    3. the ChemKAN model        kinetic core 7 -> H -> 6 (additive KAN, then LeanKAN)
    4. dynamics and solver      dz/dt = [KAN(z), 0] / t_window, integrated with Tsit5
    5. loss                     mean squared error over species and times, mean over cases
    6. optimization             Adam, one full-batch step per epoch
    7. save                     the trained parameters and the settings

    python chemkan/scripts/train.py                  # SETTINGS below: about 20 s on a laptop CPU
    python chemkan/scripts/train.py --epochs 10000   # the paper-length run

It uses the same components as the full biodiesel trainer (``train_biodiesel.py``): the
same data loader, model, initialization, dynamics, solver call and loss, so with the same
settings it computes the same losses (that trainer's ``--data canonical --solve-mode
batched --sensitivity direct_autograd``). ``train_biodiesel.py`` adds forward sensitivity
analysis, noise levels, resumption and the run records.

Adapting it to another chemical system: provide u0 with shape (cases, m + 1) (normalized
species and temperature) and targets with shape (cases, m, times), build the core with
``chemkan.model.KineticCore(species_dim=m, ...)`` as ``reference_components.build_core``
does, and keep steps 4-7 as they are. A system whose temperature evolves (combustion) also
needs the thermodynamic part of the model: see ``combustion/train_combustion.py``.
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reference_components import (NormalizedDynamics, build_core,  # noqa: E402
                                  glorot_uniform_, julia_mse)
from train_biodiesel import load_canonical_data, predict  # noqa: E402

from chemkan.solver import SolverConfig  # noqa: E402

# --------------------------------------------------------------------------- 2. settings
SETTINGS = {
    "data_file": HERE.parent / "data/generated/biodiesel_v2.npz",
    "noise_percent": None,      # None (clean) or a stored level: 1, 2, 3, 5, 7, 10, 15
    "hidden": 4,                # H: 39 * H parameters (4 = the paper's 156-parameter model)
    "seed": 0,
    "epochs": 1000,             # the paper trains 10,000
    "lr": 1e-2,                 # Adam
    "rtol": 1e-2, "atol": 1e-6,  # Tsit5 tolerances during training
    "out_dir": Path("train_example"),
}


def main(settings: dict) -> Path:
    torch.set_default_dtype(torch.float64)

    # 1. data: u0 (cases, 7) = [Y_hat (6), T_hat]; target (cases, 6, 30); train/test slices
    data = load_canonical_data("cpu", settings["data_file"], settings["noise_percent"])
    train, test = data["train"], data["test"]
    t = data["t"]

    # 3. model: kinetic core with Glorot-uniform KAN coefficients
    core = build_core(settings["hidden"])
    glorot_uniform_(core, torch.Generator().manual_seed(settings["seed"]))
    print(f"ChemKAN kinetic core: {sum(p.numel() for p in core.parameters())} parameters")

    # 4. dynamics in normalized coordinates, divided by the time window; the ODE solver
    dynamics = NormalizedDynamics(core, float(t[-1] - t[0]))
    solver = SolverConfig(method="tsit5", rtol=settings["rtol"], atol=settings["atol"],
                          sensitivity="direct_autograd")
    optimizer = torch.optim.Adam(core.parameters(), lr=settings["lr"], betas=(0.9, 0.999),
                                 eps=1e-8)

    for epoch in range(1, settings["epochs"] + 1):
        optimizer.zero_grad()
        # 5. loss of the predicted trajectories (all training cases in one solve)
        loss = julia_mse(predict(dynamics, data["u0"][train], t, solver, batched=True),
                         data["target"][train])
        # 6. one optimizer step (gradients by backpropagation through the solver)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0:
            with torch.no_grad():
                test_loss = julia_mse(predict(dynamics, data["u0"][test], t, solver, batched=True),
                                      data["target"][test])
            print(f"epoch {epoch:5d}  train loss {loss.item():.4e}  "
                  f"test loss {test_loss.item():.4e}")

    # 7. save
    out = Path(settings["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "chemkan_biodiesel.pt"
    torch.save({"model_state": core.state_dict(),
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}},
               path)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    p.add_argument("--out-dir", type=Path, default=SETTINGS["out_dir"])
    args = p.parse_args()
    main({**SETTINGS, "epochs": args.epochs, "out_dir": args.out_dir})

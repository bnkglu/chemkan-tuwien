r"""ChemKAN in one file: train a kinetic ChemKAN on an .npz dataset, top to bottom.

A small, editable example of the whole ChemKAN training workflow. By default it trains the
isothermal biodiesel system (6 species, constant temperature per trajectory):

    1. load the data            normalized initial states u0 = [Y_hat, T_hat] and targets
    2. settings                 everything below in SETTINGS, nothing hidden in a config
    3. the ChemKAN model        kinetic core (m+1) -> H -> m (additive KAN, then LeanKAN)
    4. dynamics and solver      dz/dt = [KAN(z), 0] / 50 s, integrated with Tsit5
    5. loss                     mean squared error over species and times, mean over cases
    6. optimization             Adam, one full-batch step per epoch
    7. save                     the trained parameters, the normalization and the settings

    python chemkan/scripts/train.py                          # biodiesel: about 20 s on a laptop CPU
    python chemkan/scripts/train.py --epochs 10000           # the paper-length run
    python chemkan/scripts/train.py --data my_system.npz     # your own data (format below)

Data format: one ``.npz`` with

    t              (n_t,)              observation times, increasing; t[0] is the initial time
    train_states   (n_train, n_t, m)   species trajectories used for training (m species)
    test_states    (n_test, n_t, m)    held-out trajectories on the same time grid
    train_T        (n_train,)          the constant temperature of each training trajectory
    test_T         (n_test,)           the constant temperature of each test trajectory

``chemkan/data/generated/biodiesel_v2.npz`` is such a file. Each trajectory starts from its
own state at ``t[0]``. Species and temperatures are min-max normalized with the training
set's statistics, as in the paper's Eq. 18.

Scope and limitations. This example trains the kinetic core only, for systems at constant
temperature: temperature is an input of the model, not integrated (dT/dt = 0), and
trajectories at different temperatures teach the temperature dependence. All trajectories
share one time grid and are fitted as given (clean data, or your own noisy observations).
Combustion-type systems, where temperature evolves, need the thermodynamic part of ChemKAN
and the two-stage procedure: see ``train_hydrogen.py`` / ``combustion/train_combustion.py``.
The RHS divisor (``rhs_divisor``, seconds) scales the dynamics to the time window; 50 s
suits the 30 s biodiesel window as in the authors' example, and a value of the order of
your window is a reasonable start. For forward sensitivity analysis, stored noise levels,
resumption and full run records on biodiesel, use ``train_biodiesel.py``; with the same
settings and the biodiesel data, this example computes exactly the losses of
``train_biodiesel.py --data canonical --solve-mode batched --sensitivity direct_autograd``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reference_components import (NormalizedDynamics, build_core,  # noqa: E402
                                  glorot_uniform_, julia_mse)

from chemkan.solver import SolverConfig, integrate  # noqa: E402

# --------------------------------------------------------------------------- 2. settings
SETTINGS = {
    "data_file": HERE.parent / "data/generated/biodiesel_v2.npz",
    "hidden": 4,                # H: 3 * H * (2m + 1) parameters (biodiesel, H = 4: 156)
    "seed": 0,
    "epochs": 1000,             # the paper trains 10,000
    "lr": 1e-2,                 # Adam
    "rtol": 1e-2, "atol": 1e-6,  # Tsit5 tolerances during training
    "rhs_divisor": 50.0,        # dz/dt = KAN(z) / divisor [s]; 50 as the released example,
                                # the figure runs and the biodiesel trainer's default
    "out_dir": Path("train_example"),
}
KEYS = ("t", "train_states", "test_states", "train_T", "test_T")


def load_npz(path) -> dict:
    """The data format of the module docstring -> normalized u0 (cases, m+1), targets
    (cases, m, n_t). Arrays are read as float32 and computed in float64, as the biodiesel
    trainer's loader does. Raises ValueError with the reason for a malformed file."""
    d = np.load(path)
    missing = [k for k in KEYS if k not in d.files]
    if missing:
        raise ValueError(f"{path}: missing arrays {missing}; expected {list(KEYS)}")
    t, tr, te, T_tr, T_te = (torch.as_tensor(np.asarray(d[k]), dtype=torch.float32).double()
                             for k in KEYS)
    if t.ndim != 1 or tr.ndim != 3 or te.ndim != 3:
        raise ValueError(f"{path}: t must be (n_t,), train/test_states (cases, n_t, m)")
    if tr.shape[1:] != te.shape[1:] or tr.shape[1] != len(t):
        raise ValueError(f"{path}: train {tuple(tr.shape)} / test {tuple(te.shape)} must share "
                         f"(n_t = {len(t)}, m)")
    if T_tr.shape != (len(tr),) or T_te.shape != (len(te),):
        raise ValueError(f"{path}: train_T / test_T must hold one temperature per trajectory")
    if not all(torch.isfinite(a).all() for a in (t, tr, te, T_tr, T_te)):
        raise ValueError(f"{path}: arrays contain NaN or inf")
    if not (torch.diff(t) > 0).all():
        raise ValueError(f"{path}: t must be strictly increasing")
    y_min, y_max = tr.amin(dim=(0, 1)), tr.amax(dim=(0, 1))       # train-only statistics
    if (y_max == y_min).any():
        raise ValueError(f"{path}: species {torch.where(y_max == y_min)[0].tolist()} are "
                         f"constant in the training data; drop them")
    T_min, T_max = T_tr.min(), T_tr.max()
    T_span = T_max - T_min if T_max > T_min else torch.ones(())    # one temperature: T_hat = 0

    def normalize(states, T):
        y = (states - y_min) / (y_max - y_min)
        return torch.cat([y[:, 0, :], ((T - T_min) / T_span).reshape(-1, 1)], dim=-1), y.permute(0, 2, 1)

    (u0_train, y_train), (u0_test, y_test) = normalize(tr, T_tr), normalize(te, T_te)
    return {"t": t, "m": tr.shape[2], "u0_train": u0_train, "target_train": y_train,
            "u0_test": u0_test, "target_test": y_test,
            "normalization": {"y_min": y_min, "y_max": y_max, "T_min": T_min, "T_span": T_span}}


def predict(dynamics, u0, t, solver, m):
    """Integrate all trajectories in one solve: (cases, m+1) -> species (cases, m, n_t)."""
    return integrate(dynamics, u0, t, solver)[:, :, :m].permute(1, 2, 0)


def main(settings: dict) -> dict:
    torch.set_default_dtype(torch.float64)

    # 1. data: u0 (cases, m+1) = [Y_hat (m), T_hat]; target (cases, m, n_t)
    data = load_npz(settings["data_file"])
    t, m = data["t"], data["m"]

    # 3. model: kinetic core with Glorot-uniform KAN coefficients
    core = build_core(settings["hidden"], species_dim=m)
    glorot_uniform_(core, torch.Generator().manual_seed(settings["seed"]))
    print(f"ChemKAN kinetic core for {m} species: {sum(p.numel() for p in core.parameters())} parameters")

    # 4. dynamics in normalized coordinates, divided by a time scale; the ODE solver
    dynamics = NormalizedDynamics(core, settings["rhs_divisor"])
    solver = SolverConfig(method="tsit5", rtol=settings["rtol"], atol=settings["atol"],
                          sensitivity="direct_autograd")
    optimizer = torch.optim.Adam(core.parameters(), lr=settings["lr"], betas=(0.9, 0.999),
                                 eps=1e-8)

    losses = []
    for epoch in range(1, settings["epochs"] + 1):
        optimizer.zero_grad()
        # 5. loss of the predicted trajectories (all training cases in one solve)
        loss = julia_mse(predict(dynamics, data["u0_train"], t, solver, m), data["target_train"])
        # 6. one optimizer step (gradients by backpropagation through the solver)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch == 1 or epoch % 100 == 0:
            with torch.no_grad():
                test_loss = julia_mse(predict(dynamics, data["u0_test"], t, solver, m),
                                      data["target_test"])
            print(f"epoch {epoch:5d}  train loss {loss.item():.4e}  "
                  f"test loss {test_loss.item():.4e}")

    # 7. save
    out = Path(settings["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "chemkan_kinetic.pt"
    torch.save({"model_state": core.state_dict(), "species_dim": m,
                "normalization": data["normalization"], "train_loss": losses,
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}},
               path)
    print(f"saved {path}")
    return {"checkpoint": path, "train_loss": losses}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=SETTINGS["data_file"], help="an .npz in the format above")
    p.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    p.add_argument("--out-dir", type=Path, default=SETTINGS["out_dir"])
    args = p.parse_args()
    main({**SETTINGS, "data_file": args.data, "epochs": args.epochs, "out_dir": args.out_dir})

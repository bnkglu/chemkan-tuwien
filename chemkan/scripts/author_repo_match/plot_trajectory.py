r"""Trajectory plots of a trained ChemKAN checkpoint against its reference data.

Plotting only: nothing is trained. Works for checkpoints of
``train_author_matched_hydrogen.py`` (hydrogen or methane, Stage 1 or Stage 2) and of
``train_author_repo_match.py`` runs on a dataset in ``chemkan/data/generated/`` (biodiesel).
Only ``load_model`` knows the two trainers apart; data, prediction and plotting are shared.

Data: any ``.npz`` with ``t``, ``species``, ``train_states`` and ``test_states``. A state
column after the species is the temperature and gets its own panel; a dataset without it
(isothermal biodiesel) gives the constant ``train_T`` / ``test_T`` to the model instead. A
Stage-1 model is given the observed temperature: the dense ``<system>_temperature_20000.npz``
if it exists, otherwise the dataset's own temperature column.

Each condition gets one figure: one panel per state, the prediction on a dense grid (line)
and the reference observation times (markers). The title gives the time-averaged MSE on
the observation grid in normalized coordinates (the trainers' ``mse_time_avg`` /
``train_mse``), against the clean reference.

    python plot_trajectory.py --checkpoint <run>/stage2_final.pt
    python plot_trajectory.py --checkpoint <run>/stage1_final.pt --condition 1550,1.3 \
        --species CH4 O2 CO2 H2O CO OH --out-dir <dir>
    python plot_trajectory.py --checkpoint <biodiesel run>/checkpoint_final.pt --index 0 --index 20

Conditions: ``--condition T0,phi`` (datasets with ``ics``) or ``--index i`` (training
conditions first, then test). Default: hydrogen, the paper's Fig. 7 pair (1050 K, phi 0.9;
1150 K, phi 1.3); otherwise each test condition with its nearest training condition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import train_author_matched_hydrogen as amh  # noqa: E402
from _author_match import NormalizedDynamics, build_core, rhs_divisor_from_config  # noqa: E402
from _data import DATA_DIR  # noqa: E402
from _predictions import checkpoint_sha256  # noqa: E402

from chemkan.solver import SolverConfig, integrate  # noqa: E402
from chemkan.temperature import ObservedTemperature  # noqa: E402

PAPER_CONDITIONS = {"hydrogen": [(1050.0, 0.9), (1150.0, 1.3)]}        # Fig. 7
MAIN_SPECIES = {"hydrogen": ["H2", "O2", "H2O", "H", "O", "OH", "HO2", "H2O2"],
                "methane": ["CH4", "O2", "H2O", "CO2", "CO", "H2", "OH", "CH2O"]}


def load_model(ckpt: dict) -> dict:
    """The trained right-hand side in normalized coordinates and everything needed to use it.

    ``temperature``: ``predicted`` (part of the integrated state), ``observed`` (Stage 1:
    given as a function of time) or ``constant`` (biodiesel: in the state, derivative 0).
    """
    cfg = ckpt["config"]
    solver = SolverConfig(method="tsit5", rtol=cfg["solver"]["rtol"], atol=cfg["solver"]["atol"],
                          sensitivity="direct_autograd")
    if str(cfg.get("experiment", "")).startswith("biodiesel"):
        source = cfg.get("data_source", {})
        if source.get("data_source") != "canonical" or "species_min" not in source:
            raise SystemExit("only biodiesel runs on a chemkan/data/generated dataset are supported")
        core = build_core(cfg["architecture"].get("hidden", 4)).double()
        core.load_state_dict(ckpt["model_state"])
        return {"system": "biodiesel", "data": ROOT / source["path"], "dtype": torch.float64,
                "solver": solver, "rhs": NormalizedDynamics(core, rhs_divisor_from_config(cfg)),
                "u_min": np.array(source["species_min"] + [source["T_min"]]),
                "u_max": np.array(source["species_max"] + [source["T_max"]]),
                "temperature": "constant", "noise": source.get("noise_percent") or 0,
                "label": "final", "epochs": cfg.get("epochs")}
    system = cfg.get("system", {}).get("name", "hydrogen")
    amh.configure_system(system)
    if cfg["dataset"]["sha256"] != checkpoint_sha256(DATA_DIR / f"{system}.npz"):
        raise SystemExit(f"{system}.npz differs from the file the checkpoint was trained on")
    model = amh.build_model().to(getattr(torch, cfg["dtype"]))
    model.load_state_dict(ckpt["model_state"])
    t_ref = cfg["t_ref"]["value_s"]
    stage = ckpt["stage"]
    return {"system": system, "data": DATA_DIR / f"{system}.npz",
            "dtype": getattr(torch, cfg["dtype"]), "solver": solver,
            "rhs": amh.Stage2Dynamics(model, t_ref) if stage == 2 else (model.kinetic, t_ref),
            "u_min": np.array(cfg["normalization"]["u_min"]),
            "u_max": np.array(cfg["normalization"]["u_max"]),
            "temperature": "predicted" if stage == 2 else "observed", "noise": 0,
            "label": ckpt.get("phase", f"stage{stage}"),
            "epochs": ckpt.get("epochs_completed", ckpt.get("phase_epochs_completed"))}


def load_reference(path: Path) -> dict:
    """Training then test conditions of a dataset, physical units."""
    data = np.load(path, allow_pickle=True)
    states = np.concatenate([data["train_states"], data["test_states"]])     # (N, T, k)
    n_train = len(data["train_states"])
    species = [str(s) for s in data["species"]]
    has_T = states.shape[-1] == len(species) + 1
    if has_T:
        temperature = states[..., -1].T                                      # (T, N)
    else:
        constant = np.concatenate([data["train_T"], data["test_T"]])
        temperature = np.broadcast_to(constant, (len(data["t"]), len(constant)))
    ref = {"t": np.asarray(data["t"], dtype=float), "states": states, "species": species,
           "names": species + (["T"] if has_T else []), "temperature": temperature,
           "is_test": np.arange(len(states)) >= n_train, "ics": None}
    if "train_ics" in data.files:
        ref["ics"] = np.concatenate([data["train_ics"], data["test_ics"]])
        ref["labels"] = [f"T0 = {T0:g} K, phi = {phi:g}" for T0, phi in ref["ics"]]
        ref["file_tags"] = [f"T{T0:g}_phi{phi:g}" for T0, phi in ref["ics"]]
    else:
        ref["labels"] = [f"condition {i}, T = {T:g} K" for i, T in enumerate(temperature[0])]
        ref["file_tags"] = [f"condition{i:02d}" for i in range(len(states))]
    dense_path = path.with_name(f"{path.stem}_temperature_{amh.DENSE_T_POINTS}.npz")
    ref["observed_temperature"] = (ref["t"], temperature)
    if dense_path.exists():
        dense = np.load(dense_path)
        if not np.allclose(np.concatenate([dense["train_ics"], dense["test_ics"]]), ref["ics"]):
            raise SystemExit(f"{dense_path.name}: conditions differ from {path.name}")
        ref["observed_temperature"] = (
            dense["t"], np.concatenate([dense["train_T"], dense["test_T"]], axis=1)[..., 0])
    return ref


def predict(model: dict, ref: dict, i: int, times: np.ndarray) -> np.ndarray:
    """Normalized full state [Y_hat, T_hat] of condition ``i`` at ``times`` (one solve)."""
    as_tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=model["dtype"])   # noqa: E731
    span = model["u_max"] - model["u_min"]
    u0 = np.append(ref["states"][i, 0, :len(ref["species"])], ref["temperature"][0, i])
    u0_hat = as_tensor((u0 - model["u_min"]) / span)[None]
    obs_t, obs_T = ref["observed_temperature"]
    with torch.no_grad():
        if model["temperature"] == "observed":
            kinetic, t_ref = model["rhs"]
            T_hat = as_tensor((obs_T[:, i] - model["u_min"][-1]) / span[-1])[:, None, None]
            rhs = amh.Stage1Dynamics(kinetic, ObservedTemperature(as_tensor(obs_t), T_hat), t_ref)
            species = integrate(rhs, u0_hat[:, :-1], as_tensor(times), model["solver"])[:, 0]
            T_input = np.interp(times, obs_t, obs_T[:, i])
            return np.column_stack([species.numpy(), (T_input - model["u_min"][-1]) / span[-1]])
        return integrate(model["rhs"], u0_hat, as_tensor(times), model["solver"])[:, 0].numpy()


def default_conditions(system: str, ref: dict) -> list[int]:
    if system in PAPER_CONDITIONS:
        return [find_condition(ref, T0, phi) for T0, phi in PAPER_CONDITIONS[system]]
    is_test = ref["is_test"]
    if ref["ics"] is None:
        return [int(np.flatnonzero(~is_test)[0]), int(np.flatnonzero(is_test)[0])]
    span = np.ptp(ref["ics"], axis=0)
    chosen = []
    for i in np.flatnonzero(is_test):
        distance = np.where(is_test, np.inf, (((ref["ics"] - ref["ics"][i]) / span) ** 2).sum(1))
        chosen += [int(np.argmin(distance)), int(i)]
    return chosen


def find_condition(ref: dict, T0: float, phi: float) -> int:
    if ref["ics"] is None:
        raise SystemExit("this dataset has no (T0, phi) conditions; use --index")
    hit = np.flatnonzero(np.isclose(ref["ics"][:, 0], T0) & np.isclose(ref["ics"][:, 1], phi))
    if not len(hit):
        raise SystemExit(f"no condition T0 = {T0:g} K, phi = {phi:g} in the dataset")
    return int(hit[0])


def plot_condition(ref, model, i, dense_t, dense, panels, title, path):
    ncol = 3 if len(panels) <= 6 else 4
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow), squeeze=False)
    scale, unit = (1e3, "ms") if ref["t"][-1] < 1.0 else (1.0, "s")
    for ax, name in zip(axes.flat, panels):
        j = ref["names"].index(name)
        ax.plot(ref["t"] * scale, ref["states"][i, :, j], "o", ms=2.5, color="0.35",
                label="reference")
        is_input = name == "T" and model["temperature"] == "observed"
        ax.plot(dense_t * scale, dense[:, j], "-", lw=1.4,
                color="tab:orange" if is_input else "tab:blue",
                label="observed input" if is_input else "ChemKAN")
        ax.set_title("T [K]" if name == "T" else f"Y {name}", fontsize=10)
        ax.set_xlabel(f"t [{unit}]", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--condition", action="append", default=[],
                    help="T0,phi (repeatable)")
    ap.add_argument("--index", type=int, action="append", default=[],
                    help="condition index, training conditions first (repeatable)")
    ap.add_argument("--species", nargs="+", default=None,
                    help="species panels (default: a main set, or all species if <= 8)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <checkpoint folder>/plots")
    ap.add_argument("--points", type=int, default=601, help="dense prediction grid")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.checkpoint, weights_only=False)
    model = load_model(ckpt)
    torch.set_default_dtype(model["dtype"])
    ref = load_reference(model["data"])
    species = args.species or MAIN_SPECIES.get(model["system"], ref["species"])
    unknown = [s for s in species if s not in ref["species"]]
    if unknown:
        raise SystemExit(f"unknown species {unknown}; available: {ref['species']}")
    panels = (["T"] if "T" in ref["names"] else []) + species
    indices = ([find_condition(ref, *map(float, c.split(","))) for c in args.condition]
               + args.index) or default_conditions(model["system"], ref)

    m = len(ref["species"])
    compared = m + 1 if model["temperature"] == "predicted" else m
    span = model["u_max"] - model["u_min"]
    dense_t = np.linspace(ref["t"][0], ref["t"][-1], args.points)
    out_dir = args.out_dir or args.checkpoint.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        target_hat = (ref["states"][i] - model["u_min"][:ref["states"].shape[-1]]) \
            / span[:ref["states"].shape[-1]]
        at_obs = predict(model, ref, i, ref["t"])
        mse = float(((at_obs[:, :compared] - target_hat[:, :compared]) ** 2).mean())
        dense = predict(model, ref, i, dense_t) * span + model["u_min"]
        split = "test" if ref["is_test"][i] else "train"
        epochs = f", epoch {model['epochs']}" if model["epochs"] is not None else ""
        noise = f", trained on {model['noise']} % noise" if model["noise"] else ""
        title = (f"{model['system']} {model['label']}{epochs}{noise}  |  {ref['labels'][i]} "
                 f"({split})  |  time-averaged MSE {mse:.3e}")
        path = out_dir / f"{args.checkpoint.stem}_{ref['file_tags'][i]}.png"
        plot_condition(ref, model, i, dense_t, dense, panels, title, path)
        print(f"{split:5s} {ref['labels'][i]}: time-averaged MSE {mse:.4e}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

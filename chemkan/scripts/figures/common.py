"""Small helpers shared by the ChemKAN figure scripts.

Deliberately minimal: repository paths, checkpoint loading, and figure saving. Anything
specific to one figure stays in that figure's own script, so a reader can understand a
figure without following a trail through this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Reproduction artifacts. Figure scripts default to these and accept overrides, so a
# notebook can point any figure at a different run directory.
FIGURES_BIODIESEL = ROOT / "results/reproduction/figures/biodiesel"
FIGURES_HYDROGEN = ROOT / "results/reproduction/figures/hydrogen"
TABLES = ROOT / "results/reproduction/tables"
CHEMKAN_BIODIESEL = ROOT / "results/reproduction/chemkan/biodiesel"
CHEMKAN_HYDROGEN = ROOT / "results/reproduction/chemkan/hydrogen"
DEEPONET_BIODIESEL = ROOT / "results/reproduction/baselines/deeponet/biodiesel"
DATA = ROOT / "chemkan/data/generated"

# The DeepONet architecture the corrected reproduction uses (see notebook 07).
DEEPONET_VERSION = "reference_final_trunk_relu"


def add_repo_paths() -> None:
    """Put the library, the script helpers and the DeepONet reference on ``sys.path``.

    The figure scripts import ``evaluate_biodiesel``, ``_data`` and friends, which live in
    ``chemkan/scripts`` rather than in an installed package.
    """
    for folder in ("chemkan/src", "chemkan/scripts", "deeponet"):
        path = str(ROOT / folder)
        if path not in sys.path:
            sys.path.insert(0, path)


def require_file(path, what: str = "file") -> Path:
    """Fail early and legibly when an artifact a figure needs has not been produced."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            f"Train or evaluate the corresponding run first (see "
            f"chemkan/scripts/reproduction/), or pass an explicit path.")
    return path


def load_checkpoint(path, device: str = "cpu") -> dict:
    """Load a training checkpoint. ``weights_only=False`` -- checkpoints carry metadata."""
    import torch
    return torch.load(require_file(path, "checkpoint"), map_location=device,
                      weights_only=False)


def save_figure(fig, output_path, *, dpi: int = 150, formats=("pdf", "png")) -> list[Path]:
    """Save one figure under ``output_path``'s stem in each format. Returns the paths.

    ``output_path`` may carry a suffix (it is replaced) or none. ``None`` saves nothing,
    which is what a notebook wants when it only needs the figure object.
    """
    if output_path is None:
        return []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        target = output_path.with_suffix(f".{ext}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)
    return written


def use_headless_backend() -> None:
    """Select Agg. Call from ``main()`` only -- never at import time.

    This matters for more than headlessness. With ``bbox_inches="tight"`` the crop is
    computed from the renderer's text metrics, and macOS's native backend rounds it one
    pixel differently from Agg, so a command-line run would produce a PNG one pixel wider
    than the notebook's. The notebooks' inline backend is Agg-based, so forcing Agg here
    makes the two byte-identical. Importing a figure script inside a notebook must NOT
    change its backend, which is why this is not called on import.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)


def relative_to_root(path) -> str:
    """Repo-relative path for recording in a table; absolute if it lies outside the repo."""
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def loss_reduction(n_times: int, time_averaged: bool) -> tuple[float, str, str]:
    """Divisor, axis wording and filename suffix for a full-rollout Eq. 18 loss.

    Eq. 18 averages the squared normalized-state error over the state variables and sums
    it over the ``N_t`` time points; it has no 1/N_t factor. With ``time_averaged`` the
    loss is divided by ``N_t`` as a derived diagnostic, which changes no model, stored
    table or ranking.

    ``n_times`` is the number of time points the loss sums over, taken from the grid the
    loss was evaluated on (30 for biodiesel, 50 for hydrogen; t = 0 is included in both).
    """
    if not time_averaged:
        return 1.0, "Eq. 18 (summed over the time points)", ""
    return (float(n_times), f"Time-averaged diagnostic: Eq. 18 loss / $N_t$ ($N_t$ = {n_times})",
            "_time_averaged")

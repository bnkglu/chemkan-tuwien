r"""Training orchestration -- generic, experiment-agnostic stages (ChemKAN Sec. II C 5).

Two reusable stage-level functions instead of dataset-named ones:

* ``train_kinetic_stage`` -- integrate species Y with an externally supplied
  temperature; update ONLY the kinetic core. Serves any prescribed-temperature
  system (biodiesel, hydrogen Stage 1, future systems).
* ``train_full_chemkan``  -- integrate the full ``[Y, T]`` state with the complete
  ChemKAN; update ALL parameters (the kinetic core is NOT frozen).

Both take a caller-provided ``loss_fn(pred) -> scalar`` so the experiment script,
not the library, decides MSE vs. MSE+PINN vs. any future loss. This module does not
import losses, normalization, or anything experiment-specific.

Progress reporting: when ``tqdm`` is installed and ``progress=True`` (the default), the
epoch loop shows a live bar with ETA / it-s and the current loss. ``tqdm`` is an
OPTIONAL convenience -- if it is missing (or ``progress=False``) the loop falls back to
``logging`` a line every ``log_every`` epochs, which stays clean when output is
redirected to a file.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import torch

from .solver import SolverConfig, integrate

try:                                                   # optional: nicer interactive output
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:                            # library stays usable without it
    _tqdm = None

logger = logging.getLogger(__name__)

LossFn = Callable[[torch.Tensor], torch.Tensor]


def _optimize(func: torch.nn.Module, y0: torch.Tensor, t: torch.Tensor,
              params, loss_fn: LossFn, *, epochs: int, lr: float,
              solver: SolverConfig, log_every: int = 100,
              progress: bool = True, start_epoch: int = 0,
              optimizer_state: dict | None = None,
              on_epoch: Callable[[int, float, dict, float], None] | None = None,
              checkpoint_every: int = 0,
              save_resume: Callable[[int, dict], None] | None = None) -> float:
    """Shared loop: integrate, evaluate ``loss_fn(pred)``, step Adam. Returns final loss.

    Shows a ``tqdm`` bar (updating the loss each epoch) when available and
    ``progress`` is set; otherwise logs every ``log_every`` epochs.

    The optimization math is unchanged. The optional plumbing hooks (all no-ops by
    default, so existing callers behave identically) support the run-directory layout:

    * ``loss_fn`` may return either a scalar loss or ``(scalar_loss, components_dict)``;
      only the scalar drives ``backward()``. The components are forwarded to ``on_epoch``.
    * ``on_epoch(epoch, total_loss, components, elapsed_seconds)`` -- per-epoch history hook.
    * ``start_epoch`` / ``optimizer_state`` -- resume from a saved point (loop runs
      ``range(start_epoch, epochs)`` and restores the Adam state).
    * ``checkpoint_every`` + ``save_resume(next_epoch, optimizer_state_dict)`` -- periodic
      resume snapshot.
    """
    opt = torch.optim.Adam(params, lr=lr)
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)
    loss = torch.tensor(float("nan"))
    use_bar = progress and _tqdm is not None
    rng = range(start_epoch, epochs)
    epoch_iter = (_tqdm(rng, desc="training", unit="ep", initial=start_epoch, total=epochs)
                  if use_bar else rng)
    started = time.perf_counter()
    for epoch in epoch_iter:
        opt.zero_grad()
        pred = integrate(func, y0, t, solver)          # (T, B, dim)
        out = loss_fn(pred)
        if isinstance(out, tuple):
            loss, components = out
        else:
            loss, components = out, None
        loss.backward()
        opt.step()
        if on_epoch is not None:                       # history hook (float conversions here only)
            comp_vals = {k: float(v) for k, v in (components or {}).items()}
            on_epoch(epoch, float(loss.detach()), comp_vals, time.perf_counter() - started)
        if use_bar:                                    # one .item() sync/epoch; odeint dominates
            epoch_iter.set_postfix(loss=f"{loss.item():.3e}")
        elif log_every and epoch % log_every == 0:
            logger.info("epoch %5d  loss %.6e", epoch, loss.item())
        if save_resume is not None and checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            save_resume(epoch + 1, opt.state_dict())
    if use_bar:
        epoch_iter.close()
    return loss.detach().item()                        # drop the graph before converting


def train_kinetic_stage(kinetic_dynamics: torch.nn.Module, y0: torch.Tensor,
                        t: torch.Tensor, loss_fn: LossFn, *, epochs: int,
                        lr: float, solver: SolverConfig,
                        log_every: int = 100, progress: bool = True,
                        start_epoch: int = 0, optimizer_state: dict | None = None,
                        on_epoch=None, checkpoint_every: int = 0,
                        save_resume=None) -> float:
    r"""Integrate species Y (temperature supplied externally); update the kinetic core.

        y0 : (B, m)   t : (T,)   loss_fn : (T, B, m) -> scalar

    ``lr`` (a paper experiment choice) and ``solver`` (explicit PyTorch settings) are
    required -- the reusable library never invents them. Optimizes only
    ``kinetic_dynamics.kinetic.parameters()``. ``progress`` toggles the tqdm bar. The
    optional resume/history hooks are forwarded to ``_optimize`` (see its docstring).
    """
    return _optimize(kinetic_dynamics, y0, t,
                     kinetic_dynamics.kinetic.parameters(),
                     loss_fn, epochs=epochs, lr=lr, solver=solver,
                     log_every=log_every, progress=progress,
                     start_epoch=start_epoch, optimizer_state=optimizer_state,
                     on_epoch=on_epoch, checkpoint_every=checkpoint_every,
                     save_resume=save_resume)


def train_full_chemkan(chemkan_dynamics: torch.nn.Module, u0: torch.Tensor,
                       t: torch.Tensor, loss_fn: LossFn, *, epochs: int,
                       lr: float, solver: SolverConfig,
                       log_every: int = 100, progress: bool = True,
                       start_epoch: int = 0, optimizer_state: dict | None = None,
                       on_epoch=None, checkpoint_every: int = 0,
                       save_resume=None) -> float:
    r"""Integrate the full [Y, T] state with the complete model; update ALL parameters.

        u0 : (B, m+1)   t : (T,)   loss_fn : (T, B, m+1) -> scalar

    ``lr`` and ``solver`` are required (see ``train_kinetic_stage``). Optimizes
    ``chemkan_dynamics.model.parameters()`` -- the kinetic core is NOT frozen.
    ``progress`` toggles the tqdm bar. The optional resume/history hooks are forwarded
    to ``_optimize`` (see its docstring).
    """
    return _optimize(chemkan_dynamics, u0, t,
                     chemkan_dynamics.model.parameters(),
                     loss_fn, epochs=epochs, lr=lr, solver=solver,
                     log_every=log_every, progress=progress,
                     start_epoch=start_epoch, optimizer_state=optimizer_state,
                     on_epoch=on_epoch, checkpoint_every=checkpoint_every,
                     save_resume=save_resume)

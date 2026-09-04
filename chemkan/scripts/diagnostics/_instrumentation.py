r"""Behavior-neutral training instrumentation (DIAGNOSTIC).

Two small, additive pieces used by ``train_hydrogen.py``:

* ``NFECounter``  -- counts ODE right-hand-side evaluations per epoch via a PyTorch
                     forward hook on the dynamics module. A forward hook that returns
                     ``None`` leaves the module's output object untouched and adds no
                     node to the autograd graph, so gradients and the optimizer
                     trajectory are bit-identical with the counter attached (covered by
                     ``tests/test_instrumentation.py``).
* ``Stage2Snapshot`` -- writes an immutable model snapshot at chosen epochs so a long
                     run yields intermediate checkpoints without a second training job.
                     It only reads ``state_dict()``; it never touches the optimizer.

Both hang off ``training._optimize``'s ``on_epoch`` hook and do nothing unless asked.
"""

from __future__ import annotations

from pathlib import Path

import torch


class NFECounter:
    """Counts calls to ``module.forward`` (i.e. ODE RHS evaluations) since the last reset.

    Attach to the SAME module instance the solver integrates. Other integrations that
    build their own dynamics wrapper (the Stage-2 probe, evaluation) construct a
    different instance and are therefore not counted.
    """

    def __init__(self, module: torch.nn.Module):
        self.count = 0
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):        # returns None -> output unchanged
        self.count += 1
        return None

    def reset(self) -> int:
        n, self.count = self.count, 0
        return n

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class EpochInstrumentation:
    """``on_epoch`` callback that injects ``nfe`` and ``epoch_wall_time_s`` into components.

    Must be chained FIRST, before the history writer: it mutates the components dict that
    the writer then serializes. ``elapsed_seconds`` from ``_optimize`` is cumulative for
    the current training segment, so the per-epoch wall time is its increment.
    """

    def __init__(self, counter: NFECounter | None):
        self.counter = counter
        self._prev_elapsed = 0.0

    def on_epoch(self, epoch: int, total_loss: float, components: dict, elapsed: float):
        if components is None:
            return
        if self.counter is not None:
            components["nfe"] = self.counter.reset()
        components["epoch_wall_time_s"] = round(max(elapsed - self._prev_elapsed, 0.0), 6)
        self._prev_elapsed = elapsed


class Stage2Snapshot:
    """Save ``model.state_dict()`` at chosen epochs. Analysis-only; changes no training state.

    Epoch semantics match ``Stage2Probe``: snapshot ``N`` is the model after exactly ``N``
    optimizer steps. Files are written once and never overwritten within a run.
    """

    def __init__(self, model, run_dir, epochs, *, extra: dict | None = None,
                 name_fmt: str = "checkpoint_stage2_epoch_{epoch}.pt"):
        self.model = model
        self.run_dir = Path(run_dir)
        self.epochs = {int(e) for e in epochs}
        self.extra = dict(extra or {})
        self.name_fmt = name_fmt
        self.written: list[Path] = []

    def save(self, epoch: int) -> Path | None:
        path = self.run_dir / self.name_fmt.format(epoch=epoch)
        if path.exists():                                  # never overwrite a snapshot
            return None
        payload = dict(self.extra)
        payload.update({"model_state": {k: v.detach().cpu().clone()
                                        for k, v in self.model.state_dict().items()},
                        "stage2_epoch": int(epoch),
                        "snapshot": True})
        torch.save(payload, path)
        self.written.append(path)
        return path

    def on_epoch(self, epoch: int, total_loss: float, components: dict, elapsed: float):
        done = epoch + 1                                   # after this epoch's optimizer step
        if done in self.epochs:
            self.save(done)

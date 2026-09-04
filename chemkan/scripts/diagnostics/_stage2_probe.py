r"""Stage-2 optimization probe (DIAGNOSTIC ONLY).

Records, at a few chosen epochs, whether Stage-2 training is moving toward igniting
behavior. Every measurement runs under ``torch.no_grad()`` on a deep-copied evaluation
path, so the probe can never contribute gradients or otherwise change training.

Recorded per probe epoch:

    epoch, stage2_loss, temperature_MSE (normalized), peak_T at the probe condition,
    ignites / ignition_delay_s, thermo_linear_norm, and the full coefficient vector.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

DEFAULT_PROBE_EPOCHS = (0, 10, 50, 100, 250, 500)


class Stage2Probe:
    """Per-epoch diagnostic callback for ``training._optimize``'s ``on_epoch`` hook.

    ``epoch`` semantics: probe ``0`` is the state BEFORE any optimizer step; probe ``N``
    is the state after exactly ``N`` steps.
    """

    def __init__(self, model, *, integrate_fn, input_norm, solver, full_norm, t, ref,
                 species, path, epochs=DEFAULT_PROBE_EPOCHS, condition="1050/0.9"):
        self.model = model
        self.integrate_fn = integrate_fn
        self.input_norm, self.solver, self.full_norm = input_norm, solver, full_norm
        self.t = np.asarray(t, dtype=float)
        self.ref = np.asarray(ref, dtype=float)
        self.species = list(species)
        self.epochs = set(int(e) for e in epochs)
        self.condition = condition
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = (["epoch", "stage2_loss", "temperature_MSE", "peak_T_K",
                         "ignites", "ignition_delay_s", "thermo_linear_norm"]
                        + [f"coeff_{s}" for s in self.species])
        self._fh = self.path.open("w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self._fields)
        self._w.writeheader()
        self._fh.flush()

    # -- measurement ---------------------------------------------------------
    def _ignition_delay(self, T, rise_threshold: float = 100.0):
        if float(T.max() - T[0]) < rise_threshold:
            return None
        return float(self.t[int(np.argmax(np.gradient(T, self.t)))])

    def probe(self, epoch: int, stage2_loss=None):
        """Measure and append one row. Never touches gradients or parameters."""
        was_training = self.model.training
        self.model.eval()
        u0 = torch.as_tensor(self.ref[0], dtype=torch.float32).unsqueeze(0)
        tt = torch.as_tensor(self.t, dtype=torch.float32)
        with torch.no_grad():
            pred = self.integrate_fn(self.model, self.input_norm, self.solver,
                                     u0, tt)[:, 0].cpu().numpy()
            w = self.model.thermo.linear.weight.detach().cpu().numpy().ravel().copy()
        if was_training:
            self.model.train()

        T = pred[:, -1]
        dN = (self.full_norm.normalize(torch.as_tensor(pred, dtype=torch.float32))
              - self.full_norm.normalize(torch.as_tensor(self.ref, dtype=torch.float32))).numpy()
        delay = self._ignition_delay(T)
        row = {"epoch": epoch,
               "stage2_loss": "" if stage2_loss is None else f"{float(stage2_loss):.6e}",
               "temperature_MSE": f"{float((dN[:, -1] ** 2).sum()):.6e}",
               "peak_T_K": round(float(T.max()), 1),
               "ignites": delay is not None,
               "ignition_delay_s": "" if delay is None else f"{delay:.6e}",
               "thermo_linear_norm": f"{float(np.linalg.norm(w)):.6e}"}
        row.update({f"coeff_{s}": f"{float(v):.6g}" for s, v in zip(self.species, w)})
        self._w.writerow(row)
        self._fh.flush()
        return row

    # -- training hook -------------------------------------------------------
    def on_epoch(self, epoch: int, total_loss: float, components: dict, elapsed: float):
        """Chainable ``on_epoch``: probes after step ``epoch+1`` when requested."""
        done = epoch + 1
        if done in self.epochs:
            self.probe(done, total_loss)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def chain(*callbacks):
    """Combine several ``on_epoch`` callbacks into one."""
    def _call(epoch, total_loss, components, elapsed):
        for cb in callbacks:
            if cb is not None:
                cb(epoch, total_loss, components, elapsed)
    return _call

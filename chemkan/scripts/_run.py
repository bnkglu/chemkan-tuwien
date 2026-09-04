r"""Run-directory management for reproducible ChemKAN training runs.

This module is *plumbing only* -- it does not touch model mathematics, losses, the
solver, or datasets. It gives the training scripts a single place to:

* create one directory per run (``RUN_DIR``);
* refuse to silently overwrite a completed run (``checkpoint_final.pt``);
* write a human-readable ``config.json`` (mirror of the authoritative checkpoint);
* stream a ``run.log`` alongside the interactive tqdm bar;
* append a per-epoch training ``history*.csv`` that survives interruptions;
* maintain ONE resumable ``checkpoint_resume.pt`` (overwritten periodically) and
  delete it only after ``checkpoint_final.pt`` is written successfully.

The checkpoint remains the authoritative trained-model artifact; ``config.json`` is a
convenient human-readable mirror.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

CHECKPOINT_FINAL = "checkpoint_final.pt"
CHECKPOINT_RESUME = "checkpoint_resume.pt"
CONFIG_JSON = "config.json"
RUN_LOG = "run.log"
METRICS_JSON = "metrics.json"
PREDICTIONS_DIR = "predictions"

# All artifacts a run owns -- cleared by --overwrite so a replaced run starts fresh.
_MANAGED_ARTIFACTS = [CHECKPOINT_FINAL, CHECKPOINT_RESUME, CONFIG_JSON, RUN_LOG,
                      METRICS_JSON, "history.csv", "history_stage1.csv", "history_stage2.csv"]

# Config keys allowed to differ on --resume (runtime-only, not scientific). Epoch totals
# are excluded from the strict compare and enforced separately (may grow, never shrink
# below the completed epoch); provenance stamps are added by write_config, not the caller.
_RESUME_IGNORED_CONFIG_KEYS = {"device", "created", "git_commit", "run_id",
                               "epochs", "checkpoint_every", "parameter_count"}


def check_resume_config(saved: dict, requested: dict, ignore=_RESUME_IGNORED_CONFIG_KEYS):
    """Reject a resume whose scientific configuration differs from the original run.

    Compares every config key except the runtime-only ones in ``ignore``. Raises
    ``SystemExit`` naming the first mismatch, so a resumed run can never silently change
    the chemical system, architecture, dataset/split, seed, sensitivity backend, solver
    settings, learning rate, input scaling, loss/PINN config, hydrogen Stage-1 temperature
    provider, or noise configuration. Epoch totals are handled by the caller.
    """
    for key in sorted((set(saved) | set(requested)) - set(ignore)):
        if saved.get(key) != requested.get(key):
            raise SystemExit(
                f"--resume configuration mismatch on '{key}':\n"
                f"    original  = {saved.get(key)!r}\n"
                f"    requested = {requested.get(key)!r}\n"
                f"Refusing to change the scientific configuration of an existing run. "
                f"Start a new run directory instead (or omit --resume to see overwrite options).")


def git_commit() -> str:
    """Best-effort short git SHA; 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def derive_run_id(run_dir: Path, system: str) -> str:
    """Stable, descriptive run id from the run directory path.

    If the run dir lives under ``.../reproduction/<...>``, use that relative path
    (e.g. ``chemkan/hydrogen/main/direct_autograd_seed0``); otherwise fall back to
    ``<system>/<dirname>``. Deterministic for a given directory.
    """
    parts = run_dir.resolve().parts
    if "reproduction" in parts:
        i = parts.index("reproduction")
        return "/".join(parts[i + 1:])
    return f"{system}/{run_dir.name}"


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class HistoryWriter:
    """Append-per-epoch CSV writer that flushes each row (crash-safe).

    On resume it drops any existing rows whose ``epoch`` is >= ``resume_from`` so the
    continuation neither duplicates nor silently overwrites earlier epochs.
    """

    def __init__(self, path: Path, fieldnames: list[str], resume_from: int = 0):
        self.path = Path(path)
        self.fieldnames = fieldnames
        header_needed = True
        if resume_from > 0 and self.path.exists():
            kept = []
            with self.path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(float(row["epoch"])) < resume_from:
                        kept.append(row)
            with self.path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in kept:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
            header_needed = False
        self._fh = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if header_needed and self._fh.tell() == 0:
            self._writer.writeheader()
            self._fh.flush()

    def on_epoch(self, epoch: int, total_loss: float, components: dict, elapsed_seconds: float):
        """Adapter matching ``training._optimize``'s ``on_epoch`` callback signature.

        ``elapsed_seconds`` is **segment** wall time measured from the start of the current
        training call, so it RESETS to ~0 when a run resumes (it is not cumulative across
        resumes). Use the row ordering / ``epoch`` for progress; do not sum it across
        resume boundaries.
        """
        row = {"epoch": epoch, "total_loss": total_loss,
               "elapsed_seconds": round(elapsed_seconds, 4)}
        row.update(components or {})
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class RunManager:
    """One directory per training run, with resume + overwrite protection.

    Legacy mode: if ``run_dir`` is ``None`` the manager is inert -- the caller keeps its
    old ``--out`` single-file behavior. All run-dir artifacts are produced only when a
    ``run_dir`` is supplied.
    """

    def __init__(self, run_dir, system: str, *, resume: bool = False,
                 overwrite: bool = False):
        self.system = system
        self.enabled = run_dir is not None
        self.resume = resume
        self.overwrite = overwrite
        if not self.enabled:
            self.run_dir = None
            self.run_id = None
            return
        self.run_dir = Path(run_dir)
        self.run_id = derive_run_id(self.run_dir, system)
        self._log_handler = None

    # -- paths -----------------------------------------------------------------
    @property
    def final_path(self) -> Path:
        return self.run_dir / CHECKPOINT_FINAL

    @property
    def resume_path(self) -> Path:
        return self.run_dir / CHECKPOINT_RESUME

    @property
    def config_path(self) -> Path:
        return self.run_dir / CONFIG_JSON

    @property
    def log_path(self) -> Path:
        return self.run_dir / RUN_LOG

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / PREDICTIONS_DIR

    def resume_available(self) -> bool:
        return self.enabled and self.resume_path.exists()

    def _clean_managed_artifacts(self):
        """Remove all artifacts owned by a previous run so --overwrite starts fresh."""
        removed = []
        for name in _MANAGED_ARTIFACTS:
            p = self.run_dir / name
            if p.exists():
                p.unlink()
                removed.append(name)
        if self.predictions_dir.exists():
            shutil.rmtree(self.predictions_dir)
            removed.append(PREDICTIONS_DIR + "/")
        if removed:
            # Log handler not attached yet -> goes to console only, which is intended.
            logging.info("overwrite: cleared previous run artifacts: %s", ", ".join(removed))

    # -- lifecycle -------------------------------------------------------------
    def start(self):
        """Create the run dir, enforce overwrite/resume policy, attach run.log."""
        if not self.enabled:
            return
        if self.final_path.exists() and not self.overwrite and not self.resume:
            raise SystemExit(
                f"{self.final_path} already exists -- refusing to overwrite a completed run. "
                f"Pass --overwrite to replace it, or --resume to continue an interrupted run.")
        if self.resume and not self.resume_path.exists():
            raise SystemExit(
                f"--resume requested but no resumable checkpoint at {self.resume_path}.")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # --overwrite is a genuinely fresh run: clear every prior artifact (old histories,
        # a stale resume checkpoint, predictions/metrics) BEFORE opening the new run.log.
        if self.overwrite and not self.resume:
            self._clean_managed_artifacts()

        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        self._log_handler = handler
        logging.info("run start %s | run_id=%s | system=%s | dir=%s",
                     utc_now(), self.run_id, self.system, self.run_dir)

    def write_config(self, config: dict):
        if not self.enabled:
            return
        payload = dict(config)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("git_commit", git_commit())
        payload.setdefault("created", utc_now())
        with self.config_path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)
        logging.info("wrote %s", self.config_path)

    def history(self, name: str, fieldnames: list[str], resume_from: int = 0) -> HistoryWriter:
        if not self.enabled:
            return _NullHistory()
        return HistoryWriter(self.run_dir / name, fieldnames, resume_from=resume_from)

    def save_resume(self, state: dict):
        if not self.enabled:
            return
        _atomic_torch_save(state, self.resume_path)

    def load_resume(self) -> dict | None:
        if not self.enabled or not self.resume_path.exists():
            return None
        return torch.load(self.resume_path, map_location="cpu", weights_only=False)

    def save_final(self, checkpoint: dict):
        """Write the authoritative final checkpoint, then delete the resume file."""
        if not self.enabled:
            return
        checkpoint.setdefault("run_id", self.run_id)
        _atomic_torch_save(checkpoint, self.final_path)
        logging.info("wrote %s", self.final_path)
        # Only now that the final checkpoint is on disk do we drop the resume file.
        if self.resume_path.exists():
            self.resume_path.unlink()
            logging.info("removed %s (run completed)", self.resume_path)

    def finish(self, ok: bool = True):
        if not self.enabled:
            return
        logging.info("run %s %s", "completed" if ok else "interrupted", utc_now())
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None


class _NullHistory:
    """No-op history used in legacy (no run-dir) mode."""

    def on_epoch(self, *a, **k):
        pass

    def close(self):
        pass

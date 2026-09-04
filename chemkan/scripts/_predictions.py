r"""Prediction-artifact I/O with strict provenance + compatibility checking.

A prediction artifact is a compressed ``.npz`` that stores a trained model's
predictions together with everything needed to interpret them WITHOUT guessing and to
prove which checkpoint produced them. Metadata is stored as JSON strings / UTF-8 scalars
(no pickled Python objects), so artifacts load with ``allow_pickle=False``.

Compatibility rule (enforced by ``validate``): an artifact may be used for a checkpoint
only if ALL THREE of ``run_id``, ``architecture`` (canonical JSON), and
``checkpoint_sha256`` match. A mismatch is a rejection, not a warning -- callers must
regenerate predictions from the checkpoint instead of silently using stale ones.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def checkpoint_sha256(path) -> str:
    """SHA-256 hex digest of a checkpoint file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_arch(architecture) -> str:
    """Deterministic JSON string for an architecture dict (sorted keys)."""
    return json.dumps(architecture, sort_keys=True, default=str)


def save_predictions(path, *, run_id, checkpoint_sha256, architecture, predictions,
                     reference, t, initial_conditions, species, u_min, u_max,
                     metric_convention, eval_config, force=False) -> Path:
    """Write a prediction artifact with full provenance.

    ``reference`` MUST be the canonical ground truth copied from the dataset (never
    recomputed at evaluation time). ``metric_convention`` must state the metric in words
    (e.g. "Eq. 18 species MSE: mean over normalized species, over observation times"),
    never a bare ambiguous label. Refuses to overwrite an existing file unless ``force``.
    """
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists -- pass force=True (or --force) to overwrite, or write "
            f"to a distinct path/subdirectory that encodes the evaluation configuration.")
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": str(run_id),
        "checkpoint_sha256": str(checkpoint_sha256),
        "architecture_json": canonical_arch(architecture),
        "metric_convention": str(metric_convention),
        "eval_config_json": json.dumps(eval_config, sort_keys=True, default=str),
        "species": [str(s) for s in species],
    }
    np.savez_compressed(
        path,
        predictions=np.asarray(predictions),
        reference=np.asarray(reference),
        t=np.asarray(t),
        initial_conditions=np.asarray(initial_conditions),
        u_min=np.asarray(u_min),
        u_max=np.asarray(u_max),
        species=np.asarray([str(s) for s in species]),
        # provenance as a single JSON scalar (UTF-8, no pickle needed to read it)
        provenance=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    return path


def load_predictions(path) -> dict:
    """Load a prediction artifact. Metadata is parsed from the JSON provenance scalar.

    Uses ``allow_pickle=False`` -- artifacts must not require pickle for metadata.
    """
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in ("predictions", "reference", "t",
                                 "initial_conditions", "u_min", "u_max", "species")}
        meta = json.loads(str(d["provenance"]))
    out.update(meta)
    out["architecture"] = json.loads(meta["architecture_json"])
    out["eval_config"] = json.loads(meta["eval_config_json"])
    return out


def validate(artifact: dict, *, run_id, architecture, checkpoint_sha256) -> tuple[bool, str]:
    """Return (compatible, reason). All three of run_id / architecture / sha256 must match."""
    if str(artifact.get("run_id")) != str(run_id):
        return False, (f"run_id mismatch: artifact={artifact.get('run_id')!r} "
                       f"vs checkpoint={run_id!r}")
    if artifact.get("architecture_json") != canonical_arch(architecture):
        return False, "architecture mismatch between artifact and checkpoint"
    if str(artifact.get("checkpoint_sha256")) != str(checkpoint_sha256):
        return False, "checkpoint_sha256 mismatch (artifact was made from a different checkpoint)"
    return True, "compatible"


class IncompatiblePredictionError(RuntimeError):
    """Raised when a cached prediction artifact does not belong to the checkpoint in scope."""


def load_compatible_predictions(path, *, checkpoint: dict, checkpoint_path) -> dict:
    """Load a prediction artifact ONLY if it belongs to ``checkpoint``; else raise.

    Recomputes the checkpoint's SHA-256 from ``checkpoint_path`` and compares run_id +
    architecture + checkpoint_sha256 (all three). This is the safe entry point for the
    reproduction notebooks: it cannot silently return another checkpoint's predictions.
    On incompatibility it raises ``IncompatiblePredictionError`` so the caller regenerates
    predictions from the checkpoint instead of using stale ones.
    """
    artifact = load_predictions(path)
    ok, reason = validate(
        artifact,
        run_id=checkpoint.get("run_id"),
        architecture=checkpoint["architecture"],
        checkpoint_sha256=checkpoint_sha256(checkpoint_path),
    )
    if not ok:
        raise IncompatiblePredictionError(
            f"rejected cached prediction {path}: {reason}. Regenerate from the checkpoint.")
    return artifact

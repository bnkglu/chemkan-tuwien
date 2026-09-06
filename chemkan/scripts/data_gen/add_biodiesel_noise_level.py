r"""Append one missing noise realization to an existing biodiesel.npz -- in place.

The Fig. 5 sweep needs eight noise levels (0/1/2/3/5/7/10/15 %); the shipped archive
carries seven. Regenerating the archive to add 3% would re-run the ODE solver and
re-derive the clean trajectories, so this script instead reads the archive's OWN clean
trajectories and applies ``common.add_noise`` with the generator's exact per-level
seeding rule::

    train: make_rng(seed + 1000 + round(level * 1000))
    test : make_rng(seed + 2000 + round(level * 1000))

Because that rule gives every level an independent stream, adding a level cannot alter
any existing one. The script verifies that: it re-reads the file it wrote and fails
unless every pre-existing array is bitwise identical.

Dry run by default -- ``--apply`` is required to touch the archive.

    python add_biodiesel_noise_level.py --level 0.03                # report only
    python add_biodiesel_noise_level.py --level 0.03 --apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from common import add_noise, make_rng


def tag_for(level: float) -> str:
    return f"{int(round(level * 100)):02d}"


def build(archive: dict, level: float, seed: int, mode: str) -> dict:
    """Return the two new noisy arrays for ``level`` (nothing else is touched)."""
    u_min, u_max = archive["u_min"], archive["u_max"]
    tag = tag_for(level)
    rng_tr = make_rng(seed + 1000 + int(round(level * 1000)))
    rng_te = make_rng(seed + 2000 + int(round(level * 1000)))
    return {
        f"train_states_noise{tag}": add_noise(archive["train_states"], level, rng_tr,
                                              mode, u_min, u_max),
        f"test_states_noise{tag}": add_noise(archive["test_states"], level, rng_te,
                                             mode, u_min, u_max),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", type=Path, default=Path("../../data/generated/biodiesel.npz"))
    p.add_argument("--level", type=float, required=True, help="noise fraction, e.g. 0.03")
    p.add_argument("--seed", type=int, default=0, help="generator seed (must match the archive)")
    p.add_argument("--noise-mode", choices=["multiplicative", "range"], default="multiplicative")
    p.add_argument("--apply", action="store_true", help="write the archive (default: dry run)")
    args = p.parse_args()

    with np.load(args.archive, allow_pickle=False) as f:
        archive = {k: f[k] for k in f.files}

    tag = tag_for(args.level)
    new_keys = [f"train_states_noise{tag}", f"test_states_noise{tag}"]
    present = [k for k in new_keys if k in archive]
    if present:
        raise SystemExit(f"{args.archive} already contains {present}; refusing to overwrite "
                         f"a stored noise realization.")

    added = build(archive, args.level, args.seed, args.noise_mode)
    levels = np.sort(np.unique(np.concatenate([archive["noise_levels"], [args.level]])))

    print(f"archive        : {args.archive}")
    print(f"stored levels  : {[f'{v:.2f}' for v in archive['noise_levels']]}")
    print(f"adding         : {args.level:.2f}  -> keys {new_keys}")
    print(f"seeds          : train {args.seed + 1000 + int(round(args.level * 1000))}  "
          f"test {args.seed + 2000 + int(round(args.level * 1000))}  (mode={args.noise_mode})")
    for k, v in added.items():
        clean = archive["train_states" if k.startswith("train") else "test_states"]
        print(f"  {k}: shape {v.shape}  max|noisy-clean| {np.abs(v - clean).max():.6g}  "
              f"t=0 exact {np.array_equal(v[:, 0, :], clean[:, 0, :])}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return 0

    backup = args.archive.with_suffix(".npz.bak")
    shutil.copy2(args.archive, backup)
    out = dict(archive)
    out.update(added)
    out["noise_levels"] = levels
    # np.savez_compressed appends '.npz' unless the path already ends in it, so the
    # temporary name must end in '.npz' for the atomic rename to find the file.
    tmp = args.archive.with_name(args.archive.name + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(args.archive)

    # Re-read and prove nothing pre-existing moved.
    with np.load(args.archive, allow_pickle=False) as f:
        after = {k: f[k] for k in f.files}
    for k, v in archive.items():
        if k == "noise_levels":
            continue
        if not np.array_equal(after[k], v):
            shutil.copy2(backup, args.archive)
            raise SystemExit(f"VERIFY FAILED on '{k}' -- archive restored from {backup}")
    print(f"\nwrote {args.archive}; all {len(archive) - 1} pre-existing arrays bitwise "
          f"identical. Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

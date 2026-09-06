r"""Generate the ChemKAN Figure-3 evaluation condition -- one dedicated initial state.

Figure 3 plots one explicitly published unseen condition::

    [TG]0 = 1.94   [ROH]0 = 1.43   DG = MG = GL = R'CO2R = 0   T = 334.8 K

That exact state is not in the canonical 10-case test split, so it is generated here as a
SEPARATE artifact. The canonical ``biodiesel.npz`` split, its trajectories and its stored
noise realizations are neither read for targets nor modified: this script only reuses the
same ODE right-hand side, the same solver settings and the same ``common.add_noise``
convention, so the Fig.-3 condition is produced exactly like the dataset it accompanies.

Contents of the artifact:

* ``t``, ``states``            -- the 30-point observation grid and its clean trajectory
* ``t_dense``, ``states_dense``-- a dense grid for the plotted continuous prediction and
                                 for the paper's "smooth between observations" claim
* ``states_noise{00,05,10,15}``-- deterministic noisy observations on the 30-point grid

Noise seeding is deterministic and cannot collide with the dataset's streams: the archive
uses ``seed + 1000 + round(level*1000)`` (train) and ``seed + 2000 + ...`` (test), while
this condition uses the separate ``seed + 3000 + ...`` family.

    python generate_biodiesel_fig3_condition.py                 # writes the artifact
    python generate_biodiesel_fig3_condition.py --dry-run       # report only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from common import add_noise, make_rng, metadata, save
from generate_biodiesel import SPECIES, integrate_case

# PAPER EXPLICIT (Sec. III A 1 / Fig. 3 caption).
TG0, ROH0, T_K = 1.94, 1.43, 334.8
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.15)          # the four Fig. 3 columns


def build(t_end: float, n_points: int, n_dense: int, seed: int, mode: str) -> dict:
    y0 = np.zeros(len(SPECIES))
    y0[0], y0[1] = TG0, ROH0

    t = np.linspace(0.0, t_end, n_points)                  # observation grid
    t_dense = np.linspace(0.0, t_end, n_dense)             # plotting grid
    states = np.clip(integrate_case(y0, T_K, t), 0.0, None)
    states_dense = np.clip(integrate_case(y0, T_K, t_dense), 0.0, None)

    out = {
        "t": t, "states": states,
        "t_dense": t_dense, "states_dense": states_dense,
        "species": np.array(SPECIES),
        "y0": y0, "T": np.array(T_K),
        "state_layout": np.array("species_only"),
        "noise_levels": np.array(NOISE_LEVELS),
    }
    # add_noise expects (cases, times, vars); this artifact holds a single case.
    for lvl in NOISE_LEVELS:
        tag = f"{int(round(lvl * 100)):02d}"
        rng = make_rng(seed + 3000 + int(round(lvl * 1000)))
        out[f"states_noise{tag}"] = add_noise(states[None], lvl, rng, mode)[0]
    out["metadata"] = np.array(metadata(
        system="biodiesel", generator="generate_biodiesel_fig3_condition.py", seed=seed,
        mechanism="biodiesel_transesterification", species=SPECIES,
        n_points=n_points, n_points_dense=n_dense, t_end_s=t_end,
        initial_conditions={"TG": TG0, "ROH": ROH0, "T_K": T_K},
        noise_mode=mode, noise_seed_family="seed + 3000 + round(level*1000)",
    ))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=Path("../../data/generated/biodiesel_fig3_condition.npz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-end", type=float, default=30.0)
    p.add_argument("--n-points", type=int, default=30,
                   help="observation grid; matches the canonical dataset")
    p.add_argument("--n-dense", type=int, default=601,
                   help="dense plotting grid (REPRODUCTION CHOICE: plotting only)")
    p.add_argument("--noise-mode", choices=["multiplicative", "range"],
                   default="multiplicative")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    data = build(args.t_end, args.n_points, args.n_dense, args.seed, args.noise_mode)
    print(f"Fig. 3 condition: TG0={TG0}, ROH0={ROH0}, T={T_K} K  (PAPER EXPLICIT)")
    print(f"  observations {data['states'].shape} on {args.n_points} points over "
          f"{args.t_end} s; dense {data['states_dense'].shape}")
    for i, s in enumerate(SPECIES):
        print(f"    {s:>6}: [{data['states'][:, i].min():.4f}, "
              f"{data['states'][:, i].max():.4f}]")
    for lvl in NOISE_LEVELS:
        tag = f"{int(round(lvl * 100)):02d}"
        d = np.abs(data[f"states_noise{tag}"] - data["states"]).max()
        print(f"  noise {lvl:.0%}: max|noisy-clean| {d:.6g}  "
              f"t=0 exact {np.array_equal(data[f'states_noise{tag}'][0], data['states'][0])}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return 0
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; delete it explicitly to regenerate.")
    save(args.out, **data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

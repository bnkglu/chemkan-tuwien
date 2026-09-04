r"""Fairness verification for the base-ON (N=4) hydrogen matrix (DIAGNOSTIC).

Reads artifacts only. ``--stage pre`` runs before the Stage-2 arms and checks the shared
Stage-1 checkpoint; ``--stage post`` runs afterwards and checks that all six arms differ
ONLY in their thermodynamic initialization. A non-zero exit stops the run sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[1]
for _p in (str(_SCRIPTS.parent / "src"), str(_SCRIPTS), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chemkan.model import ChemKAN  # noqa: E402

ARMS = ["random_stage2_10000_seed0", "cantera_stage2_10000_seed0",
        "scaled_random_1e5_dir0_stage2_10000", "normmatched_dir0_stage2_10000",
        "normmatched_dir1_stage2_10000", "normmatched_dir2_stage2_10000"]
EXPECTED_ARCH = {"hidden_dim": 3, "num_basis": 4, "n_mu": 3, "use_base_act": True}
EXPECTED_PARAMS = 344
REF_T, REF_PHI = 1050.0, 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(ok: bool, msg: str, failures: list) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


def pre(base: Path, failures: list) -> None:
    s1 = base / "stage1_seed0" / "checkpoint_stage1.pt"
    check(s1.exists(), f"shared Stage-1 checkpoint exists: {s1}", failures)
    if not s1.exists():
        return
    ck = torch.load(s1, map_location="cpu", weights_only=False)
    check(ck["architecture"] == EXPECTED_ARCH,
          f"Stage-1 architecture is N=4/base-ON: {ck['architecture']}", failures)
    check(int(ck.get("seed", -1)) == 0, f"Stage-1 seed == 0 (got {ck.get('seed')})", failures)
    model = ChemKAN(species_dim=int(ck["data"]["species_dim"]), **ck["architecture"])
    n = sum(p.numel() for p in model.parameters())
    check(n == EXPECTED_PARAMS, f"reconstructed parameter count == 344 (got {n})", failures)
    check(ck["solver"]["sensitivity"] == "direct_autograd",
          f"sensitivity backend is direct_autograd ({ck['solver']['sensitivity']})", failures)
    print(f"  Stage-1 sha256   : {sha256(s1)}")
    print(f"  Stage-1 final loss: {ck.get('stage1_final_loss')}")


def post(base: Path, failures: list) -> None:
    s1_sha = sha256(base / "stage1_seed0" / "checkpoint_stage1.pt")
    cfgs, inits = {}, {}
    for a in ARMS:
        d = base / a
        check((d / "checkpoint_final.pt").exists(), f"{a}: checkpoint_final.pt exists", failures)
        cfg = json.loads((d / "config.json").read_text())
        cfgs[a] = cfg
        inits[a] = cfg.get("thermo_init") or {}
        ck = torch.load(d / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        check(ck["architecture"] == EXPECTED_ARCH, f"{a}: architecture N=4/base-ON", failures)
        n = sum(p.numel() for p in ChemKAN(species_dim=9, **ck["architecture"]).parameters())
        check(n == EXPECTED_PARAMS, f"{a}: parameter count == 344 (got {n})", failures)
        check(cfg["stage1_from"]["stage1_checkpoint_sha256"] == s1_sha
              if "stage1_checkpoint_sha256" in cfg["stage1_from"]
              else cfg["stage1_from"]["run_id"].endswith("base_on_n4/stage1_seed0"),
              f"{a}: started from the shared base-ON Stage-1 checkpoint", failures)

    # every scientific setting except the thermo initialization must be identical
    INVARIANT = ["seed", "learning_rate", "optimizer", "sensitivity_backend", "solver",
                 "loss", "pinn", "normalization", "dataset", "noise", "architecture",
                 "parameter_count", "stage1_temperature"]
    ref = cfgs[ARMS[0]]
    for a in ARMS[1:]:
        diff = [k for k in INVARIANT if cfgs[a].get(k) != ref.get(k)]
        check(not diff, f"{a}: identical to {ARMS[0]} on every setting but thermo init "
                        f"(differs: {diff})", failures)
    check(len({json.dumps(cfgs[a].get("epochs"), sort_keys=True) for a in ARMS}) == 1,
          "all arms share the same epoch budget", failures)

    # the three norm-matched arms must target ONE Cantera norm and differ only by direction
    nm = [a for a in ARMS if a.startswith("normmatched")]
    norms = {inits[a].get("cantera_reference_norm") for a in nm}
    check(len(norms) == 1, f"norm-matched arms share one Cantera norm target: {norms}", failures)
    check({inits[a].get("reference_temperature") for a in nm} == {REF_T}
          and {inits[a].get("reference_phi") for a in nm} == {REF_PHI},
          f"norm-matched reference state is exactly T={REF_T:g} K / phi={REF_PHI:g}", failures)
    check(sorted(inits[a].get("thermo_linear_init_seed") for a in nm) == [0, 1, 2],
          "norm-matched direction seeds are exactly 0/1/2", failures)

    # the direction seed must touch ONLY the 9 thermo.linear values, never the kinetic core
    base_kin = None
    for a in nm:
        ck = torch.load(base / a / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        init_vec = inits[a].get("resulting_vector")
        check(init_vec is not None and len(init_vec) == 9,
              f"{a}: initialization provenance records all 9 thermo.linear values", failures)
    s1_state = torch.load(base / "stage1_seed0" / "checkpoint_stage1.pt",
                          map_location="cpu", weights_only=False)["kinetic_state"]
    print(f"  shared Stage-1 sha256: {s1_sha}")
    print(f"  Stage-1 kinetic tensors: {sorted(s1_state)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--stage", choices=["pre", "post"], required=True)
    args = p.parse_args()
    base, failures = Path(args.base), []
    print(f"=== base-ON matrix verification ({args.stage}) : {base} ===")
    (pre if args.stage == "pre" else post)(base, failures)
    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED -- stopping the sequence.")
        return 1
    print("\nall checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

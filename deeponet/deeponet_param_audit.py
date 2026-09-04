"""
DeepONet parameter-count audit for the ChemKAN biodiesel comparison (Sec. III A 3).

Point this at the unzipped lululxvi/deeponet archive. It exercises BOTH code
paths that repo contains and reports parameter counts via torch.

  PATH 1 -- src/deeponet_pde.py
      That file is TensorFlow 1.x + DeepXDE. It builds its network with

          dde.maps.OpNN([m, 40, 40], [dim_x, 40, 40], activation, initializer,
                        use_bias=True, stacked=False)

      and counts parameters with tf.trainable_variables(), so torch cannot be
      called on it directly. What transfers is the LAYER-SIZE CONVENTION: the
      lists include the input dimension, every Linear is biased, and `use_bias`
      adds ONE scalar after the branch/trunk product. This script rebuilds that
      exact network through modern DeepXDE's `dde.nn.DeepONet` on the pytorch
      backend, so the same architecture is counted with sum(p.numel()).

  PATH 2 -- seq2seq/learner/nn/{fnn,deeponet}.py
      Native PyTorch. Imported directly from the uploaded archive.

Neither path can express the ChemKAN paper's biodiesel network as-is: both
produce a SCALAR output (dot product plus one bias), while the paper needs a
6-species vector via Eq. 4's MLP_opt, and a trunk of 7-then-8 rather than one
shared width. MLP_opt is ChemKAN's own addition. The candidate readings at the
end therefore use the repo's FNN for branch and trunk, adding only that layer.

Dependency / provenance
-----------------------
PATH 2 reads the UPSTREAM reference implementation, which is NOT vendored in this
repository: https://github.com/lululxvi/deeponet (Lu, Jin, Pang, Zhang & Karniadakis,
Nat. Mach. Intell. 3, 218-229, 2021). Version/commit used: not recorded. Obtain a
checkout and pass it via --repo; without one, PATH 2 is unavailable.

The biodiesel DeepONet reconstruction itself is in ``biodiesel_deeponet_smoke.py`` and
has NO dependency on this upstream checkout.

Usage
-----
    pip install torch deepxde        # deepxde optional; PATH 1 is skipped without it
    python deeponet_param_audit.py --repo /path/to/deeponet

If --repo is omitted the script looks for ./deeponet next to itself.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import torch
import torch.nn as nn

TARGET = 308  # count stated in the ChemKAN paper for the noisy-comparison DeepONet


def find_repo(explicit=None):
    here = Path(__file__).resolve().parent
    cands = ([Path(explicit)] if explicit else []) + [
        here / "deeponet", here / "deeponet_ref", here,
    ]
    for c in cands:
        if (c / "seq2seq" / "learner" / "nn" / "fnn.py").exists():
            return c
    raise SystemExit(
        "Could not find the deeponet repo. Unzip deeponet.zip next to this "
        "script, or pass --repo /path/to/deeponet"
    )


def n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def path1_deepxde():
    try:
        import deepxde as dde
    except ImportError:
        print("\n[PATH 1] deepxde not installed -- skipping (pip install deepxde).")
        return

    print("\n" + "=" * 74)
    print("PATH 1  src/deeponet_pde.py architecture, counted with torch")
    print("=" * 74)
    print("deeponet_pde.py calls dde.maps.OpNN(...) under TF1; below is the same")
    print("network through dde.nn.DeepONet on the pytorch backend.\n")

    ref = dde.nn.DeepONet([100, 40, 40], [1, 40, 40], "relu", "Glorot normal")
    print(f"  repo default  branch [100,40,40]  trunk [1,40,40]  -> {n_params(ref)}")

    paper = dde.nn.DeepONet([7, 8, 8, 8], [1, 7, 8], "relu", "Glorot normal")
    print(f"  paper widths  branch [7,8,8,8]    trunk [1,7,8]    -> {n_params(paper)}")
    for name, p in paper.named_parameters():
        print(f"      {name:26s} {str(tuple(p.shape)):10s} {p.numel():4d}")
    print("  Scalar output: sum(branch*trunk) + b, where `b` is the trailing 1.")

    print("\n  DeepXDE multi-output strategies for a 6-species vector:")
    for strat in ["independent", "split_branch", "split_trunk", "split_both"]:
        try:
            m = dde.nn.DeepONet([7, 8, 8, 8], [1, 7, 8], "relu", "Glorot normal",
                                num_outputs=6, multi_output_strategy=strat)
            print(f"      {strat:14s} -> {n_params(m)}")
        except Exception as exc:
            print(f"      {strat:14s} -> unavailable ({type(exc).__name__})")
    print("  None reaches 308; a sweep over widths and latent sizes divisible")
    print("  by 6 found no exact match either.")


def path2_native(repo):
    sys.path.insert(0, str(repo / "seq2seq"))
    from learner.nn import FNN, DeepONet as RefDeepONet

    print("\n" + "=" * 74)
    print("PATH 2  seq2seq/learner/nn/ -- the repo's native PyTorch modules")
    print("=" * 74)
    ref = RefDeepONet(branch_dim=7, trunk_dim=1,
                      branch_depth=3, trunk_depth=3, width=8)
    print("\n  learner.nn.DeepONet(branch_dim=7, trunk_dim=1, branch_depth=3,")
    print(f"                      trunk_depth=3, width=8)  -> {n_params(ref)}")
    print("  One shared width, scalar output, no MLP_opt -- cannot express the")
    print("  paper's 7-then-8 trunk or its 6-dim output.")
    return FNN


def make_chemkan_deeponet(branch_fnn, trunk_fnn, latent=8, out_dim=6,
                          combine="hadamard", branch_dims=None, trunk_dims=None):
    """Eq. 4:  u(u0, t) = MLP_opt[ MLP_br(u0) (combine) MLP_tr(t) ]

    branch_fnn / trunk_fnn are ALREADY-BUILT library FNN modules (dde.nn.FNN or
    the repo's learner.nn.FNN). Only MLP_opt is added here: it is ChemKAN's own
    layer, absent from both DeepONet implementations, which end in a scalar
    sum(branch * trunk) + b and cannot emit a 6-species vector.
    """

    class ChemKANDeepONet(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch = branch_fnn
            self.trunk = trunk_fnn
            self.combine = combine
            self.opt = nn.Linear(latent if combine == "hadamard" else 2 * latent,
                                 out_dim)

        def forward(self, u0, t):
            b, tr = self.branch(u0), self.trunk(t)
            z = b * tr if self.combine == "hadamard" else torch.cat([b, tr], -1)
            return self.opt(z)

    m = ChemKANDeepONet()
    m.branch_dims = tuple(branch_dims)
    m.trunk_dims = tuple(trunk_dims)
    return m


def dde_candidate(branch_dims, trunk_dims, combine="hadamard"):
    """Build a candidate with DeepXDE's own FNN blocks."""
    import deepxde as dde
    return make_chemkan_deeponet(
        dde.nn.FNN(list(branch_dims), "relu", "Glorot normal"),
        dde.nn.FNN(list(trunk_dims), "relu", "Glorot normal"),
        latent=branch_dims[-1], combine=combine,
        branch_dims=branch_dims, trunk_dims=trunk_dims,
    )


def report(model, label):
    total = n_params(model)
    mark = "   <-- equals the paper's 308" if total == TARGET else ""
    print(f"\n  {label}")
    print(f"    branch  FNN{model.branch_dims}  = {n_params(model.branch):4d}")
    print(f"    trunk   FNN{model.trunk_dims}  = {n_params(model.trunk):4d}")
    print(f"    MLP_opt Linear({model.opt.in_features},{model.opt.out_features})"
          f" [{model.combine}]  = {n_params(model.opt):4d}")
    print(f"    TOTAL = {total}   (paper: {TARGET}, off by {total - TARGET:+d}){mark}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None,
                    help="path to the unzipped lululxvi/deeponet repo")
    args = ap.parse_args()

    repo = find_repo(args.repo)
    print(f"repo: {repo}")
    torch.manual_seed(0)

    path1_deepxde()
    RepoFNN = path2_native(repo)

    print("\n" + "=" * 74)
    print("ChemKAN Eq. 4 readings -- branch/trunk are dde.nn.FNN, MLP_opt added")
    print("=" * 74)
    print("Neither DeepONet class can be used whole: both return a scalar")
    print("(sum(branch*trunk) + b), so the 6-species head has to be attached.")

    cands = [
        ((7, 8, 8, 8), (1, 7, 8), "hadamard",
         "LITERAL (Eq. 4 Hadamard): branch [7,8,8,8], trunk [1,7,8]"),
        ((7, 8, 8, 8), (1, 7, 8), "concat",
         "LITERAL but concatenating the two 8-dim layers"),
        ((6, 8, 8), (1, 7, 8), "concat",
         "CANDIDATE A: branch [6,8,8] (T dropped, 2 layers), concat"),
        ((7, 7, 7, 8), (1, 7, 8), "hadamard",
         "CANDIDATE B: branch [7,7,7,8] (width 7, not 8), Hadamard"),
    ]
    for bd, td, comb, label in cands:
        report(dde_candidate(bd, td, comb), label)

    # Cross-check: the repo's own FNN must give the same branch/trunk counts.
    print("\ncross-check, dde.nn.FNN vs the repo's learner.nn.FNN:")
    for dims in [(7, 8, 8, 8), (6, 8, 8), (7, 7, 7, 8), (1, 7, 8)]:
        import deepxde as dde
        a = n_params(dde.nn.FNN(list(dims), "relu", "Glorot normal"))
        b = n_params(RepoFNN(dims[0], dims[-1], len(dims) - 1, dims[1],
                             "relu", "Glorot normal"))
        print(f"  FNN{dims}: dde.nn.FNN = {a:4d}   learner.nn.FNN = {b:4d}"
              f"   {'agree' if a == b else 'DISAGREE'}")

    m = dde_candidate((7, 8, 8, 8), (1, 7, 8), "hadamard")
    B, N = 4, 30
    u0 = torch.randn(B, 1, 7).expand(B, N, 7)
    t = torch.linspace(1, 30, N).view(1, N, 1).expand(B, N, 1)
    y = m(u0, t)
    assert y.shape == (B, N, 6)
    print(f"\nshape check: u0 {tuple(u0.shape)}, t {tuple(t.shape)} -> "
          f"{tuple(y.shape)}  (expected ({B}, {N}, 6))")

    print("\nsum(p.numel() for p in m.parameters() if p.requires_grad) = "
          f"{sum(p.numel() for p in m.parameters() if p.requires_grad)}"
          "   [literal Eq. 4 model]")


if __name__ == "__main__":
    main()
"""biodiesel_v2.npz: released-code reaction order, the authors' IC file, additive species-max
noise (authors' clarification, email B5), and the 3-input DeepONet (email B3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for sub in ("chemkan/scripts/data_gen", "chemkan/scripts", "chemkan/scripts/author_repo_match",
            "deeponet"):
    sys.path.insert(0, str(REPO / sub))

import biodiesel_deeponet as bdon  # noqa: E402
import generate_biodiesel as gb  # noqa: E402
from _data import load_biodiesel  # noqa: E402
from common import add_noise, make_rng  # noqa: E402

V2 = REPO / "chemkan/data/generated/biodiesel_v2.npz"


@pytest.fixture(scope="module")
def generated():
    """A fresh in-memory generation with the default (current) settings."""
    return gb.generate(gb.build_parser().parse_args([]))


# --------------------------------------------------------------------------- noise

def test_additive_species_max_formula_and_clipping():
    rng = np.random.default_rng(0)
    states = rng.uniform(0.0, 2.0, size=(3, 5, 4))
    states[:, :, 3] = 0.0                                     # a species that is always 0
    noisy = add_noise(states, 0.1, make_rng(7), "additive_species_max")
    z = make_rng(7).standard_normal(states.shape)
    want = np.clip(states + 0.1 * states.max(axis=1, keepdims=True) * z, 0.0, None)
    assert np.array_equal(noisy, want)
    assert (noisy >= 0).all()
    big = add_noise(states, 5.0, make_rng(1), "additive_species_max")
    assert (big == 0).any() and (big >= 0).all()             # clipping is active
    assert np.array_equal(noisy[..., 3], np.zeros((3, 5)))   # max 0 -> no noise


def test_t0_observation_noised_but_initial_condition_clean(generated):
    lvl = "05"
    clean, noisy = generated["train_states"], generated[f"train_states_noise{lvl}"]
    nonzero = clean[:, 0, :] > 0
    assert not np.allclose(noisy[:, 0, :][nonzero], clean[:, 0, :][nonzero])   # t=0 noised
    d = load_biodiesel("train", noise_percent=5, data_file=V2)
    assert torch.equal(d["Y0"], torch.as_tensor(np.load(V2)["train_states"][:, 0, :],
                                                 dtype=torch.float32))          # IC clean
    assert torch.equal(d["targets_TBm"][0],
                       torch.as_tensor(np.load(V2)["train_states_noise05"][:, 0, :],
                                       dtype=torch.float32))


def test_noise_free_test_copy_unchanged(generated):
    assert np.array_equal(generated["test_states_noise00"], generated["test_states"])
    fresh = np.stack([gb.integrate_case(y0, T, generated["t"], "released_code")
                      for y0, T in zip(generated["test_states"][:, 0, :], generated["test_T"])])
    assert np.allclose(np.clip(fresh, 0.0, None), generated["test_states"], rtol=0, atol=1e-12)
    for lvl in ("05", "15"):
        assert not np.array_equal(generated[f"test_states_noise{lvl}"], generated["test_states"])


def test_seed_reproducibility_and_recorded_seeds(generated):
    again = gb.generate(gb.build_parser().parse_args([]))
    for k, v in generated.items():
        if k != "metadata":
            assert np.array_equal(v, again[k]), k
    meta = json.loads(str(generated["metadata"]))
    for lvl, row in zip(generated["noise_levels"], generated["noise_seeds"]):
        tag = f"{int(round(lvl * 100)):02d}"
        assert list(row) == list(gb.noise_seeds(0, lvl)) == [meta["noise_seeds"][tag]["train"],
                                                             meta["noise_seeds"][tag]["test"]]
    z = make_rng(gb.noise_seeds(0, 0.05)[0]).standard_normal(generated["train_states"].shape)
    clean = generated["train_states"]
    want = np.clip(clean + 0.05 * clean.max(axis=1, keepdims=True) * z, 0.0, None)
    assert np.array_equal(generated["train_states_noise05"], want)


def test_stored_v2_file_is_what_the_generator_produces(generated):
    stored = np.load(V2)
    for k, v in generated.items():
        if k != "metadata":
            assert np.array_equal(stored[k], v), k
    meta = json.loads(str(stored["metadata"]))
    assert meta["reaction_order"] == "released_code" and meta["ic_source"] == "authors_file"
    assert meta["noise_mode"] == "additive_species_max"
    assert meta["ic_file_sha256"] == gb.file_sha256(gb.AUTHORS_IC_FILE)


# --------------------------------------------------------------------------- data

def test_reaction_orders():
    rc, pt = gb.KINETICS["released_code"], gb.KINETICS["paper_text"]
    assert rc["ln_A"].tolist() == [18.60, 19.13, 7.93] and rc["Ea_kcal"].tolist() == [14.54, 14.42, 6.47]
    assert pt["ln_A"].tolist() == [18.60, 7.93, 19.13] and pt["Ea_kcal"].tolist() == [14.54, 6.47, 14.42]
    k_rc, k_pt = gb.rate_constants(330.0, "released_code"), gb.rate_constants(330.0, "paper_text")
    assert k_rc[0] == k_pt[0] and k_rc[1] == k_pt[2] and k_rc[2] == k_pt[1]     # 2 <-> 3


def test_authors_ic_scaling_row24():
    y0, T = gb.authors_initial_conditions()
    assert y0.shape == (30, 6) and (y0[:, 2:] == 0).all()
    assert y0[23, 0] == pytest.approx(1.941, abs=5e-4)
    assert y0[23, 1] == pytest.approx(1.433, abs=5e-4)
    assert T[23] == pytest.approx(334.75, abs=5e-3)


def test_clean_trajectories_match_independent_recipe(generated):
    """Same ICs and code-order kinetics, integrated by build_reaction_order_datasets.solve."""
    import build_reaction_order_datasets as ro
    y0, T = gb.authors_initial_conditions()
    kin = ro.ORDERS["code"]
    ref = np.stack([ro.solve(y0[i], T[i], kin["lnA"], kin["Ea"], generated["t"]).T
                    for i in range(30)])
    clean = np.concatenate([generated["train_states"], generated["test_states"]])
    assert np.abs(clean - ref).max() < 1e-7


def test_per_level_normalization_from_training_data():
    stored = np.load(V2)
    d = load_biodiesel("train", noise_percent=5, data_file=V2)
    assert d["normalization_keys"] == ["u_min_noise05", "u_max_noise05"]
    noisy = stored["train_states_noise05"]
    assert np.allclose(d["u_min"].numpy(), noisy.min(axis=(0, 1)), atol=1e-6)
    legacy = load_biodiesel("train", noise_percent=5)                 # legacy archive
    assert legacy["normalization_keys"] == ["u_min", "u_max"]


# --------------------------------------------------------------------------- DeepONet

def test_deeponet_three_input_branch_has_308_parameters():
    model = bdon.build(8, seed=0)
    assert model.branch_dims == (3, 8, 8, 8) and model.trunk_dims == (1, 7, 8)
    assert bdon.n_params(model) == 308 == bdon.PAPER_PARAMS
    assert bdon.n_params(bdon.build(8, seed=0, branch_in=bdon.LEGACY_BRANCH_IN)) == 340
    for w in (2, 5, 10):
        assert bdon.n_params(bdon.build(w)) == 3 * w * w + 14 * w + 4


def test_legacy_checkpoint_still_reconstructs_with_seven_inputs():
    from evaluate_biodiesel_deeponet import build_model
    path = (REPO / "results/reproduction/legacy/biodiesel/deeponet/reference_final_trunk_relu"
            / "noise/noise00_seed0/checkpoint_final.pt")
    if not path.exists():
        pytest.skip("legacy DeepONet checkpoint not present")
    model = build_model(torch.load(path, map_location="cpu", weights_only=False), "cpu")
    assert model.branch_dims[0] == 7 and bdon.n_params(model) == 340


# --------------------------------------------------------------------------- decisions

LEGACY = REPO / "chemkan/data/generated/biodiesel_legacy.npz"
LEGACY_SHA256 = "347e39d3113896d85628c3c8ac0a85d9be30bf946e57807d229c8b9f25f32023"


def test_legacy_file_renamed_unchanged_and_default():
    assert not (REPO / "chemkan/data/generated/biodiesel.npz").exists()
    assert gb.file_sha256(LEGACY) == LEGACY_SHA256
    d = load_biodiesel("train")                                  # default = legacy file
    assert np.array_equal(d["species_TBm"].permute(1, 0, 2).numpy(),
                          np.load(LEGACY)["train_states"].astype(np.float32))


def test_v2_has_eight_additive_levels():
    meta = json.loads(str(np.load(V2)["metadata"]))
    assert meta["noise_levels"] == [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
    assert {v["noise_mode"] for v in meta["noise_seeds"].values()} == {"additive_species_max"}


@pytest.mark.parametrize("n, arch", sorted(bdon.FIG4_ARCHITECTURES.items()))
def test_fig4_deeponet_architectures(n, arch):
    w, q, p = arch
    model = bdon.build(w, trunk_hidden=q)
    assert model.branch_dims == (3, w, w, p) and model.trunk_dims == (1, q, p)
    assert bdon.n_params(model) == n == w * w + w * p + q * p + 5 * w + 2 * q + 8 * p + 6
    rec = bdon.architecture(model)
    assert (rec["w"], rec["q"], rec["p"]) == arch


def test_deeponet_trunk_hidden_round_trip():
    from evaluate_biodiesel_deeponet import build_model
    model = bdon.build(7, seed=0, trunk_hidden=6)
    rebuilt = build_model({"architecture": bdon.architecture(model),
                           "model_state": model.state_dict()}, "cpu")
    assert rebuilt.trunk_dims == (1, 6, 7) and bdon.n_params(rebuilt) == 249


@pytest.mark.parametrize("hidden, n", [(2, 78), (3, 117), (4, 156), (9, 351), (11, 429)])
def test_author_core_hidden_widths(hidden, n):
    import math

    from _author_match import build_core
    core = build_core(hidden)
    assert sum(p.numel() for p in core.parameters()) == n == 39 * hidden
    assert core.lean.n_mu == math.ceil(hidden / 2)

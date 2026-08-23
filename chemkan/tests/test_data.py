"""Train/test split loading + hydrogen species-order validation (script helpers)."""

import pytest
import torch
from _chemistry import SPECIES, assert_species_order
from _data import load_biodiesel, load_hydrogen


def _load_or_skip(fn, **kw):
    try:
        return fn(**kw)
    except FileNotFoundError:
        pytest.skip("generated .npz data not available in this environment")


# --- split behavior ---------------------------------------------------------------

def test_biodiesel_split_selects_different_states():
    train = _load_or_skip(load_biodiesel, split="train")
    test = _load_or_skip(load_biodiesel, split="test")
    # different number of trajectories (B) between splits
    assert train["species_TBm"].shape[1] != test["species_TBm"].shape[1]


def test_hydrogen_split_selects_different_states():
    train = _load_or_skip(load_hydrogen, split="train")
    test = _load_or_skip(load_hydrogen, split="test")
    assert train["full_TBm1"].shape[1] != test["full_TBm1"].shape[1]


def test_normalization_stats_are_train_only_across_splits():
    # u_min/u_max must be identical regardless of split (always train-derived).
    for loader in (load_biodiesel, load_hydrogen):
        train = _load_or_skip(loader, split="train")
        test = _load_or_skip(loader, split="test")
        assert torch.allclose(train["u_min"], test["u_min"])
        assert torch.allclose(train["u_max"], test["u_max"])


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        load_biodiesel(split="validation")


# --- hydrogen species order -------------------------------------------------------

def test_species_order_correct_passes():
    assert_species_order(SPECIES)                         # exact match -> no error


def test_species_order_wrong_fails():
    wrong = list(reversed(SPECIES))
    with pytest.raises(ValueError):
        assert_species_order(wrong)

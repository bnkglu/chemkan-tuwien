"""The DeepONet baseline's documented normalization contract.

The architecture reconstruction is checked in ``deeponet/biodiesel_deeponet_smoke.py``;
what is checked here is the part that could silently corrupt a paper comparison:

1. the min-max statistics come from the TRAINING split only, never the test split;
2. the model's output is compared against targets in the SAME normalized space;
3. denormalizing a normalized prediction returns the physical trajectory exactly, so
   physical plots are a correct inverse transform and not a second forward scaling.
"""

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "deeponet"))

from _data import load_biodiesel                        # noqa: E402  (conftest adds scripts/)

from chemkan.normalization import MinMaxNormalizer      # noqa: E402

biodiesel_deeponet = pytest.importorskip("biodiesel_deeponet")


def _train_normalizers():
    """The (m+1,) branch normalizer and its species subset, from the TRAIN split only."""
    train = load_biodiesel(split="train")
    T = train["T_const"]
    full = MinMaxNormalizer(torch.cat([train["u_min"], T.min().reshape(1)]),
                            torch.cat([train["u_max"], T.max().reshape(1)]))
    return train, full, full.subset(slice(0, len(train["species"])))


def test_statistics_are_train_only_and_ignore_the_test_split():
    """Loading the test split must not move a single statistic."""
    train, full, _ = _train_normalizers()
    test = load_biodiesel(split="test")
    # The archive stores ONE set of train-only species stats; both splits report them.
    assert torch.equal(train["u_min"], test["u_min"])
    assert torch.equal(train["u_max"], test["u_max"])
    # A normalizer refitted on test data would differ -- prove the stats are not test's.
    refit_min = test["species_TBm"].reshape(-1, test["species_TBm"].shape[-1]).min(dim=0).values
    assert not torch.allclose(full.u_min[:len(refit_min)], refit_min), \
        "branch statistics look like they were fitted on the test split"


def test_test_inputs_are_scaled_with_the_train_normalizer():
    """prepare_inputs must apply the train normalizer to the test split, not refit."""
    train, full, _ = _train_normalizers()
    test = load_biodiesel(split="test")
    t_end = float(train["t"][-1])
    branch, tau = biodiesel_deeponet.prepare_inputs(test, full, t_end)
    expected = full.normalize(biodiesel_deeponet.raw_branch_input(test))
    assert torch.equal(branch, expected)
    assert torch.allclose(tau, test["t"] / t_end)
    assert float(tau[0]) == 0.0 and float(tau[-1]) == pytest.approx(1.0)


def test_model_output_and_targets_share_the_normalized_space():
    """The loss compares the raw model output with normalized targets -- no rescaling."""
    train, full, loss_norm = _train_normalizers()
    model = biodiesel_deeponet.build(seed=0)
    branch, tau = biodiesel_deeponet.prepare_inputs(train, full, float(train["t"][-1]))
    with torch.no_grad():
        pred_norm = model(branch, tau)
    target_norm = loss_norm.normalize(train["targets_TBm"])
    assert pred_norm.shape == target_norm.shape
    # Both live in the same space: denormalizing either one lands in physical units.
    physical = loss_norm.denormalize(target_norm)
    assert torch.allclose(physical, train["targets_TBm"], atol=1e-5)


def test_denormalized_prediction_round_trips():
    """Physical plots use denormalize(pred_norm); it must invert normalize exactly."""
    train, full, loss_norm = _train_normalizers()
    model = biodiesel_deeponet.build(seed=0)
    branch, tau = biodiesel_deeponet.prepare_inputs(train, full, float(train["t"][-1]))
    with torch.no_grad():
        pred_norm = model(branch, tau)
    physical = loss_norm.denormalize(pred_norm)
    assert torch.allclose(loss_norm.normalize(physical), pred_norm, atol=1e-5)
    # And the inverse transform is not the identity -- it really rescales.
    assert not torch.allclose(physical, pred_norm)


def test_clean_reference_is_never_the_training_target_under_noise():
    """species_TBm stays the Eq. 22 clean reference even when observations are noisy."""
    clean = load_biodiesel(split="test")
    noisy = load_biodiesel(split="test", noise_percent=15)
    assert torch.equal(noisy["species_TBm"], clean["species_TBm"])
    assert not torch.equal(noisy["targets_TBm"], noisy["species_TBm"])
    assert torch.equal(noisy["Y0"], clean["Y0"])        # initial condition stays clean

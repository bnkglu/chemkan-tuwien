"""The DeepONet reproduction path: author preprocessing and the three metric definitions."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "deeponet"))

rp = pytest.importorskip("biodiesel_deeponet_repro")


@pytest.mark.parametrize("n_t", [5, 30, 47])
def test_eq18_sums_over_time_and_mean_mse_averages(n_t):
    """No hard-coded time length: Eq. 18 = n_t * mean_mse for any number of time points."""
    g = torch.Generator().manual_seed(0)
    pred, target = torch.rand(4, n_t, 6, generator=g), torch.rand(4, n_t, 6, generator=g)
    manual = ((pred - target) ** 2).mean(dim=2).sum(dim=1).mean()
    assert torch.allclose(rp.eq18(pred, target), manual)
    assert torch.allclose(rp.eq18(pred, target), n_t * rp.mean_mse(pred, target))


def test_eq18_equals_eq22_without_noise():
    data = rp.load_author_data(noise_percent=0)
    stats = rp.author_global_stats(data)
    pred = torch.rand(data["test_true"].shape, dtype=torch.float64)
    m = rp.all_metrics(pred, torch.as_tensor(data["test_obs"]), pred,
                       torch.as_tensor(rp.to_u(data["test_obs"], stats)),
                       torch.as_tensor(rp.to_u(data["test_true"], stats)))
    assert m["eq18_obs"] == m["eq22_true"]


def test_author_global_stats_cover_train_noisy_test_and_clean_test():
    data = rp.load_author_data(noise_percent=15)
    stats = rp.author_global_stats(data)
    everything = np.concatenate([data["train_obs"], data["test_obs"], data["test_true"]])
    assert np.array_equal(stats["ymin"], everything.min(axis=(0, 1)))
    assert np.array_equal(stats["ymax"], everything.max(axis=(0, 1)))
    assert stats["Tmin"] == min(data["train_T0"].min(), data["test_T0"].min())
    assert np.allclose(data["t"], np.linspace(0.0, 30.0, 30))


def test_normalized_branch_uses_species_and_temperature_scales():
    data = rp.load_author_data(noise_percent=0)
    stats = rp.author_global_stats(data)
    raw = data["train_branch_raw"]
    b = rp.branch_input(raw, "author_global_normalized_states", stats)
    assert np.allclose(b[:, 0], (raw[:, 0] - stats["ymin"][0]) / (stats["ymax"][0] - stats["ymin"][0]))
    assert np.allclose(b[:, 2], (raw[:, 2] - stats["Tmin"]) / (stats["Tmax"] - stats["Tmin"]))
    assert np.array_equal(rp.branch_input(raw, "raw_states", stats), raw)
    u = rp.encode_states(data["train_true"], "author_global_normalized_states", stats)
    assert np.allclose(rp.from_u(u, stats), data["train_true"])


def test_init_schemes_ranges_and_bias_only_arm():
    from biodiesel_deeponet import build
    base = {k: v.clone() for k, v in build(8, seed=3).state_dict().items()}
    c = rp.apply_init(build(8, seed=3), "C_xavier_normal_uniform_bias", 3).state_dict()
    for k, v in base.items():                       # C changes biases only
        if k.endswith("weight"):
            assert torch.equal(c[k], v)
        else:
            assert torch.all(v == 0) and c[k].abs().max() <= 1 / np.sqrt(c[k.replace("bias", "weight")].shape[1])
    lux = rp.apply_init(build(8, seed=3), "B_lux_default", 3)
    assert lux.trunk[0].weight.abs().max() <= 2 * np.sqrt(3)          # relu gain 2, fan_in 1
    assert lux.head.weight.abs().max() <= np.sqrt(3 / 8)              # identity gain 1
    tf = rp.apply_init(build(8, seed=3), "Aprime_tf_truncated_glorot_zero_bias", 3)
    s0 = np.sqrt(2 / 16) / 0.87962566103423978
    assert tf.branch[2].weight.abs().max() <= 2 * s0 and torch.all(tf.branch[2].bias == 0)


@pytest.mark.parametrize("scheme", ["Aprime_tf_truncated_glorot_zero_bias", "B_torch_default"])
def test_init_is_deterministic_per_seed_for_lr_pairing(scheme):
    """lr ablations rebuild the model from (seed, scheme): the tensors must be identical."""
    from biodiesel_deeponet import build
    a = rp.apply_init(build(8, seed=4), scheme, 4).state_dict()
    b = rp.apply_init(build(8, seed=4), scheme, 4).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_branch_modes_change_only_branch_columns():
    data = rp.load_author_data(noise_percent=0)
    stats = rp.author_global_stats(data)
    raw = data["train_branch_raw"]
    a = rp.branch_input_mode(raw, "normalized", "author_global_normalized_states", stats)
    c = rp.branch_input_mode(raw, "raw_species_norm_T", "author_global_normalized_states", stats)
    assert np.array_equal(a, rp.branch_input_mode(raw, "follow_state_mode",
                                                  "author_global_normalized_states", stats))
    assert np.array_equal(c[:, :2], raw[:, :2]) and np.array_equal(c[:, 2], a[:, 2])


def test_planted_knots_put_hinges_at_requested_times():
    from biodiesel_deeponet import build
    m = rp.apply_init(build(8, seed=0), "Aprime_tf_truncated_glorot_zero_bias", 0)
    rp.plant_knots(m, [4.0, 11.5], 30.0)
    kinks = rp.first_layer_kinks(m)
    assert np.allclose(kinks[:2], [4.0, 11.5], atol=1e-5)


def test_trunk_input_is_the_raw_repository_time_grid():
    data = rp.load_author_data(noise_percent=0)
    t = rp.raw_trunk_time(data)
    repo = np.load(rp.DEFAULT_DATA)["t"]
    assert torch.equal(t, torch.as_tensor(repo, dtype=torch.float32))
    assert float(t.min()) == 0.0 and float(t.max()) == 30.0
    assert not hasattr(rp, "TIME_INPUTS")


def test_no_flag_can_rescale_trunk_time(tmp_path):
    import json
    import subprocess
    import train_biodiesel_deeponet_repro as tr
    with pytest.raises(SystemExit):
        tr.build_parser().parse_args(["--state-mode", "author_global_normalized_states",
                                      "--objective", "mean_mse", "--run-dir", "x",
                                      "--time-input", "t_over_t_end"])
    assert not any("time" in a.dest and "input" in a.dest for a in tr.build_parser()._actions)
    run = tmp_path / "run"
    subprocess.run([sys.executable, str(_REPO / "deeponet" / "train_biodiesel_deeponet_repro.py"),
                    "--state-mode", "author_global_normalized_states", "--objective", "mean_mse",
                    "--epochs", "1", "--run-dir", str(run)], check=True, capture_output=True)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["preprocessing"]["trunk_input_range_s"] == [0.0, 30.0]
    assert cfg["preprocessing"]["trunk_input"].startswith("raw physical t")


def test_placements_share_parameters_and_keys():
    from biodiesel_deeponet import build, n_params
    ref = build(8, seed=1).state_dict()
    for code in rp.PLACEMENTS:
        m = rp.set_placement(build(8, seed=1), code)
        assert n_params(m) == 308 and m.state_dict().keys() == ref.keys()
        assert all(torch.equal(m.state_dict()[k], ref[k]) for k in ref)
        assert isinstance(m.trunk[-1], torch.nn.ReLU) == (code in "AC")
        assert isinstance(m.branch[-1], torch.nn.ReLU) == (code in "CD")

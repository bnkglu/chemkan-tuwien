"""Focused checks for trajectory mini-batching in the biodiesel trainer.

Scope is the one new variable: how many trajectories are grouped into a single optimizer
update. These verify the accounting (batches, steps, coverage), that the unbatched path is
untouched, that batching does not change the loss reduction, and that every solve is still
a FULL rollout from a trajectory's own initial condition.
"""

from __future__ import annotations

import csv
import itertools
import json

import pytest
import torch
import train_biodiesel as mod
from train_biodiesel import batches_per_epoch, epoch_batches

from chemkan.losses import trajectory_mse

N_TRAIN = 20                       # biodiesel train split; asserted against the data below
# The committed clean run's first epochs -- the legacy path must keep reproducing them.
LEGACY_LOSSES = [1276.571044921875, 1099.385498046875]


def gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def run_main(tmp_path, monkeypatch, *extra, epochs=1):
    """Run the trainer's ``main`` and return (history rows, config, run dir)."""
    run_dir = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["train_biodiesel.py", "--seed", "0",
                                     "--epochs", str(epochs), "--run-dir", str(run_dir),
                                     *extra])
    mod.main()
    with open(run_dir / "history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, json.loads((run_dir / "config.json").read_text()), run_dir


class CountingAdam:
    """Context-free counters for optimizer.step / zero_grad, installed by monkeypatch."""

    def __init__(self, monkeypatch):
        self.steps = 0
        self.zeros = 0
        self.order: list[str] = []
        real_step, real_zero = torch.optim.Adam.step, torch.optim.Adam.zero_grad

        def step(inner, *a, **k):
            self.steps += 1
            self.order.append("step")
            return real_step(inner, *a, **k)

        def zero(inner, *a, **k):
            self.zeros += 1
            self.order.append("zero")
            return real_zero(inner, *a, **k)

        monkeypatch.setattr(torch.optim.Adam, "step", step)
        monkeypatch.setattr(torch.optim.Adam, "zero_grad", zero)


def test_dataset_has_the_expected_number_of_training_trajectories():
    from _data import load_biodiesel
    assert load_biodiesel("train")["Y0"].shape[0] == N_TRAIN


# --------------------------------------------------------------------------------------
# 1. B = N_train preservation
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [(), ("--batch-size", "20")])
def test_full_batch_takes_the_legacy_path_and_steps_once_per_epoch(tmp_path, monkeypatch, extra):
    counter = CountingAdam(monkeypatch)
    rows, cfg, _ = run_main(tmp_path, monkeypatch, *extra, epochs=2)

    assert cfg["batching"]["batch_size"] == N_TRAIN
    assert cfg["batching"]["batches_per_epoch"] == 1
    assert cfg["batching"]["legacy_full_batch_path"] is True
    assert cfg["batching"]["shuffle_each_epoch"] is False
    assert cfg["batching"]["total_optimizer_steps"] == 2
    assert counter.steps == 2, "one optimizer update per epoch"
    # the unbatched history format is unchanged (no batching columns bolted on)
    assert set(rows[0]) == {"epoch", "total_loss", "mse_loss", "elapsed_seconds"}
    got = [float(r["total_loss"]) for r in rows]
    assert [repr(v) for v in got] == [repr(v) for v in LEGACY_LOSSES]


# --------------------------------------------------------------------------------------
# 2 & 3. mini-batch accounting
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size,want_batches", [(1, 20), (5, 4), (10, 2), (3, 7)])
def test_batch_accounting_and_full_coverage(batch_size, want_batches):
    assert batches_per_epoch(N_TRAIN, batch_size) == want_batches
    batches = epoch_batches(N_TRAIN, batch_size, gen())
    assert len(batches) == want_batches
    flat = torch.cat(batches).tolist()
    assert sorted(flat) == list(range(N_TRAIN)), "each trajectory exactly once, none dropped"
    assert len(set(flat)) == N_TRAIN, "no duplicate trajectory within an epoch"
    assert sum(int(b.numel()) for b in batches) == N_TRAIN
    # a final short batch is kept rather than discarded
    assert all(int(b.numel()) == batch_size for b in batches[:-1])
    assert 1 <= int(batches[-1].numel()) <= batch_size


@pytest.mark.parametrize("batch_size,want_steps", [(1, 20), (5, 4)])
def test_one_epoch_takes_exactly_one_step_per_batch(tmp_path, monkeypatch,
                                                    batch_size, want_steps):
    counter = CountingAdam(monkeypatch)
    rows, cfg, _ = run_main(tmp_path, monkeypatch, "--batch-size", str(batch_size), epochs=1)
    assert cfg["batching"]["batches_per_epoch"] == want_steps
    assert cfg["batching"]["total_optimizer_steps"] == want_steps
    assert counter.steps == want_steps
    assert int(rows[-1]["optimizer_step"]) == want_steps


# --------------------------------------------------------------------------------------
# 4. gradient reset -- no accumulation across trajectory batches
# --------------------------------------------------------------------------------------

def test_gradients_are_zeroed_before_every_batch(tmp_path, monkeypatch):
    counter = CountingAdam(monkeypatch)
    run_main(tmp_path, monkeypatch, "--batch-size", "5", epochs=2)
    assert counter.steps == 8                     # 4 batches x 2 epochs
    assert counter.zeros >= counter.steps, "every update is preceded by a zero_grad"
    # strict alternation: zero, step, zero, step, ... (no two steps without a reset)
    trimmed = [o for o in counter.order if o in ("zero", "step")]
    for a, b in itertools.pairwise(trimmed):
        assert not (a == "step" and b == "step"), "gradients accumulated across batches"


# --------------------------------------------------------------------------------------
# 5. full-rollout semantics: only u_i(0), whole grid, no intermediate reset
# --------------------------------------------------------------------------------------

def test_b1_solves_start_only_from_the_initial_condition(tmp_path, monkeypatch):
    from _data import load_biodiesel
    data = load_biodiesel("train")
    Y0, t, truth = data["Y0"], data["t"], data["species_TBm"]

    calls = []
    real = mod.integrate

    def recording(func, y0, tt, cfg):
        calls.append((y0.detach().clone(), tt.detach().clone()))
        return real(func, y0, tt, cfg)

    monkeypatch.setattr(mod, "integrate", recording)
    run_main(tmp_path, monkeypatch, "--batch-size", "1", epochs=1)

    single = [(y, tt) for y, tt in calls if y.shape[0] == 1]
    assert len(single) == N_TRAIN, "one full-rollout solve per trajectory"

    seen = []
    for y0, tt in single:
        assert torch.equal(tt, t), "every solve spans the COMPLETE time grid"
        matches = [i for i in range(N_TRAIN) if torch.equal(y0[0], Y0[i])]
        assert len(matches) == 1, "a solve started from something other than a u_i(0)"
        seen.append(matches[0])
    assert sorted(seen) == list(range(N_TRAIN))

    # No solve is ever started from a state observed at a LATER time -- that would be
    # observed-interval training / teacher forcing, which this formulation is not.
    later = truth[1:].reshape(-1, truth.shape[-1])
    for y0, _ in single:
        assert not any(torch.equal(y0[0], row) for row in later)


# --------------------------------------------------------------------------------------
# 6. loss semantics -- the batch reduction stays a MEAN
# --------------------------------------------------------------------------------------

def test_batch_reduction_is_a_mean_over_trajectories():
    torch.manual_seed(0)
    pred, target = torch.rand(30, 20, 6), torch.rand(30, 20, 6)
    per_traj = [float(trajectory_mse(pred[:, i:i + 1], target[:, i:i + 1]))
                for i in range(20)]
    assert float(trajectory_mse(pred, target)) == pytest.approx(sum(per_traj) / 20, rel=1e-6)
    for size in (1, 5, 10):
        sub = float(trajectory_mse(pred[:, :size], target[:, :size]))
        assert sub == pytest.approx(sum(per_traj[:size]) / size, rel=1e-6)


def test_batch_of_one_equals_that_trajectorys_own_loss():
    torch.manual_seed(0)
    pred, target = torch.rand(30, 20, 6), torch.rand(30, 20, 6)
    for i in (0, 7, 19):
        one = float(trajectory_mse(pred[:, i:i + 1], target[:, i:i + 1]))
        manual = float(((pred[:, i] - target[:, i]) ** 2).mean(dim=-1).sum())
        assert one == pytest.approx(manual, rel=1e-6)


# --------------------------------------------------------------------------------------
# 7. determinism
# --------------------------------------------------------------------------------------

def test_shuffle_is_reproducible_and_seed_dependent():
    a = [b.tolist() for b in epoch_batches(N_TRAIN, 5, gen(0))]
    b = [b.tolist() for b in epoch_batches(N_TRAIN, 5, gen(0))]
    assert a == b, "same seed must give the same trajectory ordering"
    c = [x.tolist() for x in epoch_batches(N_TRAIN, 5, gen(1))]
    assert a != c, "a different seed should give a different ordering"
    # successive epochs from one generator differ, but still cover everything
    g = gen(0)
    e1 = torch.cat(epoch_batches(N_TRAIN, 5, g)).tolist()
    e2 = torch.cat(epoch_batches(N_TRAIN, 5, g)).tolist()
    assert e1 != e2
    assert sorted(e1) == sorted(e2) == list(range(N_TRAIN))


def test_initial_parameters_are_untouched_by_batching():
    """Batching must not disturb initialization or its RNG ordering."""
    from chemkan.model import KineticCore

    def build():
        torch.manual_seed(0)
        return KineticCore(species_dim=6, hidden_dim=4, num_basis=3, n_mu=2,
                           use_base_act=False).state_dict()

    a, b = build(), build()
    assert all(torch.equal(a[k], b[k]) for k in a)
    # drawing a shuffle permutation must not consume the global RNG stream
    torch.manual_seed(0)
    before = torch.randn(4)
    torch.manual_seed(0)
    epoch_batches(N_TRAIN, 1, gen(0))
    assert torch.equal(before, torch.randn(4))


# --------------------------------------------------------------------------------------
# 8. smoke: a short B=1 run end to end
# --------------------------------------------------------------------------------------

def test_b1_smoke_run_trains_saves_and_records_metadata(tmp_path, monkeypatch):
    rows, cfg, run_dir = run_main(tmp_path, monkeypatch, "--batch-size", "1",
                                  "--eval-every", "1", epochs=2)
    assert cfg["batching"] == {**cfg["batching"], "batch_size": 1,
                               "batches_per_epoch": 20, "num_train_trajectories": 20,
                               "total_optimizer_steps": 40, "shuffle_each_epoch": True,
                               "legacy_full_batch_path": False}
    assert cfg["training_formulation"] == "full_rollout"
    assert cfg["initialization"] == "default", "the batching run must not change init"

    assert [int(r["epoch"]) for r in rows] == [0, 1]
    assert [int(r["optimizer_step"]) for r in rows] == [20, 40]
    for r in rows:                                   # every logged metric is finite
        for key in ("total_loss", "mse_loss", "batch_loss_running_mean",
                    "test_mse_noisy", "test_mse_clean"):
            assert float(r[key]) == float(r[key])

    ckpt = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu",
                      weights_only=False)
    assert ckpt["training"]["batch_size"] == 1
    assert ckpt["training"]["batches_per_epoch"] == 20
    assert ckpt["training"]["total_optimizer_steps"] == 40
    assert ckpt["training"]["training_formulation"] == "full_rollout"
    assert sum(p.numel() for p in ckpt["model_state"].values() if p.dim() >= 2) == 156
    # training actually moved the weights
    torch.manual_seed(0)
    from chemkan.model import KineticCore
    fresh = KineticCore(species_dim=6, hidden_dim=4, num_basis=3, n_mu=2,
                        use_base_act=False).state_dict()
    assert not all(torch.equal(fresh[k], ckpt["model_state"][k]) for k in fresh)


def test_epoch_metric_is_the_full_training_set_not_the_last_batch(tmp_path, monkeypatch):
    """At epoch 0 the logged metric must equal the unbatched epoch-0 loss exactly."""
    rows, _, _ = run_main(tmp_path, monkeypatch, "--batch-size", "1", epochs=1)
    assert repr(float(rows[0]["total_loss"])) == repr(LEGACY_LOSSES[0])
    # ... and it is NOT the last mini-batch's loss
    assert float(rows[0]["batch_loss_running_mean"]) != float(rows[0]["total_loss"])

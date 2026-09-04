"""Behaviour-neutrality of the training instrumentation (DIAGNOSTIC layer).

The NFE counter and the Stage-2 snapshot are attached to long, expensive runs, so the
critical property is that they change NOTHING about the optimization: same loss, same
gradients, same parameters after a step.
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "scripts", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _instrumentation import EpochInstrumentation, NFECounter, Stage2Snapshot  # noqa: E402

from chemkan.dynamics import ChemKANDynamics  # noqa: E402
from chemkan.model import ChemKAN  # noqa: E402
from chemkan.solver import SolverConfig, integrate  # noqa: E402

ARCH = dict(species_dim=3, hidden_dim=3, num_basis=4, n_mu=2)
SOLVER = SolverConfig(method="tsit5", rtol=1e-6, atol=1e-8, sensitivity="direct_autograd")


def _build(seed=0):
    torch.manual_seed(seed)
    model = ChemKAN(**ARCH, use_base_act=True)
    return model, ChemKANDynamics(model, input_normalizer=None)


def _loss_and_grads(model, dyn):
    u0 = torch.linspace(0.1, 0.4, 4).reshape(1, 4)
    t = torch.linspace(0.0, 1e-3, 5)
    pred = integrate(dyn, u0, t, SOLVER)
    loss = (pred ** 2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return float(loss.detach()), {n: p.grad.detach().clone()
                                  for n, p in model.named_parameters()}


def test_nfe_counter_does_not_change_loss_or_gradients():
    """Bit-identical loss and gradients with the forward hook attached."""
    m1, d1 = _build()
    l1, g1 = _loss_and_grads(m1, d1)

    m2, d2 = _build()                      # same seed -> same initial parameters
    counter = NFECounter(d2)
    l2, g2 = _loss_and_grads(m2, d2)

    assert l1 == l2, f"loss changed: {l1} vs {l2}"
    assert g1.keys() == g2.keys()
    for n in g1:
        assert torch.equal(g1[n], g2[n]), f"gradient changed for {n}"
    assert counter.count > 0, "counter never fired"


def test_nfe_counter_counts_rhs_calls_and_resets():
    _, dyn = _build()
    counter = NFECounter(dyn)
    u0 = torch.linspace(0.1, 0.4, 4).reshape(1, 4)
    t = torch.linspace(0.0, 1e-3, 5)
    with torch.no_grad():
        integrate(dyn, u0, t, SOLVER)
    first = counter.reset()
    assert first > 0 and counter.count == 0                # reset returns and zeroes
    with torch.no_grad():
        integrate(dyn, u0, t, SOLVER)
    assert counter.reset() > 0


def test_nfe_counter_detaches_cleanly():
    _, dyn = _build()
    counter = NFECounter(dyn)
    counter.detach()
    counter.reset()
    u0 = torch.linspace(0.1, 0.4, 4).reshape(1, 4)
    with torch.no_grad():
        integrate(dyn, u0, torch.linspace(0.0, 1e-3, 5), SOLVER)
    assert counter.count == 0, "hook still firing after detach"


def test_epoch_instrumentation_injects_nfe_and_wall_time():
    _, dyn = _build()
    counter = NFECounter(dyn)
    instr = EpochInstrumentation(counter)
    with torch.no_grad():
        integrate(dyn, torch.linspace(0.1, 0.4, 4).reshape(1, 4),
                  torch.linspace(0.0, 1e-3, 5), SOLVER)
    comp = {"state_mse": 1.0}
    instr.on_epoch(0, 1.0, comp, 2.5)
    assert comp["nfe"] > 0 and comp["epoch_wall_time_s"] == 2.5
    comp2 = {"state_mse": 1.0}
    instr.on_epoch(1, 1.0, comp2, 4.0)                     # increment, not cumulative
    assert comp2["epoch_wall_time_s"] == 1.5
    assert comp2["nfe"] == 0                               # counter was reset by epoch 0


def test_snapshot_writes_once_at_the_requested_epoch(tmp_path):
    model, _ = _build()
    snap = Stage2Snapshot(model, tmp_path, [2], extra={"tag": "t"})
    for epoch in range(4):
        snap.on_epoch(epoch, 0.0, {}, 0.0)
    files = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert files == ["checkpoint_stage2_epoch_2.pt"]       # epoch index 1 -> 2 steps done
    payload = torch.load(tmp_path / files[0], map_location="cpu", weights_only=False)
    assert payload["stage2_epoch"] == 2 and payload["snapshot"] is True
    assert payload["tag"] == "t"
    assert payload["model_state"].keys() == model.state_dict().keys()


def test_snapshot_never_overwrites(tmp_path):
    model, _ = _build()
    snap = Stage2Snapshot(model, tmp_path, [1])
    assert snap.save(1) is not None
    assert snap.save(1) is None                            # second call is a no-op


def test_snapshot_does_not_touch_parameters(tmp_path):
    model, _ = _build()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    Stage2Snapshot(model, tmp_path, [1]).save(1)
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p.detach())

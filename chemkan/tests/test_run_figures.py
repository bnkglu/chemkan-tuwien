"""run_figures.py (figure-run plan, shards, skip/resume). No real training."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "chemkan/scripts/author_repo_match"))

import run_figures as rf  # noqa: E402



# --------------------------------------------------------------------------- run_figures

def test_plan_contents():
    runs = rf.plan()
    names = [r["name"] for r in runs]
    assert len(runs) == 27 and len(set(names)) == 27
    fig05 = [r for r in runs if r["name"].startswith("fig05")]
    assert len(fig05) == 16 and all(r["epochs"] == 10000 for r in fig05)
    ck4 = [r for r in runs if r["name"].startswith("fig04_chemkan")]
    assert [r["args"][-1] for r in ck4] == ["2", "3", "4", "9", "11"]
    assert all(r["epochs"] == 5000 for r in ck4)
    dn4 = [r for r in runs if r["name"].startswith("fig04_deeponet")]
    assert all(r["epochs"] == 50000 for r in dn4) and len(dn4) == 6
    for r in (r for r in runs if r["model"] == "chemkan"):
        cmd = rf.command(r, None)
        assert cmd[cmd.index("--data-file") + 1] == rf.DATA
        assert cmd[cmd.index("--seed") + 1] == "0"
        assert "--noise-percent" not in cmd or r["name"].startswith("fig05")
    with pytest.raises(ValueError, match="run_figure_runs.sh"):
        rf.command(dn4[0], None)
    ck = rf.command(runs[0], None)
    assert ck[ck.index("--solve-mode") + 1] == "batched" and ck[ck.index("--sensitivity") + 1] == "fsa"


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_shards_partition_and_balance(n):
    runs = rf.plan()
    groups = rf.shards(runs, n)
    flat = [r["name"] for g in groups for r in g]
    assert sorted(flat) == sorted(r["name"] for r in runs)             # each run exactly once
    loads = [sum(r["estimate_s"] for r in g) for g in groups]
    assert max(loads) - min(loads) <= max(r["estimate_s"] for r in runs)


def test_skip_resume_and_incomplete(tmp_path, monkeypatch):
    run = dict(rf.plan()[0], dir=tmp_path / "chemkan_noise00_seed0")
    assert rf.status(run) == "new"
    run["dir"].mkdir()
    (run["dir"] / "history.csv").write_text("x")
    assert rf.status(run) == "incomplete"
    (run["dir"] / "checkpoint_resume.pt").write_text("x")
    assert rf.status(run) == "resume" and rf.command(run, None)[-1] == "--resume"
    (run["dir"] / "checkpoint_final.pt").write_text("x")
    assert rf.status(run) == "done"
    monkeypatch.setattr(rf.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    assert "skipped" in rf.execute(run, None, dry_run=False)


def test_dry_run_starts_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rf, "OUT", tmp_path)                     # no finished runs here
    monkeypatch.setattr(rf.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(rf, "log_line", lambda *a: pytest.fail("must not log"))
    assert rf.main(["--dry-run", "--only", "fig04_chemkan_h04", "fig04_deeponet_p308"]) == 0
    out = capsys.readouterr().out
    assert "--hidden 4" in out and "biodiesel_v2.npz" in out
    assert rf.DEEPONET_LAUNCHER in out and "--trunk-hidden" not in out

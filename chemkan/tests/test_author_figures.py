"""Figures and paper table from the biodiesel_v2 figure runs (skipped if the runs are absent)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
REPO = Path(__file__).resolve().parents[2]
FIGURES = str(REPO / "chemkan/scripts/figures")


def _figures_module(name: str):
    """Import a figures-folder module with the figures' own ``common`` (other tests put
    ``data_gen/common.py`` into sys.modules under the same name)."""
    shadow = sys.modules.pop("common", None)
    sys.path.insert(0, FIGURES)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(FIGURES)
        sys.modules.pop("common", None)
        if shadow is not None:
            sys.modules["common"] = shadow


ar = _figures_module("author_runs")

pytestmark = pytest.mark.skipif(not (ar.RUNS / "fig05").exists(), reason="figure runs absent")


def test_all_runs_verify():
    rows, problems = ar.verify()
    assert len(rows) == 27 and problems == []


def test_deeponet_histories_are_time_averaged():
    import csv
    legacy = {r["name"]: r for r in ar.plan("relu_legacy")}["fig05_deeponet_noise00"]
    raw = [float(r["mse_loss"]) for r in csv.DictReader(open(legacy["dir"] / "history.csv"))]
    assert ar.history(legacy)["train"][-1] == pytest.approx(raw[-1] / 30)       # logged Eq. 18
    tanh = ar.by_name()["fig05_deeponet_noise00"]                                # default variant
    raw = [float(r["train_mse_fig"]) for r in csv.DictReader(open(tanh["dir"] / "history.csv"))]
    assert ar.history(tanh)["train"][-1] == pytest.approx(raw[-1])               # already averaged


@pytest.mark.parametrize("module", ["fig03_biodiesel_trajectories", "fig04_biodiesel_scaling",
                                    "fig05_biodiesel_noise", "fig05_biodiesel_loss",
                                    "fig06_biodiesel_profiles"])
def test_author_figures_write_png_and_pdf(tmp_path, module):
    mod = _figures_module(module)
    mod.make_author_figure(output_path=tmp_path / "out")
    assert (tmp_path / "out.png").exists() and (tmp_path / "out.pdf").exists()


def test_paper_table():
    table = _figures_module("author_paper_table")
    rows = table.comparison()
    assert len(rows) == 9 and all(r["section"] in ("III A 2", "III A 3") for r in rows)
    assert "Eq. 18" in rows[4]["ours"]

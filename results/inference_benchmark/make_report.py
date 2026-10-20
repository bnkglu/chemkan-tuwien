r"""Result table and runtime-vs-accuracy figure from the Python and Julia benchmark outputs.

Python: ``<system>/results.json`` (benchmark.py). Julia, when present: ``julia/julia_chemkan_<system>.json``
and ``julia/julia_reference_<system>.json`` with their ``*_pred.npz``; their accuracy is computed
here against the same reference trajectories (``julia/inputs_<system>.npz``, the Python
reference at its data-generation tolerance). Writes ``results_table.{md,csv}`` and
``runtime_vs_accuracy.{png,pdf}`` here. No timing is measured.

    ~/uni_projects/chemkan-venv/bin/python results/inference_benchmark/make_report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
JL = HERE / "julia"
ORDER = ["hydrogen", "methane", "biodiesel"]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#6b6b6b"
REF_NAME = {"hydrogen": "Cantera", "methane": "Cantera", "biodiesel": "LSODA"}


def summary(v) -> dict:
    v = np.asarray(v, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": float(med), "q25": float(q1), "q75": float(q3)}


def python_entries(system: str, res: dict) -> list[dict]:
    lat, thr, acc = res["timing"]["latency"], res["timing"]["throughput"], res["accuracy_vs_reference"]
    out = []
    for key, L in lat.items():
        kan = key.startswith("chemkan")
        a = acc[key]
        out.append({
            "impl": "Python", "kind": "ChemKAN" if kan else "reference",
            "solver": ("PyTorch/torchdiffeq Tsit5" if kan else
                       "Cantera CVODES" if system != "biodiesel" else "SciPy LSODA"),
            "key": key, "rtol": L["rtol"], "atol": L["atol"],
            "precision": res["precision"]["chemkan" if kan else "reference"],
            "latency": L, "per_condition": L["per_condition_median_s"],
            "sequential": thr[f"{key}_sequential"], "batched": thr.get(f"{key}_batched"),
            "mse": np.array(a["per_condition_mse"]),
            "mse_batched": a.get("batched_time_averaged_normalized_mse", {}).get("all", {}).get("median"),
            "physical": a.get("physical", {}),
        })
    for key, f in res.get("reference_settings_that_failed", {}).items():
        out.append({"impl": "Python", "kind": "reference", "key": key, "rtol": f["rtol"],
                    "atol": f["atol"], "failed": f["failed_conditions"],
                    "solver": "Cantera CVODES" if system != "biodiesel" else "SciPy LSODA"})
    return out


def julia_entries(system: str) -> list[dict]:
    inp = JL / f"inputs_{system}.npz"
    if not inp.exists():
        return []
    d = np.load(inp)
    ref = d["reference"]
    k = ref.shape[-1]
    span = (d["u_max"] - d["u_min"])[:k]
    mse = lambda p: (((p - ref) / span) ** 2).mean(axis=(0, 2))       # noqa: E731
    out = []
    path = JL / f"julia_chemkan_{system}.json"
    if path.exists():
        res = json.loads(path.read_text())
        pred = np.load(JL / f"julia_chemkan_{system}_pred.npz")
        for key, r in res["runs"].items():
            j = key.rsplit("tol", 1)[1]
            n = len(r["latency_per_condition_median_s"])
            out.append({"impl": "Julia", "kind": "ChemKAN", "solver": "OrdinaryDiffEq Tsit5",
                        "key": key, "rtol": r["rtol"], "atol": r["atol"],
                        "precision": res["precision"], "latency": r["latency"],
                        "per_condition": r["latency_per_condition_median_s"],
                        "sequential": r["throughput_sequential"] | {"trajectories_per_s": n / r["throughput_sequential"]["median"]},
                        "batched": r["throughput_batched"] | {"trajectories_per_s": n / r["throughput_batched"]["median"]},
                        "mse": mse(pred[f"single_{j}"]),
                        "mse_batched": float(np.median(mse(pred[f"batched_{j}"])))})
    path = JL / f"julia_reference_{system}.json"
    if path.exists():
        res = json.loads(path.read_text())
        pred = np.load(JL / f"julia_reference_{system}_pred.npz")
        solver = "Arrhenius.jl CVODE_BDF" if system != "biodiesel" else "released kinetics, Julia"
        for key, r in res["runs"].items():
            name = (solver if "cvode" in key or system != "biodiesel"
                    else "released kinetics, AutoTsit5(Rosenbrock23)")
            if system == "biodiesel" and "cvode" in key:
                name = "released kinetics, CVODE_BDF"
            e = {"impl": "Julia", "kind": "reference", "solver": name, "key": key,
                 "rtol": r["rtol"], "atol": r["atol"], "precision": "Float64"}
            if "failed_conditions" in r:
                e["failed"] = r["failed_conditions"]
            else:
                n = len(r["latency_per_condition_median_s"])
                e |= {"latency": r["latency"], "per_condition": r["latency_per_condition_median_s"],
                      "sequential": r["throughput_sequential"] | {"trajectories_per_s": n / r["throughput_sequential"]["median"]},
                      "batched": None, "mse": mse(pred[key]), "mse_batched": None}
            out.append(e)
    return out


def tol(e) -> str:
    if e["rtol"] is None:
        return "solver default"
    return f"{e['rtol']:.0e} / {e['atol']:.0e}".replace("e-0", "e-")


def fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, (float, np.floating)):
        if v == 0.0:
            return "0"
        return f"{v:.3g}" if 1e-2 <= abs(v) < 1e4 else f"{v:.2e}"
    return str(v)


def table_rows(system: str, entries: list[dict], res: dict) -> list[dict]:
    base_py = next(e for e in entries if e["impl"] == "Python" and e["key"] == "reference_generator")
    jl_refs = [e for e in entries if e["impl"] == "Julia" and e["kind"] == "reference"
               and "latency" in e and "cvode" in e["key"]]
    base_jl = min(jl_refs, key=lambda e: e["rtol"]) if jl_refs else None   # generation tolerance
    rows = []
    for e in entries:
        row = {"system": system, "implementation": e["impl"], "method": e["kind"],
               "solver": e["solver"], "rtol / atol": tol(e)}
        if "failed" in e:
            row["note"] = f"failed (non-finite rates / solver error) for {len(e['failed'])} conditions; not timed"
            rows.append(row)
            continue
        bat = e.get("batched")
        best = min(e["sequential"]["median"], bat["median"] if bat else np.inf)
        row |= {
            "precision": e["precision"],
            "latency median [s]": e["latency"]["median"], "latency IQR [s]": e["latency"]["iqr"],
            "throughput sequential [traj/s]": e["sequential"]["trajectories_per_s"],
            "throughput batched [traj/s]": bat["trajectories_per_s"] if bat else None,
            f"speed-up latency vs {REF_NAME[system]} ref.": base_py["latency"]["median"] / e["latency"]["median"],
            f"speed-up throughput vs {REF_NAME[system]} ref.": base_py["sequential"]["median"] / best,
            "speed-up latency vs Julia ref.": (base_jl["latency"]["median"] / e["latency"]["median"]) if base_jl else None,
            "speed-up throughput vs Julia ref.": (base_jl["sequential"]["median"] / best) if base_jl else None,
            "MSE median (all)": float(np.median(e["mse"])),
            "MSE median (test)": float(np.median(e["mse"][np.array(res["is_test"])])),
            "note": "",
        }
        phys = e.get("physical", {})
        if system in ("hydrogen", "methane") and phys:
            row["note"] = (f"peak-T err {phys['peak_T_abs_error_K']['all']['median']:.3g} K, "
                           f"delay rel. err {phys['ignition_delay_rel_error']['all']['median']:.3g}")
        if system == "biodiesel" and phys:
            row["note"] = f"final FAME rel. err {phys['final_FAME_rel_error']['all']['median']:.3g}"
        rows.append(row)
    return rows


def write_tables(results: dict, entries: dict) -> None:
    all_rows = [r for s in results for r in table_rows(s, entries[s], results[s])]
    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with open(HERE / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    lines = ["# ChemKAN inference benchmark: results", "",
             "Speed-up = reference runtime / method runtime (< 1: slower than the reference). Two "
             "baselines: the Python reference at its data-generation tolerance (Cantera rtol 1e-9 "
             "or LSODA rtol 1e-10) and the Julia reference at the same tolerance (Arrhenius.jl "
             "CVODE_BDF rtol 1e-9, or the released kinetics with CVODE_BDF rtol 1e-10). "
             "Throughput speed-up uses the method's faster mode (sequential or batched) against the "
             "baseline solving the conditions sequentially. MSE: time-averaged, model-normalized, "
             "100 output points, against the Python reference at its generation tolerance. "
             "Latency: one trajectory per call, median and IQR over conditions x repetitions.", ""]
    for s, res in results.items():
        rows = table_rows(s, entries[s], res)
        env = res["environment"]
        lines += [f"## {s}", "",
                  f"Checkpoint `{res['checkpoint']}` (epochs {res['checkpoint_epochs']}); "
                  f"{res['conditions']} conditions ({res['train_conditions']} train / "
                  f"{res['test_conditions']} test); {res['output_grid']['points']} points over "
                  f"{res['output_grid']['t_end_s']:g} s. {env['cpu']}, 1 thread, "
                  f"{res['timing_protocol']['warmup_calls']} warm-up + "
                  f"{res['timing_protocol']['timed_repetitions']} timed repetitions.", ""]
        cols = [k for k in dict.fromkeys(k for r in rows for k in r) if k != "system"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in rows:
            lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
        lines += ["", f"Python reference vs stored dataset: {json.dumps(res['reference_vs_dataset'])}", ""]
    (HERE / "results_table.md").write_text("\n".join(lines) + "\n")


MSE_FLOOR = 1e-12            # reference solves below this are drawn on the floor (labelled)
TITLES = {"hydrogen": "Hydrogen (9 species, 29 reactions)",
          "methane": "Methane (GRI-Mech 3.0: 52 species, 325 reactions)",
          "biodiesel": "Biodiesel (6 species, 3 reactions)"}
STYLE = {("Python", "ChemKAN"): (BLUE, "o", "ChemKAN PyTorch"),
         ("Julia", "ChemKAN"): (BLUE, "D", "ChemKAN Julia"),
         ("Python", "reference"): (ORANGE, "s", None),
         ("Julia", "reference"): (ORANGE, "^", None)}
REF_LABEL = {("Python", "hydrogen"): "Cantera", ("Python", "methane"): "Cantera",
             ("Python", "biodiesel"): "LSODA", ("Julia", "hydrogen"): "Arrhenius.jl",
             ("Julia", "methane"): "Arrhenius.jl", ("Julia", "biodiesel"): "Julia kinetics"}


def _place_labels(ax, items):
    """Greedy label placement: try a few offsets per point, keep the first whose box does not
    overlap an earlier label or leave the axes."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken, axbox = [], ax.get_window_extent(renderer)
    cands = [(9, 7, "left", "bottom"), (9, -7, "left", "top"), (-9, 7, "right", "bottom"),
             (-9, -7, "right", "top"), (0, 13, "center", "bottom"), (0, -13, "center", "top"),
             (16, 0, "left", "center"), (-16, 0, "right", "center")]
    for x, y, text, color in sorted(items, key=lambda it: (-it[1], it[0])):
        for dx, dy, ha, va in cands:
            t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                            fontsize=8.5, color=color,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            bb = t.get_window_extent(renderer)
            inside = (bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1)
            if inside and not any(bb.overlaps(o) for o in taken):
                taken.append(bb)
                break
            t.remove()
        else:
            t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(9, 7), fontsize=8.5,
                            color=color, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            taken.append(t.get_window_extent(renderer))


DARK, LIGHT = "#1c5cab", "#86b6ef"     # ChemKAN at its training tolerance / at the other one
TRAINED_RTOL = {"hydrogen": 1e-6, "methane": 1e-6, "biodiesel": 1e-2}


def _panel(ax, s, res, entries):
    n = res["conditions"]
    labels = []
    ref = next(e for e in entries if e["impl"] == "Python" and e["key"] == "reference_generator")
    xr = ref["latency"]["median"]
    ax.axvspan(1e-7, xr, color="#f3f3f3", zorder=0)
    ax.axvline(xr, color=ORANGE, lw=1.6, ls="--", zorder=1)
    xs = [xr]
    for e in entries:
        if "failed" in e or e is ref:
            continue
        L, m = e["latency"], summary(e["mse"])
        y = max(m["median"], MSE_FLOOR)
        on_floor = m["median"] < MSE_FLOOR
        xs.append(L["median"])
        if e["kind"] == "ChemKAN":
            color = DARK if e["rtol"] is not None and np.isclose(e["rtol"], TRAINED_RTOL[s]) else LIGHT
            marker = "o" if e["impl"] == "Python" else "D"
        else:
            color, marker = ORANGE, ("s" if e["impl"] == "Python" else "^")
        ax.errorbar(L["median"], y, xerr=[[L["median"] - L["q25"]], [L["q75"] - L["median"]]],
                    yerr=None if on_floor else [[m["median"] - max(m["q25"], MSE_FLOOR * 0.999)],
                                                [m["q75"] - m["median"]]],
                    fmt=marker, ms=11, color=color, mec="white", mew=1.3, elinewidth=1.2, zorder=4)
        if e["kind"] == "reference":
            tol_txt = "default tol." if e["rtol"] is None else f"{e['rtol']:g} / {e['atol']:g}"
            labels.append((L["median"], y, tol_txt + (" (MSE<1e-12)" if on_floor else ""), "#8a3a12"))
        if e.get("batched"):
            xb, yb = e["batched"]["median"] / n, max(e["mse_batched"], MSE_FLOOR)
            xs.append(xb)
            ax.plot(xb, yb, marker, ms=11, mfc="white", mec=color, mew=1.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(xs) / 3, max(xs) * 10)
    ax.set_ylim(MSE_FLOOR / 3, 1.0)
    ax.axhline(MSE_FLOOR, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("wall time per trajectory [s]")
    ax.set_ylabel("time-averaged normalized MSE vs reference")
    rt = ref["rtol"]
    ax.set_title(f"{TITLES.get(s, s)}\n"
                 f"dashed: {REF_LABEL[('Python', s)]} at rtol {rt:g} ({1e3 * xr:.3g} ms, accuracy target); "
                 f"shaded: faster than it", fontsize=10, color=INK, loc="left")
    ax.grid(True, which="major", color="#e4e4e4", lw=0.7)
    _place_labels(ax, labels)


def plot(results: dict, entries: dict) -> None:
    rc = {"font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
          "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False}
    H = lambda **k: plt.Line2D([], [], ls="", ms=10, **k)          # noqa: E731
    handles = [H(marker="o", color=DARK, mec="white", label="ChemKAN, PyTorch"),
               H(marker="D", color=DARK, mec="white", label="ChemKAN, Julia port"),
               matplotlib.patches.Patch(color=DARK, label="dark blue: ChemKAN at its training tolerance"),
               matplotlib.patches.Patch(color=LIGHT, label="light blue: at the other tolerance (H2/CH4 rtol 1e-2, biodiesel 1e-10)"),
               H(marker="o", mfc="white", mec=DARK, mew=1.8, label="hollow: all conditions in one batched solve"),
               H(marker="s", color=ORANGE, mec="white", label="reference, Python (Cantera / LSODA), labels rtol / atol"),
               H(marker="^", color=ORANGE, mec="white", label="reference, Julia (Arrhenius.jl / released kinetics)"),
               plt.Line2D([], [], color=ORANGE, ls="--", label="Python reference at its generation tolerance")]
    note = ("Markers: median over conditions x 25 repetitions; bars: IQR. MSE against the reference at its "
            "generation tolerance; values below 1e-12 drawn on the dotted floor.")
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(results), figsize=(7.2 * len(results), 6.6), squeeze=False)
        for ax, s in zip(axes[0], results):
            _panel(ax, s, results[s], entries[s])
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
        fig.suptitle("ChemKAN vs detailed-chemistry reference: runtime and accuracy", fontsize=14, color=INK)
        fig.text(0.5, 0.115, note, ha="center", fontsize=9, color=MUTED)
        fig.tight_layout(rect=(0, 0.13, 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(HERE / f"runtime_vs_accuracy.{ext}", dpi=200)
        plt.close(fig)
        for s in results:
            fig, ax = plt.subplots(figsize=(8.2, 7.2))
            _panel(ax, s, results[s], entries[s])
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
            fig.text(0.5, 0.165, note, ha="center", fontsize=8, color=MUTED, wrap=True)
            fig.tight_layout(rect=(0, 0.2, 1, 1))
            for ext in ("png", "pdf"):
                fig.savefig(HERE / f"runtime_vs_accuracy_{s}.{ext}", dpi=200)
            plt.close(fig)


def main() -> int:
    results = {s: json.loads((HERE / s / "results.json").read_text())
               for s in ORDER if (HERE / s / "results.json").exists()}
    if not results:
        raise SystemExit("no <system>/results.json yet")
    entries = {s: python_entries(s, r) + julia_entries(s) for s, r in results.items()}
    write_tables(results, entries)
    plot(results, entries)
    print("systems:", ", ".join(results), "-> results_table.md, results_table.csv, runtime_vs_accuracy[_<system>].png/pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

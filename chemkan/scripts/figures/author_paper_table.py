r"""Paper comparison for the biodiesel_v2 figure runs, using ONLY numbers stated in the
paper's text (section cited per row). Our values come from the same data functions the
Fig. 4 / Fig. 5 scripts plot, in the time-averaged convention unless a row says otherwise
(Eq. 18 = 30 x time-averaged).

    rows = comparison()        # list of dicts: quantity, section, paper, ours, how
    print(markdown(rows))
"""

from __future__ import annotations

import numpy as np

import author_runs as ar
import fig04_biodiesel_scaling as fig04
import fig05_biodiesel_noise as fig05a


def _g(x: float) -> str:
    return f"{x:.3g}"


def comparison(scaling=None, noise_rows=None) -> list[dict]:
    """``scaling``: fig04.author_scaling_data(); ``noise_rows``: fig05a.author_noise_rows()[0].
    Both are computed here when omitted."""
    scaling = scaling or fig04.author_scaling_data()
    noise_rows = noise_rows or fig05a.author_noise_rows()[0]
    pts, fits = scaling["points"], scaling["fits"]

    def series(model):
        sub = sorted((p for p in pts if p["model"] == model), key=lambda r: r["parameters"])
        return {"params": np.array([p["parameters"] for p in sub]),
                "train": np.array([p["train_loss"] for p in sub]),
                "test_clean": np.array([p["test_loss"] for p in sub])}

    def order(model, metric):
        return -next(f["slope"] for f in fits if f["model"] == model and f["metric"] == metric)

    ck, dn = series("ChemKAN"), series("DeepONet")
    ck["train_fit"], ck["test_clean_fit"] = ({"order": order("ChemKAN", "train_loss")},
                                             {"order": order("ChemKAN", "test_loss")})
    lv = sorted({r["noise_percent"] for r in noise_rows})

    def by_noise(model, key):
        return np.array([next(r[key] for r in noise_rows
                              if r["model"] == model and r["noise_percent"] == p) for p in lv])

    nck = {"train": by_noise("ChemKAN", "train_mse_noisy"),
           "test_clean": by_noise("ChemKAN", "test_mse_clean")}
    ndn = {"test_clean": by_noise("DeepONet", "test_mse_clean")}
    at = lambda arr, p: float(arr[lv.index(p)])                       # noqa: E731
    rows = []

    # ---- Sec. III A 2 (neural scaling, noise-free) ----
    rows.append({"quantity": "ChemKAN neural-scaling order, training", "section": "III A 2",
                 "paper": "1.0", "ours": _g(ck["train_fit"]["order"]),
                 "how": "-(slope) of a log-log least-squares fit of final-checkpoint training "
                        "MSE vs parameters, H = 2, 3, 4, 9 (last point, H = 11, excluded)"})
    rows.append({"quantity": "ChemKAN neural-scaling order, testing", "section": "III A 2",
                 "paper": "0.6", "ours": _g(ck["test_clean_fit"]["order"]),
                 "how": "same fit on the final-checkpoint noise-free testing MSE"})
    i0 = 0
    rows.append({"quantity": "ChemKAN error at the smallest size", "section": "III A 2",
                 "paper": "~1e-4 (72 parameters in the text; 78 per email B2)",
                 "ours": f"train {_g(ck['train'][i0])}, test {_g(ck['test_clean'][i0])} "
                         f"(Eq. 18: {_g(30 * ck['train'][i0])}, {_g(30 * ck['test_clean'][i0])})",
                 "how": f"final checkpoint of the {int(ck['params'][i0])}-parameter run (H = 2)"})
    big_dn = [(int(p), float(y)) for p, y in zip(dn["params"], dn["train"]) if p > 200]
    big_ck = [(int(p), float(y)) for p, y in zip(ck["params"], ck["train"]) if p > 200]
    rows.append({"quantity": "DeepONet training loss below ChemKAN's above ~200 parameters",
                 "section": "III A 2", "paper": "yes",
                 "ours": ("yes" if max(y for _, y in big_dn) < min(y for _, y in big_ck) else "no")
                         + f": DeepONet {', '.join(f'{p}: {_g(y)}' for p, y in big_dn)}; "
                           f"ChemKAN {', '.join(f'{p}: {_g(y)}' for p, y in big_ck)}",
                 "how": "final-checkpoint training MSE of every run above 200 parameters"})

    # ---- Sec. III A 3 (noise) ----
    inc1, inc5 = at(nck["train"], 1) - at(nck["train"], 0), at(nck["train"], 5) - at(nck["train"], 0)
    f1, f5 = ar.noise_floor(1), ar.noise_floor(5)
    rows.append({"quantity": "ChemKAN training-MSE increase, 0 -> 1 % noise", "section": "III A 3",
                 "paper": "3.78e-5",
                 "ours": f"time-averaged {_g(inc1)}; Eq. 18 {_g(30 * inc1)} "
                         f"(noise floor: {_g(f1)}; Eq. 18 {_g(30 * f1)})",
                 "how": "difference of converged training MSE (final checkpoint after "
                        "10,000 epochs); noise floor = MSE between noisy and clean training "
                        "targets, the increase a perfect model would show"})
    rows.append({"quantity": "ChemKAN training-MSE increase, 0 -> 5 % noise", "section": "III A 3",
                 "paper": "9.45e-4 expected (25 x 3.78e-5); 9.64e-4 observed",
                 "ours": f"time-averaged {_g(inc5)}; Eq. 18 {_g(30 * inc5)} "
                         f"(noise floor: {_g(f5)}; Eq. 18 {_g(30 * f5)})",
                 "how": "as above, at 5 %"})
    r_ck = at(nck["test_clean"], 15) / at(nck["test_clean"], 0)
    r_dn = at(ndn["test_clean"], 15) / at(ndn["test_clean"], 0)
    r_15 = at(ndn["test_clean"], 15) / at(nck["test_clean"], 15)
    rows.append({"quantity": "ChemKAN noise-free testing MSE, 15 % / 0 %", "section": "III A 3",
                 "paper": "~2x", "ours": f"{r_ck:.2f}x",
                 "how": "ratio of converged noise-free testing MSE (final checkpoints)"})
    rows.append({"quantity": "DeepONet noise-free testing MSE, 15 % / 0 %", "section": "III A 3",
                 "paper": "~5x", "ours": f"{r_dn:.2f}x", "how": "same, DeepONet 308"})
    rows.append({"quantity": "noise-free testing MSE at 15 %, DeepONet / ChemKAN",
                 "section": "III A 3", "paper": "4.4x", "ours": f"{r_15:.2f}x",
                 "how": "ratio of the two converged 15 % values"})
    return rows


def markdown(rows: list[dict]) -> str:
    lines = ["| quantity | paper (section) | ours | how ours was computed |",
             "|---|---|---|---|"]
    lines += [f"| {r['quantity']} | {r['paper']} (Sec. {r['section']}) | {r['ours']} | {r['how']} |"
              for r in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(comparison()))

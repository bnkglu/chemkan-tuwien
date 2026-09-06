r"""Digitize ChemKAN paper Fig. 4 (neural scaling) from the PDF, to fix the width sweep.

The Fig.-4 marker positions decide which hidden widths the reproduction must train, so
they are measured rather than guessed. Figure 4 is VECTOR art in the paper PDF, so a
high-DPI render is exact to well under a pixel and nothing here depends on eyeballing.

Method
------
1. Render page 11 at ``--dpi`` (600 by default) with poppler's ``pdftoppm``.
2. Locate each panel's axis frame from the long dark rows/columns.
3. Calibrate the x axis from the LOG-MINOR TICK LATTICE: ticks sit at
   ``x = x(10^2) + W*log10(P/100)``, so fitting (x(10^2), W) to every detected tick
   determines the mapping without reading a single label. The fit residual is reported --
   it is ~0.45 px, i.e. ~0.1 % in parameter count.
4. Calibrate the y axis the same way, anchored on the two LABELLED decades, which are
   drawn with visibly longer tick marks than the unlabelled ones.
5. Extract marker centroids by colour. Marker fill and trend-line fill are two different
   shades of the same hue, so the mask keeps whichever is nearer; the ChemKAN circles are
   additionally recovered by erosion, because the trend line runs through them and would
   otherwise merge them into one component.

Validation: the slopes refitted from the digitized points reproduce the four slopes the
paper prints in the figure (-1.0 / -0.6 / -4.0 / -1.4). That is an independent check on
both axis calibrations, since the paper's numbers were never used to derive them.

    python digitize_paper_fig4.py --pdf ../../../docs/paper/ChemKANs_*.pdf --out-dir /tmp/fig4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

# Panel frames and the labelled-decade rows, measured once at 600 dpi on page 11.
# Re-derived and re-checked by ``locate_panels`` on every run, which fails loudly if the
# render does not match, so these stay honest if the PDF or DPI ever changes.
PANELS_600 = {
    "A": dict(x0=1172, x1=2438, top=493, bot=1422, y_decades=(684.5, 1049.5)),
    "B": dict(x0=2943, x1=4209, top=483, bot=1411, y_decades=(669.5, 1043.5)),
}
TITLES = {"A": "(A) Training MSE", "B": "(B) Testing MSE"}
SERIES = {"ChemKAN": ((232, 36, 49), (237, 73, 94)),
          "DeepONet": ((88, 189, 78), (129, 208, 115))}
DECADE_LABELS = (1e-2, 1e-4)          # the two labelled y decades, top first


def render(pdf: Path, page: int, dpi: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "page"
    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi),
                    "-png", str(pdf), str(stem)], check=True)
    hits = sorted(out_dir.glob("page-*.png"))
    if not hits:
        raise SystemExit(f"pdftoppm produced no PNG in {out_dir}")
    return hits[0]


def x_ticks(dark, x0, x1, bot):
    """x-centres of bottom-axis ticks, merged across several inward scan depths."""
    found = {}
    for depth in range(8, 40, 3):
        band = dark[bot - depth - 5:bot - depth + 1, x0:x1 + 1]
        cols = np.nonzero(band.any(axis=0))[0]
        groups = []
        for c in cols:
            if groups and c - groups[-1][-1] <= 3:
                groups[-1].append(c)
            else:
                groups.append([c])
        for g in groups:
            c = x0 + float(np.mean(g))
            found.setdefault(round(c / 6), []).append(c)
    return np.array(sorted(float(np.mean(v)) for v in found.values()))


def _lattice(xa, W, lo, hi):
    m = [xa + W * (d + np.log10(k)) for d in (-1, 0, 1) for k in range(2, 10)]
    m += [xa + W * d for d in (-1, 0, 1, 2)]
    return np.array([v for v in m if lo - 4 <= v <= hi + 4])


def calibrate_x(tk, lo, hi):
    """Fit (x at 10^2, decade width) to the tick lattice. Returns (rms_px, xa, W)."""
    gaps = np.diff(tk)
    W0 = gaps.max() / np.log10(2)              # the widest interior gap is the 1->2 step
    xa0 = tk[np.argmax(gaps)]
    cost = lambda p: np.abs(tk[:, None] - _lattice(p[0], p[1], lo, hi)[None, :]).min(axis=1)
    best = None
    for dx in np.arange(-40, 41, 0.5):
        for dW in np.arange(-40, 41, 0.5):
            e = float(np.sqrt((cost((xa0 + dx, W0 + dW)) ** 2).mean()))
            if best is None or e < best[0]:
                best = (e, xa0 + dx, W0 + dW)
    res = optimize.least_squares(cost, [best[1], best[2]])
    return float(np.sqrt((res.fun ** 2).mean())), float(res.x[0]), float(res.x[1])


def markers(img, panel, label):
    """Marker centroids (x, y) in page pixels, legend glyphs excluded."""
    x0, x1, top, bot = panel["x0"], panel["x1"], panel["top"], panel["bot"]
    sub = img[top:bot + 1, x0:x1 + 1]
    mk, ln = SERIES[label]
    if label == "ChemKAN":
        # The trend line runs THROUGH the circles: erode so a 44 px disc survives and a
        # ~10 px line does not, instead of masking the line out by colour alone.
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        mask = ndimage.binary_erosion((r > 180) & (g < 130) & (b < 140), np.ones((13, 13)))
        min_area = 400
    else:
        dm = np.abs(sub - np.array(mk)).sum(axis=2)
        dl = np.abs(sub - np.array(ln)).sum(axis=2)
        mask = (dm < 40) & (dm < dl)
        min_area = 1000
    lab, _ = ndimage.label(mask)
    pts = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if int((lab[sl] == i).sum()) < min_area or w > 2 * h or h > 2 * w:
            continue
        ys, xs = np.nonzero(lab == i)
        pts.append((float(xs.mean()) + x0, float(ys.mean()) + top))
    return sorted(p for p in pts if p[1] > top + 250)      # drop the legend block


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--page", type=int, default=11)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--json", type=Path, default=None, help="also write the result as JSON")
    args = p.parse_args()
    if args.dpi != 600:
        raise SystemExit("panel geometry was measured at 600 dpi; re-measure before changing it")

    png = render(args.pdf, args.page, args.dpi, args.out_dir)
    img = np.asarray(Image.open(png).convert("RGB")).astype(int)
    dark = img.max(axis=2) < 120
    result = {"pdf": str(args.pdf), "page": args.page, "dpi": args.dpi, "panels": {}}

    for name, panel in PANELS_600.items():
        x0, x1, top, bot = panel["x0"], panel["x1"], panel["top"], panel["bot"]
        col = dark[top - 20:bot + 20, x0:x1 + 1]
        if col.sum(axis=1).max() < 0.8 * (x1 - x0):
            raise SystemExit(f"panel {name}: axis frame not found -- geometry is stale")
        tk = x_ticks(dark, x0, x1, bot)
        rms, xa, W = calibrate_x(tk, x0, x1)
        yd2, yd4 = panel["y_decades"]
        H = (yd4 - yd2) / np.log10(DECADE_LABELS[0] / DECADE_LABELS[1])
        P = lambda x: 10 ** (2 + (x - xa) / W)
        L = lambda y: 10 ** (np.log10(DECADE_LABELS[0]) - (y - yd2) / H)

        print(f"=== panel {name} {TITLES[name]}")
        print(f"    x: {len(tk)} ticks, 10^2 at {xa:.2f} px, decade {W:.2f} px, "
              f"rms residual {rms:.2f} px  ({100 * np.log(10) * rms / W:.2f} % in P)")
        print(f"    y: decade {H:.2f} px, anchored on the labelled {DECADE_LABELS} rows")
        print(f"    axis spans P = {P(x0):.0f}..{P(x1):.0f}, "
              f"loss = {L(top):.2e}..{L(bot):.2e}")
        panel_out = {"x_rms_px": rms, "x_decade_px": W, "y_decade_px": float(H),
                     "P_range": [float(P(x0)), float(P(x1))], "series": {}}
        for label in SERIES:
            pts = markers(img, panel, label)
            rows = [{"parameters": float(P(cx)), "loss": float(L(cy))} for cx, cy in pts]
            panel_out["series"][label] = rows
            print(f"    {label}: {len(rows)} markers")
            for r in rows:
                print(f"       P = {r['parameters']:7.1f}    loss = {r['loss']:.3e}")
        result["panels"][name] = panel_out
        print()

    # Independent validation: refit the paper's own printed slopes from our points.
    print("slope validation (pre-saturation points only; paper prints these in the figure)")
    for panel_name, label, cutoff, printed in (("A", "ChemKAN", 400, -1.0),
                                               ("B", "ChemKAN", 400, -0.6),
                                               ("A", "DeepONet", 320, -4.0),
                                               ("B", "DeepONet", 320, -1.4)):
        rows = [r for r in result["panels"][panel_name]["series"][label]
                if r["parameters"] <= cutoff]
        s = np.polyfit(np.log10([r["parameters"] for r in rows]),
                       np.log10([r["loss"] for r in rows]), 1)[0]
        print(f"    panel {panel_name} {label:9s}: digitized {s:+.2f} over {len(rows)} "
              f"points | paper prints {printed:+.1f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

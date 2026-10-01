"""Kink transmission check for planted ReLU kinks (diagnostic). Works on any model state."""
import json
import numpy as np
import torch
from common import rp

DELTA = 0.05          # s, one-sided slope window around each knot
VISIBLE = 1e-4        # |change of slope| (per s) in at least one final trunk coordinate


def transmission(model, knots, t_end=30.0):
    t = torch.linspace(0, t_end, 3001).reshape(-1, 1)
    with torch.no_grad():
        z1 = model.trunk[0](t); a1 = torch.relu(z1)
        z2 = model.trunk[2](a1); a2 = torch.relu(z2)
    rows = []
    for k in knots:
        tk = torch.tensor([[k - DELTA], [k], [k + DELTA]])
        with torch.no_grad():
            y = model.trunk(tk).numpy()                       # (3, 8)
        dslope = (y[2] - y[1]) / DELTA - (y[1] - y[0]) / DELTA
        rows.append({"knot_s": float(k), "max_abs_slope_change": float(np.abs(dslope).max()),
                     "visible": bool(np.abs(dslope).max() > VISIBLE),
                     "coords_carrying": [int(j) for j in np.flatnonzero(np.abs(dslope) > VISIBLE)]})
    return {"layer1_active": int((a1.amax(0) > 0).sum()), "layer1_dead": int((a1.amax(0) == 0).sum()),
            "layer2_active": int((a2.amax(0) > 0).sum()), "layer2_dead": int((a2.amax(0) == 0).sum()),
            "layer2_preact_std_over_t": z2.std(0).numpy().round(4).tolist(),
            "trunk_coord_range_over_t": (a2.amax(0) - a2.amin(0)).numpy().round(4).tolist(),
            "knots": rows, "delta_s": DELTA, "visible_threshold": VISIBLE}


if __name__ == "__main__":
    from common import build
    knots = {e["K"]: e["knots_s"] for e in json.loads(open("stage2_knots/knots.json").read())["fits"]}
    for K in (3, 4):
        for s in range(5):
            m = rp.apply_init(build(8, seed=s), "Aprime_tf_truncated_glorot_zero_bias", s)
            rp.plant_knots(m, knots[K], 30.0)
            r = transmission(m, knots[K])
            print(f"P{K} seed {s}: L1 active {r['layer1_active']}/7 L2 active {r['layer2_active']}/8 | "
                  + " ".join(f"k={x['knot_s']:.2f}:{'vis' if x['visible'] else 'HIDDEN'}({x['max_abs_slope_change']:.1e})" for x in r["knots"]))

"""Fig. 6 (15 % noise, test row 24) on the PAPER reaction-order dataset
(biodiesel_v2_paper_order.npz: identical to biodiesel_v2 except ``--reaction-order paper_text``).

Same drawing (fig06_biodiesel_profiles.plot_profiles), same models and settings as the 07b
Fig. 6: ChemKAN batched FSA (H = 4, seed 0) and DeepONet tanh lr 1e-2 (R1 settings, seed 0),
both trained on the paper-order data. Noise-free MSE is time-averaged in ChemKAN's train-only
normalization, as in 07b.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "chemkan/scripts/figures"))
import author_runs as ar                                          # noqa: E402
import fig06_biodiesel_profiles as fig06                          # noqa: E402
from common import save_figure, use_headless_backend              # noqa: E402

DATA = ROOT / "chemkan/data/generated/biodiesel_v2_paper_order.npz"


def main():
    use_headless_backend()
    d = dict(np.load(DATA))
    i = ar.FIG3_TEST_INDEX
    y0, T, t = d["test_states"][i, 0], float(d["test_T"][i]), d["t"]
    clean, noisy = d["test_states"][i], d["test_states_noise15"][i]
    ck = {"dir": HERE / "fig05/chemkan_noise15_seed0", "model": "chemkan"}
    don = {"dir": HERE / "fig05/deeponet_noise15_seed0", "model": "deeponet",
           "variant": "tanh_r1_lr1e-2"}
    src = ar.config(ck)["data_source"]
    assert Path(src["path"]).name == DATA.name, src["path"]
    assert Path(ar.config(don)["dataset_file"]["path"]).name == DATA.name
    rng = np.array(src["species_max"]) - np.array(src["species_min"])
    pred = {"chemkan": ar.chemkan_predict(ck, y0, T, t), "deeponet": ar.deeponet_predict(don, y0, T, t)}
    mse = {k: float(np.mean(((v - clean) / rng) ** 2)) for k, v in pred.items()}
    fig = fig06.plot_profiles(
        t, list(d["species"]), clean, noisy, pred["chemkan"], pred["deeponet"],
        f"Fig. 6 - 15% noise, test row 24 (TG0={y0[0]:.3f}, ROH0={y0[1]:.3f}, T={T:.2f} K), "
        f"PAPER reaction order\nnoise-free {ar.CONVENTION_SHORT}: ChemKAN {mse['chemkan']:.2e}, "
        f"DeepONet (tanh, lr 1e-2) {mse['deeponet']:.2e}")
    save_figure(fig, HERE / "fig06_profiles_15pct_paper_order")
    print(mse)


if __name__ == "__main__":
    main()

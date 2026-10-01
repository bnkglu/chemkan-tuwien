import json
import numpy as np
import torch
from common import DIAG, at, build, load_model, rp, trunk_diag

RR = DIAG / "relu_reachability"
def runs(arm):
    if arm == "R1":
        return [DIAG / "init_ablation" / "Aprime_tf_truncated_glorot_zero_bias" / f"seed{s}" for s in range(10)]
    if arm == "R5":
        return ([RR / "stage1_branch" / "normalized" / f"seed{s}" for s in range(5)]
                + [RR / "stage3_screen" / "R5_torchinit_lr2e-3" / f"seed{s}" for s in range(5, 10)])
    return [RR / "stage3_screen" / {"R2": "R2_refinit_lr2e-3", "R3": "R3_glorot_uniform_lr1e-3",
                                    "R4": "R4_glorot_uniform_lr2e-3"}[arm] / f"seed{s}" for s in range(10)]

LABEL = {"R1": "Lu/DeepXDE reference (TF Glorot normal, zero bias, lr 1e-3)",
         "R2": "reference-init lr ablation (lr 2e-3)",
         "R3": "Glorot-uniform zero bias, lr 1e-3", "R4": "Glorot-uniform zero bias, lr 2e-3",
         "R5": "PyTorch-default control, lr 2e-3"}
# pairing: R2 vs the deterministic R1 initialization; R3 vs R4; R5 = torch default draw
for s in range(10):
    r1 = rp.apply_init(build(8, seed=s), "Aprime_tf_truncated_glorot_zero_bias", s).state_dict()
    r2 = torch.load(runs("R2")[s] / "checkpoint_init.pt", weights_only=False)["model_state"]
    assert all(torch.equal(r1[k], r2[k]) for k in r1)
    r3 = torch.load(runs("R3")[s] / "checkpoint_init.pt", weights_only=False)["model_state"]
    r4 = torch.load(runs("R4")[s] / "checkpoint_init.pt", weights_only=False)["model_state"]
    assert all(torch.equal(r3[k], r4[k]) for k in r3)
print("pairing OK: R2 == R1 initialization (rebuilt deterministically; the reused R1 runs predate "
      "checkpoint_init.pt), R3 == R4, every seed")

summary = {}
for arm in LABEL:
    print(f"\n== {arm}: {LABEL[arm]}")
    tr, te = [], []
    for s, run in enumerate(runs(arm)):
        tr.append(at(run, 2500)); te.append(at(run, 2500, "test_mean_mse"))
        d = trunk_diag(load_model(run / "checkpoint_final.pt", s))
        print(f"seed {s}: train {tr[-1]:.2e} test {te[-1]:.2e} | dead1 {d['dead1']} dead2 {d['dead2']} | "
              f"useful bp {d['n_useful']} at {[round(x, 1) for x in d['useful_s']]} (interior total {d['n_interior']})")
    tr, te = np.array(tr), np.array(te)
    c = {th: int((tr <= th).sum()) for th in (1e-3, 5e-4, 3e-4, 1e-4)}
    summary[arm] = {"median_train": float(np.median(tr)), "best": float(tr.min()), "worst": float(tr.max()),
                    "median_test": float(np.median(te)), "counts": {str(k): v for k, v in c.items()}}
    print(f"median train {np.median(tr):.2e} best {tr.min():.2e} worst {tr.max():.2e} | median test {np.median(te):.2e} | "
          + " ".join(f"<= {k:g}: {v}/10" for k, v in c.items()))
b = sorted(summary, key=lambda a: (-summary[a]["counts"]["0.0003"], summary[a]["median_train"]))[0]
print(f"\nStage-4 B (most seeds <= 3e-4 at 2.5k, tie -> lower median): {b}")
(RR / "stage3_screen" / "stage3_summary.json").write_text(json.dumps({"arms": summary, "stage4_B": b}, indent=2))

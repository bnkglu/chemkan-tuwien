"""Fig. 4: evaluate every scaling checkpoint, fit slopes with EXPLICIT masks, plot."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
for directory in ("chemkan/src", "chemkan/scripts", "deeponet"):
    sys.path.insert(0, str(ROOT / directory))
import csv, numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, StrMethodFormatter, LogLocator, LogFormatterSciNotation
from evaluate_biodiesel import evaluate_biodiesel
import evaluate_biodiesel_deeponet as don

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n-mu", choices=("scaled", "2"), default="scaled")
parser.add_argument("--deeponet-version", choices=("reference", "legacy"), default="reference")
args = parser.parse_args()
suffix = ("_nmu2" if args.n_mu == "2" else "") + ("_legacy" if args.deeponet_version == "legacy" else "")
CKS=ROOT/"results/reproduction/chemkan/biodiesel/scaling"
DOS=ROOT/"results/reproduction/baselines/deeponet/biodiesel"
DON_VERSION = ("reference_final_trunk_relu" if args.deeponet_version == "reference"
               else "legacy_final_trunk_linear")
DOS = (DOS / DON_VERSION if args.deeponet_version == "reference" else DOS) / "scaling"
REPLAY=ROOT/"results/reproduction/chemkan/biodiesel/noise/clean_replay_seed0"
T=ROOT/"results/reproduction/tables"; FIG=ROOT/"results/reproduction/figures/biodiesel"

def late_band(run_dir, upto):
    """(min, max, median) of the RAW training loss over the last 20% of epochs up to `upto`.

    The ChemKAN training loss oscillates strongly, so a single fixed-checkpoint value is one
    draw from this band. The median is a robust secondary estimate; it is reported ALONGSIDE
    the final-checkpoint value, never in place of it. Only the training loss has a per-epoch
    record here -- the scaling runs were launched without in-training test evaluation.
    """
    rows=list(csv.DictReader(open(run_dir/"history.csv")))
    L=np.array([float(r["total_loss"]) for r in rows if int(float(r["epoch"]))<upto])
    late=L[int(.8*len(L)):]
    return float(late.min()), float(late.max()), float(np.median(late))

pts=[]
CK_SPEC=[(h, CKS/f"h{h:02d}_seed0"/"checkpoint_final.pt", False) for h in (2,3,10,17)]
CK_SPEC.append((4, REPLAY/"checkpoint_epoch_5000.pt", True))
if args.n_mu == "2":
    manifest = json.loads((CKS.parent/"scaling_nmu2/manifest_scaling_seed0.json").read_text())
    CK_SPEC = [(j["width"], ROOT/j["checkpoint"], "scaling_nmu2" not in j["checkpoint"])
               for j in manifest["jobs"]]
for h,ckp,reused in sorted(CK_SPEC):
    d = ckp.parent
    if not ckp.exists(): raise FileNotFoundError(ckp)
    tr=evaluate_biodiesel(str(ckp),"train","cpu"); te=evaluate_biodiesel(str(ckp),"test","cpu")
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    expected_nmu = 2 if args.n_mu == "2" else (h+1)//2
    assert ck["architecture"]["n_mu"] == expected_nmu
    assert ck.get("stage2_epoch", ck["training"]["epochs"]) == 5000
    lo,hi,med = late_band(d, 5000)
    pts.append(dict(model="ChemKAN",width=h,parameters=tr["n_params"],
                    train_loss=tr["mse"],test_loss=te["mse"],
                    train_loss_late_median=med, train_late_min=lo, train_late_max=hi,
                    source=str(ckp.relative_to(ROOT)),
                    reused=reused, architecture_version="", n_mu=expected_nmu))
for w in (3,5,6,8,10,13):
    d=DOS/f"w{w:02d}_seed0"; ckp=d/"checkpoint_final.pt"
    if not ckp.exists(): raise FileNotFoundError(ckp)
    tr=don.evaluate(str(ckp),"train","cpu"); te=don.evaluate(str(ckp),"test","cpu")
    assert tr["model"].architecture_version == DON_VERSION
    assert tr["ckpt"]["training"]["epochs"] == 50000
    lo,hi,med = late_band(d, 50000)
    pts.append(dict(model="DeepONet",width=w,parameters=tr["n_params"],
                    train_loss=tr["mse"],test_loss=te["mse"],
                    train_loss_late_median=med, train_late_min=lo, train_late_max=hi,
                    source=str(ckp.relative_to(ROOT)),reused=False,
                    architecture_version=DON_VERSION, n_mu=""))

def fit(P,L):
    x,y=np.log10(P),np.log10(L); n=len(x)
    s,i=np.polyfit(x,y,1); yh=s*x+i
    ss=float(((y-yh)**2).sum()); r2=1-ss/float(((y-y.mean())**2).sum())
    se=float(np.sqrt(ss/(n-2)/((x-x.mean())**2).sum())) if n>2 else float("nan")
    return float(s),float(i),float(r2),se

# EXPLICIT MASK: ALL points are included.
# The paper fits "prior to saturation"; our curves have no identifiable saturation regime
# to cut at -- they are NON-MONOTONIC in parameter count. A minimum-based mask degenerates
# (it left the ChemKAN training fit with 2 points and an undefined standard error), so it
# is not used. Every point is fitted, and the poor R^2 is reported rather than improved by
# dropping points after the fact.
MASK_RULE = ("all measured points, with no post-hoc exclusions; descriptive regression; "
             "fitting scope differs from the paper's pre-saturation subset")
fits=[]
for model in ("ChemKAN","DeepONet"):
    sub=sorted([p for p in pts if p["model"]==model], key=lambda r:r["parameters"])
    if len(sub)<3: continue
    for metric in ("train_loss","test_loss","train_loss_late_median"):
        if metric=="train_loss_late_median" and any(r.get(metric) is None for r in sub): continue
        P=np.array([r["parameters"] for r in sub]); L=np.array([r[metric] for r in sub])
        s,i,r2,se=fit(P,L)
        fits.append(dict(model=model,metric=metric,slope=s,intercept=i,r_squared=r2,
                         std_error=se,n_included=len(P),
                         included=",".join(str(int(v)) for v in P),
                         excluded="(none)", mask_rule=MASK_RULE))
with (T/f"biodiesel_fig4_points{suffix}.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(pts[0])); w.writeheader(); w.writerows(pts)
with (T/f"biodiesel_fig4_fits{suffix}.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(fits[0])); w.writeheader(); w.writerows(fits)

# Reported paper slopes are comparison values; the lines retain our measured fits.
PAPER_SLOPES = {
    "train_loss": {"ChemKAN": -1.0, "DeepONet": -4.0},
    "test_loss": {"ChemKAN": -0.6, "DeepONet": -1.4},
}
PARAMETER_COUNTS = sorted({int(p["parameters"]) for p in pts})
loss_values = [p[metric] for p in pts for metric in ("train_loss", "test_loss")]
LOSS_LIMITS = (10 ** np.floor(np.log10(min(loss_values))),
               10 ** np.ceil(np.log10(max(loss_values))))

fig,axes=plt.subplots(1,2,figsize=(14,7),sharex=True,sharey=True)
for ax,metric,panel,tt in ((axes[0],"train_loss","A","Training MSE"),(axes[1],"test_loss","B","Testing MSE")):
    slope_rows = []
    for model,col in (("ChemKAN","crimson"),("DeepONet","seagreen")):
        sub=sorted([p for p in pts if p["model"]==model],key=lambda r:r["parameters"])
        if not sub: continue
        P=np.array([r["parameters"] for r in sub]); L=np.array([r[metric] for r in sub])
        ax.plot(P,L,"o",color=col,ms=8,label=f"{model}: our results")
        fr=next((f for f in fits if f["model"]==model and f["metric"]==metric),None)
        if fr:
            inc=np.array([int(v) for v in fr["included"].split(",")])
            xs=np.linspace(np.log10(inc.min()),np.log10(inc.max()),20)
            ax.plot(10**xs,10**(fr["slope"]*xs+fr["intercept"]),"-",color=col,lw=1.6,
                    label=f"{model}: fit to our results")
            slope_rows.append([model, f"{fr['slope']:.2f}", f"{PAPER_SLOPES[metric][model]:.1f}"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of trainable parameters", fontsize=11, labelpad=8)
    ax.set_xlim(70, 1000)
    ax.set_ylim(LOSS_LIMITS)
    # Decade labels match the y-axis notation; minor ticks identify every measured size.
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=True))
    ax.xaxis.set_minor_locator(FixedLocator(PARAMETER_COUNTS))
    ax.xaxis.set_minor_formatter(StrMethodFormatter("{x:.0f}"))
    ax.tick_params(axis="both", which="major", labelsize=10, labelleft=True)
    ax.tick_params(axis="x", which="major", pad=29, length=6)
    ax.tick_params(axis="x", which="minor", labelsize=8, labelrotation=60, pad=3, length=3)
    ax.set_title(f"({panel}) {tt}", fontsize=13)
    ax.grid(alpha=.25,which="both")
    if slope_rows:
        comparison = ax.table(cellText=slope_rows,
                              colLabels=["Model", r"Our $\Delta$", r"Paper $\Delta$"],
                              cellLoc="center", colWidths=[0.42, 0.29, 0.29],
                              bbox=(0.03, 0.025, 0.59, 0.21), zorder=5)
        comparison.auto_set_font_size(False)
        comparison.set_fontsize(10)
        for (row, col), cell in comparison.get_celld().items():
            cell.set_edgecolor("0.8")
            cell.set_linewidth(0.6)
            cell.set_facecolor("0.95" if row == 0 else "white")
            if row == 0:
                cell.get_text().set_fontweight("bold")
            else:
                cell.get_text().set_color({"ChemKAN": "crimson", "DeepONet": "seagreen"}[slope_rows[row - 1][0]])
axes[0].set_ylabel("Loss (Eq. 18)", fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.045),
           fontsize=9, handlelength=3, columnspacing=2.5, framealpha=1)
ck_label = "fixed n_mu=2" if args.n_mu == "2" else "n_mu=ceil(h/2)"
fig.suptitle(f"Fig. 4 - neural scaling: ChemKAN {ck_label}", fontsize=15, y=0.99)
fig.text(0.5, 0.94, f"DeepONet: {DON_VERSION}. Circles: measured results; lines: fitted trends.",
         ha="center", fontsize=10)
fig.text(0.5, 0.012, r"$\Delta$ is the slope of log(loss) vs log(parameters). "
         "Our fits use all runs; the paper reports fits before saturation.",
         ha="center", fontsize=9, color="0.3")
fig.tight_layout(rect=(0, 0.20, 1, 0.91))
for e in ("pdf","png"): fig.savefig(FIG/f"fig04_biodiesel_neural_scaling{suffix}.{e}",dpi=150,bbox_inches="tight")
plt.close(fig)

print(f"{'model':9s} {'w':>3s} {'params':>7s} {'train(final)':>13s} {'test(final)':>12s} "
      f"{'train late median':>18s} {'late band':>22s}")
for r in sorted(pts,key=lambda r:(r["model"],r["parameters"])):
    band=f"[{r['train_late_min']:.2e},{r['train_late_max']:.2e}]"
    print(f"{r['model']:9s} {r['width']:3d} {r['parameters']:7d} {r['train_loss']:13.4e} "
          f"{r['test_loss']:12.4e} {r['train_loss_late_median']:18.4e} {band:>22s}"
          f"{'  REUSED' if r['reused'] else ''}")
print()
for f_ in fits:
    print(f"{f_['model']:9s} {f_['metric']:10s} slope {f_['slope']:+.3f}  R2 {f_['r_squared']:.3f}  "
          f"SE {f_['std_error']:.3f}  included [{f_['included']}]  excluded [{f_['excluded']}]")

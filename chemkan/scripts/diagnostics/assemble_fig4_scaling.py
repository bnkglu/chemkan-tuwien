"""Fig. 4: evaluate every scaling checkpoint, fit slopes with EXPLICIT masks, plot."""
import sys; sys.path.insert(0,"."); sys.path.insert(0,"../../deeponet")
import csv, json, numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from evaluate_biodiesel import evaluate_biodiesel
import evaluate_biodiesel_deeponet as don

ROOT=Path("../..").resolve()
CKS=ROOT/"results/reproduction/chemkan/biodiesel/scaling"
DOS=ROOT/"results/reproduction/baselines/deeponet/biodiesel/scaling"
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
CK_SPEC=[(2,CKS/"h02_seed0"),(3,CKS/"h03_seed0"),
         (4,REPLAY),                                  # epoch-5000 snapshot, not a new run
         (10,CKS/"h10_seed0"),(17,CKS/"h17_seed0")]
for h,d in CK_SPEC:
    ckp = d/"checkpoint_epoch_5000.pt" if d is REPLAY else d/"checkpoint_final.pt"
    if not ckp.exists(): print("MISSING", ckp); continue
    tr=evaluate_biodiesel(str(ckp),"train","cpu"); te=evaluate_biodiesel(str(ckp),"test","cpu")
    lo,hi,med = late_band(d, 5000)
    pts.append(dict(model="ChemKAN",width=h,parameters=tr["n_params"],
                    train_loss=tr["mse"],test_loss=te["mse"],
                    train_loss_late_median=med, train_late_min=lo, train_late_max=hi,
                    source=str(ckp.relative_to(ROOT)),
                    reused=bool(d is REPLAY)))
for w in (3,5,6,8,10,13):
    d=DOS/f"w{w:02d}_seed0"; ckp=d/"checkpoint_final.pt"
    if not ckp.exists(): print("MISSING", ckp); continue
    tr=don.evaluate(str(ckp),"train","cpu"); te=don.evaluate(str(ckp),"test","cpu")
    lo,hi,med = late_band(d, 50000)
    pts.append(dict(model="DeepONet",width=w,parameters=tr["n_params"],
                    train_loss=tr["mse"],test_loss=te["mse"],
                    train_loss_late_median=med, train_late_min=lo, train_late_max=hi,
                    source=str(ckp.relative_to(ROOT)),reused=False))

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
MASK_RULE = ("all points; the paper's pre-saturation mask has no analogue here because our "
             "loss-vs-parameter curves are non-monotonic")
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
with (T/"biodiesel_fig4_points.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(pts[0])); w.writeheader(); w.writerows(pts)
with (T/"biodiesel_fig4_fits.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(fits[0])); w.writeheader(); w.writerows(fits)

fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True)
paper=json.loads((T/"paper_fig4_digitized.json").read_text())
for ax,metric,panel,tt in ((axes[0],"train_loss","A","Training MSE"),(axes[1],"test_loss","B","Testing MSE")):
    for model,col in (("ChemKAN","crimson"),("DeepONet","seagreen")):
        sub=sorted([p for p in pts if p["model"]==model],key=lambda r:r["parameters"])
        if not sub: continue
        P=np.array([r["parameters"] for r in sub]); L=np.array([r[metric] for r in sub])
        ax.plot(P,L,"o",color=col,ms=8,label=f"{model} (ours)")
        fr=next((f for f in fits if f["model"]==model and f["metric"]==metric),None)
        if fr:
            inc=np.array([int(v) for v in fr["included"].split(",")])
            xs=np.linspace(np.log10(inc.min()),np.log10(inc.max()),20)
            ax.plot(10**xs,10**(fr["slope"]*xs+fr["intercept"]),"-",color=col,lw=1.6,
                    label=f"{model} fit $\\Delta$={fr['slope']:.2f} (n={fr['n_included']})")
        pp=paper["panels"][panel]["series"][model]
        ax.plot([r["parameters"] for r in pp],[r["loss"] for r in pp],"^",color=col,
                mfc="none",alpha=.55,ms=7,label=f"{model} (paper, digitized)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("# parameters")
    ax.set_title(f"({panel}) {tt}"); ax.grid(alpha=.3,which="both")
axes[0].set_ylabel("loss (Eq. 18)"); axes[0].legend(fontsize=6.5)
fig.suptitle("Fig. 4 - neural scaling. Filled = our runs; open = the paper's own digitized "
             "points (different absolute scale; shown for shape, not agreement)")
fig.tight_layout()
for e in ("pdf","png"): fig.savefig(FIG/f"fig04_biodiesel_neural_scaling.{e}",dpi=150,bbox_inches="tight")

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

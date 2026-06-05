"""
Phase 1 / R2 + R3 — cross-year transfer and the pure-year (seen vs unseen) probe.
Uses the R1-best preprocessing (sg2). Models EN (R1-best) + PLS (cross-check), tuned by
inner CV on the TRAINING YEAR only (no peeking at the test year).
R2: train 2021 -> test 2022 and reverse. R3: split the test year into the 34 genotypes also
present in the training year ("seen") vs the rest ("unseen") to separate year vs new-genotype effect.
Also: relative-tertile screening metrics (can predicted DON flag the top-tertile high-DON lines?).
Outputs: results/04_r2_r3.md  +  results/04_r2_r3.tsv  +  results/FIG_cross_year_scatter.png
"""
import os, sys, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import make_preprocessor
warnings.filterwarnings("ignore")
try:                                   # UnicodeEncodeError guard for legacy (GBK) consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file is in scripts/)
DATA = os.environ.get("WHEAT_DON_DATA", os.path.join(_ROOT, "data"))
RES  = os.environ.get("WHEAT_DON_RESULTS", os.path.join(_ROOT, "results"))
SEED=42; PREP="sg2"
F={"2021":"2021_Phenomic_Data_VIS-NIR.csv","2022":"2022_Phenomic_Data_VIS-NIR.csv"}
# R1-best within-year R2 (from script 03) for the gap table:
R1_REF={("2021","EN"):0.667,("2022","EN"):0.517,("2021","PLS"):0.653,("2022","PLS"):0.481}

def load(fn):
    df=pd.read_csv(os.path.join(DATA,fn))
    wl=[c for c in df.columns if str(c).startswith("R")]
    return df["Genotype"].astype(str).values, df[wl].values.astype(float), df["DON"].values.astype(float)

def build(model):
    steps=make_preprocessor(PREP)
    if model=="PLS":
        steps+=[("model",PLSRegression())]; grid={"model__n_components":[2,4,6,8,10,12]}
    else:
        steps+=[("scale",StandardScaler()),("model",ElasticNet(max_iter=20000))]
        grid={"model__alpha":np.logspace(-3,1,7),"model__l1_ratio":[0.2,0.5,0.8]}
    return Pipeline(steps),grid

def fit_predict(Xtr,ytr,Xte,model):
    pipe,grid=build(model)
    gs=GridSearchCV(pipe,grid,cv=KFold(5,shuffle=True,random_state=SEED),scoring="r2",n_jobs=-1)
    gs.fit(Xtr,ytr)
    return np.asarray(gs.predict(Xte)).ravel(), gs.best_params_

def metrics(y,pred):
    resid=y-pred; r2=1-np.sum(resid**2)/np.sum((y-y.mean())**2)
    rmse=np.sqrt(np.mean(resid**2)); bias=np.mean(pred-y); r=pearsonr(y,pred)[0]
    return r2,rmse,bias,r

def screening(y,pred):
    # truth = top tertile of the test batch (relative high-DON); score = predicted DON
    thr=np.quantile(y,2/3); truth=(y>thr).astype(int)
    auc=roc_auc_score(truth,pred)
    pthr=np.quantile(pred,2/3); flag=(pred>pthr).astype(int)
    tp=int(((flag==1)&(truth==1)).sum()); fn=int(((flag==0)&(truth==1)).sum())
    tn=int(((flag==0)&(truth==0)).sum()); fp=int(((flag==1)&(truth==0)).sum())
    sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan
    return auc,sens,spec

g21,X21,y21=load(F["2021"]); g22,X22,y22=load(F["2022"])
overlap=set(g21)&set(g22)
rows=[]; scatter={}
for (trn,tst,gtr,Xtr,ytr,gte,Xte,yte) in [
        ("2021","2022",g21,X21,y21,g22,X22,y22),
        ("2022","2021",g22,X22,y22,g21,X21,y21)]:
    for model in ["EN","PLS"]:
        pred,bp=fit_predict(Xtr,ytr,Xte,model)
        r2,rmse,bias,r=metrics(yte,pred); auc,sens,spec=screening(yte,pred)
        seen=np.array([g in overlap for g in gte])
        # R3 breakdown
        def sub(mask,label):
            if mask.sum()<5: return dict(subset=label,n=int(mask.sum()),r2=np.nan,rmse=np.nan,r=np.nan)
            s_r2,s_rmse,_,s_r=metrics(yte[mask],pred[mask])
            return dict(subset=label,n=int(mask.sum()),r2=s_r2,rmse=s_rmse,r=s_r)
        all_m=sub(np.ones_like(seen,bool),"all"); seen_m=sub(seen,"seen(R3)"); uns_m=sub(~seen,"unseen")
        rows.append(dict(regime="R2_cross", train=trn, test=tst, model=model, prep=PREP,
                         R2=r2, RMSE=rmse, bias=bias, pearson_r=r, auc_screen=auc,
                         sens_topT=sens, spec_topT=spec,
                         R1_ref=R1_REF[(tst,model)], R1_minus_R2=R1_REF[(tst,model)]-r2,
                         n_seen=int(seen.sum()), r2_seen=seen_m["r2"], r2_unseen=uns_m["r2"],
                         rmse_seen=seen_m["rmse"], rmse_unseen=uns_m["rmse"]))
        if model=="EN": scatter[(trn,tst)]=(yte,pred,seen)

res=pd.DataFrame(rows)
os.makedirs(RES,exist_ok=True)
res.to_csv(os.path.join(RES,"04_r2_r3.tsv"),sep="\t",index=False,float_format="%.4f")

md=["# R2 cross-year transfer + R3 seen/unseen probe", "",
    f"Preprocessing = {PREP} (R1-best). Hyperparameters tuned on TRAINING year only.",
    f"Overlap genotypes (R3 'seen') = {len(overlap)}.", "",
    "## Headline — random CV (R1) vs cross-year transfer (R2)",
    "| train→test | model | R1 within-yr R2 | R2 cross-yr R2 | **drop (R1−R2)** | cross-yr RMSE | bias | Pearson r |",
    "|---|---|---|---|---|---|---|---|"]
for _,x in res.iterrows():
    md.append(f"| {x.train}→{x.test} | {x.model} | {x.R1_ref:.3f} | {x.R2:.3f} | **{x.R1_minus_R2:+.3f}** | {x.RMSE:.2f} | {x.bias:+.2f} | {x.pearson_r:.3f} |")
md+=["", "## R3 — was the drop *year* or *new genotype*? (seen vs unseen within test year)",
     "| train→test | model | n seen | R2 seen | R2 unseen | RMSE seen | RMSE unseen |",
     "|---|---|---|---|---|---|---|"]
for _,x in res.iterrows():
    md.append(f"| {x.train}→{x.test} | {x.model} | {x.n_seen} | {x.r2_seen:.3f} | {x.r2_unseen:.3f} | {x.rmse_seen:.2f} | {x.rmse_unseen:.2f} |")
md+=["", "## Relative high-DON screening (top tertile of test batch; score = predicted DON)",
     "| train→test | model | ROC-AUC | sensitivity@topT | specificity@topT |","|---|---|---|---|---|"]
for _,x in res.iterrows():
    md.append(f"| {x.train}→{x.test} | {x.model} | {x.auc_screen:.3f} | {x.sens_topT:.3f} | {x.spec_topT:.3f} |")
md+=["","Full table: results/04_r2_r3.tsv"]
open(os.path.join(RES,"04_r2_r3.md"),"w",encoding="utf-8").write("\n".join(md))

# scatter figure
fig,axes=plt.subplots(1,2,figsize=(10,4.6))
for ax,(k,(yte,pred,seen)) in zip(axes,scatter.items()):
    ax.scatter(yte[~seen],pred[~seen],s=14,alpha=.55,label="unseen",color="#3D6E70")
    ax.scatter(yte[seen],pred[seen],s=26,alpha=.9,label=f"seen (n={int(seen.sum())})",color="#C77B53",edgecolor="#33302C",linewidth=.4)
    lo=min(yte.min(),pred.min()); hi=max(yte.max(),pred.max())
    ax.plot([lo,hi],[lo,hi],"k--",lw=1)
    r2,rmse,_,r=metrics(yte,pred)
    ax.set_title(f"train {k[0]} -> test {k[1]}  (EN, sg2)\nR2={r2:.2f}  r={r:.2f}  RMSE={rmse:.1f}")
    ax.set_xlabel("observed DON (ppm)"); ax.set_ylabel("predicted DON (ppm)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES,"FIG_cross_year_scatter.png"),dpi=300); fig.savefig(os.path.join(RES,"FIG_cross_year_scatter.pdf"))
print("\n".join(md)); print("\n[written] results/04_r2_r3.{md,tsv} + FIG_cross_year_scatter.png")

"""
Phase 1 / waveband audit — which bands carry the DON signal, are they stable across years,
VIS vs NIR vs full, and is the signal in the red ~600 nm / FHB-damage region (signal nature)?
Uses sg2 preprocessing. Outputs: results/05_waveband_audit.md + results/FIG_waveband_importance.png
"""
import os, sys, numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedKFold, GridSearchCV, KFold, cross_val_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import make_preprocessor, _sg
warnings.filterwarnings("ignore")
try:                                   # UnicodeEncodeError guard for legacy (GBK) consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file is in scripts/)
DATA = os.environ.get("WHEAT_DON_DATA", os.path.join(_ROOT, "data"))
RES  = os.environ.get("WHEAT_DON_RESULTS", os.path.join(_ROOT, "results"))
SEED=42
F={"2021":"2021_Phenomic_Data_VIS-NIR.csv","2022":"2022_Phenomic_Data_VIS-NIR.csv"}

def load(fn):
    df=pd.read_csv(os.path.join(DATA,fn))
    wl=[c for c in df.columns if str(c).startswith("R")]
    nm=np.array([float(c[1:]) for c in wl])
    return df[wl].values.astype(float), df["DON"].values.astype(float), nm

def en_coef_importance(X,y):
    # sg2 -> standardize -> EN; |standardized coef| as band importance
    Xs=_sg(X,2); sc=StandardScaler().fit(Xs); Xz=sc.transform(Xs)
    gs=GridSearchCV(ElasticNet(max_iter=20000),
                    {"alpha":np.logspace(-3,1,7),"l1_ratio":[0.2,0.5,0.8]},
                    cv=KFold(5,shuffle=True,random_state=SEED),scoring="r2",n_jobs=-1).fit(Xz,y)
    return np.abs(gs.best_estimator_.coef_)

def boot_importance(X,y,n=200):
    rng=np.random.default_rng(SEED); imp=np.zeros(X.shape[1])
    Xs=_sg(X,2)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        sc=StandardScaler().fit(Xs[idx]); Xz=sc.transform(Xs[idx])
        en=ElasticNet(alpha=0.1,l1_ratio=0.5,max_iter=20000).fit(Xz,y[idx])
        imp += (np.abs(en.coef_)>1e-8)
    return imp/n  # selection frequency per band

def region_cv(X,y,nm,lo,hi):
    m=(nm>=lo)&(nm<hi)
    if m.sum()<3: return np.nan
    steps=make_preprocessor("sg2")+[("scale",StandardScaler()),("model",ElasticNet(max_iter=20000))]
    pipe=Pipeline(steps)
    gs=GridSearchCV(pipe,{"model__alpha":np.logspace(-3,1,7),"model__l1_ratio":[0.2,0.5,0.8]},
                    cv=KFold(5,shuffle=True,random_state=SEED),scoring="r2",n_jobs=-1)
    sc=cross_val_score(gs,X[:,m],y,cv=RepeatedKFold(n_splits=5,n_repeats=3,random_state=SEED),scoring="r2",n_jobs=-1)
    return sc.mean()

X21,y21,nm=load(F["2021"]); X22,y22,_=load(F["2022"])
imp21=en_coef_importance(X21,y21); imp22=en_coef_importance(X22,y22)
sel21=boot_importance(X21,y21); sel22=boot_importance(X22,y22)

# cross-year stability
rho,_=spearmanr(imp21,imp22)
k=20
top21=set(np.argsort(imp21)[-k:]); top22=set(np.argsort(imp22)[-k:])
jacc=len(top21&top22)/len(top21|top22)
def topnm(imp,k=8):
    idx=np.argsort(imp)[-k:][::-1]; return [(round(float(nm[i]),1),round(float(imp[i]),3)) for i in idx]

# region comparison
REG=[("VIS 400-700",400,700),("NIR 700-1000",700,1004),("full 397-1004",397,1004),("red 560-680",560,680)]
reg_rows=[]
for name,lo,hi in REG:
    reg_rows.append((name,region_cv(X21,y21,nm,lo,hi),region_cv(X22,y22,nm,lo,hi)))

md=["# Phase 1 — waveband audit (sg2, EN)","",
    "## Cross-year stability of band importance",
    f"- Spearman rho(importance_2021, importance_2022) = **{rho:.3f}**",
    f"- top-{k} band overlap (Jaccard) = **{jacc:.3f}**  ({len(top21&top22)}/{len(top21|top22)})",
    f"- 2021 top bands (nm, |coef|): {topnm(imp21)}",
    f"- 2022 top bands (nm, |coef|): {topnm(imp22)}","",
    "## Region comparison (within-year repeated CV R2, sg2/EN)",
    "| region | 2021 R2 | 2022 R2 |","|---|---|---|"]
for name,a,b in reg_rows: md.append(f"| {name} | {a:.3f} | {b:.3f} |")
md+=["", "## Signal-nature read",
     "- If top bands cluster in the visible/red (~550-680 nm) and VIS≈full while NIR is weaker,",
     "  the model is reading FHB kernel discolouration/damage (a DON-correlated phenotype),",
     "  not a DON molecular absorption. Compare against Sci Rep 2024 (397-673 nm correlate with DON).",
     "", "Source: results/05_waveband_audit.md"]
os.makedirs(RES,exist_ok=True)
open(os.path.join(RES,"05_waveband_audit.md"),"w",encoding="utf-8").write("\n".join(md))

# figure
fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
ax[0].plot(nm,imp21,label="2021",color="#4C72B0"); ax[0].plot(nm,imp22,label="2022",color="#C44E52",alpha=.8)
ax[0].axvspan(400,700,color="gold",alpha=.08,label="VIS 400-700"); ax[0].axvspan(560,680,color="red",alpha=.08)
ax[0].set_ylabel("|EN coef| (sg2)"); ax[0].legend(fontsize=8); ax[0].set_title(f"Band importance — cross-year Spearman rho={rho:.2f}, top-20 Jaccard={jacc:.2f}")
ax[1].plot(nm,sel21,label="2021",color="#4C72B0"); ax[1].plot(nm,sel22,label="2022",color="#C44E52",alpha=.8)
ax[1].axvspan(560,680,color="red",alpha=.08)
ax[1].set_ylabel("bootstrap selection freq"); ax[1].set_xlabel("wavelength (nm)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES,"FIG_waveband_importance.png"),dpi=130)
# console-safe print (ascii)
print("rho=%.3f  jaccard=%.3f"%(rho,jacc))
for name,a,b in reg_rows: print("%-16s 2021=%.3f 2022=%.3f"%(name,a,b))
print("[written] results/05_waveband_audit.md + FIG_waveband_importance.png")

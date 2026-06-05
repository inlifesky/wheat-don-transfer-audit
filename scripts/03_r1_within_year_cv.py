"""
Phase 1 / R1 — within-year random nested CV (the optimistic baseline).
For each year x preprocessing x model: nested CV (outer RepeatedKFold 5x3, inner GridSearchCV 5)
report R2, RMSE, bias. Folds are genotype-disjoint by construction (one row per genotype per year).
Outputs: results/03_r1_within_year.tsv  +  results/03_r1_summary.md
"""
import os, sys, numpy as np, pandas as pd, warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold, GridSearchCV, KFold
from sklearn.base import clone
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import make_preprocessor, PREPROCESSINGS
warnings.filterwarnings("ignore")
try:                                   # UnicodeEncodeError guard for legacy (GBK) consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file is in scripts/)
DATA = os.environ.get("WHEAT_DON_DATA", os.path.join(_ROOT, "data"))
RES  = os.environ.get("WHEAT_DON_RESULTS", os.path.join(_ROOT, "results"))
SEED = 42
YEARS = {"2021":"2021_Phenomic_Data_VIS-NIR.csv", "2022":"2022_Phenomic_Data_VIS-NIR.csv"}

def load(fn):
    df = pd.read_csv(os.path.join(DATA, fn))
    wl = [c for c in df.columns if str(c).startswith("R")]
    return df[wl].values.astype(float), df["DON"].values.astype(float)

def build(prep, model):
    steps = make_preprocessor(prep)
    if model == "PLS":
        steps += [("model", PLSRegression())]
        grid = {"model__n_components": [2,4,6,8,10,12,15,20]}
    else:  # EN
        steps += [("scale", StandardScaler()), ("model", ElasticNet(max_iter=20000))]
        grid = {"model__alpha": np.logspace(-3,1,7), "model__l1_ratio": [0.2,0.5,0.8]}
    return Pipeline(steps), grid

def nested(X, y, prep, model):
    pipe, grid = build(prep, model)
    outer = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2s, rmses, biases, ncomp = [], [], [], []
    for tr, te in outer.split(X):
        gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="r2", n_jobs=-1)
        gs.fit(X[tr], y[tr])
        pred = np.asarray(gs.predict(X[te])).ravel()
        resid = y[te] - pred
        ss_res = np.sum(resid**2); ss_tot = np.sum((y[te]-y[te].mean())**2)
        r2s.append(1 - ss_res/ss_tot)
        rmses.append(np.sqrt(np.mean(resid**2)))
        biases.append(np.mean(pred - y[te]))
        ncomp.append(gs.best_params_.get("model__n_components", np.nan))
    return np.array(r2s), np.array(rmses), np.array(biases), ncomp

rows = []
for yr, fn in YEARS.items():
    X, y = load(fn)
    print(f"[{yr}] X={X.shape} DON n={len(y)}")
    for prep in PREPROCESSINGS:
        for model in ["PLS","EN"]:
            r2,rmse,bias,nc = nested(X,y,prep,model)
            rows.append(dict(regime="R1_within", year=yr, prep=prep, model=model,
                             R2_mean=r2.mean(), R2_sd=r2.std(), RMSE_mean=rmse.mean(),
                             bias_mean=bias.mean(),
                             ncomp_median=(np.nanmedian([c for c in nc if c==c]) if model=="PLS" else np.nan)))
            print(f"  {yr} {prep:8s} {model:3s}  R2={r2.mean():.3f}±{r2.std():.3f}  RMSE={rmse.mean():.2f}  bias={bias.mean():+.2f}")

res = pd.DataFrame(rows).sort_values(["year","model","R2_mean"], ascending=[True,True,False])
os.makedirs(RES, exist_ok=True)
res.to_csv(os.path.join(RES,"03_r1_within_year.tsv"), sep="\t", index=False, float_format="%.4f")

# markdown summary: best prep per (year, model)
md = ["# R1 — within-year random nested CV", "",
      "Outer RepeatedKFold(5x3), inner GridSearchCV(5). DON ppm. Genotype-disjoint folds.", ""]
md += ["| year | model | best prep | R2 | RMSE(ppm) | bias | PLS ncomp |",
       "|---|---|---|---|---|---|---|"]
for (yr,model),g in res.groupby(["year","model"]):
    b = g.iloc[0]
    md.append(f"| {yr} | {model} | **{b['prep']}** | {b.R2_mean:.3f}±{b.R2_sd:.3f} | {b.RMSE_mean:.2f} | {b.bias_mean:+.2f} | {b.ncomp_median if model=='PLS' else '-'} |")
md += ["", "Full grid: results/03_r1_within_year.tsv"]
open(os.path.join(RES,"03_r1_summary.md"),"w",encoding="utf-8").write("\n".join(md))
print("\n".join(md))
print("\n[written] results/03_r1_within_year.tsv + 03_r1_summary.md")

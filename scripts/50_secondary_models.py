"""
Secondary-model robustness check (Paper 1 review response): back the Methods claim that
tree/kernel learners overfit the p>>n collinear spectra and transfer no better than the
PLSR/EN primaries. Random forest and SVR (RBF) under R1 within-year nested CV and R2 cross-year
transfer, SG2 preprocessing (R1-best), genotype-disjoint, inner-tuned on training only.
Outputs: results/50_secondary_models.tsv + .md
"""
import os, sys, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, GridSearchCV, KFold
from sklearn.base import clone
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
SEED = 42; PREP = "sg2"
F = {"2021": "2021_Phenomic_Data_VIS-NIR.csv", "2022": "2022_Phenomic_Data_VIS-NIR.csv"}


def load(fn):
    df = pd.read_csv(os.path.join(DATA, fn))
    wl = [c for c in df.columns if str(c).startswith("R")]
    return df[wl].values.astype(float), df["DON"].values.astype(float)


def build(model):
    steps = make_preprocessor(PREP) + [("scale", StandardScaler())]
    if model == "RF":
        steps += [("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))]
        grid = {"model__n_estimators": [400], "model__max_depth": [None, 8], "model__max_features": ["sqrt", 0.3]}
    else:  # SVR
        steps += [("model", SVR(kernel="rbf"))]
        grid = {"model__C": [1, 10, 100], "model__gamma": ["scale", 0.01], "model__epsilon": [0.5, 1.0]}
    return Pipeline(steps), grid


def regm(y, p):
    resid = y - p; r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    return r2, np.sqrt(np.mean(resid**2)), np.mean(p - y), (pearsonr(y, p)[0] if np.std(p) > 0 else np.nan)


X21, y21 = load(F["2021"]); X22, y22 = load(F["2022"])
rows = []

# R1 within-year nested CV
for yr, (X, y) in {"2021": (X21, y21), "2022": (X22, y22)}.items():
    for model in ["RF", "SVR"]:
        pipe, grid = build(model)
        outer = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
        inner = KFold(5, shuffle=True, random_state=SEED)
        r2s = []
        for tr, te in outer.split(X):
            gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="r2", n_jobs=-1)
            gs.fit(X[tr], y[tr]); p = np.asarray(gs.predict(X[te])).ravel()
            r2s.append(regm(y[te], p)[0])
        rows.append(dict(regime="R1_within", train=yr, test=yr, model=model,
                         R2=np.mean(r2s), R2_sd=np.std(r2s), RMSE=np.nan, bias=np.nan, pearson_r=np.nan))

# R2 cross-year transfer
for (trn, tst, Xtr, ytr, Xte, yte) in [("2021", "2022", X21, y21, X22, y22), ("2022", "2021", X22, y22, X21, y21)]:
    for model in ["RF", "SVR"]:
        pipe, grid = build(model)
        gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2", n_jobs=-1)
        gs.fit(Xtr, ytr); p = np.asarray(gs.predict(Xte)).ravel()
        r2, rmse, bias, r = regm(yte, p)
        rows.append(dict(regime="R2_cross", train=trn, test=tst, model=model,
                         R2=r2, R2_sd=np.nan, RMSE=rmse, bias=bias, pearson_r=r))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "50_secondary_models.tsv"), sep="\t", index=False, float_format="%.4f")
md = ["# Secondary models (RF, SVR) — robustness check vs PLSR/EN primaries", "",
      "SG2 preprocessing; genotype-disjoint; inner-tuned on training only.", "",
      "## R1 within-year (nested CV R2) — compare to EN sg2 0.667/0.517",
      "| year | model | R2 (sd) |", "|---|---|---|"]
for _, x in res[res.regime == "R1_within"].iterrows():
    md.append(f"| {x.train} | {x.model} | {x.R2:.3f} ({x.R2_sd:.3f}) |")
md += ["", "## R2 cross-year transfer (R2) — compare to EN sg2 -2.73 / -2.15",
       "| train→test | model | R2 | RMSE | bias | r |", "|---|---|---|---|---|---|"]
for _, x in res[res.regime == "R2_cross"].iterrows():
    md.append(f"| {x.train}→{x.test} | {x.model} | {x.R2:.3f} | {x.RMSE:.2f} | {x.bias:+.2f} | {x.pearson_r:.3f} |")
md += ["", "Source: results/50_secondary_models.tsv"]
open(os.path.join(RES, "50_secondary_models.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md)); print("\n[written] results/50_secondary_models.{tsv,md}")

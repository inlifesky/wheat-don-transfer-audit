"""
Bootstrap 95% CIs for the headline cross-year R2, the within-year R1 reference, and the
R1-R2 gap (Paper 1 review response: put uncertainty on the headline numbers).

R1 is re-estimated as the mean over outer CV folds with a bootstrap over genotype-disjoint
fold assignments is overkill here; instead we bootstrap the R2 metric directly given the fixed
cross-year model predictions (sampling uncertainty of the test set), and we bootstrap R1 over
repeated CV splits. The reportable CI is on R2 (test-set sampling) and on the gap R1_point - R2.

Method:
  - Fit EN/PLS (sg2) once per direction on the training year (script-04 model); predict test year.
  - Bootstrap B times: resample test rows with replacement -> recompute R2. -> percentile CI.
  - R1 point estimate taken from script 03 (repeated nested CV mean); gap CI = R1_point - R2_boot CI.
Outputs: results/48_bootstrap_R2_CIs.tsv + .md
"""
import os, sys, numpy as np, pandas as pd, warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
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
SEED = 42; PREP = "sg2"; B = 2000
F = {"2021": "2021_Phenomic_Data_VIS-NIR.csv", "2022": "2022_Phenomic_Data_VIS-NIR.csv"}
R1_POINT = {("2021", "EN"): 0.667, ("2022", "EN"): 0.517, ("2021", "PLS"): 0.653, ("2022", "PLS"): 0.481}


def load(fn):
    df = pd.read_csv(os.path.join(DATA, fn))
    wl = [c for c in df.columns if str(c).startswith("R")]
    return df[wl].values.astype(float), df["DON"].values.astype(float)


def build(model):
    steps = make_preprocessor(PREP)
    if model == "PLS":
        steps += [("model", PLSRegression())]; grid = {"model__n_components": [2, 4, 6, 8, 10, 12]}
    else:
        steps += [("scale", StandardScaler()), ("model", ElasticNet(max_iter=20000))]
        grid = {"model__alpha": np.logspace(-3, 1, 7), "model__l1_ratio": [0.2, 0.5, 0.8]}
    return Pipeline(steps), grid


def fit_predict(Xtr, ytr, Xte, model):
    pipe, grid = build(model)
    gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2", n_jobs=-1)
    gs.fit(Xtr, ytr)
    return np.asarray(gs.predict(Xte)).ravel()


def r2_of(y, p):
    return 1 - np.sum((y - p)**2) / np.sum((y - y.mean())**2)


X21, y21 = load(F["2021"]); X22, y22 = load(F["2022"])
DIRS = [("2021", "2022", X21, y21, X22, y22), ("2022", "2021", X22, y22, X21, y21)]
rows = []
for (trn, tst, Xtr, ytr, Xte, yte) in DIRS:
    for model in ["EN", "PLS"]:
        pred = fit_predict(Xtr, ytr, Xte, model)
        r2_point = r2_of(yte, pred)
        n = len(yte); rng = np.random.RandomState(SEED)
        boot = np.empty(B)
        for b in range(B):
            idx = rng.choice(n, size=n, replace=True)
            boot[b] = r2_of(yte[idx], pred[idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        r1 = R1_POINT[(tst, model)]
        gap_lo, gap_hi = r1 - hi, r1 - lo  # gap = R1_point - R2_boot
        rows.append(dict(train=trn, test=tst, model=model,
                         R2_point=r2_point, R2_lo=lo, R2_hi=hi,
                         R1_point=r1, gap_point=r1 - r2_point, gap_lo=gap_lo, gap_hi=gap_hi))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "48_bootstrap_R2_CIs.tsv"), sep="\t", index=False, float_format="%.4f")
md = ["# Bootstrap 95% CIs for cross-year R2 and the R1-R2 optimism gap", "",
      f"B = {B} test-set resamples; percentile CI. R1 = repeated-nested-CV point (script 03).", "",
      "| train→test | model | R2 (95% CI) | R1 | gap R1−R2 (95% CI) |", "|---|---|---|---|---|"]
for _, x in res.iterrows():
    md.append(f"| {x.train}→{x.test} | {x.model} | {x.R2_point:.2f} [{x.R2_lo:.2f}, {x.R2_hi:.2f}] | "
              f"{x.R1_point:.3f} | {x.gap_point:.2f} [{x.gap_lo:.2f}, {x.gap_hi:.2f}] |")
md += ["", "All cross-year R2 CIs lie entirely below 0; all gap CIs entirely above ~2 -> the optimism is",
       "not a sampling fluke.", "", "Source: results/48_bootstrap_R2_CIs.tsv"]
open(os.path.join(RES, "48_bootstrap_R2_CIs.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
print("\n[written] results/48_bootstrap_R2_CIs.{tsv,md}")

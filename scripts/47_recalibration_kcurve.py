"""
Phase 2 / per-year recalibration k-curve (Paper 1, review response to ChatGPT point on §4.3).

Question: cross-year transfer R2 is strongly negative because of a ~15-20 ppm offset.
If a lab spends k reference GC/MS assays on the NEW year, can a light recalibration
(bias-only or full linear) restore usable prediction? And how large must k be?

Design (leakage-safe):
  - Fit EN/PLS (sg2) ONCE on the full training year, tuned by inner CV on the training year
    only -> exactly the script-04 model. Predict the whole test year -> (yte, pred).
  - For each k in {5,10,20} and each of N_SEEDS random draws:
      * draw k "reference" samples from the test year (the spent assays),
      * fit recalibration on those k:
          bias_only : pred' = pred + mean(yref - pred_ref)              (1 param)
          linear    : pred' = a + b*pred   from OLS(yref ~ pred_ref)    (2 params)
      * evaluate on the REMAINING test samples (anchors excluded -> no leakage):
          R2, RMSE, bias, Pearson r, screening ROC-AUC + sens/spec @ top-tertile.
  - Aggregate over seeds: mean and sd. Compare to no-recalibration baseline.

Note: affine recalibration is monotonic -> ROC-AUC is invariant by construction; we report it
to confirm the recovery is offset-correction, not new discrimination.

Outputs: results/47_recalibration_kcurve.tsv (raw per-seed)
         results/47_recalibration_agg.tsv   (aggregated)
         results/47_recalibration.md
         results/FIG_recalibration_kcurve.png
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
SEED = 42; PREP = "sg2"
KS = [5, 10, 20]; N_SEEDS = 200
F = {"2021": "2021_Phenomic_Data_VIS-NIR.csv", "2022": "2022_Phenomic_Data_VIS-NIR.csv"}


def load(fn):
    df = pd.read_csv(os.path.join(DATA, fn))
    wl = [c for c in df.columns if str(c).startswith("R")]
    return df["Genotype"].astype(str).values, df[wl].values.astype(float), df["DON"].values.astype(float)


def build(model):
    steps = make_preprocessor(PREP)
    if model == "PLS":
        steps += [("model", PLSRegression())]
        grid = {"model__n_components": [2, 4, 6, 8, 10, 12]}
    else:
        steps += [("scale", StandardScaler()), ("model", ElasticNet(max_iter=20000))]
        grid = {"model__alpha": np.logspace(-3, 1, 7), "model__l1_ratio": [0.2, 0.5, 0.8]}
    return Pipeline(steps), grid


def fit_predict(Xtr, ytr, Xte, model):
    pipe, grid = build(model)
    gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2", n_jobs=-1)
    gs.fit(Xtr, ytr)
    return np.asarray(gs.predict(Xte)).ravel()


def reg_metrics(y, pred):
    resid = y - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    rmse = np.sqrt(np.mean(resid**2)); bias = np.mean(pred - y)
    r = pearsonr(y, pred)[0] if len(y) > 2 and np.std(pred) > 0 else np.nan
    return r2, rmse, bias, r


def screening(y, score):
    # truth = top tertile of THIS sample set; score = (recalibrated) predicted DON
    thr = np.quantile(y, 2/3); truth = (y > thr).astype(int)
    if truth.sum() == 0 or truth.sum() == len(truth):
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(truth, score)
    pthr = np.quantile(score, 2/3); flag = (score > pthr).astype(int)
    tp = int(((flag == 1) & (truth == 1)).sum()); fn = int(((flag == 0) & (truth == 1)).sum())
    tn = int(((flag == 0) & (truth == 0)).sum()); fp = int(((flag == 1) & (truth == 0)).sum())
    sens = tp/(tp+fn) if tp+fn else np.nan; spec = tn/(tn+fp) if tn+fp else np.nan
    return auc, sens, spec


g21, X21, y21 = load(F["2021"]); g22, X22, y22 = load(F["2022"])
DIRS = [("2021", "2022", X21, y21, X22, y22), ("2022", "2021", X22, y22, X21, y21)]

raw_rows = []; base_rows = []
for (trn, tst, Xtr, ytr, Xte, yte) in DIRS:
    for model in ["EN", "PLS"]:
        pred = fit_predict(Xtr, ytr, Xte, model)
        # no-recalibration baseline on the FULL test year
        b_r2, b_rmse, b_bias, b_r = reg_metrics(yte, pred)
        b_auc, b_sens, b_spec = screening(yte, pred)
        base_rows.append(dict(train=trn, test=tst, model=model, method="none(baseline)", k=0,
                              R2=b_r2, RMSE=b_rmse, bias=b_bias, pearson_r=b_r,
                              auc=b_auc, sens=b_sens, spec=b_spec))
        n = len(yte)
        for k in KS:
            for s in range(N_SEEDS):
                rng = np.random.RandomState(1000 * k + s)  # deterministic, varies by k and seed
                anchor = rng.choice(n, size=k, replace=False)
                rest = np.setdiff1d(np.arange(n), anchor)
                pr_a, y_a = pred[anchor], yte[anchor]
                pr_r, y_r = pred[rest], yte[rest]
                # --- bias-only recalibration (1 param) ---
                off = np.mean(y_a - pr_a)
                rec_b = pr_r + off
                # --- full linear recalibration (2 params): y ~ a + b*pred ---
                if np.std(pr_a) > 1e-9 and k >= 2:
                    b1, b0 = np.polyfit(pr_a, y_a, 1)   # y = b1*pred + b0
                    rec_l = b0 + b1 * pr_r
                else:
                    rec_l = rec_b.copy()
                for method, rec in [("bias_only", rec_b), ("linear", rec_l)]:
                    r2, rmse, bias, r = reg_metrics(y_r, rec)
                    auc, sens, spec = screening(y_r, rec)
                    raw_rows.append(dict(train=trn, test=tst, model=model, method=method, k=k, seed=s,
                                         R2=r2, RMSE=rmse, bias=bias, pearson_r=r,
                                         auc=auc, sens=sens, spec=spec))

raw = pd.DataFrame(raw_rows); base = pd.DataFrame(base_rows)
os.makedirs(RES, exist_ok=True)
raw.to_csv(os.path.join(RES, "47_recalibration_kcurve.tsv"), sep="\t", index=False, float_format="%.4f")

agg = (raw.groupby(["train", "test", "model", "method", "k"])
          .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
               RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
               bias_mean=("bias", "mean"), bias_sd=("bias", "std"),
               r_mean=("pearson_r", "mean"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
               sens_mean=("sens", "mean"), spec_mean=("spec", "mean"))
          .reset_index())
agg.to_csv(os.path.join(RES, "47_recalibration_agg.tsv"), sep="\t", index=False, float_format="%.4f")

# ---- markdown summary ----
md = ["# Per-year recalibration k-curve (Paper 1 §4.3 review response)", "",
      f"Preprocessing = {PREP}. Model fit ONCE on training year (script-04 model); recalibration fit on",
      f"k random reference samples from the TEST year; metrics on the held-out remainder (anchors excluded).",
      f"N_SEEDS = {N_SEEDS} draws per k. Affine recalibration is monotonic => ROC-AUC invariant by construction.", "",
      "## No-recalibration baseline (full test year)",
      "| train→test | model | R2 | RMSE | bias | Pearson r | AUC |", "|---|---|---|---|---|---|---|"]
for _, x in base.iterrows():
    md.append(f"| {x.train}→{x.test} | {x.model} | {x.R2:.3f} | {x.RMSE:.2f} | {x.bias:+.2f} | {x.pearson_r:.3f} | {x.auc:.3f} |")
md += ["", "## After recalibration (mean over draws; held-out remainder)",
       "| train→test | model | method | k | R2 (sd) | RMSE | bias | AUC |", "|---|---|---|---|---|---|---|---|"]
for _, x in agg.iterrows():
    md.append(f"| {x.train}→{x.test} | {x.model} | {x.method} | {int(x.k)} | "
              f"{x.R2_mean:.3f} ({x.R2_sd:.3f}) | {x.RMSE_mean:.2f} | {x.bias_mean:+.2f} | {x.auc_mean:.3f} |")
md += ["", "## Read",
       "- If R2 flips from strongly negative to positive after a 1-parameter bias correction, the cross-year",
       "  failure is dominated by a removable offset (batch/session and/or sample-condition), not lost information.",
       "- The recalibrated R2 ceiling = cross-year r^2 (linear recal), so a residual gap vs within-year R2 means",
       "  the within-cloud information is genuinely degraded (cf. unstable cross-year bands) -> recalibration is",
       "  NECESSARY but NOT SUFFICIENT.",
       "- AUC unchanged across k/method confirms ranking was already the surviving part; recalibration buys",
       "  calibrated ppm values, not new discrimination.", "",
       "Full per-seed: results/47_recalibration_kcurve.tsv ; aggregated: results/47_recalibration_agg.tsv"]
open(os.path.join(RES, "47_recalibration.md"), "w", encoding="utf-8").write("\n".join(md))

# ---- figure: R2 vs k, per direction, bias_only vs linear, EN ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (trn, tst) in zip(axes, [("2021", "2022"), ("2022", "2021")]):
    base_r2 = base[(base.train == trn) & (base.model == "EN")].R2.values[0]
    ax.axhline(base_r2, ls=":", color="#999", lw=1.4, label=f"no recal (R²={base_r2:.2f})")
    ax.axhline(0, ls="-", color="k", lw=0.6)
    for method, col in [("bias_only", "#4C72B0"), ("linear", "#C44E52")]:
        sub = agg[(agg.train == trn) & (agg.test == tst) & (agg.model == "EN") & (agg.method == method)].sort_values("k")
        ax.errorbar(sub.k, sub.R2_mean, yerr=sub.R2_sd, marker="o", capsize=3, color=col, label=method)
    ax.set_title(f"train {trn} → test {tst}  (EN, sg2)")
    ax.set_xlabel("k reference samples from test year"); ax.set_ylabel("recalibrated R² (held-out remainder)")
    ax.set_xticks(KS); ax.legend(fontsize=8)
fig.suptitle("Per-year recalibration recovers calibration from a few reference assays", y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(RES, "FIG_recalibration_kcurve.png"), dpi=130, bbox_inches="tight")

print("\n".join(md))
print("\n[written] results/47_recalibration.{md,tsv,agg.tsv} + FIG_recalibration_kcurve.png")

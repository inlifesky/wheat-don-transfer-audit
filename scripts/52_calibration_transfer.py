"""
Phase 2 / Paper §3.7 — standard calibration-transfer methods (PDS, EPO, GLSW)
under the locked R2 protocol, using the 34 cross-year overlap genotypes as
surrogate paired standards. Compared to the §3.6 prediction-space corrections
(bias-only, linear/slope+intercept) computed deterministically from the same 34.

Design (Option A, per Wei Yuan 2026-06-03):
  - Train EN(sg2) on the FULL training year (298 or 285) — same as §3.2.
  - Identify 34 cross-year overlap genotypes; use as paired surrogate standards.
  - Evaluate ALL methods on the non-overlap test holdout (n_test - 34) so all
    rows are directly comparable.
  - PDS: per-band local ridge map with window sweep [0, 1, 3, 5, 7, 10].
  - EPO: paired-difference SVD, project orthogonal complement; K sweep [0, 1, 2, 3, 5, 8].
  - GLSW: paired-difference clutter covariance, downweighted filter; alpha sweep.
  - bias-only: pred' = pred - mean(pred_anchor - y_anchor).
  - linear:    pred' = a + b * pred, OLS on the 34.
Outputs: results/52_calibration_transfer.{tsv,md} + results/FIG_calibration_transfer.png
"""
import os, sys, warnings, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
warnings.filterwarnings("ignore")
try:                                   # UnicodeEncodeError guard for legacy (GBK) consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file is in scripts/)
DATA = os.environ.get("WHEAT_DON_DATA", os.path.join(_ROOT, "data"))
RES  = os.environ.get("WHEAT_DON_RESULTS", os.path.join(_ROOT, "results"))
SEED = 42
F = {"2021": "2021_Phenomic_Data_VIS-NIR.csv", "2022": "2022_Phenomic_Data_VIS-NIR.csv"}
PDS_WINDOWS = [0, 1, 3, 5, 7, 10]
EPO_KS      = [0, 1, 2, 3, 5, 8]
GLSW_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3]


# ─── data load + sg2 preprocessing (per-row, no leakage) ──────────────────
def load(fn):
    df = pd.read_csv(os.path.join(DATA, fn))
    wl = [c for c in df.columns if str(c).startswith("R")]
    return (df["Genotype"].astype(str).values,
            df[wl].values.astype(float),
            df["DON"].values.astype(float),
            np.array([float(c[1:]) for c in wl]))


def train_en_on_sg2(X_sg2, y):
    """Train StandardScaler + EN on pre-sg2 spectra (consistent with §3.2 grid choice)."""
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", ElasticNet(max_iter=20000))])
    gs = GridSearchCV(pipe,
                      {"model__alpha": np.logspace(-3, 1, 7),
                       "model__l1_ratio": [0.2, 0.5, 0.8]},
                      cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="r2", n_jobs=-1).fit(X_sg2, y)
    return gs


def metrics(y, pred):
    if len(y) < 3 or np.std(pred) == 0:
        return dict(R2=np.nan, RMSE=np.nan, bias=np.nan, pearson_r=np.nan, AUC=np.nan)
    resid = y - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    rmse = np.sqrt(np.mean(resid**2))
    bias = float(np.mean(pred - y))
    r = pearsonr(y, pred)[0]
    truth = (y > np.quantile(y, 2/3)).astype(int)
    auc = roc_auc_score(truth, pred) if 0 < truth.sum() < len(truth) else np.nan
    return dict(R2=r2, RMSE=rmse, bias=bias, pearson_r=r, AUC=auc)


# ─── PDS, EPO, GLSW core ──────────────────────────────────────────────────
def build_pds_map(X_target_ov, X_source_ov, window):
    """For each target band j, regress X_target[:,j] on X_source[:, j-w : j+w+1] (ridge)."""
    n, p = X_target_ov.shape
    M = np.zeros((p, p))
    for j in range(p):
        lo, hi = max(0, j - window), min(p, j + window + 1)
        Xw = X_source_ov[:, lo:hi]
        r = Ridge(alpha=1e-3, fit_intercept=False).fit(Xw, X_target_ov[:, j])
        M[lo:hi, j] = r.coef_
    return M


def epo_projection(X_a_ov, X_b_ov, K):
    """Return P = I - W.T@W where W = top-K right SVs of (X_b_ov - X_a_ov)."""
    D = X_b_ov - X_a_ov
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    W = Vt[:max(K, 0)]
    P = np.eye(X_a_ov.shape[1]) - W.T @ W
    return P


def glsw_filter(X_a_ov, X_b_ov, alpha):
    """Build W s.t. X@W downweights paired-difference covariance directions."""
    D = X_b_ov - X_a_ov
    Sigma = (D.T @ D) / max(len(D) - 1, 1)
    U, s, _ = np.linalg.svd(Sigma)
    s_filt = 1.0 / np.sqrt(1.0 + s / alpha)
    return U @ np.diag(s_filt) @ U.T


# ─── load ─────────────────────────────────────────────────────────────────
g21, X21_raw, y21, nm = load(F["2021"])
g22, X22_raw, y22, _  = load(F["2022"])
X21_sg, X22_sg = _sg(X21_raw, 2), _sg(X22_raw, 2)
overlap = sorted(set(g21) & set(g22))
idx21 = {g: i for i, g in enumerate(g21)}; idx22 = {g: i for i, g in enumerate(g22)}
ov21 = np.array([idx21[g] for g in overlap])
ov22 = np.array([idx22[g] for g in overlap])
X21_ov_sg, X22_ov_sg = X21_sg[ov21], X22_sg[ov22]
y21_ov, y22_ov = y21[ov21], y22[ov22]
mask21_holdout = ~np.isin(g21, overlap)   # use as test when train=2022
mask22_holdout = ~np.isin(g22, overlap)   # use as test when train=2021
print(f"[data] 2021 n={len(g21)} (holdout n={mask21_holdout.sum()}); 2022 n={len(g22)} (holdout n={mask22_holdout.sum()}); overlap=34")


# ─── train one EN on each FULL training year (matches §3.2) ──────────────
gs21 = train_en_on_sg2(X21_sg, y21)   # trained on full 2021 (n=298)
gs22 = train_en_on_sg2(X22_sg, y22)   # trained on full 2022 (n=285)


def eval_baseline_on_holdout(model, X_test_sg, y_test, mask_test):
    pred = np.asarray(model.predict(X_test_sg[mask_test])).ravel()
    return metrics(y_test[mask_test], pred), pred


# ─── run all methods, both directions, on the (test - 34) holdout ────────
rows = []
DIRS = [("2021->2022", gs21, X21_sg, X22_sg, X21_ov_sg, X22_ov_sg,
        y22, mask22_holdout, ov22, y22_ov),
        ("2022->2021", gs22, X22_sg, X21_sg, X22_ov_sg, X21_ov_sg,
        y21, mask21_holdout, ov21, y21_ov)]

for (label, gs_train, X_tr_sg, X_te_sg, X_tr_ov, X_te_ov,
     y_te_full, mask_te, ov_te, y_te_ov) in DIRS:

    # baseline on full test year (for context, matches §3.2 numbers)
    pred_full = np.asarray(gs_train.predict(X_te_sg)).ravel()
    m_full = metrics(y_te_full, pred_full)
    m_full.update(direction=label, method="no_correction", scope="full_test", param=np.nan)
    rows.append(m_full)

    # baseline on (test - 34) holdout — apples-to-apples scope for all transfer methods
    pred_hold = pred_full[mask_te]
    y_hold    = y_te_full[mask_te]
    m_base = metrics(y_hold, pred_hold)
    m_base.update(direction=label, method="no_correction", scope="holdout_minus_overlap", param=np.nan)
    rows.append(m_base)

    # anchor predictions on test-year overlap (deterministic, all 34)
    pred_anchor = np.asarray(gs_train.predict(X_te_ov)).ravel()
    y_anchor    = y_te_ov

    # bias-only (1 parameter) — deterministic from 34
    off = float(np.mean(pred_anchor - y_anchor))
    pred_bias = pred_hold - off
    mb = metrics(y_hold, pred_bias)
    mb.update(direction=label, method="bias_only_k34", scope="holdout_minus_overlap", param=np.nan)
    rows.append(mb)

    # linear / slope+intercept (2 parameters) — deterministic from 34
    if np.std(pred_anchor) > 1e-9:
        b1, b0 = np.polyfit(pred_anchor, y_anchor, 1)
        pred_lin = b0 + b1 * pred_hold
    else:
        pred_lin = pred_bias.copy()
    ml = metrics(y_hold, pred_lin)
    ml.update(direction=label, method="linear_k34", scope="holdout_minus_overlap", param=np.nan)
    rows.append(ml)

    # PDS window sweep — map test-year spectra to look like train-year
    # We need F such that  X_te_translated  ≈  X_tr-like ; standards are pairs (X_tr_ov, X_te_ov)
    # build_pds_map(target, source) regresses target on source → so target=X_tr_ov, source=X_te_ov
    for w in PDS_WINDOWS:
        M = build_pds_map(X_target_ov=X_tr_ov, X_source_ov=X_te_ov, window=w)
        X_te_translated = X_te_sg @ M
        pred_pds = np.asarray(gs_train.predict(X_te_translated[mask_te])).ravel()
        mp = metrics(y_hold, pred_pds)
        mp.update(direction=label, method="PDS", scope="holdout_minus_overlap", param=float(w))
        rows.append(mp)

    # EPO K sweep — orthogonalize BOTH years against year-perturbation subspace, then retrain
    for K in EPO_KS:
        P = epo_projection(X_a_ov=X_tr_ov, X_b_ov=X_te_ov, K=K)
        X_tr_clean = X_tr_sg @ P
        X_te_clean = X_te_sg @ P
        gs_clean = train_en_on_sg2(X_tr_clean, y21 if label.startswith("2021") else y22)
        pred_epo = np.asarray(gs_clean.predict(X_te_clean[mask_te])).ravel()
        me = metrics(y_hold, pred_epo)
        me.update(direction=label, method="EPO", scope="holdout_minus_overlap", param=float(K))
        rows.append(me)

    # GLSW alpha sweep — downweight clutter directions, then retrain
    for alpha in GLSW_ALPHAS:
        W = glsw_filter(X_tr_ov, X_te_ov, alpha)
        X_tr_filt = X_tr_sg @ W
        X_te_filt = X_te_sg @ W
        gs_filt = train_en_on_sg2(X_tr_filt, y21 if label.startswith("2021") else y22)
        pred_g = np.asarray(gs_filt.predict(X_te_filt[mask_te])).ravel()
        mg = metrics(y_hold, pred_g)
        mg.update(direction=label, method="GLSW", scope="holdout_minus_overlap", param=float(alpha))
        rows.append(mg)


res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "52_calibration_transfer.tsv"), sep="\t", index=False, float_format="%.4f")


# ─── markdown summary ────────────────────────────────────────────────────
def best_of(df, method):
    """Best row (highest R²) of a swept method, per direction."""
    return df[df.method == method].sort_values("R2", ascending=False).groupby("direction").head(1)

md = ["# §3.7 Calibration-transfer comparison (PDS / EPO / GLSW vs prediction-space recalibration)", "",
      "All methods use the 34 cross-year overlap genotypes as paired surrogate standards (Option A, deterministic).",
      "Training: EN(sg2) on full training year (n=298/285), identical to §3.2.",
      "Evaluation: the non-overlap test holdout (n_test - 34) — same set for all methods so rows are directly comparable.", "",
      "## Baselines (no correction)",
      "| direction | scope | R2 | RMSE | bias | r | AUC |",
      "|---|---|---|---|---|---|---|"]
for _, x in res[res.method == "no_correction"].iterrows():
    md.append(f"| {x.direction} | {x.scope} | {x.R2:+.3f} | {x.RMSE:.2f} | {x.bias:+.2f} | {x.pearson_r:.3f} | {x.AUC:.3f} |")

md += ["", "## Headline comparison on the (test - 34) holdout",
       "| direction | method | param | R2 | RMSE | bias | r | AUC | note |",
       "|---|---|---|---|---|---|---|---|---|"]
keep = res[(res.scope == "holdout_minus_overlap") & (res.method != "no_correction")].copy()
# annotate with "best" tag for PDS / EPO / GLSW best param
best_tags = {}
for mth in ["PDS", "EPO", "GLSW"]:
    best_tags[mth] = set((r.direction, r.param) for _, r in best_of(res, mth).iterrows())
for _, x in keep.iterrows():
    note = ""
    if x.method in best_tags and (x.direction, x.param) in best_tags[x.method]:
        note = "← best param"
    p = "—" if pd.isna(x.param) else f"{x.param:g}"
    md.append(f"| {x.direction} | {x.method} | {p} | {x.R2:+.3f} | {x.RMSE:.2f} | {x.bias:+.2f} | {x.pearson_r:.3f} | {x.AUC:.3f} | {note} |")

md += ["", "## Read",
       "- bias-only (1 parameter) and linear/slope+intercept (2 parameters) using the same 34 standards",
       "  recover R2 from strongly negative to ~0 — they do not restore within-year accuracy but they do remove the offset.",
       "- PDS (per-band local map), EPO (orthogonal projection), and GLSW (clutter downweighting) DO NOT help and several",
       "  destroy ranking — Pearson r can go negative under PDS, and AUC drops below the no-correction baseline.",
       "- Mechanistic reading: spectral-level transfer methods presume a structured spectral perturbation (instrument drift,",
       "  illumination shift, multiplicative scatter) that can be learned from paired standards. The cross-year difference",
       "  in this dataset evidently is NOT such a perturbation — the 34 'same-cultivar' pairs are field-grown samples whose",
       "  spectra differ for real biological reasons (FHB severity, kernel discolouration), so transfer maps fit on those",
       "  pairs alias real signal into 'clutter' and remove it. This is independent evidence for the §4.2 FHB-colour proxy",
       "  interpretation: if the model were tracking a DON-specific molecular feature, paired-standard transfer would correct",
       "  a cross-year drift; the failure of every spectral-level method here means the difference is in the kernel state itself.", "",
       "Source: results/52_calibration_transfer.tsv"]
open(os.path.join(RES, "52_calibration_transfer.md"), "w", encoding="utf-8").write("\n".join(md))


# ─── figure: best-of-method R^2 comparison, both directions ──────────────
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
ORDER = [("no_correction (holdout)", "no_correction_holdout"),
         ("bias-only (k=34)", "bias_only_k34"),
         ("linear (k=34)", "linear_k34"),
         ("PDS (best w)", "PDS_best"),
         ("EPO (best K)", "EPO_best"),
         ("GLSW (best α)", "GLSW_best")]
COLORS = ["#7f7f7f", "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

for ax, dir_ in zip(axes, ["2021->2022", "2022->2021"]):
    vals = []
    for _, key in ORDER:
        if key == "no_correction_holdout":
            v = res[(res.direction == dir_) & (res.method == "no_correction") & (res.scope == "holdout_minus_overlap")].R2.iloc[0]
        elif key.endswith("_best"):
            mth = key.replace("_best", "")
            v = best_of(res[res.direction == dir_], mth).R2.iloc[0]
        else:
            v = res[(res.direction == dir_) & (res.method == key)].R2.iloc[0]
        vals.append(v)
    bars = ax.bar(range(len(ORDER)), vals, color=COLORS, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([lbl for lbl, _ in ORDER], rotation=30, ha="right", fontsize=9)
    ax.set_title(f"train {dir_.split('->')[0]} → test {dir_.split('->')[1]}  (EN, sg2; evaluated on test − 34)")
    ax.set_ylabel("cross-year R²" if dir_.startswith("2021") else "")
    # value labels above/below bars
    for b, v in zip(bars, vals):
        offset = 0.06 if v >= 0 else -0.06
        ax.text(b.get_x() + b.get_width()/2, v + offset, f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

fig.suptitle("Spectral-level calibration transfer fails; only prediction-space recalibration helps", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(RES, "FIG_calibration_transfer.png"), dpi=130, bbox_inches="tight")

print("\n".join(md))
print("\n[written] results/52_calibration_transfer.{tsv,md} + FIG_calibration_transfer.png")

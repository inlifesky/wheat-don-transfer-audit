"""
Phase 0 — data audit for project_wheat_DON_2A.
Verifies (does NOT assume) the structure of the 3 Dryad VIS-NIR CSVs:
  - true row/col counts (resolve 298/307 & 285/288 & overlap discrepancies)
  - whether a DON target column exists IN these files (make-or-break for 2A)
  - column naming / waveband count / wavelength range
  - genotype ID column + cross-year overlap (the year/genotype confound)
  - missing values
Outputs a markdown report to results/01_data_audit.md
"""
import pandas as pd, numpy as np, os, re, io, sys
try:                                   # avoid UnicodeEncodeError on legacy (GBK) consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Resolve data/ and results/ relative to the repository root (this file lives in scripts/),
# overridable by env vars, so the pipeline runs on any machine after a Dryad download.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("WHEAT_DON_DATA", os.path.join(_ROOT, "data"))
OUT  = os.path.join(os.environ.get("WHEAT_DON_RESULTS", os.path.join(_ROOT, "results")), "01_data_audit.md")
FILES = {
    "2021":   "2021_Phenomic_Data_VIS-NIR.csv",
    "2022":   "2022_Phenomic_Data_VIS-NIR.csv",
    "across": "2021-2022_Across_Years_Phenomic_Data_VIS-NIR.csv",
}

DON_PAT = re.compile(r"don|deoxy|vomitoxin|mycotox|ppm|trait|pheno", re.I)

# A waveband header looks like "R397.32" / "397.32" / "X397.32nm": an optional leading
# band-prefix letter (R/X), a number, and an optional "nm" suffix. The earlier check stripped
# only "X"/"nm", so "R397.32" failed float() and was misfiled as a meta column.
WL_RE = re.compile(r"^[XR]?\d+(?:\.\d+)?(?:nm)?$", re.I)
def wl_value(c):
    m = re.search(r"\d+(?:\.\d+)?", str(c))
    return float(m.group()) if m else None

lines = ["# Phase 0 — Data audit (project_wheat_DON_2A)", ""]
summary = {}

for tag, fn in FILES.items():
    path = os.path.join(DATA, fn)
    df = pd.read_csv(path)
    cols = list(df.columns)
    # classify columns: numeric-looking wavelength headers vs non-spectral (id/meta/target)
    wl_cols, meta_cols = [], []
    for c in cols:
        if WL_RE.match(str(c).strip()):
            wl_cols.append(c)
        else:
            meta_cols.append(c)
    don_hits = [c for c in cols if DON_PAT.search(str(c))]
    wl_numeric = sorted(wl_value(c) for c in wl_cols) if wl_cols else []

    lines += [
        f"## {tag}  (`{fn}`)",
        f"- shape: **{df.shape[0]} rows  x  {df.shape[1]} cols**",
        f"- non-spectral/meta columns ({len(meta_cols)}): `{meta_cols[:10]}`",
        f"- waveband-like numeric columns: **{len(wl_cols)}**"
        + (f"  range **{wl_numeric[0]:.2f} – {wl_numeric[-1]:.2f} nm**" if wl_numeric else ""),
        f"- columns matching DON/target pattern: **{don_hits if don_hits else 'NONE FOUND'}**",
        f"- first 5 column names: `{cols[:5]}`",
        f"- last 3 column names: `{cols[-3:]}`",
        f"- total missing cells: {int(df.isna().sum().sum())}",
        "",
    ]
    summary[tag] = dict(df=df, meta=meta_cols, wl=wl_cols, don=don_hits)

# ---- genotype ID + cross-year overlap ----
def id_col(meta):
    # heuristic: first meta column is the genotype/sample id
    return meta[0] if meta else None

lines += ["## Cross-year overlap (year vs genotype confound)"]
id21 = id_col(summary["2021"]["meta"]); id22 = id_col(summary["2022"]["meta"])
if id21 and id22:
    g21 = set(summary["2021"]["df"][id21].astype(str))
    g22 = set(summary["2022"]["df"][id22].astype(str))
    ov = g21 & g22
    lines += [
        f"- id column used: 2021=`{id21}`, 2022=`{id22}`",
        f"- |2021|={len(g21)}, |2022|={len(g22)}, **overlap={len(ov)}**, union={len(g21|g22)}",
        f"- example overlapping IDs: {sorted(ov)[:8]}",
        "",
    ]
else:
    lines += ["- could not identify a genotype id column heuristically; inspect meta columns above.", ""]

# ---- verdict on DON availability ----
any_don = any(summary[t]["don"] for t in summary)
lines += ["## VERDICT — is the DON target present in this deposit?"]
if any_don:
    lines += ["- **YES** — at least one file carries a DON/target-like column (see above). 2A modeling can proceed on this deposit alone."]
else:
    lines += [
        "- **NO DON COLUMN DETECTED in any of the 3 phenomic CSVs.**",
        "- These files appear to be spectra-only (genotype id + wavebands).",
        "- => The DON ppm target must be sourced elsewhere (paper supplement / author request / companion file).",
        "- This is a Phase-0 BLOCKER for the prediction task and must be resolved before modeling.",
    ]
lines += [""]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print(f"\n[written] {OUT}")

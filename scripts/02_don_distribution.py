"""
Phase 0 — DON target distribution + across-file reconciliation.
Drives the screening cut-off decision (and flags the breeding-data caveat).
Outputs: results/02_don_distribution.md  +  results/FIG_don_distribution.png
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
try:                                   # UnicodeEncodeError guard for legacy (GBK) consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file is in scripts/)
DATA = os.environ.get("WHEAT_DON_DATA", os.path.join(_ROOT, "data"))
RES  = os.environ.get("WHEAT_DON_RESULTS", os.path.join(_ROOT, "results"))
F = {"2021":"2021_Phenomic_Data_VIS-NIR.csv",
     "2022":"2022_Phenomic_Data_VIS-NIR.csv",
     "across":"2021-2022_Across_Years_Phenomic_Data_VIS-NIR.csv"}
dfs = {k: pd.read_csv(os.path.join(DATA,v)) for k,v in F.items()}

# Reference cut-offs (COMMODITY limits — breeding lines differ; for context only)
REFS = {"US FDA advisory finished wheat (1 ppm)":1.0,
        "EU unprocessed wheat (1.25 ppm)":1.25,
        "EU unprocessed durum/oats (1.75 ppm)":1.75}

lines = ["# Phase 0 — DON distribution & cut-off basis", ""]

def desc(s):
    s = s.astype(float)
    q = s.quantile([.10,.25,.50,.75,.90,.95]).round(3)
    return (f"n={s.size}, min={s.min():.3f}, max={s.max():.3f}, mean={s.mean():.3f}, "
            f"median={s.median():.3f}, sd={s.std():.3f}, skew={s.skew():.2f}\n"
            f"  quantiles p10={q.loc[.10]} p25={q.loc[.25]} p50={q.loc[.50]} "
            f"p75={q.loc[.75]} p90={q.loc[.90]} p95={q.loc[.95]}")

for k in ["2021","2022","across"]:
    s = dfs[k]["DON"].astype(float)
    lines += [f"## {k}", "- DON ppm: " + desc(s)]
    lines += ["- fraction above reference cut-offs:"]
    for name,thr in REFS.items():
        frac = (s>thr).mean()
        lines += [f"    - >{thr} ppm — {name}: {(s>thr).sum()}/{s.size} = {frac*100:.1f}%"]
    # data-driven tertiles
    t1,t2 = s.quantile([1/3,2/3]).round(3)
    lines += [f"- data-driven tertile thresholds: low<={t1} | mid | high>{t2}", ""]

# ---- reconcile across (558) vs union of single years (549) ----
g21=set(dfs["2021"]["Genotype"].astype(str)); g22=set(dfs["2022"]["Genotype"].astype(str))
ga =dfs["across"]["Genotype"].astype(str)
ga_set=set(ga)
lines += ["## Across-file reconciliation",
          f"- across rows={len(ga)}, unique genotypes={ga_set.__len__()}, dup rows={len(ga)-len(ga_set)}",
          f"- union(2021,2022)={len(g21|g22)}",
          f"- in across but NOT in either single year: {len(ga_set-(g21|g22))} -> {sorted(ga_set-(g21|g22))[:10]}",
          f"- in either single year but NOT in across: {len((g21|g22)-ga_set)} -> {sorted((g21|g22)-ga_set)[:10]}",
          f"- overlap(2021,2022)={len(g21&g22)}", ""]

# ---- figure: DON histograms ----
fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharex=True)
for ax,k in zip(axes,["2021","2022","across"]):
    s=dfs[k]["DON"].astype(float)
    ax.hist(s,bins=25,color="#8DA1A8",edgecolor="white")
    ax.axvline(1.0,color="#B5503C",ls="--",lw=1.1,label="FDA / EU current (1.0)")
    ax.axvline(1.25,color="#8C7A5B",ls=":",lw=1.1,label="EU former (1.25)")
    ax.axvline(s.median(),color="#3D6E70",ls="-",lw=1,label=f"median {s.median():.2f}")
    ax.set_title(f"{k}  (n={s.size})"); ax.set_xlabel("DON (ppm)"); ax.legend(fontsize=7)
axes[0].set_ylabel("genotypes")
fig.suptitle("Wheat DON distribution by trial (Dryad d2547d8bx)")
fig.tight_layout()
os.makedirs(RES,exist_ok=True)
fig.savefig(os.path.join(RES,"FIG_don_distribution.png"),dpi=300)
fig.savefig(os.path.join(RES,"FIG_don_distribution.pdf"))

out=os.path.join(RES,"02_don_distribution.md")
open(out,"w",encoding="utf-8").write("\n".join(lines))
print("\n".join(lines)); print("\n[written]",out)

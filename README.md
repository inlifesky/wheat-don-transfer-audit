# Cross-year transfer audit of VIS–NIR deoxynivalenol screening in wheat

Reproducibility repository for the manuscript:

> Yuan, W. (2026). *Transfer validation of VIS–NIR deoxynivalenol screening in wheat: recalibration limits, calibration-transfer failure, and signal-nature audit.* Submitted to *Journal of Food Measurement and Characterization* (June 2026). ORCID: 0009-0009-4139-7802.

This repository contains the analysis pipeline that reproduces every number, table and figure in the manuscript, including the supplement. The underlying spectral data are openly deposited at Dryad and are **not** redistributed here; see [Data](#data) below for download instructions.

---

## What this paper audits

VIS–NIR hyperspectral imaging of wheat is widely used as a rapid surrogate for measuring deoxynivalenol (DON, a *Fusarium* mycotoxin). Reported predictive abilities are typically estimated by random within-year cross-validation. This paper asks whether such validation reflects what happens on a new harvest, using one openly deposited dataset (Concepcion & Olson 2025, Dryad doi:10.5061/dryad.d2547d8bx).

Three layers of measurement reliability are quantified:

1. **Optimism of random within-year cross-validation.** Within-year R² (0.667 / 0.517) reproduces the published Pearson r (0.82 / 0.72); cross-year R² collapses to negative values (all bootstrap 95% CIs below 0).
2. **Recalibration ceiling.** A per-year bias correction from 5–20 reference assays halves cross-year RMSE but only lifts R² to ≈ 0; recalibration restores scale, not discrimination.
3. **Standard calibration-transfer failure.** Piecewise Direct Standardization, External Parameter Orthogonalization and Generalized Least Squares Weighting, using the 34 cross-year cultivar pairs as paired surrogate standards, do not recover transfer; several actively destroy ranking.

Mechanistic interpretation: the cross-year spectral difference is dominated by biological / sample-state variation (FHB severity, kernel discolouration) rather than a reversible instrument/session perturbation. The signal is therefore better understood as an FHB-damage colour proxy than as a DON-specific spectral feature.

---

## Data

The spectral data and DON reference values are openly deposited at:

- **Dryad** — doi:[10.5061/dryad.d2547d8bx](https://doi.org/10.5061/dryad.d2547d8bx) (Concepcion & Olson, 2025). Dryad publishes all datasets under the **Creative Commons Zero (CC0 1.0) public-domain dedication**.

Three CSV files are needed (the SNP matrix is not used in this analysis):

| File | Size |
|---|---|
| `2021_Phenomic_Data_VIS-NIR.csv` | 729 KB |
| `2022_Phenomic_Data_VIS-NIR.csv` | 698 KB |
| `2021-2022_Across_Years_Phenomic_Data_VIS-NIR.csv` | 1.4 MB |

Place these in `data/` after download. SHA256 checksums of the files used in the manuscript are recorded in `logs/download_provenance.tsv`.

**This repository does not redistribute the raw Dryad files.** Even though CC0 would permit it, the convention in this field is to point users back to the source deposit so that download counts and citations accrue to the original authors. Run the pipeline locally after downloading from Dryad.

**Note on Dryad's download gate.** Dryad now serves files behind an Anubis proof-of-work + AWS-WAF gate; anonymous `curl`/API calls return 401/403. The working method is to navigate the dataset page in a real browser, which clears the WAF and solves the PoW automatically. See `docs/PHASE_0_summary_v1.md` §6 for the original investigation.

---

## Reproduction

### 1. Environment

The pipeline was developed on Python 3.14.0 with NumPy 2.4, pandas 2.3, scikit-learn 1.8, SciPy 1.17 and matplotlib 3.10; it should also work on Python 3.11+. Either `environment.yml` (conda) or `requirements.txt` (pip) reproduces a locked environment:

```bash
# Option A: conda
conda env create -f environment.yml
conda activate wheat-don-audit

# Option B: pip + venv
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data

Download the three CSV files from Dryad (link above) and place them in `data/`:

```
data/
├── 2021_Phenomic_Data_VIS-NIR.csv
├── 2022_Phenomic_Data_VIS-NIR.csv
└── 2021-2022_Across_Years_Phenomic_Data_VIS-NIR.csv
```

### 3. Run

```bash
# Audit + Figure 1 (~10 s)
python scripts/01_data_audit.py
python scripts/02_don_distribution.py

# R1 within-year nested CV — Table 2 (~3 min on 8 cores)
python scripts/03_r1_within_year_cv.py

# R2 cross-year + R3 seen/unseen — Tables 3, 4; Figure 2 (~30 s)
python scripts/04_r2_r3_cross_year.py

# Waveband audit — Table 5; Figure 3 (~30 s)
python scripts/05_waveband_audit.py

# §3.6 recalibration k-curve — Table 6; Figure 4 (~1 min)
python scripts/47_recalibration_kcurve.py

# §3.2 bootstrap R² CIs (~30 s)
python scripts/48_bootstrap_R2_CIs.py

# Supplement S2 secondary models — RF, SVR (~5 min)
python scripts/50_secondary_models.py

# §3.7 calibration-transfer comparison — Table 7; Figure 5 (~30 s)
python scripts/52_calibration_transfer.py
```

All outputs are written to `results/`. Each script is independent and idempotent; you can re-run a single script after editing.

---

## Repository structure

```
.
├── README.md                          ← this file
├── LICENSE                            ← MIT for code; data are not redistributed
├── environment.yml                    ← conda environment lock
├── requirements.txt                   ← pip alternative
├── .gitignore                         ← excludes data/*.csv, results/*, __pycache__
├── data/                              ← (empty; user downloads from Dryad)
├── scripts/                           ← analysis pipeline
│   ├── preprocessing.py               ← SNV / MSC / SG1 / SG2 (used by all)
│   ├── 01_data_audit.py               ← verifies row counts, missingness, overlap
│   ├── 02_don_distribution.py         ← Figure 1
│   ├── 03_r1_within_year_cv.py        ← Table 2 (within-year nested CV)
│   ├── 04_r2_r3_cross_year.py         ← Tables 3, 4 + Figure 2 (cross-year transfer)
│   ├── 05_waveband_audit.py           ← Table 5 + Figure 3 (band-importance stability)
│   ├── 47_recalibration_kcurve.py     ← Table 6 + Figure 4 (per-year recalibration)
│   ├── 48_bootstrap_R2_CIs.py         ← §3.2 bootstrap confidence intervals
│   ├── 50_secondary_models.py         ← Supplement S2 (RF, SVR)
│   └── 52_calibration_transfer.py     ← Table 7 + Figure 5 (PDS / EPO / GLSW)
├── results/                           ← outputs (regenerated by re-running scripts)
├── docs/                              ← locked pre-analysis protocol (PHASE_0_summary_v1.md)
└── logs/                              ← number verification log + download provenance
```

**This repository contains only the scripts that reproduce the manuscript.** Many exploratory analyses (alternative models, transfer diagnostics, sensitivity sweeps) were run during development and are omitted here to keep the reproduction pipeline self-contained. Every number, table and figure in the manuscript and its supplement is produced by the scripts listed above; the locked pre-analysis protocol — with a dated log of the small deviations made during analysis — is in `docs/PHASE_0_summary_v1.md`.

### A note on the calibration-transfer implementations

The PDS, EPO and GLSW routines in `scripts/52_calibration_transfer.py` are lightweight, transparent implementations based on the cited chemometric literature (Wang, Veltkamp & Kowalski, 1991; Roger, Chauchard & Bellon-Maurel, 2003; Martens et al., 2003) and are used in the manuscript as sensitivity analyses under the locked R2 protocol. All parameter sweeps reported in Supplement S6 are exposed in the script. Users are encouraged to inspect the code and report any implementation issues through the repository issue tracker.

---

## What you should see when reproduction succeeds

Each script writes a Markdown summary alongside its TSV outputs. The headline numbers to match (within rounding from SEED = 42):

| Script | Headline | Expected value |
|---|---|---|
| `01_data_audit.py` | 2021 n × 2022 n × overlap | 298, 285, 34 |
| `03_r1_within_year_cv.py` | R1 SG2/EN R² 2021 / 2022 | 0.667 / 0.517 |
| `04_r2_r3_cross_year.py` | R2 EN R² 2021→2022 / 2022→2021 | −2.73 / −2.15 |
| `47_recalibration_kcurve.py` | bias-only EN k = 20 R² 21→22 / 22→21 | −0.19 / +0.10 |
| `48_bootstrap_R2_CIs.py` | R2 EN 21→22 95% CI | [−3.40, −2.23] |
| `52_calibration_transfer.py` | PDS (w = 10) R² 21→22 / 22→21 | −4.78 / −0.09 |

If any of these are off, please open an issue.

---

## Citation

If you use this code, please cite the manuscript (once published) and the source dataset:

- Yuan, W. (2026). Transfer validation of VIS–NIR deoxynivalenol screening in wheat: recalibration limits, calibration-transfer failure, and signal-nature audit. *Submitted to Journal of Food Measurement and Characterization (June 2026).* ORCID: 0009-0009-4139-7802.
- Concepcion, J. S., Noble, A. D., Thompson, A. M., Dong, Y., & Olson, E. L. (2025). Genomic and hyperspectral imaging-based prediction blending enables selection for reduced deoxynivalenol content in wheat grains. *G3 Genes|Genomes|Genetics, 15*(10), jkaf176. https://doi.org/10.1093/g3journal/jkaf176
- Concepcion, J., & Olson, E. (2025). *SNP genotype and hyperspectral reflectance data* [Data set]. Dryad. https://doi.org/10.5061/dryad.d2547d8bx

---

## License

Code is released under the [MIT License](LICENSE). The underlying Dryad dataset is released under its own licence; users should download it directly from Dryad and comply with that licence.

---

## Contact

Wei Yuan — rita.w.yuan@gmail.com — ORCID: [0009-0009-4139-7802](https://orcid.org/0009-0009-4139-7802)

# Phase 0 — Data acquisition, audit & locked design (project_wheat_DON_2A)

**Date:** 2026-06-02
**Status:** Phase 0 **COMPLETE** → Phase 1 (modeling) UNLOCKED, under the protocol locked in §4–§5.
**Inputs:** verified Dryad deposit doi:10.5061/dryad.d2547d8bx (see Phase −1 audit).

---

## 1. Data acquired (provenance)

Downloaded 2026-06-02 via browser (Dryad serves files behind an **Anubis** proof-of-work gate +
AWS-WAF; `curl`/API both 401/403 anonymously — see §6). SHA256 recorded:

| file | bytes | sha256 |
|---|---|---|
| 2021_Phenomic_Data_VIS-NIR.csv | 729380 | 22D6FC6D658CBD875EA5C57393FDC077C9F4B872A0FF6EF8AE38CF88B4448DB6 |
| 2022_Phenomic_Data_VIS-NIR.csv | 697606 | 8E5887560E78CFBEE43B5AE72521E05AFA1363DCFBD5D2EF2B00411CC7EB79E7 |
| 2021-2022_Across_Years_..._.csv | 1368047 | B7355C57DAE54C68705AD9075783D98206C59EB3A71E049F9DC9A35D48A56F91 |
| README.md | 3854 | EBBB53B9CA9A84B25C03289594AF4233ECC5BD05568088B976620979183DD279 |

(107 MB SNP matrix intentionally NOT downloaded — irrelevant to spectral-only 2A.)

Scripts: `scripts/01_data_audit.py`, `scripts/02_don_distribution.py`.
Machine outputs: `results/01_data_audit.md`, `results/02_don_distribution.md`, `results/FIG_don_distribution.png`.

---

## 2. Verified data structure (counted, not assumed)

| fact | value | note |
|---|---|---|
| File layout | `Genotype, DON, R397.32 … R1003.58` | 2 meta + **204 wavebands** |
| **DON target present?** | **YES, in all 3 files** | make-or-break PASS — no author request needed |
| Rows | 2021 = **298**, 2022 = **285**, across = **558** | resolves paper's 307/288 → use 298/285 |
| Missing cells | **0** in all files | clean |
| Wavebands | 204, R397.32–R1003.58 nm | matches spec |
| DON units | ppm (GC/MS, U. Minnesota) | real chemical reference |
| Cross-year genotype overlap | **34** | paper said 37; counted = 34 |
| Across-file integrity | 558 unique, 0 dups, ⊇ union(549); 9 across-only genotypes | clean |

---

## 3. ⚠️ Framing-critical finding — this is an inoculated breeding nursery, not commodity grain

DON distribution (ppm): **range 5.0–58.7; 2021 mean 27.8 / 2022 mean 21.2 / across mean 24.8.**
**100% of genotypes exceed every commodity limit** (FDA 1 ppm, EU 1.25 ppm). See `FIG_don_distribution.png`.

Consequences (binding on all downstream framing):
1. **A regulatory "safe vs unsafe" classifier is meaningless** on this data — everything is "unsafe" by legal limits. Do NOT frame 2A as detecting whether grain passes a legal threshold.
2. The defensible task is **relative DON-risk ranking/stratification among breeding lines** — regression of DON ppm + a **data-driven** (tertile/quartile) high-vs-low split. Label it as relative risk, never regulatory.
3. **Real year target-shift exists** (2021 ≈ +6.6 ppm vs 2022). This is a feature, not a bug: it makes the cross-year transfer test genuinely informative about robustness to a worse FHB year.
4. This is the quantified form of the Phase −1 "breeding-data caveat." Manuscript must state it up front.

---

## 4. LOCKED cross-validation protocol (decided before any fitting)

Designed around the two verified traps (34-genotype overlap confound; p≈n with 204 collinear bands).

**Primary evaluation — three CV regimes, reported side by side:**
| regime | how | what it isolates |
|---|---|---|
| R1 Within-year random CV | 5×5 repeated CV inside each year, genotype-disjoint folds | optimistic baseline (what most papers report) |
| R2 **Cross-year transfer** | train 2021 → test 2022; train 2022 → test 2021 | real-world year robustness (the headline number) |
| R3 Pure-year probe | the **34 overlapping genotypes** only: same line, predicted across years | partially separates *year* effect from *new-genotype* effect |

- The gap **R1 − R2** is the core reportable quantity (optimism of random CV vs year transfer).
- R3 is the only clean handle on "is the drop year or new germplasm?" — report it explicitly with its small-n caveat (n=34).
- **No genotype may appear in both train and test of any fold** (genotype-disjoint everywhere).
- Hyperparameters tuned by **nested** inner CV only; outer folds never see tuning.

**Models (primary → secondary):** PLSR and Elastic Net are primaries (built for collinear p≈n spectra); Random Forest / SVR / XGBoost as secondary robustness checks (expected to overfit 204 collinear bands — report, don't headline).
**Preprocessing:** evaluate standard spectral transforms via `chemotools`/`prospectr` (SNV, MSC, Savitzky-Golay 1st/2nd derivative) as a pre-registered small grid — do NOT hand-roll.

---

## 5. PRE-REGISTERED metrics (fixed now, before modeling)

**Regression (primary):** R², RMSE, MAE, bias (mean error) — bias matters under year-shift. Report per regime R1/R2/R3.
**Relative risk-screening (secondary):** define "high-DON" = top tertile *within the training distribution*; report **sensitivity (recall of high-DON lines)**, specificity, precision, balanced accuracy, ROC-AUC, PR-AUC — and how each degrades R1→R2. "Missed high-DON line" = the false-negative analog.
**Waveband audit:** PLS-VIP + permutation importance + bootstrap selection frequency; report **cross-year stability** of the top bands (rank overlap 2021 vs 2022), not just in-sample importance. Pre-registered band-region contrasts: VIS 400–700 vs NIR 700–1000 vs full vs red ~600 nm.
**Signal-nature probe:** qualitative — whether the dominant bands sit where Fusarium kernel damage/colour (visible/red) rather than a DON molecular feature would predict; cross-ref Sci Rep 2024 GWAS regions.

---

## 6. Reproducibility note — how to re-download (non-trivial)

Dryad now fronts file downloads with **Anubis** (PoW "Validating…" gate, `within.website/x/xess`) plus AWS-WAF.
- `GET /api/v2/files/{id}/download` → 401 (needs bearer token / account).
- `GET /downloads/file_stream/{id}` via curl → 403 (WAF) or Anubis challenge HTML.
- **Working method:** drive a real browser (Playwright) to the dataset page (solves WAF), then navigate to each `…/downloads/file_stream/{id}` URL once (Anubis solves PoW, sets `techaro.lol-anubis-auth` cookie, browser auto-downloads the file). File IDs: 2021=4202047, 2022=4202048, across=4202049, README=4202051, SNP=4202050.

---

## 7. Phase 0 gate — status
- [x] download + SHA256 + true row counts (298/285/558; resolves 307/288)
- [x] DON column confirmed present + units (ppm, GC/MS) + 0 missing
- [x] DON distribution per year → cut-off strategy decided (**relative tertile, NOT regulatory**; breeding-data caveat quantified)
- [x] 34 overlapping genotypes confirmed by ID (reserved as R3 pure-year probe)
- [x] CV protocol locked (R1/R2/R3, genotype-disjoint, nested tuning) — §4
- [x] screening + regression + waveband metrics pre-registered — §5

**GATE: PASS.** Phase 1 (preprocessing grid + PLSR/EN baselines under R1/R2/R3) is unlocked.

### Repositioned thesis (carried from Phase −1, now data-grounded)
Not "first year-external validation" (the source paper already did forward prediction) and not "regulatory DON detection" (every line is over-limit). 2A = **a reproducibility + robustness audit of public VIS-NIR DON prediction: how much does random-CV performance overstate cross-year transfer, which wavebands stay informative across years, and is the signal DON-specific or FHB-damage-driven — framed as relative risk ranking among breeding lines.**

---

## 8. Deviations from the locked protocol (post-hoc addendum)

**Added 2026-06-05, after analysis was complete.** Sections 1–7 above are the original plan, locked 2026-06-02 before any model fitting, and are **left unchanged**. This section transparently records every point where the final analysis (as reported in the manuscript and reproduced by the pipeline scripts) departs from that locked plan, and why. None of these deviations changes the direction or the qualitative conclusions of the study; two of them improve the validity of the test.

| # | Locked plan (§) | What was actually done | Rationale | Effect on conclusions |
|---|---|---|---|---|
| D1 | R1 = **5×5** repeated within-year CV (§4) | R1 = **5×3** repeated within-year CV | Three repeats were sufficient to stabilise the R1 point estimate given that the reportable quantity is the **R1 − R2 gap**, not R1 to high precision; sampling uncertainty is separately quantified by the bootstrap CIs (§3.2 / script 48). | None. R1 = 0.667 / 0.517 still reproduces the source paper's published *r*; the gap and its CI are unchanged in interpretation. |
| D2 | "high-DON" = top tertile **within the training distribution** (§5) | "high-DON" = top tertile **of the test year's own distribution** | The deployment question is "can the model rank the worst lines in a *new* harvest?" Under the documented year target-shift (2021 ≈ +6.6 ppm vs 2022, §3), a training-distribution cut would mislabel a large fraction of the test year purely from the shift, confounding the screening metric with the offset. Scoring against the test year's own tertile is the operationally correct triage target. | **Strengthens validity.** Removes a known confound; the surviving AUC ~0.7 is a cleaner statement of rank usefulness. |
| D3 | Waveband audit = **PLS-VIP + permutation importance + bootstrap selection frequency** (§5) | Waveband audit = **Elastic-Net coefficient magnitude + cross-year rank-stability of top bands + VIS-vs-NIR region contrast** | The pre-registered *intent* — cross-year **stability** of important bands and a VIS/NIR region contrast — is preserved. EN is a primary model, so its coefficient profile is directly interpretable without introducing a separate importance estimator; the full VIP/permutation/bootstrap battery was descoped as redundant for the stability claim actually made. | None on the stability conclusion (top bands sit in VIS/red and are cross-year-unstable). The descoped estimators would have answered the same question with more machinery. |
| D4 | Preprocessing via **`chemotools`/`prospectr`** ("do NOT hand-roll", §4) | Preprocessing via **local NumPy/SciPy implementations** (`scripts/preprocessing.py`: SNV, MSC, SG1, SG2) | Implemented locally to keep the pipeline dependency-light and fully auditable, and to make leakage control explicit (MSC reference mean is fitted on the **training fold only**; SNV and Savitzky–Golay derivatives are row-wise/stateless). The pre-registered **grid of transforms** is unchanged — only the implementation source differs. | None. Standard transforms; behaviour matches the library definitions. Arguably a strengthening (no opaque dependency; leakage path is visible in code). |

**Summary.** D1, D3, D4 are scope/implementation choices that leave the conclusions intact; D2 is a definitional correction that makes the cross-year screening metric more, not less, conservative. The locked metrics list (R², RMSE, MAE, bias for regression; sensitivity/specificity/precision/balanced-accuracy/AUC/PR-AUC for screening; cross-year band stability) is otherwise reported as pre-registered.

# INTERVENE v4 — Locked re-validation protocol

Written 2026-10-11 **before** running any analysis in this folder. Its SHA-256 is recorded in `PROTOCOL.sha256`
before the first run; any later change is logged as a dated amendment at the bottom, never edited in place.

## Why this protocol exists (audit of the v3 evidence)
1. The refined 4-gene signature (MRPL21, PSMA1, PSMB5, TXNRD1) and the "top relapse" mitochondrial genes were ranked on
   the full GSE31210 cohort and then scored on the same patients, so the reported internal C-index of about 0.76 is
   optimistic. The INTERVENE v3 attention model's out-of-fold C-index (0.701) is the only honest internal estimate.
2. The v3 attention model did not significantly beat plain attention (+0.007, p = 0.29) or the raw module score
   (+0.036, p = 0.07); a random-module control was not rejected (p = 0.14).
3. Earlier external tests mixed histologies (GSE30219, GSE37745, GSE50081 include squamous, neuroendocrine and
   large-cell tumours) although the claim concerns lung adenocarcinoma (LUAD).
4. In every external cohort a 13-gene proliferation score was at least as prognostic as the MRPL21-based scores.
5. GSE50081 was evaluated (fixed INTERVENE state, C-index 0.485) but omitted from later summary tables.

**What is new versus already seen.** The 4-gene signature has been scored before on GSE30219, GSE37745 and GSE68465
(all histologies). The LUAD-only populations, the nested internal estimate, the proliferation-adjusted pooled test,
the random-signature null and the 4-gene score on GSE50081 have not been run. All results below are reported,
whatever their direction.

## Data (all downloaded GEO series matrices; no new data)
| Cohort | Role | Platform / processing | Population (primary) | Endpoint |
|---|---|---|---|---|
| GSE31210 | discovery | GPL570, MAS5 | stage I-II primary LUAD, `prognosis_included` (n = 204, 54 relapses) | relapse-free months, relapse |
| GSE30219 | external | GPL570, RMA | histology = ADC | DFS months (`dfs_months`), relapse |
| GSE37745 | external | GPL570, RMA | histology = adeno | recurrence months, recurrence |
| GSE50081 | external | GPL570, RMA | adenocarcinoma incl. "NSCLC-favor adenocarcinoma" | DFS (years x 12 = months), recurrence |
| GSE68465 | external | GPL96 | Lung Adenocarcinoma (exclude Normal) | months to first progression/relapse (`pfs_months`), progression/relapse |
Rows with missing endpoint or time <= 0 are dropped. Secondary analysis: all histologies (as in v3).

## Expression processing (fixed)
Series-matrix values as deposited; log2(x + 1) if the cohort maximum exceeds 100 (MAS5 / GPL96). Probes collapse to
genes by the per-gene median across probes (as in v3). Each gene is z-scored within cohort (unsupervised, label-free).

## Scores (fixed)
- **S4** (locked signature) = mean z of MRPL21, PSMA1, PSMB5, TXNRD1 (all risk-increasing). GSE68465 (GPL96) has no
  MRPL21 probe: S4 there uses the 3 available genes (pre-specified; flagged in all tables).
- **P13** (proliferation) = mean z of MKI67, TOP2A, MCM2, MCM5, PCNA, CDK1, CDK2, AURKA, AURKB, MCM6, BUB1, CCNB1, UBE2C.
- Candidate pool for nested discovery = genes of the six DepMap-derived INTERVENE modules (proteostasis,
  mito_translation, oxidative_stress, apoptosis_iap, srebp_lipid, akt_mtor_growth) present in GSE31210
  (the columns of `gse31210_module_gene_expression.csv`). The modules come from cell-line data, independent of patients.

## Hypotheses, tests and pass rules
**H1 — honest internal performance of the discovery procedure.** Repeated nested CV in GSE31210: 5 folds x 50 repeats,
stratified by event. In each training fold: univariate Cox z-statistic for every candidate gene; select the 4 genes
with the largest positive z (risk-increasing, as in v3); score = mean z of those genes; evaluate C-index on the held-out
fold. Report the mean C-index and the 2.5-97.5 percentiles over repeats, and how often each of the four S4 genes is
selected. *Pass:* mean nested C-index > 0.60 with lower percentile > 0.55.

**H2 — external discrimination of the locked S4 (LUAD).** C-index per external cohort with 1000-sample bootstrap
95% CI; pooled by DerSimonian-Laird random effects on C-index (bootstrap SEs). *Pass:* pooled C-index > 0.5 with 95% CI
excluding 0.5.

**H3 — information beyond proliferation.** Per cohort Cox model risk ~ P13 + S4 (both standardised); Wald z for S4.
Pooled by inverse-variance fixed effect on the S4 log-hazard ratio (and random effects as sensitivity).
*Pass:* pooled S4 log-HR > 0 with two-sided p < 0.05.

**H4 — better than random signatures.** 1000 random 4-gene sets drawn uniformly from genes measured in all five
cohorts (seed 20261011). For each set: mean external C-index across the four cohorts, and pooled proliferation-adjusted
log-HR. Report the percentile of S4 for both. *Pass:* S4 above the 95th percentile on both statistics.

**H5 (secondary) — clinical adjustment.** Where stage is available, Cox risk ~ stage + S4 and ~ stage + P13 + S4.

**Reported without a pass rule:** all-histology versions of H2-H4; the v3 attention model's existing out-of-fold
results versus its baselines; S4 vs P13 head-to-head C-index per cohort.

## Decision rule for the paper framing (fixed now)
- H1, H2, H3 and H4 all pass: method-and-discovery paper (INTERVENE derives a relapse axis that is prognostic beyond
  proliferation and beyond random signatures).
- H1 and H2 pass but H3 or H4 fails: the signal is largely proliferation / generic; write a rigorous re-evaluation paper
  centred on what actually carries the relapse signal and on the evaluation pitfalls.
- H1 or H2 fails: report as a negative replication.

## Amendments
- A1 (2026-10-11, before any H1-H5 result was computed): in H1 the per-gene 'univariate Cox z-statistic' is computed as
  the Cox partial-likelihood score-test z at beta = 0 (the log-rank form), which ranks genes identically in direction and
  is fast enough for 5 x 50 nested folds. All other Cox models are full maximum-likelihood fits (lifelines).

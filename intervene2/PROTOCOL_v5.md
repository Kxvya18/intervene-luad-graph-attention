# INTERVENE-2 — locked model-centric protocol (v5)

Written 2026-10-11, before any full run. Committed to GitHub before the Colab run starts; the pipeline then executes
end to end without human choices, so nothing about the sealed cohorts can influence any design decision.

## Motivation (from the v4 locked re-validation, `v4_locked_validation/`)
The original INTERVENE (6 module tokens, 204 patients) did not beat standard attention (+0.007 C-index, p = 0.29) and
its derived 4-gene signature carried no relapse information beyond proliferation in four external LUAD cohorts.
INTERVENE-2 is redesigned so that (i) the interleaved-head architecture has a task rich enough to matter and (ii) the
model is forced to look for relapse biology beyond proliferation.

## Data
- Development (all design choices): GSE31210 (204 pts / 54 events), GSE30219 (83/27), GSE37745 (53/26), GSE68465 (333/178);
  LUAD only, relapse / recurrence / DFS / progression endpoints as in v4.
- Sealed (evaluated once, at the end, automatically): **TCGA-LUAD** RNA-seq (UCSC Xena HiSeqV2, primary tumours;
  progression-free interval PFI; 502 pts / 206 events) — never used before in this project; **GSE50081** LUAD
  (125/37) — its expression was used in v4 only to score the locked 4-gene signature, never for INTERVENE-2.
- Genes z-scored within cohort; 3,921 genes common to all platforms.
- Tokens (57): 50 MSigDB Hallmark gene sets, the 6 DepMap-derived INTERVENE modules, and a 13-gene proliferation token.

## Model (INTERVENE-2)
Gene-to-token encoder with learned within-token gene attention and L1-sparse signed gene loadings; stacked
interleaved-head attention blocks (Meta-IHA-inspired head mixing through 2 pseudo-heads per head) over tokens with a
soft graph bias from the gene-overlap coefficient |A & B| / min(|A|, |B|) between token gene sets (prior knowledge only); attention-pooled Cox risk head. Proliferation enters as an explicit Cox
covariate (risk = f(x) + beta * proliferation), so f(x) must explain relapse beyond proliferation.
Stratified (by cohort) Cox partial likelihood; AdamW; early stopping on a stratified 15% validation split.

## Comparators (identical encoder, data, budget, seeds)
- **MHA**: same network with standard multi-head attention (no head interleaving), same grid.
- **MLP**: same token encoder, no attention.  **LINEAR**: Cox on token mean scores.  **PROLIFERATION** alone.

## Development (automatic)
Leave-one-cohort-out over the 4 development cohorts, 5 seeds, grid layers {1, 2} x graph bias {0, 0.5} for both IHA
and MHA. **Selection rule (fixed):** within each family, the config with the highest mean held-out
proliferation-(and stage-)adjusted z-statistic of f(x); ties by mean held-out C-index.

## Final (automatic, once)
Selected IHA (INTERVENE-2), selected MHA, MLP and LINEAR trained on all development cohorts with 10 seeds; seed
ensembles evaluated on the sealed cohorts.

## Claims and pass rules (sealed cohorts; pooled by inverse variance across TCGA-LUAD and GSE50081)
- **C1 beyond proliferation:** INTERVENE-2's f(x) adds to proliferation (+ stage where available): pooled log-HR > 0,
  two-sided p < 0.05.
- **C2 better than standard attention:** pooled paired-bootstrap delta C-index (INTERVENE-2 minus MHA) > 0, one-sided p < 0.05.
- **C3 better than non-attention and classical models:** same test versus MLP and LINEAR.
- **C4 interaction discovery:** of the 10 token pairs with the highest INTERVENE-2 attention (learned on development
  data only; pairs involving proliferation programs - the proliferation token, Hallmark E2F targets, G2-M checkpoint,
  mitotic spindle - are excluded), at least one has a sealed pooled Cox interaction term (adjusted for proliferation, stage and main effects)
  with BH q < 0.05.
- **C5 stability (reported, no pass rule):** mean pairwise Spearman correlation of interaction maps across seeds,
  INTERVENE-2 vs MHA.
All outcomes are reported whatever their direction; the paper's headline claims are only those that pass.

## Amendments
(none)

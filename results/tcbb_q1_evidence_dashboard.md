# TCBB / Q1 Evidence Dashboard

## Current Verdict

The project is Q1-plausible as a method-led computational biology paper, but not yet ready as a pure biological breakthrough paper. The central contribution should be INTERVENE as an interpretable physics-guided attention framework; MRPL21/proteasome is the case-study discovery output.

## Readiness Counts

- complete: 11
- partial: 2
- limitation: 1
- missing: 1

## Evidence Checklist

| area | item | status | evidence | source_file |
| --- | --- | --- | --- | --- |
| AI benchmark | Refined gene axis outperforms raw broad module score | complete | best=MRPL21 + proteasome-core + proliferation C-index=0.761; V3 OOF=0.701; raw module=0.696 | results/mrpl21_intervene_ai_ablation.csv |
| Statistics | Permutation-label control | complete | observed C=0.701, null mean=0.499, p=0.000333 | results/final_permutation_control.csv |
| Statistics | Clinical-adjusted survival value | complete | best adjusted model=stage_age_smoking_driver, delta C=0.0941, p=4.02e-07 | results/final_clinical_adjusted_cox.csv |
| Statistics | Paired bootstrap model improvement | partial | V3 vs raw delta=0.0362, p=0.0716; V3 vs MHA delta=0.00748 | results/final_paired_bootstrap_c_index.csv |
| Interpretability | Stable sparse attention edges | complete | 7 soft-model edges appear in top10 across all 5 seeds | results/final_v3_edge_top10_seed_stability.csv |
| Interpretability | Module-removal importance | complete | top removed module=proteostasis, C-index drop=0.042 | results/final_v3_module_removal_importance.csv |
| Biology | Combined proteasome + mito/redox state | complete | high-both relapse=34/77 (44.2%); low-both=7/77 (9.09%) | results/gse31210_proteasome_mito_redox_combination_groups.csv |
| External validation | MRPL21 external validation | partial | GSE30219 MRPL21 C=0.603, p=0.000822; GSE37745 C=0.576, p=0.126 | results/tcbb_final_external_score_confidence_intervals.csv |
| Statistics | Bootstrap confidence intervals for internal scores | complete | refined C=0.759 CI=0.7-0.8; V3 C=0.701 CI=0.638-0.77 | results/tcbb_final_internal_score_confidence_intervals.csv |
| AI benchmark | Classical Cox survival baselines | complete | best=ridge_cox_refined_genes C-index=0.764; broad-module ridge/L1 also included | results/tcbb_final_classical_survival_baselines.csv |
| Data quality | MRPL21 probe/platform audit | limitation | GPL570 MRPL21 probes per cohort=GSE31210:1; GSE30219:1; GSE37745:1; GPL96=GSE68465:0 | results/mrpl21_probe_platform_stability.csv |
| Submission readiness | Wet-lab/causal validation | missing | No perturbation experiment; frame as computational discovery lead, not causal proof |  |
| Submission readiness | Final manuscript figures | complete | Final SVG panels generated for architecture, AI ablation, biology mechanism, and external validation/falsification | figures/tcbb_figure1_intervene_architecture.svg; figures/tcbb_figure2_ai_ablation_attention.svg; figures/intervene_mrpl21_proteasome_redox_biology.svg; figures/tcbb_figure4_external_validation_falsification.svg |
| Submission readiness | One-command reproducibility | complete | Final runner and reproducibility manifest created | src/run_tcbb_final_pipeline.sh; paper_notes/tcbb_reproducibility_manifest.md |
| Submission readiness | Manuscript draft | complete | Draft Methods, Results, Discussion, limitations, and novelty comparison created | paper_notes/tcbb_manuscript_draft.md |

## Final Table Registry

| table | purpose | source_files |
| --- | --- | --- |
| Table 1 | Dataset/cohort summary | final_external_dataset_scout.csv; mrpl21_external_validation.csv |
| Table 2 | AI ablation and benchmark | mrpl21_intervene_ai_ablation.csv; tcbb_final_classical_survival_baselines.csv; relapse_intervene_survival_model_comparison.csv |
| Table 3 | Robustness statistics and confidence intervals | tcbb_final_internal_score_confidence_intervals.csv; tcbb_final_external_score_confidence_intervals.csv; final_paired_bootstrap_c_index.csv; final_permutation_control.csv; final_clinical_adjusted_cox.csv |
| Table 4 | Attention-to-biology evidence | mrpl21_attention_to_biology_evidence.csv; final_v3_edge_top10_seed_stability.csv |
| Table 5 | External validation and probe audit | mrpl21_external_validation.csv; mrpl21_probe_platform_stability.csv |

## Final Figure Registry

| figure | purpose | status | source_files |
| --- | --- | --- | --- |
| Figure 1 | INTERVENE architecture | exists | figures/tcbb_figure1_intervene_architecture.svg |
| Figure 2 | AI ablation and sparse attention stability | exists | figures/tcbb_figure2_ai_ablation_attention.svg |
| Figure 3 | MRPL21/proteasome biological mechanism | exists | figures/intervene_mrpl21_proteasome_redox_biology.svg |
| Figure 4 | External validation/falsification | exists | figures/tcbb_figure4_external_validation_falsification.svg |

## Immediate Work Needed For TCBB/Q1

1. Increase bootstrap iterations to 1000 before final submission tables.
2. Add one stronger external RNA-seq relapse cohort with direct MRPL21 measurement if available.
3. Add nested cross-validation if model tuning choices are presented as optimized.
4. Keep the manuscript method-led and avoid claiming MRPL21 as a proven causal relapse driver.
5. Convert SVG figures to journal-resolution PDF/TIFF after final content decisions.
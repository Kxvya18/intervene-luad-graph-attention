# INTERVENE

**INTERVENE** means **Interleaved Thermodynamic Vulnerability Network Embedding**.

Current Q1/TCBB-facing title:

**INTERVENE: An IHA-Inspired Graph-Sparse Attention Framework for Physics-Guided Relapse Mechanism Discovery in Lung Adenocarcinoma**

## Current Framing

This project is now a **method-led computational biology project**, not only a cancer-gene finding exercise.

INTERVENE extends an original thermodynamic vulnerability framework by adding an IHA-inspired graph-sparse attention model. The thermodynamic framework provides physics-guided biological priors; INTERVENE learns relapse-associated interactions among biological module tokens and extracts gene-level discovery leads.

Best current claim:

> INTERVENE identifies and stress-tests a relapse-associated mitochondrial-translation/proteasome-core state in lung adenocarcinoma, nominating MRPL21 and PSMA1/PSMB5 as candidate relapse-associated drivers.

Careful biological wording:

> MRPL21/proteasome-core is a computational discovery lead, not a proven cause of relapse. External validation is partial and platform-limited, so the paper should be framed as interpretable AI-driven mechanism discovery with falsification boundaries.

## Main Evidence

Discovery cohort:

- **GSE31210** lung adenocarcinoma relapse cohort.

External/falsification cohorts:

- **GSE30219**: partial MRPL21/proteasome support.
- **GSE37745**: weak/non-significant MRPL21 support.
- **GSE68465**: large LUAD cohort, but current GPL96 mapping does not directly support MRPL21 validation.

Key biological lead:

- MRPL21-linked mitochondrial translation.
- PSMA1/PSMB5 proteasome-core burden.
- TXNRD1/redox support.
- Combined proteasome + mito/redox high group has substantially higher relapse burden in GSE31210.

Key AI/method evidence:

- IHA-inspired interleaved module-token attention.
- GraphSparse biological edge regularization.
- Multi-seed stable attention edges.
- Module-removal importance.
- Proliferation-residual and clinical-adjusted survival tests.
- External validation/falsification tables.

## TCBB / Q1 Package

Start here:

- `results/tcbb_q1_evidence_dashboard.md`
- `paper_notes/tcbb_q1_readiness_plan.md`
- `paper_notes/tcbb_methods_equations.md`
- `paper_notes/tcbb_manuscript_skeleton.md`
- `paper_notes/tcbb_final_tables_and_figures.md`

Important result files:

- `results/mrpl21_intervene_ai_biology_validation_summary.txt`
- `results/mrpl21_intervene_ai_ablation.csv`
- `results/mrpl21_external_validation.csv`
- `results/mrpl21_probe_platform_stability.csv`
- `results/mrpl21_attention_to_biology_evidence.csv`
- `results/gse31210_refined_mechanism_axes_summary.txt`
- `results/final_intervene_ieee_readiness_tests_summary.txt`

Important figure:

- `figures/intervene_mrpl21_proteasome_redox_biology.svg`

## Main Commands

Use the bundled Codex Python runtime:

```bash
/Users/boolu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/run_relapse_intervene_graph_sparse_v3.py
/Users/boolu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/run_final_intervene_robustness_tests.py
/Users/boolu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/run_gse31210_refined_mechanism_axes.py
/Users/boolu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/run_mrpl21_intervene_ai_validation.py
/Users/boolu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/make_tcbb_q1_readiness_package.py
```

## IHA Relationship

INTERVENE is **IHA-inspired**, not a claim of improving or reproducing Meta's IHA.

Paper-safe wording:

> INTERVENE adapts the IHA idea of interleaved head mixing from sequence tokens to biological module tokens, allowing module-level biological interactions to be learned before gene-level refinement.

## Limitations

- External validation is partial.
- MRPL21 is one-probe on GPL570 and absent under current GSE68465/GPL96 mapping.
- No wet-lab perturbation or causal validation yet.
- Retrospective cohorts only.
- The current best submission angle is Q1 computational biology / biomedical AI, not pure oncology breakthrough biology.

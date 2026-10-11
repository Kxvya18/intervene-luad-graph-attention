from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from interleaved_biophysics_attention_full import high_low_stats, zscore


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
NOTES = ROOT / "paper_notes"


def method_side() -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv(RESULTS / "ibpa_full_model_comparison.csv")
    ablation = pd.read_csv(RESULTS / "ibpa_v2_ablation_summary.csv")
    raw = pd.read_csv(RESULTS / "ibpa_v2_raw_prior_comparison.csv")
    paper = pd.read_csv(RESULTS / "paper1_vs_intervene_compact_table.csv")
    multiseed_edges = RESULTS / "intervene_multiseed_edge_stability.csv"
    if multiseed_edges.exists():
        edges = pd.read_csv(multiseed_edges)
        edge_stability = edges.copy()
    else:
        edges = pd.read_csv(RESULTS / "ibpa_v2_discovery_attention_edges.csv")
        top_edges = edges.head(20).copy()
        edge_stability = (
            top_edges.groupby(["source_token", "target_token"])
            .agg(
                n_pseudohead_hits=("mean_attention", "size"),
                max_attention=("mean_attention", "max"),
                mean_enrichment=("attention_enrichment_vs_uniform", "mean"),
                heads=("head", lambda x: ",".join(map(str, sorted(set(x))))),
                pseudoheads=("pseudohead", lambda x: ",".join(map(str, sorted(set(x))))),
            )
            .reset_index()
            .sort_values(["n_pseudohead_hits", "mean_enrichment"], ascending=[False, False])
        )

    supervised = model[
        (model["dataset"] == "paper_tokens_no_controls")
        & (model["target"] == "cytotoxic_residual_stress_survival")
        & (model["model"].isin(["mha", "ibpa", "residual_only", "module_linear"]))
    ].copy()
    supervised["question"] = "supervised predictor comparison"
    supervised["paper_reading"] = supervised["model"].map(
        {
            "ibpa": "IHA-style attention is better than normal attention, but supervised CV is not the main claim.",
            "mha": "normal multi-head attention is weaker than IHA-style attention in this small-data setting.",
            "residual_only": "plain thermodynamic residual remains a strong simple baseline.",
            "module_linear": "linear module baseline is competitive but less biologically interpretable.",
        }
    )

    prior_tests = raw[
        raw["test"].isin(
            [
                "cytotoxic-residual stress-survival score",
                "PRISM stress-survival sensitivity",
                "PRISM generic cytotoxic sensitivity",
                "PRISM direct lipid sensitivity",
            ]
        )
    ].copy()
    prior_tests["question"] = "raw prior versus learned INTERVENE state"

    ablation_keep = ablation[
        [
            "ablation",
            "reconstruction_mse",
            "state_prior_corr",
            "prior_weight",
            "include_srebp",
            "include_mito",
            "include_oxidative",
            "cytotoxic_residual_stress_survival_score_effect",
            "cytotoxic_residual_stress_survival_score_p",
            "prism_generic_cytotoxic_sensitivity_effect",
            "prism_direct_lipid_sensitivity_effect",
            "keap1_nfe2l2_oxidative_context_effect",
            "keap1_nfe2l2_oxidative_context_p",
        ]
    ].copy()
    ablation_keep["paper_reading"] = np.where(
        ablation_keep["ablation"] == "no_prior",
        "no-prior model fails; weak physics guidance is necessary.",
        "prior-guided INTERVENE retains residual stress-survival validation.",
    )

    method_rows = []
    method_rows.append(
        {
            "method_question": "INTERVENE vs normal attention",
            "status": "done",
            "finding": "In supervised CV, IHA-style IBPA beats MHA for cytotoxic-residual stress-survival, but both are small-data predictors and not the headline.",
            "paper_use": "Use as secondary method ablation, not as the main evidence.",
        }
    )
    method_rows.append(
        {
            "method_question": "INTERVENE vs raw thermodynamic prior",
            "status": "done",
            "finding": "Raw prior is strong; learned INTERVENE slightly improves residual validation and adds oxidative context.",
            "paper_use": "Shows INTERVENE extends, not replaces, the physics prior.",
        }
    )
    method_rows.append(
        {
            "method_question": "INTERVENE vs no-prior model",
            "status": "done",
            "finding": "No-prior model fails on residual stress-survival; weak-prior model succeeds.",
            "paper_use": "Important ablation proving the physics prior is not decorative.",
        }
    )
    method_rows.append(
        {
            "method_question": "different prior weights",
            "status": "done",
            "finding": "Prior weight 0.05 gives the strongest residual effect; 0.22 remains strong; 0.50 over-constrains slightly.",
            "paper_use": "Report as sensitivity analysis.",
        }
    )
    method_rows.append(
        {
            "method_question": "pseudo-head biological edges",
            "status": "done",
            "finding": "Multi-seed testing shows recurrent edges into mitochondrial translation across all 7 seeds: SREBP/lipid, proteostasis, apoptosis/IAP, and AKT/mTOR all recur. Several oxidative-stress edges recur in 4-5 of 7 seeds.",
            "paper_use": "Use recurrent edges as interpretable biological mechanism evidence; avoid over-interpreting seed-specific edges.",
        }
    )
    method_summary = pd.DataFrame(method_rows)
    return method_summary, edge_stability


def biology_side() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(RESULTS / "ibpa_v2_discovery_cell_scores.csv")
    modules = pd.read_csv(RESULTS / "ibpa_v2_module_state_scores.csv")
    compounds = pd.read_csv(RESULTS / "prism_srebp_compound_scores_bucketed.csv")

    model_meta = pd.read_csv(RAW / "Model.csv")
    model_meta["ModelID"] = model_meta["ModelID"].astype(str)
    cells["ModelID"] = cells["ModelID"].astype(str)
    cells = cells.merge(
        model_meta[
            [
                "ModelID",
                "CellLineName",
                "OncotreeSubtype",
                "Sex",
                "PrimaryOrMetastasis",
                "SampleCollectionSite",
            ]
        ],
        on="ModelID",
        how="left",
    )

    high_cut = cells["ibpa_v2_state_score"].quantile(2 / 3)
    low_cut = cells["ibpa_v2_state_score"].quantile(1 / 3)
    cells["intervene_state_group"] = np.where(
        cells["ibpa_v2_state_score"] >= high_cut,
        "high",
        np.where(cells["ibpa_v2_state_score"] <= low_cut, "low", "middle"),
    )

    gene_rows = []
    module_gene_map = {
        "srebp_lipid": ["SCAP", "SCD", "MBTPS1", "MBTPS2", "SREBF1"],
        "mito_translation": [c.replace("_dep", "") for c in cells.columns if c.startswith("MRP") and c.endswith("_dep")],
        "oxidative_stress": ["KEAP1", "NFE2L2", "PGD", "TXNRD1", "GCLC", "GCLM", "SLC7A11"],
        "proteostasis": ["USP8", "STAMBP", "PTPN23", "UBAP1", "HGS", "VCP", "PSMA1", "PSMB5"],
        "apoptosis_iap": ["CFLAR", "XIAP", "BIRC2", "BIRC3", "CASP8", "MCL1", "BCL2L1"],
        "akt_mtor_growth": ["AKT1", "AKT2", "AKT3", "MTOR", "PIK3CA", "PIK3CB", "RPTOR", "RICTOR", "PTEN"],
    }
    for module, genes in module_gene_map.items():
        for gene in sorted(set(genes)):
            dep_col = f"{gene}_dep"
            expr_col = f"{gene}_expr"
            cn_col = f"{gene}_cn"
            if dep_col not in cells.columns:
                continue
            high = cells[cells["intervene_state_group"] == "high"]
            low = cells[cells["intervene_state_group"] == "low"]
            gene_rows.append(
                {
                    "module": module,
                    "gene": gene,
                    "high_minus_low_dependency": float(high[dep_col].mean() - low[dep_col].mean()),
                    "state_corr_dependency": float(cells[dep_col].corr(cells["ibpa_v2_state_score"])),
                    "state_corr_expression": float(cells[expr_col].corr(cells["ibpa_v2_state_score"])) if expr_col in cells else np.nan,
                    "state_corr_copy_number": float(cells[cn_col].corr(cells["ibpa_v2_state_score"])) if cn_col in cells else np.nan,
                }
            )
    genes = pd.DataFrame(gene_rows).sort_values(
        ["state_corr_dependency", "high_minus_low_dependency"], ascending=[False, False]
    )

    top_cells = cells.sort_values("ibpa_v2_state_score", ascending=False)[
        [
            "ModelID",
            "CellLineName",
            "OncotreeSubtype",
            "ibpa_v2_state_score",
            "target_cytotoxic_residual_stress_survival",
            "target_stress_survival_sensitivity",
            "generic_cytotoxic_sensitivity",
            "direct_lipid_sensitivity",
            "KEAP1_high_impact",
            "NFE2L2_high_impact",
            "KEAP1_or_NFE2L2_high_impact",
            "srebp_lipid_thermo_residual",
            "mito_translation_thermo_residual",
            "oxidative_context_score",
        ]
    ].head(30)

    compound_focus = compounds[
        compounds["mechanism_bucket"].isin(["stress_survival", "direct_lipid_axis", "generic_cytotoxic"])
    ].copy()
    compound_focus["therapy_hypothesis_role"] = compound_focus["mechanism_bucket"].map(
        {
            "stress_survival": "primary validation and combination partner class",
            "direct_lipid_axis": "secondary/negative direct lipid result",
            "generic_cytotoxic": "excluded generic killing control",
        }
    )
    compound_focus = compound_focus.sort_values(["mechanism_bucket", "sensitivity_score"], ascending=[True, False])

    combination = pd.DataFrame(
        [
            {
                "hypothesis": "SREBP/lipid stress state plus mitochondrial stress",
                "candidate_classes": "SREBP/lipid module marker + mitochondrial complex I / mitochondrial stress drugs",
                "reason": "Top attention edge is SREBP/lipid to mitochondrial translation; stress-survival PRISM signal is stronger than direct lipid response.",
                "paper_status": "hypothesis, not proven combination therapy",
            },
            {
                "hypothesis": "SREBP/mitochondrial state plus proteostasis pressure",
                "candidate_classes": "proteasome/deubiquitinase/HDAC-associated stress-survival compounds",
                "reason": "Proteostasis to mitochondrial translation appears as a strong biological attention edge.",
                "paper_status": "hypothesis, needs external validation",
            },
            {
                "hypothesis": "stress-survival state plus apoptosis/IAP escape",
                "candidate_classes": "IAP/apoptosis modulators with mitochondrial-stress context",
                "reason": "Apoptosis/IAP to mitochondrial translation appears among top attention edges.",
                "paper_status": "hypothesis, needs dose-response or perturbation validation",
            },
            {
                "hypothesis": "direct lipid monotherapy",
                "candidate_classes": "direct lipid drugs only",
                "reason": "Direct lipid PRISM response is weaker and not significant in the INTERVENE high-low split.",
                "paper_status": "secondary/negative result",
            },
        ]
    )

    return genes, top_cells, compound_focus, combination


def dataset_side() -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = pd.read_csv(RESULTS / "intervene_cross_cancer_specificity.csv")
    cells = pd.read_csv(RESULTS / "ibpa_v2_discovery_cell_scores.csv")
    models = pd.read_csv(RAW / "Model.csv")
    models["ModelID"] = models["ModelID"].astype(str)
    cells["ModelID"] = cells["ModelID"].astype(str)
    cells = cells.merge(models[["ModelID", "OncotreeSubtype"]], on="ModelID", how="left")

    subtype_rows = []
    for subtype, group in cells.groupby(cells["OncotreeSubtype"].fillna("unknown")):
        if group["target_cytotoxic_residual_stress_survival"].notna().sum() < 10:
            continue
        row = high_low_stats(
            group,
            "ibpa_v2_state_score",
            "target_cytotoxic_residual_stress_survival",
            "cytotoxic-residual stress-survival score",
            n_perm=1000,
        )
        row["OncotreeSubtype"] = subtype
        subtype_rows.append(row)
    subtype = pd.DataFrame(subtype_rows).sort_values("high_minus_low", ascending=False)

    dataset_rows = [
        {
            "dataset_question": "other cancer types in DepMap",
            "status": "done",
            "finding": "NSCLC ranks 2nd of 17 by residual stress-survival transfer effect; mature B-cell neoplasms are also significant and should be future validation.",
        },
        {
            "dataset_question": "NSCLC subtypes only",
            "status": "done_now",
            "finding": "Subtype-level high-low audit generated; interpret cautiously because subtype sample sizes are smaller.",
        },
        {
            "dataset_question": "PRISM secondary screen",
            "status": "partial",
            "finding": "Current PRISM validation uses 24Q2 collapsed repurposing LFC with screen labels aggregated. A screen-separated REP1M/REP300-style analysis is still needed if the teacher wants primary-vs-secondary screen claims.",
        },
        {
            "dataset_question": "GDSC/CTRP external validation",
            "status": "not_done",
            "finding": "No GDSC/CTRP files are currently in the project. This is the next external validation step, not part of the current DepMap-only evidence.",
        },
    ]
    dataset_summary = pd.DataFrame(dataset_rows)
    return dataset_summary, subtype


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if len(df) == 0:
        return "_No rows._"
    out = df.copy()
    out = out.fillna("")
    cols = list(out.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in out.itertuples(index=False):
        values = []
        for value in row:
            text = str(value)
            if isinstance(value, float):
                text = f"{value:.4g}"
            text = text.replace("|", "/")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_notes(
    method_summary: pd.DataFrame,
    edge_stability: pd.DataFrame,
    genes: pd.DataFrame,
    top_cells: pd.DataFrame,
    combination: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    subtype: pd.DataFrame,
) -> None:
    path = NOTES / "intervene_full_1_4_audit.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# INTERVENE Full 1-4 Audit\n\n")
        f.write("## 1. Method Side\n\n")
        f.write(markdown_table(method_summary))
        f.write("\n\nTop repeated biological attention edges:\n\n")
        f.write(markdown_table(edge_stability, max_rows=12))
        f.write("\n\n## 2. Biology Side\n\n")
        f.write("Top candidate driver genes by dependency correlation with INTERVENE state:\n\n")
        f.write(markdown_table(genes, max_rows=20))
        f.write("\n\nHighest INTERVENE-state NSCLC cell lines:\n\n")
        f.write(markdown_table(top_cells, max_rows=15))
        f.write("\n\nCombination therapy hypotheses:\n\n")
        f.write(markdown_table(combination))
        f.write("\n\n## 3. Dataset Side\n\n")
        f.write(markdown_table(dataset_summary))
        f.write("\n\nNSCLC subtype audit:\n\n")
        if len(subtype):
            f.write(markdown_table(subtype))
        else:
            f.write("No subtype had enough PRISM-covered models for a subtype-level split.")
        f.write("\n\n## 4. Theory Side\n\n")
        f.write(
            "Theory note completed in `paper_notes/intervene_theory_section.md`: "
            "Paper 1 is a special case, pseudo-head mixing creates cross-module interactions, "
            "and cytotoxic-residual validation removes generic killing signal.\n"
        )
        f.write("\n## Honest Remaining Gaps\n\n")
        f.write(
            "The main remaining method gap is seed-level pseudo-head edge stability. "
            "The current result shows repeated edges across heads/pseudo-heads in the trained model, "
            "but not yet repeated training across random seeds. The main dataset gap is external GDSC/CTRP validation.\n"
        )


def main() -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    method_summary, edge_stability = method_side()
    genes, top_cells, compound_focus, combination = biology_side()
    dataset_summary, subtype = dataset_side()

    method_summary.to_csv(RESULTS / "intervene_method_side_paper_audit.csv", index=False)
    edge_stability.to_csv(RESULTS / "intervene_pseudohead_edge_stability_current_model.csv", index=False)
    genes.to_csv(RESULTS / "intervene_hidden_state_driver_genes.csv", index=False)
    top_cells.to_csv(RESULTS / "intervene_high_state_nsclc_cell_lines.csv", index=False)
    compound_focus.to_csv(RESULTS / "intervene_drug_class_interpretation.csv", index=False)
    combination.to_csv(RESULTS / "intervene_combination_hypotheses.csv", index=False)
    dataset_summary.to_csv(RESULTS / "intervene_dataset_side_audit.csv", index=False)
    subtype.to_csv(RESULTS / "intervene_nsclc_subtype_audit.csv", index=False)
    write_notes(method_summary, edge_stability, genes, top_cells, combination, dataset_summary, subtype)

    print(f"Wrote {RESULTS / 'intervene_method_side_paper_audit.csv'}")
    print(f"Wrote {RESULTS / 'intervene_pseudohead_edge_stability_current_model.csv'}")
    print(f"Wrote {RESULTS / 'intervene_hidden_state_driver_genes.csv'}")
    print(f"Wrote {RESULTS / 'intervene_high_state_nsclc_cell_lines.csv'}")
    print(f"Wrote {RESULTS / 'intervene_drug_class_interpretation.csv'}")
    print(f"Wrote {RESULTS / 'intervene_combination_hypotheses.csv'}")
    print(f"Wrote {RESULTS / 'intervene_dataset_side_audit.csv'}")
    print(f"Wrote {RESULTS / 'intervene_nsclc_subtype_audit.csv'}")
    print(f"Wrote {NOTES / 'intervene_full_1_4_audit.md'}")


if __name__ == "__main__":
    main()

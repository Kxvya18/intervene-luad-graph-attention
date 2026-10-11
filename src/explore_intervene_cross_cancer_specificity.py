from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from classify_prism_mechanisms import assign_bucket
from interleaved_biophysics_attention_full import (
    FEATURES,
    PAPER_TOKENS,
    discovery_weak_prior,
    high_low_stats,
    module_specs,
    read_wide_subset,
    zscore,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PRISM = RAW / "PRISM"
RESULTS = ROOT / "results"

MIN_LINEAGE_MODELS = 25
MIN_LINEAGE_PRISM_MODELS = 20
CHUNKSIZE = 500_000


def clean_broad_id(value: object) -> str:
    return str(value).replace(" - QC Failure", "").strip()


def load_model_lineages() -> pd.DataFrame:
    models = pd.read_csv(RAW / "Model.csv", usecols=["ModelID", "OncotreePrimaryDisease", "OncotreeLineage"])
    models["ModelID"] = models["ModelID"].astype(str)
    models["lineage_label"] = models["OncotreePrimaryDisease"].fillna(models["OncotreeLineage"])
    models = models.dropna(subset=["lineage_label"]).copy()
    counts = models["lineage_label"].value_counts()
    keep = counts[counts >= MIN_LINEAGE_MODELS].index
    return models[models["lineage_label"].isin(keep)].copy()


def build_all_cancer_feature_tensor(models: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    specs = [spec for spec in module_specs() if spec.token in PAPER_TOKENS]
    model_ids = set(models["ModelID"].astype(str))
    genes = sorted(set(g for spec in specs for g in spec.genes))

    dep = read_wide_subset(RAW / "CRISPRGeneEffect.csv", model_ids, genes, "dep")
    dep_cols = [c for c in dep.columns if c != "ModelID"]
    dep[dep_cols] = -dep[dep_cols].apply(pd.to_numeric, errors="coerce")
    expr = read_wide_subset(RAW / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", model_ids, genes, "expr")
    cn = read_wide_subset(RAW / "OmicsCNGene.csv", model_ids, genes, "cn")

    frame = models[["ModelID", "lineage_label", "OncotreeLineage", "OncotreePrimaryDisease"]].copy()
    for table in [dep, expr, cn]:
        frame = frame.merge(table, on="ModelID", how="left")

    residual = pd.read_csv(RESULTS / "selected_residual_vulnerability_gene_scores.csv")
    residual = residual.drop_duplicates("gene_symbol").set_index("gene_symbol")

    all_tokens = [spec.token for spec in specs]
    for spec in specs:
        for measure, feature_name in {"dep": "dependency", "expr": "expression", "cn": "copy_number"}.items():
            cols = [f"{g}_{measure}" for g in spec.genes if f"{g}_{measure}" in frame.columns]
            frame[f"{spec.token}_{feature_name}"] = frame[cols].mean(axis=1) if cols else np.nan

        genes_present = [g for g in spec.genes if g in residual.index]
        if genes_present:
            weights = residual.loc[genes_present, "thermo_residual_score"].astype(float)
            dep_cols = [f"{g}_dep" for g in genes_present if f"{g}_dep" in frame.columns]
            if dep_cols and float(weights.abs().sum()) != 0:
                values = frame[dep_cols].apply(zscore)
                w = weights.loc[[c.replace("_dep", "") for c in dep_cols]].abs()
                w.index = dep_cols
                frame[f"{spec.token}_thermo_residual"] = values.mul(w / w.sum(), axis=1).sum(axis=1)
            else:
                frame[f"{spec.token}_thermo_residual"] = float(weights.mean())
            frame[f"{spec.token}_selectivity"] = float(residual.loc[genes_present, "nsclc_selectivity"].astype(float).mean())
        else:
            frame[f"{spec.token}_thermo_residual"] = np.nan
            frame[f"{spec.token}_selectivity"] = np.nan

        frame[f"{spec.token}_context"] = frame[f"{spec.token}_selectivity"]
        frame[f"{spec.token}_generic_cytotoxic_prism_control"] = np.nan
        frame[f"{spec.token}_direct_lipid_prism_control"] = np.nan

    x = np.zeros((len(frame), len(all_tokens), len(FEATURES)), dtype=np.float32)
    for ti, token in enumerate(all_tokens):
        for fi, feature in enumerate(FEATURES):
            col = f"{token}_{feature}"
            values = frame[col] if col in frame.columns else pd.Series(np.nan, index=frame.index)
            values = values.fillna(values.median() if values.notna().any() else 0.0)
            x[:, ti, fi] = zscore(values).to_numpy(dtype=np.float32)

    frame["intervene_transfer_state"] = discovery_weak_prior(x, all_tokens, FEATURES)
    frame["srebp_mito_dependency_coupling"] = (
        x[:, all_tokens.index("srebp_lipid"), FEATURES.index("dependency")]
        * x[:, all_tokens.index("mito_translation"), FEATURES.index("dependency")]
    )
    frame["oxidative_context_score"] = x[:, all_tokens.index("oxidative_stress"), FEATURES.index("context")]
    return frame, x, all_tokens, FEATURES


def load_bucketed_prism_targets(model_ids: set[str]) -> pd.DataFrame:
    cell_meta = pd.read_csv(PRISM / "Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv", usecols=["row_id", "depmap_id"])
    cell_meta["depmap_id"] = cell_meta["depmap_id"].astype(str)
    cell_meta = cell_meta[cell_meta["depmap_id"].isin(model_ids)].copy()
    row_to_model = dict(zip(cell_meta["row_id"], cell_meta["depmap_id"]))
    keep_rows = set(row_to_model)

    treatment = pd.read_csv(PRISM / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
    treatment["broad_id_clean"] = treatment["broad_id"].map(clean_broad_id)
    treatment = treatment[treatment["perturbation_type"] == "trt_cp"].copy()
    treatment = treatment.rename(columns={"name": "drug_name"})

    compound_list = PRISM / "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv"
    if compound_list.exists():
        compound = pd.read_csv(compound_list)
        compound["broad_id_clean"] = compound["IDs"].astype(str).str.replace("BRD:", "", regex=False)
        compound = compound.rename(
            columns={
                "Drug.Name": "compound_list_drug_name",
                "MOA": "moa",
                "Synonyms": "synonyms",
            }
        )
        compound = compound[["broad_id_clean", "compound_list_drug_name", "repurposing_target", "moa", "synonyms"]]
        treatment = treatment.merge(compound.drop_duplicates("broad_id_clean"), on="broad_id_clean", how="left")
        treatment["drug_name"] = treatment["drug_name"].fillna(treatment["compound_list_drug_name"])

    for col in ["drug_name", "repurposing_target", "moa", "synonyms"]:
        if col not in treatment:
            treatment[col] = ""
    treatment["mechanism_bucket"] = treatment.apply(assign_bucket, axis=1)
    bucket_map = treatment.drop_duplicates("broad_id_clean").set_index("broad_id_clean")["mechanism_bucket"].to_dict()

    rows = []
    usecols = ["row_id", "broad_id", "LFC"]
    for chunk in pd.read_csv(PRISM / "Repurposing_Public_24Q2_LFC_COLLAPSED.csv", usecols=usecols, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["row_id"].isin(keep_rows)].copy()
        if len(chunk) == 0:
            continue
        chunk["ModelID"] = chunk["row_id"].map(row_to_model)
        chunk["broad_id_clean"] = chunk["broad_id"].map(clean_broad_id)
        chunk["mechanism_bucket"] = chunk["broad_id_clean"].map(bucket_map).fillna("unclassified")
        chunk["sensitivity"] = -pd.to_numeric(chunk["LFC"], errors="coerce")
        rows.append(chunk[["ModelID", "mechanism_bucket", "sensitivity"]])

    if not rows:
        return pd.DataFrame({"ModelID": sorted(model_ids)})

    prism = pd.concat(rows, ignore_index=True)
    target = pd.DataFrame({"ModelID": sorted(model_ids)})
    for bucket, col in [
        ("stress_survival", "target_stress_survival_sensitivity"),
        ("generic_cytotoxic", "generic_cytotoxic_sensitivity"),
        ("direct_lipid_axis", "direct_lipid_sensitivity"),
    ]:
        agg = prism[prism["mechanism_bucket"] == bucket].groupby("ModelID")["sensitivity"].mean().rename(col)
        target = target.merge(agg.reset_index(), on="ModelID", how="left")
    target["target_specific_stress_survival"] = (
        zscore(target["target_stress_survival_sensitivity"]) - 0.35 * zscore(target["generic_cytotoxic_sensitivity"])
    )
    valid = target[["target_stress_survival_sensitivity", "generic_cytotoxic_sensitivity"]].dropna()
    if len(valid) >= 10:
        slope, intercept = np.polyfit(
            valid["generic_cytotoxic_sensitivity"].to_numpy(),
            valid["target_stress_survival_sensitivity"].to_numpy(),
            deg=1,
        )
        residual = target["target_stress_survival_sensitivity"] - (
            intercept + slope * target["generic_cytotoxic_sensitivity"]
        )
        target["target_cytotoxic_residual_stress_survival"] = zscore(residual)
    else:
        target["target_cytotoxic_residual_stress_survival"] = np.nan
    return target


def summarize_lineages(frame: pd.DataFrame) -> pd.DataFrame:
    tests = [
        ("target_stress_survival_sensitivity", "PRISM stress-survival sensitivity"),
        ("target_cytotoxic_residual_stress_survival", "cytotoxic-residual stress-survival score"),
        ("target_specific_stress_survival", "stress-survival minus cytotoxic score"),
        ("generic_cytotoxic_sensitivity", "PRISM generic cytotoxic sensitivity"),
        ("direct_lipid_sensitivity", "PRISM direct lipid sensitivity"),
        ("srebp_mito_dependency_coupling", "SREBP x mitochondrial dependency coupling"),
    ]
    rows = []
    for lineage, group in frame.groupby("lineage_label"):
        if group["target_stress_survival_sensitivity"].notna().sum() < MIN_LINEAGE_PRISM_MODELS:
            continue
        for value_col, label in tests:
            stats = high_low_stats(group, "intervene_transfer_state", value_col, label, n_perm=1000)
            stats["lineage_label"] = lineage
            rows.append(stats)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    wide = out.pivot_table(
        index="lineage_label",
        columns="test",
        values=["n", "high_minus_low", "effect_size", "perm_p_two_sided"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{test}" for metric, test in wide.columns]
    wide = wide.reset_index()
    rank_col = "high_minus_low__cytotoxic-residual stress-survival score"
    if rank_col in wide:
        wide = wide.sort_values(rank_col, ascending=False)
    return wide


def write_summary(lineage_summary: pd.DataFrame, detailed: pd.DataFrame) -> None:
    path = RESULTS / "intervene_cross_cancer_specificity_summary.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("INTERVENE cross-cancer specificity audit\n")
        f.write("=======================================\n\n")
        f.write(
            "Question: is the SREBP/mitochondrial/oxidative stress-survival state NSCLC-specific, "
            "or is it a generic pan-cancer cytotoxic signal?\n\n"
        )
        f.write(
            "Method: the NSCLC-derived INTERVENE transfer state was computed in every cancer lineage "
            "with enough DepMap and PRISM coverage. Each lineage was tested by high-vs-low state split "
            "against PRISM stress-survival sensitivity, cytotoxic-residual stress-survival, direct lipid "
            "response, and generic cytotoxic response.\n\n"
        )
        if len(lineage_summary):
            f.write("Top lineages by cytotoxic-residual stress-survival effect\n")
            f.write("--------------------------------------------------------\n")
            cols = [c for c in lineage_summary.columns if c == "lineage_label" or "cytotoxic-residual" in c or "generic cytotoxic" in c or "direct lipid" in c]
            f.write(lineage_summary[cols].head(20).to_string(index=False))
            f.write("\n\n")
            nsclc = lineage_summary[lineage_summary["lineage_label"] == "Non-Small Cell Lung Cancer"]
            if len(nsclc):
                rank = int(lineage_summary.reset_index(drop=True).index[lineage_summary["lineage_label"] == "Non-Small Cell Lung Cancer"][0]) + 1
                f.write(f"NSCLC rank by residual stress-survival effect: {rank} of {len(lineage_summary)} tested lineages.\n\n")
        else:
            f.write("No lineage passed the PRISM coverage threshold.\n\n")
        f.write("Interpretation rule\n")
        f.write("-------------------\n")
        f.write(
            "Keep NSCLC as the main disease context if it has a positive residual effect and generic cytotoxic/direct lipid "
            "controls do not dominate. Treat other positive lineages as future validation contexts, not as the primary claim.\n"
        )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    models = load_model_lineages()
    features, _, _, _ = build_all_cancer_feature_tensor(models)
    targets = load_bucketed_prism_targets(set(features["ModelID"].astype(str)))
    frame = features.merge(targets, on="ModelID", how="left")
    detailed_rows = []
    for lineage, group in frame.groupby("lineage_label"):
        if group["target_stress_survival_sensitivity"].notna().sum() < MIN_LINEAGE_PRISM_MODELS:
            continue
        for value_col, label in [
            ("target_stress_survival_sensitivity", "PRISM stress-survival sensitivity"),
            ("target_cytotoxic_residual_stress_survival", "cytotoxic-residual stress-survival score"),
            ("target_specific_stress_survival", "stress-survival minus cytotoxic score"),
            ("generic_cytotoxic_sensitivity", "PRISM generic cytotoxic sensitivity"),
            ("direct_lipid_sensitivity", "PRISM direct lipid sensitivity"),
            ("srebp_mito_dependency_coupling", "SREBP x mitochondrial dependency coupling"),
        ]:
            row = high_low_stats(group, "intervene_transfer_state", value_col, label, n_perm=1000)
            row["lineage_label"] = lineage
            detailed_rows.append(row)
    detailed = pd.DataFrame(detailed_rows)
    summary = summarize_lineages(frame)
    frame.to_csv(RESULTS / "intervene_cross_cancer_cell_scores.csv", index=False)
    detailed.to_csv(RESULTS / "intervene_cross_cancer_detailed_tests.csv", index=False)
    summary.to_csv(RESULTS / "intervene_cross_cancer_specificity.csv", index=False)
    write_summary(summary, detailed)
    print(f"Wrote {RESULTS / 'intervene_cross_cancer_cell_scores.csv'}")
    print(f"Wrote {RESULTS / 'intervene_cross_cancer_detailed_tests.csv'}")
    print(f"Wrote {RESULTS / 'intervene_cross_cancer_specificity.csv'}")
    print(f"Wrote {RESULTS / 'intervene_cross_cancer_specificity_summary.txt'}")


if __name__ == "__main__":
    main()

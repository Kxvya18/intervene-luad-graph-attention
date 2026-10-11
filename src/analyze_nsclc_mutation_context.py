from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

PRIMARY_DISEASE = "Non-Small Cell Lung Cancer"
MUTATION_GENES = ["KRAS", "NFE2L2", "KEAP1", "STK11", "EGFR", "TP53"]
SREBP_GENES = ["SREBF1", "SCAP", "SCD", "MBTPS1", "MBTPS2"]
CANDIDATE_GENES = [
    "NFE2L2",
    "KRAS",
    "UBIAD1",
    "UXS1",
    "SMARCA2",
    "TRPM7",
    "CFLAR",
    "PTDSS1",
    "PGD",
    "SREBF1",
    "SCAP",
    "SCD",
    "MBTPS1",
    "MBTPS2",
]


def gene_column_map(columns: list[str]) -> dict[str, str]:
    mapping = {}
    for col in columns:
        if " (" in col and col.endswith(")"):
            symbol = col.split(" (", 1)[0]
            mapping[symbol] = col
    return mapping


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(RAW / "Model.csv")
    models["ModelID"] = models["ModelID"].astype(str)
    gene_effect = pd.read_csv(RAW / "CRISPRGeneEffect.csv")
    gene_effect = gene_effect.rename(columns={gene_effect.columns[0]: "ModelID"})
    gene_effect["ModelID"] = gene_effect["ModelID"].astype(str)
    return models, gene_effect, pd.DataFrame()


def load_relevant_mutations(nsclc_ids: set[str]) -> pd.DataFrame:
    usecols = [
        "ModelID",
        "HugoSymbol",
        "ProteinChange",
        "VariantType",
        "MolecularConsequence",
        "VepImpact",
        "OncogeneHighImpact",
        "TumorSuppressorHighImpact",
        "LikelyLoF",
        "Hotspot",
        "HessDriver",
        "VariantInfo",
    ]
    chunks = []
    for chunk in pd.read_csv(RAW / "OmicsSomaticMutations.csv", usecols=usecols, chunksize=100_000):
        chunk["ModelID"] = chunk["ModelID"].astype(str)
        filtered = chunk[
            chunk["ModelID"].isin(nsclc_ids) & chunk["HugoSymbol"].isin(MUTATION_GENES)
        ].copy()
        if len(filtered):
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def build_mutation_matrix(nsclc_models: pd.DataFrame, mutations: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame({"ModelID": nsclc_models["ModelID"].astype(str)})
    for gene in MUTATION_GENES:
        gene_mut = mutations[mutations["HugoSymbol"] == gene]
        any_ids = set(gene_mut["ModelID"].astype(str))
        high_impact_ids = set(
            gene_mut[
                gene_mut[
                    [
                        "OncogeneHighImpact",
                        "TumorSuppressorHighImpact",
                        "LikelyLoF",
                        "Hotspot",
                        "HessDriver",
                    ]
                ]
                .fillna(False)
                .astype(bool)
                .any(axis=1)
            ]["ModelID"].astype(str)
        )
        matrix[f"{gene}_mut"] = matrix["ModelID"].isin(any_ids)
        matrix[f"{gene}_high_impact"] = matrix["ModelID"].isin(high_impact_ids)
    return matrix


def dependency_frame(gene_effect: pd.DataFrame, nsclc_ids: set[str]) -> pd.DataFrame:
    mapping = gene_column_map(list(gene_effect.columns))
    needed_symbols = sorted(set(SREBP_GENES + CANDIDATE_GENES + MUTATION_GENES))
    cols = ["ModelID"] + [mapping[s] for s in needed_symbols if s in mapping]
    dep = gene_effect.loc[gene_effect["ModelID"].isin(nsclc_ids), cols].copy()
    rename = {mapping[s]: s for s in needed_symbols if s in mapping}
    dep = dep.rename(columns=rename).set_index("ModelID")
    dep = -dep
    dep["SREBP_seed_dependency_activity"] = dep[[g for g in SREBP_GENES if g in dep]].mean(axis=1)
    return dep.reset_index()


def compare_groups(dep: pd.DataFrame, mut_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = dep.merge(mut_matrix, on="ModelID", how="left")
    rows = []
    gene_rows = []
    for marker in MUTATION_GENES:
        for flag in [f"{marker}_mut", f"{marker}_high_impact"]:
            if flag not in merged:
                continue
            mask = merged[flag].fillna(False).astype(bool)
            n_mut = int(mask.sum())
            n_wt = int((~mask).sum())
            if n_mut < 3 or n_wt < 3:
                rows.append(
                    {
                        "marker": marker,
                        "flag": flag,
                        "n_mut": n_mut,
                        "n_wt": n_wt,
                        "mean_srebp_activity_mut": np.nan,
                        "mean_srebp_activity_wt": np.nan,
                        "difference_mut_minus_wt": np.nan,
                        "note": "insufficient group size",
                    }
                )
                continue
            mut_mean = float(merged.loc[mask, "SREBP_seed_dependency_activity"].mean())
            wt_mean = float(merged.loc[~mask, "SREBP_seed_dependency_activity"].mean())
            rows.append(
                {
                    "marker": marker,
                    "flag": flag,
                    "n_mut": n_mut,
                    "n_wt": n_wt,
                    "mean_srebp_activity_mut": mut_mean,
                    "mean_srebp_activity_wt": wt_mean,
                    "difference_mut_minus_wt": mut_mean - wt_mean,
                    "note": "ok",
                }
            )
            for gene in CANDIDATE_GENES:
                if gene not in merged:
                    continue
                gene_rows.append(
                    {
                        "marker": marker,
                        "flag": flag,
                        "candidate_gene": gene,
                        "n_mut": n_mut,
                        "n_wt": n_wt,
                        "mean_dependency_mut": float(merged.loc[mask, gene].mean()),
                        "mean_dependency_wt": float(merged.loc[~mask, gene].mean()),
                        "difference_mut_minus_wt": float(
                            merged.loc[mask, gene].mean() - merged.loc[~mask, gene].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(gene_rows)


def mutation_details(mutations: pd.DataFrame) -> pd.DataFrame:
    if len(mutations) == 0:
        return mutations
    cols = [
        "ModelID",
        "HugoSymbol",
        "ProteinChange",
        "VariantType",
        "MolecularConsequence",
        "VepImpact",
        "OncogeneHighImpact",
        "TumorSuppressorHighImpact",
        "LikelyLoF",
        "Hotspot",
        "HessDriver",
    ]
    return mutations[cols].sort_values(["HugoSymbol", "ModelID"])


def write_summary(group_stats: pd.DataFrame, candidate_stats: pd.DataFrame, details: pd.DataFrame) -> None:
    summary_path = RESULTS / "nsclc_mutation_context_srebp_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("NSCLC mutation context for SREBP/lipid axis\n")
        f.write("==========================================\n\n")
        f.write("SREBP seed dependency activity by mutation group\n")
        f.write("-----------------------------------------------\n")
        f.write(group_stats.sort_values("difference_mut_minus_wt", ascending=False).to_string(index=False))
        f.write("\n\nTop candidate dependency shifts by mutation group\n")
        f.write("-------------------------------------------------\n")
        if len(candidate_stats):
            f.write(
                candidate_stats.sort_values("difference_mut_minus_wt", ascending=False)
                .head(40)
                .to_string(index=False)
            )
        else:
            f.write("No candidate group comparisons had enough samples.")
        f.write("\n\nMutation counts and examples\n")
        f.write("----------------------------\n")
        if len(details):
            counts = details.groupby("HugoSymbol")["ModelID"].nunique().sort_values(ascending=False)
            f.write(counts.to_string())
            f.write("\n\n")
            f.write(details.head(80).to_string(index=False))
        else:
            f.write("No relevant NSCLC mutations found.")
        f.write("\n")
    print(f"Wrote {summary_path}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    models, gene_effect, _ = load_inputs()
    screened_ids = set(gene_effect["ModelID"].astype(str))
    nsclc_models = models[
        (models["OncotreePrimaryDisease"] == PRIMARY_DISEASE)
        & (models["ModelID"].isin(screened_ids))
    ].copy()
    nsclc_ids = set(nsclc_models["ModelID"].astype(str))

    mutations = load_relevant_mutations(nsclc_ids)
    mut_matrix = build_mutation_matrix(nsclc_models, mutations)
    dep = dependency_frame(gene_effect, nsclc_ids)
    group_stats, candidate_stats = compare_groups(dep, mut_matrix)
    details = mutation_details(mutations)

    mut_matrix.to_csv(RESULTS / "nsclc_driver_mutation_matrix.csv", index=False)
    group_stats.to_csv(RESULTS / "nsclc_srebp_activity_by_mutation.csv", index=False)
    candidate_stats.to_csv(RESULTS / "nsclc_candidate_dependency_by_mutation.csv", index=False)
    details.to_csv(RESULTS / "nsclc_relevant_mutation_details.csv", index=False)
    write_summary(group_stats, candidate_stats, details)

    print(f"NSCLC screened models: {len(nsclc_models)}")
    print(f"Relevant mutation rows: {len(mutations)}")
    print(f"Wrote {RESULTS / 'nsclc_srebp_activity_by_mutation.csv'}")
    print(f"Wrote {RESULTS / 'nsclc_candidate_dependency_by_mutation.csv'}")


if __name__ == "__main__":
    main()

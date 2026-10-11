from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

PRIMARY_DISEASE = "Non-Small Cell Lung Cancer"
TOP_GENES = 1500
TOP_NEIGHBORS = 10
MIN_CORR = 0.35
MIN_VALID_FRACTION = 0.90


GENE_RE = re.compile(r"^(?P<symbol>.+?) \((?P<entrez>\d+)\)$")


def parse_gene_column(col: str) -> tuple[str, str]:
    match = GENE_RE.match(col)
    if not match:
        return col, ""
    return match.group("symbol"), match.group("entrez")


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def load_nsclc_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(RAW / "Model.csv")
    nsclc_models = models[models["OncotreePrimaryDisease"] == PRIMARY_DISEASE].copy()
    nsclc_ids = set(nsclc_models["ModelID"].astype(str))

    gene_effect = pd.read_csv(RAW / "CRISPRGeneEffect.csv")
    first_col = gene_effect.columns[0]
    gene_effect = gene_effect.rename(columns={first_col: "ModelID"})
    gene_effect["ModelID"] = gene_effect["ModelID"].astype(str)
    gene_effect = gene_effect[gene_effect["ModelID"].isin(nsclc_ids)].copy()
    gene_effect = gene_effect.set_index("ModelID")
    nsclc_models = nsclc_models[nsclc_models["ModelID"].astype(str).isin(gene_effect.index)].copy()

    return gene_effect, nsclc_models


def choose_genes(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_fraction = matrix.notna().mean(axis=0)
    matrix = matrix.loc[:, valid_fraction >= MIN_VALID_FRACTION].copy()

    means = matrix.mean(axis=0, skipna=True)
    stds = matrix.std(axis=0, skipna=True)
    mins = matrix.min(axis=0, skipna=True)

    stats = pd.DataFrame(
        {
            "gene_column": matrix.columns,
            "mean_gene_effect": means.values,
            "std_gene_effect": stds.values,
            "min_gene_effect": mins.values,
            "valid_fraction": valid_fraction.loc[matrix.columns].values,
        }
    )
    parsed = stats["gene_column"].map(parse_gene_column)
    stats["gene_symbol"] = [p[0] for p in parsed]
    stats["entrez_id"] = [p[1] for p in parsed]

    stats["dependency_strength"] = -stats["mean_gene_effect"]
    stats["selectivity"] = stats["std_gene_effect"]
    stats["screen_score"] = (
        zscore(stats["dependency_strength"])
        + 0.75 * zscore(stats["selectivity"])
        + 0.25 * zscore(-stats["min_gene_effect"])
    )
    stats = stats.sort_values("screen_score", ascending=False).head(TOP_GENES).copy()

    selected = matrix.loc[:, stats["gene_column"]].copy()
    selected = selected.apply(lambda col: col.fillna(col.mean()), axis=0)
    return selected, stats


def build_edges(selected: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = selected.to_numpy(dtype=float)
    values = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    values = values / std

    corr = (values.T @ values) / (values.shape[0] - 1)
    np.fill_diagonal(corr, -np.inf)

    gene_columns = selected.columns.to_numpy()
    edges: dict[tuple[int, int], float] = {}
    for i in range(corr.shape[0]):
        top_idx = np.argpartition(corr[i], -TOP_NEIGHBORS)[-TOP_NEIGHBORS:]
        for j in top_idx:
            weight = float(corr[i, j])
            if weight < MIN_CORR:
                continue
            a, b = sorted((i, int(j)))
            if a == b:
                continue
            edges[(a, b)] = max(edges.get((a, b), -np.inf), weight)

    edge_rows = [
        {
            "source": gene_columns[i],
            "target": gene_columns[j],
            "weight": weight,
        }
        for (i, j), weight in edges.items()
    ]
    edge_df = pd.DataFrame(edge_rows).sort_values("weight", ascending=False)

    node_df = stats.copy().set_index("gene_column")
    node_df["degree"] = 0
    node_df["weighted_degree"] = 0.0
    for _, row in edge_df.iterrows():
        s = row["source"]
        t = row["target"]
        w = float(row["weight"])
        node_df.loc[s, "degree"] += 1
        node_df.loc[t, "degree"] += 1
        node_df.loc[s, "weighted_degree"] += w
        node_df.loc[t, "weighted_degree"] += w

    node_df["boltzmann_beta1_weight"] = (node_df["degree"] + 1.0) / (node_df["degree"] + 1.0).sum()
    node_df["thermodynamic_energy"] = -np.log(node_df["degree"] + 1.0)
    node_df["baseline_thermo_importance"] = (
        zscore(node_df["weighted_degree"])
        + zscore(node_df["dependency_strength"])
        + 0.5 * zscore(node_df["selectivity"])
    )
    node_df["hidden_nonhub_score"] = (
        zscore(node_df["dependency_strength"])
        + zscore(node_df["selectivity"])
        - zscore(node_df["degree"])
    )

    node_df = node_df.reset_index().sort_values("baseline_thermo_importance", ascending=False)
    return node_df, edge_df


def write_summary(node_df: pd.DataFrame, edge_df: pd.DataFrame, nsclc_models: pd.DataFrame) -> None:
    summary = RESULTS / "nsclc_network_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("NSCLC DepMap co-dependency network\n")
        f.write("==================================\n\n")
        f.write(f"Primary disease: {PRIMARY_DISEASE}\n")
        f.write(f"NSCLC models: {len(nsclc_models)}\n")
        f.write(f"Selected genes: {len(node_df)}\n")
        f.write(f"Edges: {len(edge_df)}\n")
        f.write(f"Top neighbors per gene: {TOP_NEIGHBORS}\n")
        f.write(f"Minimum positive correlation: {MIN_CORR}\n\n")
        f.write("Top genes by baseline thermodynamic importance\n")
        f.write("----------------------------------------------\n")
        cols = [
            "gene_symbol",
            "entrez_id",
            "degree",
            "weighted_degree",
            "dependency_strength",
            "selectivity",
            "baseline_thermo_importance",
        ]
        f.write(node_df[cols].head(30).to_string(index=False))
        f.write("\n\nTop hidden non-hub candidates\n")
        f.write("-----------------------------\n")
        hidden = node_df.sort_values("hidden_nonhub_score", ascending=False)
        cols = [
            "gene_symbol",
            "entrez_id",
            "degree",
            "weighted_degree",
            "dependency_strength",
            "selectivity",
            "hidden_nonhub_score",
        ]
        f.write(hidden[cols].head(30).to_string(index=False))
        f.write("\n")
    print(f"Wrote {summary}")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    matrix, nsclc_models = load_nsclc_matrix()
    selected, stats = choose_genes(matrix)
    node_df, edge_df = build_edges(selected, stats)

    node_path = PROCESSED / "nsclc_dependency_network_nodes.csv"
    edge_path = PROCESSED / "nsclc_dependency_network_edges.csv"
    rank_path = RESULTS / "nsclc_top_ranked_genes.csv"

    node_df.to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)
    nsclc_models.to_csv(PROCESSED / "nsclc_model_metadata.csv", index=False)
    node_df.to_csv(rank_path, index=False)
    write_summary(node_df, edge_df, nsclc_models)

    print(f"Wrote {node_path}")
    print(f"Wrote {edge_path}")
    print(f"Wrote {rank_path}")


if __name__ == "__main__":
    main()

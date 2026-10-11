from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

PRIMARY_DISEASE = "Non-Small Cell Lung Cancer"
LIPID_SEEDS = {"SCAP", "SCD", "MBTPS1", "SREBF1", "MBTPS2"}
MIN_VALID_FRACTION = 0.90
MIN_MODULE_SIZE = 4
BIOLOGY_GENES = [
    "KRAS",
    "NFE2L2",
    "KEAP1",
    "STK11",
    "EGFR",
    "SMARCA2",
    "CFLAR",
    "TRPM7",
    "UBIAD1",
    "UXS1",
    "PGD",
    "PTDSS1",
]

GENE_RE = re.compile(r"^(?P<symbol>.+?) \((?P<entrez>\d+)\)$")


def parse_gene_column(col: str) -> tuple[str, str]:
    match = GENE_RE.match(col)
    if not match:
        return col, ""
    return match.group("symbol"), match.group("entrez")


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def fit_linear(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    return coef, y - pred


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_effect = pd.read_csv(RAW / "CRISPRGeneEffect.csv")
    gene_effect = gene_effect.rename(columns={gene_effect.columns[0]: "ModelID"})
    gene_effect["ModelID"] = gene_effect["ModelID"].astype(str)

    models = pd.read_csv(RAW / "Model.csv")
    models["ModelID"] = models["ModelID"].astype(str)
    return gene_effect, models


def disease_ids(models: pd.DataFrame, matrix: pd.DataFrame, disease: str) -> list[str]:
    available = set(matrix.index.astype(str))
    ids = models.loc[models["OncotreePrimaryDisease"] == disease, "ModelID"].astype(str)
    return [model_id for model_id in ids if model_id in available]


def gene_metadata(columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        symbol, entrez = parse_gene_column(col)
        rows.append({"gene_column": col, "gene_symbol": symbol, "entrez_id": entrez})
    return pd.DataFrame(rows)


def choose_genes(matrix: pd.DataFrame, ids: list[str], top_genes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = matrix.loc[ids].copy()
    valid_fraction = subset.notna().mean(axis=0)
    subset = subset.loc[:, valid_fraction >= MIN_VALID_FRACTION].copy()

    means = subset.mean(axis=0, skipna=True)
    stds = subset.std(axis=0, skipna=True)
    mins = subset.min(axis=0, skipna=True)

    stats = gene_metadata(list(subset.columns)).set_index("gene_column")
    stats["mean_gene_effect"] = means
    stats["std_gene_effect"] = stds
    stats["min_gene_effect"] = mins
    stats["valid_fraction"] = valid_fraction.loc[subset.columns]
    stats["dependency_strength"] = -stats["mean_gene_effect"]
    stats["selectivity"] = stats["std_gene_effect"]
    stats["screen_score"] = (
        zscore(stats["dependency_strength"])
        + 0.75 * zscore(stats["selectivity"])
        + 0.25 * zscore(-stats["min_gene_effect"])
    )

    selected_stats = stats.sort_values("screen_score", ascending=False).head(top_genes).copy()
    selected = subset.loc[:, selected_stats.index].apply(lambda col: col.fillna(col.mean()), axis=0)
    return selected, selected_stats.reset_index()


def build_edges(selected: pd.DataFrame, neighbors: int, min_corr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = selected.to_numpy(dtype=float)
    values = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    values = values / std
    corr_mat = (values.T @ values) / (values.shape[0] - 1)
    np.fill_diagonal(corr_mat, -np.inf)

    genes = selected.columns.to_numpy()
    edge_weights: dict[tuple[int, int], float] = {}
    for i in range(corr_mat.shape[0]):
        top_idx = np.argpartition(corr_mat[i], -neighbors)[-neighbors:]
        for j in top_idx:
            weight = float(corr_mat[i, j])
            if weight < min_corr:
                continue
            a, b = sorted((i, int(j)))
            if a == b:
                continue
            edge_weights[(a, b)] = max(edge_weights.get((a, b), -np.inf), weight)

    edges = pd.DataFrame(
        [
            {"source": genes[i], "target": genes[j], "weight": weight}
            for (i, j), weight in edge_weights.items()
        ]
    )
    if len(edges):
        edges = edges.sort_values("weight", ascending=False)
    return edges, pd.DataFrame(corr_mat, index=genes, columns=genes)


def add_network_scores(
    stats: pd.DataFrame,
    edges: pd.DataFrame,
    matrix: pd.DataFrame,
    ids: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    nodes = stats.copy().set_index("gene_column")
    nodes["degree"] = 0
    nodes["weighted_degree"] = 0.0
    if len(edges):
        for row in edges.itertuples(index=False):
            nodes.loc[row.source, "degree"] += 1
            nodes.loc[row.target, "degree"] += 1
            nodes.loc[row.source, "weighted_degree"] += float(row.weight)
            nodes.loc[row.target, "weighted_degree"] += float(row.weight)

    group_ids = set(ids)
    group = matrix.loc[matrix.index.isin(group_ids), nodes.index]
    other = matrix.loc[~matrix.index.isin(group_ids), nodes.index]
    nodes["group_dependency"] = -group.mean(axis=0, skipna=True)
    nodes["pan_cancer_dependency"] = -matrix[nodes.index].mean(axis=0, skipna=True)
    nodes["non_group_dependency"] = -other.mean(axis=0, skipna=True)
    nodes["group_selectivity"] = nodes["group_dependency"] - nodes["non_group_dependency"]
    nodes["nsclc_selectivity"] = nodes["group_selectivity"]

    y = nodes["dependency_strength"].astype(float).to_numpy()
    x = np.column_stack(
        [
            np.ones(len(nodes)),
            np.log(nodes["degree"].astype(float).to_numpy() + 1.0),
            np.log(nodes["weighted_degree"].astype(float).to_numpy() + 1.0),
            nodes["pan_cancer_dependency"].astype(float).to_numpy(),
        ]
    )
    coef, residual = fit_linear(y, x)
    nodes["expected_dependency_from_topology"] = x @ coef
    nodes["thermo_residual_dependency"] = residual
    nodes["thermo_residual_score"] = (
        zscore(nodes["thermo_residual_dependency"])
        + zscore(nodes["group_selectivity"])
        - 0.5 * zscore(nodes["degree"])
    )
    nodes["selected_residual_vulnerability_score"] = (
        1.25 * zscore(nodes["thermo_residual_dependency"])
        + 1.50 * zscore(nodes["group_selectivity"])
        - 1.00 * zscore(nodes["pan_cancer_dependency"])
        - 0.75 * zscore(nodes["degree"])
        - 0.50 * zscore(nodes["weighted_degree"])
    )
    return nodes.reset_index(), {
        "intercept": float(coef[0]),
        "log_degree": float(coef[1]),
        "log_weighted_degree": float(coef[2]),
        "pan_cancer_dependency": float(coef[3]),
    }


def connected_components(nodes: pd.DataFrame, edges: pd.DataFrame, threshold: float) -> pd.Series:
    node_set = set(nodes["gene_column"])
    parent = {node: node for node in node_set}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if len(edges):
        for row in edges[edges["weight"] >= threshold].itertuples(index=False):
            if row.source in node_set and row.target in node_set:
                union(row.source, row.target)

    roots = {node: find(node) for node in node_set}
    counts = pd.Series(roots).value_counts()
    root_to_id: dict[str, str] = {}
    module_index = 1
    for root, size in counts.items():
        if size >= MIN_MODULE_SIZE:
            root_to_id[root] = f"M{module_index:03d}"
            module_index += 1
    return pd.Series({node: root_to_id.get(root, "singleton") for node, root in roots.items()})


def score_modules(nodes: pd.DataFrame, edges: pd.DataFrame, module_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = nodes.copy()
    module_map = connected_components(out, edges, module_threshold)
    out["module_id"] = out["gene_column"].map(module_map).fillna("singleton")
    modules = (
        out[out["module_id"] != "singleton"]
        .groupby("module_id")
        .agg(
            n_genes=("gene_symbol", "size"),
            mean_dependency=("dependency_strength", "mean"),
            mean_selectivity=("group_selectivity", "mean"),
            mean_degree=("degree", "mean"),
            mean_residual_score=("thermo_residual_score", "mean"),
            top_genes=("gene_symbol", lambda x: ", ".join(x.astype(str).head(12))),
        )
        .reset_index()
    )
    if len(modules):
        modules["module_stress_adaptation_score"] = (
            zscore(modules["mean_dependency"])
            + zscore(modules["mean_selectivity"])
            + zscore(modules["mean_residual_score"])
            - 0.25 * zscore(modules["mean_degree"])
        )
        modules = modules.sort_values("module_stress_adaptation_score", ascending=False).reset_index(drop=True)
        modules["module_rank"] = np.arange(1, len(modules) + 1)
    return out, modules


def seed_module_info(nodes: pd.DataFrame, modules: pd.DataFrame) -> dict[str, object]:
    seed_nodes = nodes[nodes["gene_symbol"].isin(LIPID_SEEDS)].copy()
    present = set(seed_nodes["gene_symbol"])
    module_counts = (
        seed_nodes[seed_nodes["module_id"] != "singleton"]["module_id"].value_counts()
        if "module_id" in seed_nodes
        else pd.Series(dtype=int)
    )
    if len(module_counts) == 0:
        return {
            "present_seed_count": len(present),
            "present_seeds": ", ".join(sorted(present)),
            "seed_module_id": "",
            "seed_module_seed_count": 0,
            "seed_module_rank": np.nan,
            "seed_module_score": np.nan,
            "seed_module_genes": "",
        }
    module_id = str(module_counts.index[0])
    module_seed_count = int(module_counts.iloc[0])
    row = modules[modules["module_id"] == module_id]
    module_rank = float(row["module_rank"].iloc[0]) if len(row) else np.nan
    module_score = float(row["module_stress_adaptation_score"].iloc[0]) if len(row) else np.nan
    module_genes = ", ".join(nodes.loc[nodes["module_id"] == module_id, "gene_symbol"].astype(str))
    return {
        "present_seed_count": len(present),
        "present_seeds": ", ".join(sorted(present)),
        "seed_module_id": module_id,
        "seed_module_seed_count": module_seed_count,
        "seed_module_rank": module_rank,
        "seed_module_score": module_score,
        "seed_module_genes": module_genes,
    }


def run_pipeline(
    matrix: pd.DataFrame,
    models: pd.DataFrame,
    disease: str,
    top_genes: int,
    neighbors: int,
    min_corr: float,
    module_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ids = disease_ids(models, matrix, disease)
    selected, stats = choose_genes(matrix, ids, top_genes)
    edges, _ = build_edges(selected, neighbors, min_corr)
    nodes, coeffs = add_network_scores(stats, edges, matrix, ids)
    nodes, modules = score_modules(nodes, edges, module_threshold)
    return nodes, edges, modules, coeffs


def robustness_check(matrix: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for top_genes in [1000, 1500, 2000]:
        for min_corr in [0.30, 0.35, 0.40, 0.45]:
            for neighbors in [5, 10, 15]:
                nodes, edges, modules, _ = run_pipeline(
                    matrix,
                    models,
                    PRIMARY_DISEASE,
                    top_genes,
                    neighbors,
                    min_corr,
                    module_threshold=0.45,
                )
                info = seed_module_info(nodes, modules)
                rows.append(
                    {
                        "top_genes": top_genes,
                        "min_corr": min_corr,
                        "neighbors": neighbors,
                        "n_edges": len(edges),
                        "n_modules": len(modules),
                        **info,
                    }
                )
                print(
                    "robustness",
                    top_genes,
                    min_corr,
                    neighbors,
                    "rank",
                    info["seed_module_rank"],
                    "seeds",
                    info["seed_module_seed_count"],
                )
    return pd.DataFrame(rows)


def specificity_check(matrix: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    screened_ids = set(matrix.index.astype(str))
    disease_counts = (
        models[models["ModelID"].isin(screened_ids)]["OncotreePrimaryDisease"]
        .value_counts()
        .head(12)
    )
    rows = []
    for disease, n_models in disease_counts.items():
        if n_models < 40 or disease == "Non-Cancerous":
            continue
        nodes, edges, modules, _ = run_pipeline(
            matrix,
            models,
            disease,
            top_genes=1500,
            neighbors=10,
            min_corr=0.35,
            module_threshold=0.45,
        )
        info = seed_module_info(nodes, modules)
        top_selected = ", ".join(
            nodes.sort_values("selected_residual_vulnerability_score", ascending=False)
            .head(10)["gene_symbol"]
            .astype(str)
        )
        rows.append(
            {
                "disease": disease,
                "n_models": int(n_models),
                "n_edges": len(edges),
                "n_modules": len(modules),
                "top_selected_genes": top_selected,
                **info,
            }
        )
        print("specificity", disease, "rank", info["seed_module_rank"], "score", info["seed_module_score"])
    return pd.DataFrame(rows)


def biological_interpretation(matrix: pd.DataFrame, models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes, edges, modules, _ = run_pipeline(
        matrix,
        models,
        PRIMARY_DISEASE,
        top_genes=1500,
        neighbors=10,
        min_corr=0.35,
        module_threshold=0.45,
    )
    info = seed_module_info(nodes, modules)
    module_genes = [g.strip() for g in str(info["seed_module_genes"]).split(",") if g.strip()]
    module_nodes = nodes[nodes["gene_symbol"].isin(module_genes)].copy()
    seed_nodes = nodes[nodes["gene_symbol"].isin(LIPID_SEEDS)].copy()

    ids = disease_ids(models, matrix, PRIMARY_DISEASE)
    nsclc = matrix.loc[ids]
    symbol_to_col = dict(zip(nodes["gene_symbol"], nodes["gene_column"]))
    seed_cols = [symbol_to_col[s] for s in sorted(LIPID_SEEDS) if s in symbol_to_col]
    module_activity = -nsclc[seed_cols].mean(axis=1)

    corr_rows = []
    for gene in BIOLOGY_GENES:
        col = symbol_to_col.get(gene)
        if col is None:
            continue
        corr_rows.append(
            {
                "gene_symbol": gene,
                "corr_dependency_with_srebp_seed_activity": corr(-nsclc[col], module_activity),
                "dependency_strength": float(nodes.loc[nodes["gene_symbol"] == gene, "dependency_strength"].iloc[0]),
                "selected_residual_vulnerability_score": float(
                    nodes.loc[nodes["gene_symbol"] == gene, "selected_residual_vulnerability_score"].iloc[0]
                ),
            }
        )
    correlations = pd.DataFrame(corr_rows).sort_values(
        "corr_dependency_with_srebp_seed_activity", ascending=False
    )

    nsclc_models = models[models["ModelID"].isin(ids)].copy()
    text = (
        nsclc_models[["PatientSubtypeFeatures", "ModelSubtypeFeatures"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.upper()
    )
    mutation_rows = []
    for marker in ["KRAS", "NFE2L2", "KEAP1", "STK11", "EGFR"]:
        mask = text.str.contains(marker, regex=False)
        if int(mask.sum()) < 3 or int((~mask).sum()) < 3:
            mutation_rows.append(
                {
                    "marker": marker,
                    "n_marker_models": int(mask.sum()),
                    "n_other_models": int((~mask).sum()),
                    "mean_srebp_activity_marker": np.nan,
                    "mean_srebp_activity_other": np.nan,
                    "difference": np.nan,
                    "note": "insufficient metadata-coded models",
                }
            )
            continue
        marker_ids = nsclc_models.loc[mask, "ModelID"]
        other_ids = nsclc_models.loc[~mask, "ModelID"]
        marker_mean = float(module_activity.loc[marker_ids].mean())
        other_mean = float(module_activity.loc[other_ids].mean())
        mutation_rows.append(
            {
                "marker": marker,
                "n_marker_models": int(mask.sum()),
                "n_other_models": int((~mask).sum()),
                "mean_srebp_activity_marker": marker_mean,
                "mean_srebp_activity_other": other_mean,
                "difference": marker_mean - other_mean,
                "note": "metadata-coded comparison",
            }
        )
    marker_activity = pd.DataFrame(mutation_rows)

    seed_edges = edges[
        edges["source"].isin(seed_cols) & edges["target"].isin(seed_cols)
        | edges["source"].isin(seed_cols) & edges["target"].isin(module_nodes["gene_column"])
        | edges["target"].isin(seed_cols) & edges["source"].isin(module_nodes["gene_column"])
    ].copy()
    col_to_symbol = dict(zip(nodes["gene_column"], nodes["gene_symbol"]))
    if len(seed_edges):
        seed_edges["source_symbol"] = seed_edges["source"].map(col_to_symbol)
        seed_edges["target_symbol"] = seed_edges["target"].map(col_to_symbol)
        seed_edges = seed_edges[
            ["source_symbol", "target_symbol", "weight", "source", "target"]
        ].sort_values("weight", ascending=False)

    seed_nodes.to_csv(RESULTS / "biology_srebp_seed_gene_scores.csv", index=False)
    module_nodes.to_csv(RESULTS / "biology_srebp_full_module_gene_scores.csv", index=False)
    seed_edges.to_csv(RESULTS / "biology_srebp_module_edges.csv", index=False)
    return seed_nodes, correlations, marker_activity


def write_summary(
    robustness: pd.DataFrame,
    specificity: pd.DataFrame,
    seed_nodes: pd.DataFrame,
    correlations: pd.DataFrame,
    marker_activity: pd.DataFrame,
) -> None:
    summary_path = RESULTS / "srebp_lipid_axis_validation_summary.txt"
    robust_found = robustness["seed_module_seed_count"].fillna(0).ge(3)
    robust_top3 = robustness["seed_module_rank"].fillna(999).le(3)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("SREBP/lipid axis validation\n")
        f.write("==========================\n\n")
        f.write("Step 1: Robustness check\n")
        f.write("------------------------\n")
        f.write(f"Parameter settings tested: {len(robustness)}\n")
        f.write(f"SREBP seed module found in settings: {int(robust_found.sum())}/{len(robustness)}\n")
        f.write(f"SREBP seed module rank <= 3: {int(robust_top3.sum())}/{len(robustness)}\n")
        f.write("\nTop robustness rows by seed module score\n")
        f.write(robustness.sort_values("seed_module_score", ascending=False).head(15).to_string(index=False))
        f.write("\n\n")

        f.write("Step 2: Cross-cancer specificity check\n")
        f.write("--------------------------------------\n")
        f.write(specificity.sort_values("seed_module_score", ascending=False).to_string(index=False))
        f.write("\n\n")

        f.write("Step 3: Biological interpretation check\n")
        f.write("---------------------------------------\n")
        f.write("SREBP/lipid seed gene scores\n")
        seed_cols = [
            "gene_symbol",
            "degree",
            "weighted_degree",
            "dependency_strength",
            "pan_cancer_dependency",
            "group_selectivity",
            "thermo_residual_dependency",
            "selected_residual_vulnerability_score",
        ]
        f.write(seed_nodes[seed_cols].sort_values("selected_residual_vulnerability_score", ascending=False).to_string(index=False))
        f.write("\n\nDependency-profile correlation with SREBP seed activity\n")
        f.write(correlations.to_string(index=False))
        f.write("\n\nMetadata-coded marker comparison\n")
        f.write(marker_activity.to_string(index=False))
        f.write("\n")
    print(f"Wrote {summary_path}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    gene_effect, models = load_data()
    matrix = gene_effect.set_index("ModelID")

    robustness = robustness_check(matrix, models)
    specificity = specificity_check(matrix, models)
    seed_nodes, correlations, marker_activity = biological_interpretation(matrix, models)

    robustness.to_csv(RESULTS / "srebp_lipid_axis_robustness.csv", index=False)
    specificity.to_csv(RESULTS / "srebp_lipid_axis_specificity.csv", index=False)
    correlations.to_csv(RESULTS / "biology_srebp_activity_correlations.csv", index=False)
    marker_activity.to_csv(RESULTS / "biology_metadata_marker_activity.csv", index=False)
    write_summary(robustness, specificity, seed_nodes, correlations, marker_activity)

    print(f"Wrote {RESULTS / 'srebp_lipid_axis_robustness.csv'}")
    print(f"Wrote {RESULTS / 'srebp_lipid_axis_specificity.csv'}")
    print(f"Wrote {RESULTS / 'biology_srebp_activity_correlations.csv'}")
    print(f"Wrote {RESULTS / 'biology_metadata_marker_activity.csv'}")


if __name__ == "__main__":
    main()

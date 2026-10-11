from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

PRIMARY_DISEASE = "Non-Small Cell Lung Cancer"
MODULE_EDGE_THRESHOLD = 0.45
MIN_MODULE_SIZE = 4
EPS = 1e-9


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def fit_linear(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    residual = y - pred
    return coef, residual


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(PROCESSED / "nsclc_dependency_network_nodes.csv")
    edges = pd.read_csv(PROCESSED / "nsclc_dependency_network_edges.csv")
    models = pd.read_csv(RAW / "Model.csv")

    selected_cols = ["Unnamed: 0"] + nodes["gene_column"].tolist()
    gene_effect = pd.read_csv(RAW / "CRISPRGeneEffect.csv", usecols=selected_cols)
    gene_effect = gene_effect.rename(columns={"Unnamed: 0": "ModelID"})
    gene_effect["ModelID"] = gene_effect["ModelID"].astype(str)
    models["ModelID"] = models["ModelID"].astype(str)

    return nodes, edges, gene_effect, models


def add_pan_cancer_penalty(
    nodes: pd.DataFrame, gene_effect: pd.DataFrame, models: pd.DataFrame
) -> pd.DataFrame:
    nsclc_ids = set(
        models.loc[models["OncotreePrimaryDisease"] == PRIMARY_DISEASE, "ModelID"].astype(str)
    )
    matrix = gene_effect.set_index("ModelID")
    nsclc = matrix.loc[matrix.index.isin(nsclc_ids), nodes["gene_column"]]
    non_nsclc = matrix.loc[~matrix.index.isin(nsclc_ids), nodes["gene_column"]]

    nsclc_dependency = -nsclc.mean(axis=0, skipna=True)
    pan_dependency = -matrix[nodes["gene_column"]].mean(axis=0, skipna=True)
    non_nsclc_dependency = -non_nsclc.mean(axis=0, skipna=True)
    nsclc_selectivity = nsclc_dependency - non_nsclc_dependency

    mapped = nodes.set_index("gene_column").copy()
    mapped["nsclc_dependency_recomputed"] = nsclc_dependency
    mapped["pan_cancer_dependency"] = pan_dependency
    mapped["non_nsclc_dependency"] = non_nsclc_dependency
    mapped["nsclc_selectivity"] = nsclc_selectivity
    mapped["generic_essentiality_penalty"] = zscore(mapped["pan_cancer_dependency"])
    return mapped.reset_index()


def add_idea_1_fitness_weighted_pa(nodes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    degree_term = np.log(nodes["degree"].astype(float).to_numpy() + 1.0)
    dependency_term = zscore(nodes["dependency_strength"]).to_numpy()
    observed_mass = (
        nodes["weighted_degree"].clip(lower=0).astype(float)
        + nodes["dependency_strength"].clip(lower=0).astype(float)
        + EPS
    )
    observed_mass = observed_mass / observed_mass.sum()
    y = np.log(observed_mass.to_numpy())
    x = np.column_stack([np.ones(len(nodes)), degree_term, dependency_term])
    coef, residual = fit_linear(y, x)

    out = nodes.copy()
    out["fitness_pa_log_mass_residual"] = residual
    out["fitness_pa_score"] = (
        zscore(out["fitness_pa_log_mass_residual"])
        + zscore(out["nsclc_selectivity"])
        - 0.5 * zscore(out["pan_cancer_dependency"])
    )
    params = {
        "fitness_pa_intercept": float(coef[0]),
        "fitness_pa_beta_degree": float(coef[1]),
        "fitness_pa_gamma_dependency": float(coef[2]),
    }
    return out, params


def add_idea_2_antihub(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    out["anti_hub_vulnerability_score"] = (
        zscore(out["dependency_strength"])
        + zscore(out["selectivity"])
        + zscore(out["nsclc_selectivity"])
        - zscore(out["degree"])
        - 0.5 * zscore(out["weighted_degree"])
    )
    return out


def add_idea_3_thermodynamic_residual(nodes: pd.DataFrame) -> pd.DataFrame:
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
    out = nodes.copy()
    out["expected_dependency_from_topology"] = x @ coef
    out["thermo_residual_dependency"] = residual
    out["thermo_residual_score"] = (
        zscore(out["thermo_residual_dependency"])
        + zscore(out["nsclc_selectivity"])
        - 0.5 * zscore(out["degree"])
    )
    out.attrs["thermo_residual_coefficients"] = {
        "intercept": float(coef[0]),
        "log_degree": float(coef[1]),
        "log_weighted_degree": float(coef[2]),
        "pan_cancer_dependency": float(coef[3]),
    }
    return out


def add_idea_4_free_energy(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    energy = -np.log(out["degree"].astype(float) + 1.0)
    fitness = zscore(out["dependency_strength"]) + 0.75 * zscore(out["nsclc_selectivity"])
    entropy_like = np.log(out["weighted_degree"].astype(float) + 1.0)
    out["network_energy"] = energy
    out["fitness_term"] = fitness
    out["entropy_like_connectivity"] = entropy_like
    out["free_energy_vulnerability"] = energy - fitness - 0.25 * entropy_like
    out["free_energy_score"] = -zscore(out["free_energy_vulnerability"])
    return out


def connected_components(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
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

    strong_edges = edges[edges["weight"] >= MODULE_EDGE_THRESHOLD]
    for row in strong_edges.itertuples(index=False):
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


def add_idea_5_modules(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = nodes.copy()
    modules = connected_components(out, edges)
    out["module_id"] = out["gene_column"].map(modules).fillna("singleton")

    module_df = (
        out[out["module_id"] != "singleton"]
        .groupby("module_id")
        .agg(
            n_genes=("gene_symbol", "size"),
            mean_dependency=("dependency_strength", "mean"),
            mean_selectivity=("nsclc_selectivity", "mean"),
            mean_degree=("degree", "mean"),
            mean_residual_score=("thermo_residual_score", "mean"),
            top_genes=("gene_symbol", lambda x: ", ".join(x.astype(str).head(12))),
        )
        .reset_index()
    )
    if len(module_df):
        module_df["module_stress_adaptation_score"] = (
            zscore(module_df["mean_dependency"])
            + zscore(module_df["mean_selectivity"])
            + zscore(module_df["mean_residual_score"])
            - 0.25 * zscore(module_df["mean_degree"])
        )
        module_df = module_df.sort_values("module_stress_adaptation_score", ascending=False)
    return out, module_df


def add_selected_novelty_penalty(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    out["selected_residual_vulnerability_score"] = (
        1.25 * zscore(out["thermo_residual_dependency"])
        + 1.50 * zscore(out["nsclc_selectivity"])
        - 1.00 * zscore(out["pan_cancer_dependency"])
        - 0.75 * zscore(out["degree"])
        - 0.50 * zscore(out["weighted_degree"])
    )
    return out


def add_idea_6_novelty_penalty(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    out["novel_nsclc_vulnerability_score"] = (
        zscore(out["thermo_residual_score"])
        + zscore(out["anti_hub_vulnerability_score"])
        + zscore(out["free_energy_score"])
        + zscore(out["fitness_pa_score"])
        + 1.5 * zscore(out["nsclc_selectivity"])
        - zscore(out["pan_cancer_dependency"])
        - 0.75 * zscore(out["degree"])
    )
    return out


def extract_srebp_lipid_module(nodes: pd.DataFrame) -> pd.DataFrame:
    lipid_seeds = {"SCAP", "SCD", "MBTPS1", "SREBF1", "MBTPS2"}
    seed_modules = set(
        nodes.loc[nodes["gene_symbol"].isin(lipid_seeds), "module_id"].dropna().astype(str)
    )
    seed_modules.discard("singleton")
    if not seed_modules:
        return pd.DataFrame(columns=nodes.columns)
    return (
        nodes[nodes["module_id"].isin(seed_modules)]
        .sort_values("selected_residual_vulnerability_score", ascending=False)
        .copy()
    )


def write_top_table(f, title: str, nodes: pd.DataFrame, score_col: str) -> None:
    cols = [
        "gene_symbol",
        "entrez_id",
        "degree",
        "weighted_degree",
        "dependency_strength",
        "pan_cancer_dependency",
        "nsclc_selectivity",
        score_col,
    ]
    f.write(f"{title}\n")
    f.write("-" * len(title) + "\n")
    f.write(nodes.sort_values(score_col, ascending=False)[cols].head(25).to_string(index=False))
    f.write("\n\n")


def write_summary(nodes: pd.DataFrame, modules: pd.DataFrame, params: dict[str, float]) -> None:
    summary_path = RESULTS / "extension_ideas_1_6_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Extension ideas 1-6 comparison\n")
        f.write("==============================\n\n")
        f.write("Idea 1: Fitness-weighted thermodynamic PA\n")
        f.write(
            "P(i) proportional to (k_i + 1)^beta * exp(gamma * dependency_i), "
            "fit as a log-mass regression.\n"
        )
        for key, value in params.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")

        write_top_table(f, "Idea 1 top fitness-weighted PA residual genes", nodes, "fitness_pa_score")
        write_top_table(f, "Idea 2 top anti-hub vulnerability genes", nodes, "anti_hub_vulnerability_score")
        write_top_table(f, "Idea 3 top thermodynamic residual genes", nodes, "thermo_residual_score")
        write_top_table(f, "Idea 4 top free-energy vulnerability genes", nodes, "free_energy_score")
        write_top_table(f, "Idea 6 top novelty-penalized NSCLC vulnerability genes", nodes, "novel_nsclc_vulnerability_score")

        f.write("Idea 5 top module-level stress-adaptation modules\n")
        f.write("-------------------------------------------------\n")
        if len(modules):
            f.write(modules.head(20).to_string(index=False))
        else:
            f.write("No modules passed threshold.")
        f.write("\n")

    print(f"Wrote {summary_path}")


def write_selected_summary(
    nodes: pd.DataFrame,
    modules: pd.DataFrame,
    srebp_module: pd.DataFrame,
    residual_coefficients: dict[str, float],
) -> None:
    summary_path = RESULTS / "selected_extension_residual_srebp_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Selected extension: thermodynamic residual vulnerability\n")
        f.write("=======================================================\n\n")
        f.write("Active extension\n")
        f.write("----------------\n")
        f.write(
            "Score genes whose NSCLC dependency is stronger than expected after "
            "correcting for graph topology and pan-cancer generic essentiality.\n\n"
        )
        f.write("Residual model\n")
        f.write("--------------\n")
        f.write(
            "dependency_strength ~ log(degree + 1) + "
            "log(weighted_degree + 1) + pan_cancer_dependency\n"
        )
        for key, value in residual_coefficients.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")

        write_top_table(
            f,
            "Top selected residual vulnerability genes",
            nodes,
            "selected_residual_vulnerability_score",
        )

        f.write("Top stress-adaptation modules\n")
        f.write("-----------------------------\n")
        if len(modules):
            f.write(modules.head(20).to_string(index=False))
        else:
            f.write("No modules passed threshold.")
        f.write("\n\n")

        f.write("SREBP/lipid module genes\n")
        f.write("------------------------\n")
        if len(srebp_module):
            cols = [
                "module_id",
                "gene_symbol",
                "entrez_id",
                "degree",
                "weighted_degree",
                "dependency_strength",
                "pan_cancer_dependency",
                "nsclc_selectivity",
                "thermo_residual_dependency",
                "selected_residual_vulnerability_score",
            ]
            f.write(srebp_module[cols].to_string(index=False))
        else:
            f.write("SREBP/lipid seed genes did not form a non-singleton module.")
        f.write("\n")

    print(f"Wrote {summary_path}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    nodes, edges, gene_effect, models = load_inputs()
    nodes = add_pan_cancer_penalty(nodes, gene_effect, models)

    # Optional ideas from the first brainstorming pass. Kept here for later, but
    # intentionally disabled so the chosen extension is not mixed with other
    # scores.
    # nodes, fitness_params = add_idea_1_fitness_weighted_pa(nodes)
    # nodes = add_idea_2_antihub(nodes)
    # nodes = add_idea_4_free_energy(nodes)
    # nodes = add_idea_6_novelty_penalty(nodes)

    nodes = add_idea_3_thermodynamic_residual(nodes)
    residual_coefficients = nodes.attrs.get("thermo_residual_coefficients", {})
    nodes = add_selected_novelty_penalty(nodes)
    nodes, modules = add_idea_5_modules(nodes, edges)
    srebp_module = extract_srebp_lipid_module(nodes)

    nodes = nodes.sort_values("selected_residual_vulnerability_score", ascending=False)
    nodes.to_csv(RESULTS / "selected_residual_vulnerability_gene_scores.csv", index=False)
    modules.to_csv(RESULTS / "selected_residual_vulnerability_module_scores.csv", index=False)
    srebp_module.to_csv(RESULTS / "selected_srebp_lipid_module_genes.csv", index=False)
    write_selected_summary(nodes, modules, srebp_module, residual_coefficients)
    print(f"Wrote {RESULTS / 'selected_residual_vulnerability_gene_scores.csv'}")
    print(f"Wrote {RESULTS / 'selected_residual_vulnerability_module_scores.csv'}")
    print(f"Wrote {RESULTS / 'selected_srebp_lipid_module_genes.csv'}")


if __name__ == "__main__":
    main()

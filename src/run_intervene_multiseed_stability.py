from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import interleaved_biophysics_attention_full as ibpa


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEEDS = [7, 17, 29, 43, 61, 83, 101]
TOP_K_EDGES = 12
PRIOR_WEIGHT = 0.22


def prepare_inputs() -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    frame, specs = ibpa.load_feature_table()
    x, tokens, feature_names, rows = ibpa.build_tensor(frame, specs)
    x_paper, paper_tokens = ibpa.subset_tensor(x, tokens, ibpa.PAPER_TOKENS)
    x_paper, paper_feature_names = ibpa.remove_control_features(x_paper, feature_names)
    return x_paper, paper_tokens, paper_feature_names, rows


def run_seed(
    seed: int,
    x_paper: np.ndarray,
    paper_tokens: list[str],
    paper_feature_names: list[str],
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    old_seed = ibpa.SEED
    ibpa.SEED = seed
    try:
        _, state, recon, attn, metrics = ibpa.train_discovery_model(
            x_paper,
            paper_tokens,
            paper_feature_names,
            prior_weight=PRIOR_WEIGHT,
            include_srebp=True,
            include_mito=True,
            include_oxidative=True,
        )
        edges = ibpa.attention_edges(attn, paper_tokens, f"intervene_seed_{seed}")
        validation = ibpa.discovery_validation_for_score(
            rows,
            x_paper,
            paper_tokens,
            paper_feature_names,
            state,
            "multiseed_state_score",
            n_perm=1000,
        )
    finally:
        ibpa.SEED = old_seed

    edges["seed"] = seed
    edges["edge"] = edges["source_token"] + "->" + edges["target_token"]
    edges["rank_within_seed"] = np.arange(1, len(edges) + 1)
    validation["seed"] = seed
    for key, value in metrics.items():
        validation[key] = value
    return edges, validation, metrics


def summarize_edges(all_edges: pd.DataFrame) -> pd.DataFrame:
    top = all_edges[all_edges["rank_within_seed"] <= TOP_K_EDGES].copy()
    summary = (
        top.groupby(["source_token", "target_token", "edge"])
        .agg(
            seed_count=("seed", "nunique"),
            mean_rank=("rank_within_seed", "mean"),
            best_rank=("rank_within_seed", "min"),
            mean_attention=("mean_attention", "mean"),
            mean_enrichment=("attention_enrichment_vs_uniform", "mean"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
            heads=("head", lambda x: ",".join(map(str, sorted(set(x))))),
            pseudoheads=("pseudohead", lambda x: ",".join(map(str, sorted(set(x))))),
        )
        .reset_index()
    )
    summary["seed_fraction"] = summary["seed_count"] / len(SEEDS)
    return summary.sort_values(
        ["seed_count", "mean_enrichment", "mean_attention"],
        ascending=[False, False, False],
    )


def summarize_validation(validation: pd.DataFrame) -> pd.DataFrame:
    key_tests = [
        "cytotoxic-residual stress-survival score",
        "PRISM stress-survival sensitivity",
        "PRISM generic cytotoxic sensitivity",
        "PRISM direct lipid sensitivity",
        "KEAP1/NFE2L2 oxidative context",
    ]
    rows = []
    for test in key_tests:
        sub = validation[validation["test"] == test].copy()
        if len(sub) == 0:
            continue
        rows.append(
            {
                "test": test,
                "n_seeds": int(sub["seed"].nunique()),
                "mean_effect": float(sub["high_minus_low"].mean()),
                "sd_effect": float(sub["high_minus_low"].std(ddof=0)),
                "min_effect": float(sub["high_minus_low"].min()),
                "max_effect": float(sub["high_minus_low"].max()),
                "median_perm_p": float(sub["perm_p_two_sided"].median()),
                "significant_seed_count_p_lt_0.05": int((sub["perm_p_two_sided"] < 0.05).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_summary(edge_summary: pd.DataFrame, validation_summary: pd.DataFrame) -> None:
    path = RESULTS / "intervene_multiseed_stability_summary.txt"
    stable = edge_summary[edge_summary["seed_fraction"] >= 0.5].copy()
    with path.open("w", encoding="utf-8") as f:
        f.write("INTERVENE multi-seed pseudo-head stability\n")
        f.write("=========================================\n\n")
        f.write(f"Seeds: {', '.join(map(str, SEEDS))}\n")
        f.write(f"Top edges considered per seed: {TOP_K_EDGES}\n")
        f.write(f"Prior weight: {PRIOR_WEIGHT}\n\n")
        f.write("Validation stability across seeds\n")
        f.write("---------------------------------\n")
        f.write(validation_summary.to_string(index=False))
        f.write("\n\n")
        f.write("Stable biological edges appearing in at least half of seeds\n")
        f.write("----------------------------------------------------------\n")
        cols = [
            "edge",
            "seed_count",
            "seed_fraction",
            "mean_rank",
            "mean_attention",
            "mean_enrichment",
            "seeds",
            "heads",
            "pseudoheads",
        ]
        if len(stable):
            f.write(stable[cols].head(25).to_string(index=False))
        else:
            f.write("No edge appeared in at least half of seeds.")
        f.write("\n\nPaper-facing interpretation\n")
        f.write("---------------------------\n")
        if len(stable):
            f.write(
                "The pseudo-head edges are not fully random: several module-module edges recur across seeds. "
                "Use only recurrent edges as biological interpretation, and avoid over-interpreting seed-specific edges.\n"
            )
        else:
            f.write(
                "The validation state may be stable, but individual attention edges are not stable enough for strong mechanistic claims.\n"
            )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    x_paper, paper_tokens, paper_feature_names, rows = prepare_inputs()
    edge_tables = []
    validation_tables = []
    metric_rows = []
    for seed in SEEDS:
        edges, validation, metrics = run_seed(seed, x_paper, paper_tokens, paper_feature_names, rows)
        edge_tables.append(edges)
        validation_tables.append(validation)
        metric_row = {"seed": seed}
        metric_row.update(metrics)
        metric_rows.append(metric_row)
        print(f"Finished seed {seed}")

    all_edges = pd.concat(edge_tables, ignore_index=True)
    validation = pd.concat(validation_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    edge_summary = summarize_edges(all_edges)
    validation_summary = summarize_validation(validation)

    all_edges.to_csv(RESULTS / "intervene_multiseed_attention_edges.csv", index=False)
    edge_summary.to_csv(RESULTS / "intervene_multiseed_edge_stability.csv", index=False)
    validation.to_csv(RESULTS / "intervene_multiseed_validation.csv", index=False)
    validation_summary.to_csv(RESULTS / "intervene_multiseed_validation_summary.csv", index=False)
    metrics.to_csv(RESULTS / "intervene_multiseed_training_metrics.csv", index=False)
    write_summary(edge_summary, validation_summary)

    print(f"Wrote {RESULTS / 'intervene_multiseed_attention_edges.csv'}")
    print(f"Wrote {RESULTS / 'intervene_multiseed_edge_stability.csv'}")
    print(f"Wrote {RESULTS / 'intervene_multiseed_validation.csv'}")
    print(f"Wrote {RESULTS / 'intervene_multiseed_validation_summary.csv'}")
    print(f"Wrote {RESULTS / 'intervene_multiseed_training_metrics.csv'}")
    print(f"Wrote {RESULTS / 'intervene_multiseed_stability_summary.txt'}")


if __name__ == "__main__":
    main()

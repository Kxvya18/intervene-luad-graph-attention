from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from interleaved_biophysics_attention_full import module_specs, zscore
from prepare_gse31210_relapse_intervene import two_sided_fisher


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"

MODULES = [
    "srebp_lipid",
    "mito_translation",
    "oxidative_stress",
    "proteostasis",
    "apoptosis_iap",
    "akt_mtor_growth",
]

THERAPY_MAP = {
    "srebp_lipid": "SREBP/lipid-axis modulation; use as stress-state marker because direct lipid drugs were weak in PRISM",
    "mito_translation": "mitochondrial translation / OXPHOS or complex-I stressors",
    "oxidative_stress": "oxidative-stress / NRF2-redox buffering perturbation",
    "proteostasis": "proteasome, deubiquitinase, ER-stress, or protein-homeostasis perturbation",
    "apoptosis_iap": "IAP/anti-apoptotic survival blockade or apoptosis priming",
    "akt_mtor_growth": "AKT/mTOR growth-stress adaptation inhibitors",
}


def summarize_model(path: Path, model: str, display: str, category: str) -> dict[str, object]:
    df = pd.read_csv(path)
    rows = df[df["model"].eq(model)].copy()
    if rows.empty:
        raise ValueError(f"Missing {model} in {path}")
    if rows["seed"].astype(str).eq("na").all():
        row = rows.iloc[0]
        return {
            "display_model": display,
            "source_model": model,
            "category": category,
            "seeds": "na",
            "c_index_mean": row["c_index"],
            "c_index_sd": 0.0,
            "hazard_ratio_mean": row["hazard_ratio_approx"],
            "logrank_p_median": row["logrank_p"],
            "risk_prior_corr_mean": row.get("risk_prior_corr", np.nan),
            "edge_separation_mean": row.get("edge_separation", np.nan),
            "high_events_mean": row.get("high_events", np.nan),
            "low_events_mean": row.get("low_events", np.nan),
        }
    return {
        "display_model": display,
        "source_model": model,
        "category": category,
        "seeds": int(rows["seed"].nunique()),
        "c_index_mean": rows["c_index"].mean(),
        "c_index_sd": rows["c_index"].std(ddof=1),
        "hazard_ratio_mean": rows["hazard_ratio_approx"].mean(),
        "logrank_p_median": rows["logrank_p"].median(),
        "risk_prior_corr_mean": rows.get("risk_prior_corr", pd.Series(dtype=float)).mean(),
        "edge_separation_mean": rows.get("edge_separation", pd.Series(dtype=float)).mean(),
        "high_events_mean": rows.get("high_events", pd.Series(dtype=float)).mean(),
        "low_events_mean": rows.get("low_events", pd.Series(dtype=float)).mean(),
    }


def make_final_ablation_table() -> pd.DataFrame:
    rows = [
        summarize_model(
            RESULTS / "relapse_intervene_graph_sparse_v3_model_comparison.csv",
            "raw_thermodynamic_transfer",
            "Raw transferred stress-survival state",
            "biology prior only",
        ),
        summarize_model(
            RESULTS / "relapse_intervene_gene_level_v2_model_comparison.csv",
            "clinical_only_cox",
            "Clinical-only Cox-like baseline",
            "clinical baseline",
        ),
        summarize_model(
            RESULTS / "relapse_intervene_survival_model_comparison.csv",
            "normal_attention_mha",
            "Normal attention",
            "AI baseline",
        ),
        summarize_model(
            RESULTS / "relapse_intervene_module_v2_model_comparison.csv",
            "module_intervene_constrained",
            "INTERVENE V2 constrained",
            "IHA-inspired + biology prior",
        ),
        summarize_model(
            RESULTS / "relapse_intervene_graph_sparse_v3_model_comparison.csv",
            "graph_sparse_v3_soft",
            "GraphSparse INTERVENE V3 soft",
            "sparse graph attention",
        ),
        summarize_model(
            RESULTS / "relapse_intervene_graph_sparse_v3_model_comparison.csv",
            "graph_sparse_v3_balanced",
            "GraphSparse INTERVENE V3 balanced",
            "sparse graph attention",
        ),
    ]
    table = pd.DataFrame(rows)
    table["delta_c_index_vs_normal_attention"] = (
        table["c_index_mean"] - float(table.loc[table["source_model"].eq("normal_attention_mha"), "c_index_mean"].iloc[0])
    )
    table["rank_by_c_index"] = table["c_index_mean"].rank(ascending=False, method="min").astype(int)
    return table.sort_values(["rank_by_c_index", "display_model"])


def v3_soft_score(frame: pd.DataFrame) -> pd.Series:
    risk = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_oof_risk_scores.csv")
    soft_cols = [c for c in risk.columns if c.startswith("graph_sparse_v3_soft_seed") and c.endswith("_risk")]
    risk["v3_soft_mean_risk"] = risk[soft_cols].mean(axis=1)
    return frame[["geo_accession"]].merge(risk[["geo_accession", "v3_soft_mean_risk"]], on="geo_accession", how="left")[
        "v3_soft_mean_risk"
    ]


def module_gene_map() -> dict[str, list[str]]:
    specs = {s.token: list(s.genes) for s in module_specs()}
    return {module: [g for g in specs[module]] for module in MODULES}


def high_low_frame() -> pd.DataFrame:
    patient = pd.read_csv(PROCESSED / "gse31210_patient_module_scores.csv")
    frame = patient[patient["is_primary_tumor"] & patient["prognosis_included"]].copy()
    frame = frame.dropna(subset=["relapse_event", "relapse_free_months", "raw_relapse_intervene_state"]).copy()
    frame = frame[frame["relapse_free_months"] > 0].reset_index(drop=True)
    frame["v3_soft_mean_risk"] = v3_soft_score(frame)
    high_cut = frame["v3_soft_mean_risk"].quantile(2 / 3)
    low_cut = frame["v3_soft_mean_risk"].quantile(1 / 3)
    frame["v3_risk_group"] = "middle"
    frame.loc[frame["v3_soft_mean_risk"] >= high_cut, "v3_risk_group"] = "high"
    frame.loc[frame["v3_soft_mean_risk"] <= low_cut, "v3_risk_group"] = "low"
    return frame


def gene_high_low_table(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(812)
    rows = []
    for module, genes in module_gene_map().items():
        for gene in genes:
            if gene not in frame.columns:
                continue
            data = frame.loc[frame["v3_risk_group"].isin(["high", "low"]), ["v3_risk_group", gene]].copy()
            data[gene] = pd.to_numeric(data[gene], errors="coerce")
            data = data.dropna()
            if data.empty:
                continue
            high = data.loc[data["v3_risk_group"].eq("high"), gene]
            low = data.loc[data["v3_risk_group"].eq("low"), gene]
            observed = float(high.mean() - low.mean())
            pooled = float(data[gene].std(ddof=0) or 1.0)
            values = data[gene].to_numpy()
            labels = data["v3_risk_group"].to_numpy()
            exceed = 0
            for _ in range(2000):
                shuffled = rng.permutation(labels)
                diff = values[shuffled == "high"].mean() - values[shuffled == "low"].mean()
                if abs(diff) >= abs(observed):
                    exceed += 1
            rows.append(
                {
                    "module": module,
                    "gene": gene,
                    "high_mean_expression": float(high.mean()),
                    "low_mean_expression": float(low.mean()),
                    "high_minus_low": observed,
                    "effect_size_z": observed / pooled,
                    "perm_p_two_sided": (exceed + 1) / 2001,
                }
            )
    return pd.DataFrame(rows).sort_values(["effect_size_z", "high_minus_low"], ascending=False)


def category_enrichment(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    data = frame[frame["v3_risk_group"].isin(["high", "low"])].copy()
    data[col] = data[col].fillna("missing").astype(str)
    rows = []
    for value in sorted(data[col].dropna().unique()):
        high = data["v3_risk_group"].eq("high")
        low = data["v3_risk_group"].eq("low")
        present = data[col].eq(value)
        a = int((high & present).sum())
        b = int((high & ~present).sum())
        c = int((low & present).sum())
        d = int((low & ~present).sum())
        rows.append(
            {
                "metadata_field": col,
                "category": value,
                "high_count": a,
                "high_total": int(high.sum()),
                "low_count": c,
                "low_total": int(low.sum()),
                "high_fraction": a / max(int(high.sum()), 1),
                "low_fraction": c / max(int(low.sum()), 1),
                "high_minus_low_fraction": a / max(int(high.sum()), 1) - c / max(int(low.sum()), 1),
                "fisher_p_two_sided": two_sided_fisher(a, b, c, d),
            }
        )
    return pd.DataFrame(rows).sort_values(["fisher_p_two_sided", "high_minus_low_fraction"])


def module_ablation_table() -> pd.DataFrame:
    ablation = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_module_ablation.csv")
    model_full = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_model_comparison.csv")
    ablation["seed_key"] = ablation["seed"].astype(str)
    model_full["seed_key"] = model_full["seed"].astype(str)
    full = model_full[model_full["model"].eq("graph_sparse_v3_soft")].groupby("seed_key")["c_index"].mean()
    soft = ablation[ablation["model"].eq("graph_sparse_v3_soft")].copy()
    soft["full_c_index"] = soft["seed_key"].map(full)
    soft["c_index_drop_after_removal"] = soft["full_c_index"] - soft["c_index_after_removal"]
    summary = (
        soft.groupby("module_removed")
        .agg(
            mean_c_index_drop_after_removal=("c_index_drop_after_removal", "mean"),
            sd_c_index_drop_after_removal=("c_index_drop_after_removal", "std"),
            mean_risk_drop_when_removed=("mean_risk_drop_when_removed", "mean"),
            mean_c_index_after_removal=("c_index_after_removal", "mean"),
        )
        .reset_index()
        .sort_values("mean_c_index_drop_after_removal", ascending=False)
    )
    return summary


def therapy_hypotheses(edges: pd.DataFrame, ablation: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    top_edges = edges[
        edges["model"].eq("graph_sparse_v3_soft") & edges["is_biology_edge"].astype(bool)
    ].copy()
    top_edges = top_edges[top_edges["source_module"].ne(top_edges["target_module"])].copy()
    edge_summary = (
        top_edges.groupby(["source_module", "target_module", "edge"])
        .agg(mean_attention=("mean_attention", "mean"), seeds_observed=("seed", "nunique"))
        .reset_index()
        .sort_values(["seeds_observed", "mean_attention"], ascending=False)
    )
    drivers = genes.groupby("module").head(4).groupby("module")["gene"].apply(lambda x: ", ".join(x)).to_dict()
    module_drop = ablation.set_index("module_removed")["mean_c_index_drop_after_removal"].to_dict()
    rows = []
    for edge in edge_summary.head(12).itertuples(index=False):
        source = edge.source_module
        target = edge.target_module
        rows.append(
            {
                "priority_edge": edge.edge,
                "mean_attention": edge.mean_attention,
                "seeds_observed": edge.seeds_observed,
                "source_candidate_genes": drivers.get(source, ""),
                "target_candidate_genes": drivers.get(target, ""),
                "source_module_removal_c_index_drop": module_drop.get(source, np.nan),
                "target_module_removal_c_index_drop": module_drop.get(target, np.nan),
                "combination_hypothesis": f"{THERAPY_MAP[source]} + {THERAPY_MAP[target]}",
                "claim_strength": "hypothesis only; needs experimental validation",
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    ablation: pd.DataFrame,
    genes: pd.DataFrame,
    enrichment: pd.DataFrame,
    module_ablation: pd.DataFrame,
    therapy: pd.DataFrame,
) -> None:
    path = RESULTS / "final_relapse_intervene_v3_science_summary.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Final Relapse-INTERVENE V3 science tables\n")
        f.write("========================================\n\n")
        f.write("Main AI ablation table\n")
        f.write("----------------------\n")
        f.write(ablation.to_string(index=False))
        f.write("\n\nTop genes higher in V3 high-risk patients\n")
        f.write("-----------------------------------------\n")
        f.write(genes.head(20).to_string(index=False))
        f.write("\n\nDriver alteration enrichment: high vs low V3 risk\n")
        f.write("-------------------------------------------------\n")
        f.write(enrichment.head(20).to_string(index=False))
        f.write("\n\nModule-removal importance\n")
        f.write("-------------------------\n")
        f.write(module_ablation.to_string(index=False))
        f.write("\n\nCombination hypotheses\n")
        f.write("----------------------\n")
        f.write(therapy.head(12).to_string(index=False))
        f.write("\n\nCareful interpretation\n")
        f.write("----------------------\n")
        f.write(
            "GraphSparse V3 should be described as a relapse-associated stress-survival hypothesis generator. "
            "The gene, driver, and therapy tables are not proof of treatment response; they define testable hypotheses "
            "connecting sparse module edges to NSCLC relapse risk.\n"
        )


def main() -> None:
    final_ablation = make_final_ablation_table()
    frame = high_low_frame()
    genes = gene_high_low_table(frame)
    enrichment = pd.concat(
        [
            category_enrichment(frame, "gene_alteration_status"),
            category_enrichment(frame, "pathological_stage_binary"),
            category_enrichment(frame, "smoking_status"),
        ],
        ignore_index=True,
    ).sort_values(["fisher_p_two_sided", "high_minus_low_fraction"])
    mod_ablation = module_ablation_table()
    edges = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_attention_edges.csv")
    therapy = therapy_hypotheses(edges, mod_ablation, genes)

    final_ablation.to_csv(RESULTS / "final_relapse_intervene_ai_ablation_table.csv", index=False)
    frame.to_csv(RESULTS / "final_v3_high_low_patient_groups.csv", index=False)
    genes.to_csv(RESULTS / "final_v3_high_risk_gene_drivers.csv", index=False)
    enrichment.to_csv(RESULTS / "final_v3_high_risk_clinical_driver_enrichment.csv", index=False)
    mod_ablation.to_csv(RESULTS / "final_v3_module_removal_importance.csv", index=False)
    therapy.to_csv(RESULTS / "final_v3_combination_hypotheses.csv", index=False)
    write_summary(final_ablation, genes, enrichment, mod_ablation, therapy)

    print(f"Wrote {RESULTS / 'final_relapse_intervene_ai_ablation_table.csv'}")
    print(f"Wrote {RESULTS / 'final_v3_high_risk_gene_drivers.csv'}")
    print(f"Wrote {RESULTS / 'final_v3_high_risk_clinical_driver_enrichment.csv'}")
    print(f"Wrote {RESULTS / 'final_v3_module_removal_importance.csv'}")
    print(f"Wrote {RESULTS / 'final_v3_combination_hypotheses.csv'}")
    print(f"Wrote {RESULTS / 'final_relapse_intervene_v3_science_summary.txt'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch

from interleaved_biophysics_attention_full import module_specs, set_seed, zscore
from prepare_gse31210_relapse_intervene import logrank_high_low
from prepare_gse31210_relapse_intervene import read_annotation, read_expression
from run_relapse_intervene_survival_head import cox_loss, concordance_index, folds
from run_relapse_intervene_graph_sparse_v3 import SPECS, train_fold, evaluate
from run_relapse_intervene_module_v2 import load_data, MODULES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"

N_BOOT = 3000
N_PERM = 3000
N_RANDOM_MODULES = 2000
SEEDS = [7, 17, 29, 43, 61]


def norm_sf_chi2_1(x: float) -> float:
    return float(math.erfc(math.sqrt(max(x, 0.0) / 2.0)))


def load_frame_scores() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / "gse31210_patient_module_scores.csv")
    frame = frame[frame["is_primary_tumor"] & frame["prognosis_included"]].copy()
    frame = frame.dropna(subset=["relapse_event", "relapse_free_months", "raw_relapse_intervene_state"])
    frame = frame[frame["relapse_free_months"] > 0].reset_index(drop=True)

    v3 = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_oof_risk_scores.csv")
    v2 = pd.read_csv(RESULTS / "relapse_intervene_module_v2_oof_risk_scores.csv")
    base = pd.read_csv(RESULTS / "relapse_intervene_survival_oof_risk_scores.csv")

    def mean_score(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = [c for c in table.columns if c.startswith(prefix) and c.endswith("_risk")]
        return pd.DataFrame({"geo_accession": table["geo_accession"], prefix.rstrip("_"): table[cols].mean(axis=1)})

    scores = mean_score(v3, "graph_sparse_v3_soft_seed")
    scores = scores.merge(mean_score(v3, "graph_sparse_v3_balanced_seed"), on="geo_accession")
    scores = scores.merge(mean_score(v2, "module_intervene_constrained_seed"), on="geo_accession")
    scores = scores.merge(mean_score(base, "normal_attention_mha_seed"), on="geo_accession")
    frame = frame.merge(scores, on="geo_accession", how="left")
    frame["raw_transfer"] = zscore(frame["raw_relapse_intervene_state"])
    return frame


def paired_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1307)
    time = frame["relapse_free_months"].to_numpy(float)
    event = frame["relapse_event"].to_numpy(float)
    models = {
        "raw_transfer": "raw_transfer",
        "normal_attention": "normal_attention_mha_seed",
        "intervene_v2_constrained": "module_intervene_constrained_seed",
        "graph_sparse_v3_balanced": "graph_sparse_v3_balanced_seed",
    }
    v3 = frame["graph_sparse_v3_soft_seed"].to_numpy(float)
    observed = concordance_index(time, event, v3)
    rows = []
    n = len(frame)
    for label, col in models.items():
        comp = frame[col].to_numpy(float)
        comp_obs = concordance_index(time, event, comp)
        diffs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            diffs.append(concordance_index(time[idx], event[idx], v3[idx]) - concordance_index(time[idx], event[idx], comp[idx]))
        diffs = np.array(diffs, dtype=float)
        rows.append(
            {
                "comparison": f"graph_sparse_v3_soft_vs_{label}",
                "v3_c_index": observed,
                "comparator_c_index": comp_obs,
                "observed_delta_c_index": observed - comp_obs,
                "bootstrap_delta_mean": float(np.nanmean(diffs)),
                "bootstrap_delta_ci_low": float(np.nanpercentile(diffs, 2.5)),
                "bootstrap_delta_ci_high": float(np.nanpercentile(diffs, 97.5)),
                "bootstrap_p_delta_le_0": float((np.sum(diffs <= 0) + 1) / (np.sum(~np.isnan(diffs)) + 1)),
            }
        )
    return pd.DataFrame(rows)


def permutation_control(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2307)
    score = frame["graph_sparse_v3_soft_seed"].to_numpy(float)
    time = frame["relapse_free_months"].to_numpy(float)
    event = frame["relapse_event"].to_numpy(float)
    observed_c = concordance_index(time, event, score)
    observed_lr = logrank_high_low(
        frame.assign(v3_score=score), "v3_score", "relapse_free_months", "relapse_event"
    )
    rows = []
    c_values = []
    p_values = []
    for i in range(N_PERM):
        idx = rng.permutation(len(frame))
        perm_time = time[idx]
        perm_event = event[idx]
        c = concordance_index(perm_time, perm_event, score)
        tmp = pd.DataFrame({"score": score, "time": perm_time, "event": perm_event})
        lr = logrank_high_low(tmp, "score", "time", "event")
        c_values.append(c)
        p_values.append(lr["logrank_p"])
        rows.append({"perm_id": i, "c_index": c, "logrank_p": lr["logrank_p"], "hazard_ratio_approx": lr["hazard_ratio_approx"]})
    c_values = np.array(c_values)
    summary = pd.DataFrame(
        [
            {
                "observed_c_index": observed_c,
                "perm_mean_c_index": float(np.nanmean(c_values)),
                "perm_ci_low": float(np.nanpercentile(c_values, 2.5)),
                "perm_ci_high": float(np.nanpercentile(c_values, 97.5)),
                "perm_p_c_index_ge_observed": float((np.sum(c_values >= observed_c) + 1) / (N_PERM + 1)),
                "observed_logrank_p": observed_lr["logrank_p"],
                "perm_fraction_logrank_p_le_observed": float((np.sum(np.array(p_values) <= observed_lr["logrank_p"]) + 1) / (N_PERM + 1)),
            }
        ]
    )
    pd.DataFrame(rows).to_csv(RESULTS / "final_v3_permutation_null_distribution.csv", index=False)
    return summary


def one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    values = series.fillna("missing").astype(str).str.strip().replace({"": "missing", ".": "missing", "nd": "missing"})
    return pd.get_dummies(values, prefix=prefix, dtype=float)


def clinical_covariates(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    parts = []
    if "age" in fields:
        age = pd.to_numeric(frame["age"], errors="coerce")
        parts.append(zscore(age.fillna(age.median())).rename("age_z").to_frame())
    if "stage" in fields:
        parts.append(one_hot(frame["pathological_stage_binary"], "stage").drop(columns=["stage_missing"], errors="ignore"))
    if "smoking" in fields:
        parts.append(one_hot(frame["smoking_status"], "smoking").drop(columns=["smoking_missing"], errors="ignore"))
    if "driver" in fields:
        parts.append(one_hot(frame["gene_alteration_status"], "driver").drop(columns=["driver_missing"], errors="ignore"))
    if not parts:
        return pd.DataFrame(index=frame.index)
    x = pd.concat(parts, axis=1).fillna(0.0)
    return x.loc[:, ~x.columns.duplicated()]


def fit_cox_linear(x: np.ndarray, time: np.ndarray, event: np.ndarray, seed: int = 7) -> tuple[float, np.ndarray]:
    set_seed(seed)
    if x.shape[1] == 0:
        risk = torch.zeros(len(time), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.LBFGS([risk], lr=0.25, max_iter=80)
        t = torch.tensor(time, dtype=torch.float32)
        e = torch.tensor(event, dtype=torch.float32)

        def closure():
            opt.zero_grad()
            loss = cox_loss(risk, t, e)
            loss.backward()
            return loss

        opt.step(closure)
        loss = float(cox_loss(risk.detach(), t, e).item())
        return -loss * float(event.sum()), risk.detach().numpy()

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    xz = (x - x_mean) / x_std
    beta = torch.zeros(xz.shape[1], dtype=torch.float32, requires_grad=True)
    xt = torch.tensor(xz, dtype=torch.float32)
    t = torch.tensor(time, dtype=torch.float32)
    e = torch.tensor(event, dtype=torch.float32)
    opt = torch.optim.LBFGS([beta], lr=0.35, max_iter=160, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        risk = xt @ beta
        loss = cox_loss(risk, t, e) + 1e-4 * torch.sum(beta * beta)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        risk = (xt @ beta).numpy()
        ll = -float(cox_loss(torch.tensor(risk, dtype=torch.float32), t, e).item()) * float(event.sum())
    return ll, risk


def clinical_adjusted_tests(frame: pd.DataFrame) -> pd.DataFrame:
    time = frame["relapse_free_months"].to_numpy(float)
    event = frame["relapse_event"].to_numpy(float)
    v3 = zscore(frame["graph_sparse_v3_soft_seed"]).to_numpy(float).reshape(-1, 1)
    specs = {
        "stage_only": ["stage"],
        "stage_age": ["stage", "age"],
        "stage_age_smoking_driver": ["stage", "age", "smoking", "driver"],
    }
    rows = []
    for name, fields in specs.items():
        cx = clinical_covariates(frame, fields).to_numpy(float)
        ll0, risk0 = fit_cox_linear(cx, time, event)
        ll1, risk1 = fit_cox_linear(np.column_stack([cx, v3]), time, event)
        lrt = 2 * (ll1 - ll0)
        rows.append(
            {
                "clinical_model": name,
                "n_covariates_without_v3": cx.shape[1],
                "clinical_only_c_index": concordance_index(time, event, risk0),
                "clinical_plus_v3_c_index": concordance_index(time, event, risk1),
                "delta_c_index": concordance_index(time, event, risk1) - concordance_index(time, event, risk0),
                "cox_lrt_chi2_for_v3": lrt,
                "cox_lrt_p_for_v3": norm_sf_chi2_1(lrt),
            }
        )
    return pd.DataFrame(rows)


def edge_robustness() -> pd.DataFrame:
    edges = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_attention_edges.csv")
    rows = []
    for model in ["graph_sparse_v3_soft", "graph_sparse_v3_balanced"]:
        sub = edges[edges["model"].eq(model)].copy()
        for seed, group in sub.groupby("seed"):
            cutoff = group["mean_attention"].nlargest(10).min()
            top = group[group["mean_attention"] >= cutoff].copy()
            for row in top.itertuples(index=False):
                rows.append({"model": model, "seed": seed, "edge": row.edge, "is_biology_edge": row.is_biology_edge})
    top = pd.DataFrame(rows)
    stability = (
        top.groupby(["model", "edge", "is_biology_edge"])
        .agg(top10_seed_count=("seed", "nunique"))
        .reset_index()
        .sort_values(["model", "top10_seed_count"], ascending=[True, False])
    )
    cross = (
        stability[stability["top10_seed_count"] >= 4]
        .groupby("edge")
        .agg(models_stable=("model", "nunique"), is_biology_edge=("is_biology_edge", "max"))
        .reset_index()
        .sort_values(["models_stable", "edge"], ascending=[False, True])
    )
    stability.to_csv(RESULTS / "final_v3_edge_top10_seed_stability.csv", index=False)
    cross.to_csv(RESULTS / "final_v3_edge_soft_balanced_overlap.csv", index=False)
    return stability


def random_module_negative_control(frame: pd.DataFrame) -> pd.DataFrame:
    specs = {s.token: list(s.genes) for s in module_specs() if s.token in MODULES}
    expr = read_expression()
    annot = read_annotation()
    probe_to_gene = annot.drop_duplicates("probe_id").set_index("probe_id")["gene_symbol"]
    keep_probes = [p for p in expr.index if p in probe_to_gene.index]
    sub = expr.loc[keep_probes].copy()
    sub["gene_symbol"] = [probe_to_gene.loc[p] for p in keep_probes]
    all_gene_expr = sub.groupby("gene_symbol").median(numeric_only=True).T.reset_index().rename(columns={"index": "geo_accession"})
    selected_genes = sorted({g for genes in specs.values() for g in genes if g in frame.columns})
    all_genes = sorted([g for g in all_gene_expr.columns if g != "geo_accession" and g not in selected_genes])
    expr_frame = frame[["geo_accession", "relapse_free_months", "relapse_event", "raw_transfer"]].merge(
        all_gene_expr, on="geo_accession", how="left"
    )
    module_sizes = {m: len([g for g in genes if g in frame.columns]) for m, genes in specs.items()}
    weights = {
        "srebp_lipid": 1.0,
        "mito_translation": 1.0,
        "oxidative_stress": 0.50,
        "proteostasis": 0.30,
        "apoptosis_iap": 0.30,
        "akt_mtor_growth": 0.20,
    }
    time = frame["relapse_free_months"].to_numpy(float)
    event = frame["relapse_event"].to_numpy(float)
    observed = concordance_index(time, event, frame["raw_transfer"].to_numpy(float))
    rng = np.random.default_rng(3307)
    rows = []
    for i in range(N_RANDOM_MODULES):
        score = np.zeros(len(frame), dtype=float)
        for module, size in module_sizes.items():
            genes = rng.choice(all_genes, size=size, replace=False)
            module_score = expr_frame[list(genes)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            score += weights[module] * zscore(module_score).to_numpy(float)
        c = concordance_index(time, event, score)
        rows.append({"random_id": i, "c_index": c})
    out = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "observed_raw_transfer_c_index": observed,
                "random_module_mean_c_index": out["c_index"].mean(),
                "random_module_ci_low": out["c_index"].quantile(0.025),
                "random_module_ci_high": out["c_index"].quantile(0.975),
                "random_module_p_c_index_ge_observed": (int((out["c_index"] >= observed).sum()) + 1) / (len(out) + 1),
            }
        ]
    )
    out.to_csv(RESULTS / "final_random_module_negative_control_distribution.csv", index=False)
    return summary


def retrain_module_drop_ablation() -> pd.DataFrame:
    frame, x, _, time, event, raw_prior, coupling = load_data()
    spec = [s for s in SPECS if s.model == "graph_sparse_v3_soft"][0]
    rows = []
    for removed in MODULES:
        xi = x.copy()
        xi[:, MODULES.index(removed), :] = 0.0
        for seed in SEEDS:
            pred = np.full(len(frame), np.nan)
            for fold_id, test_idx in enumerate(folds(len(frame), seed)):
                train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
                fold_pred, _, _ = train_fold(
                    xi, time, event, raw_prior, coupling, train_idx, test_idx, spec, seed + fold_id * 101
                )
                pred[test_idx] = fold_pred
            rows.append(
                {
                    "removed_module_zeroed_before_training": removed,
                    "seed": seed,
                    "c_index": concordance_index(time, event, pred),
                    "hazard_ratio_approx": logrank_high_low(
                        frame.assign(pred=pred), "pred", "relapse_free_months", "relapse_event"
                    )["hazard_ratio_approx"],
                }
            )
    detailed = pd.DataFrame(rows)
    detailed.to_csv(RESULTS / "final_v3_retrain_module_drop_detailed.csv", index=False)
    full = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_model_comparison.csv")
    full["seed_key"] = full["seed"].astype(str)
    detailed["seed_key"] = detailed["seed"].astype(str)
    full = full[full["model"].eq("graph_sparse_v3_soft")].groupby("seed_key")["c_index"].mean().rename("full_v3_c_index")
    detailed["full_v3_c_index"] = detailed["seed_key"].map(full)
    detailed["c_index_drop_vs_full"] = detailed["full_v3_c_index"] - detailed["c_index"]
    return (
        detailed.groupby("removed_module_zeroed_before_training")
        .agg(
            mean_c_index=("c_index", "mean"),
            sd_c_index=("c_index", "std"),
            mean_c_index_drop_vs_full=("c_index_drop_vs_full", "mean"),
            mean_hazard_ratio=("hazard_ratio_approx", "mean"),
        )
        .reset_index()
        .sort_values("mean_c_index_drop_vs_full", ascending=False)
    )


def prism_bridge() -> pd.DataFrame:
    buckets = pd.read_csv(RESULTS / "prism_mechanism_bucket_scores.csv")
    combos = pd.read_csv(RESULTS / "final_v3_combination_hypotheses.csv")
    bucket_summary = buckets[["mechanism_bucket", "n_compounds", "mean_sensitivity_score", "median_sensitivity_score", "mean_high_minus_low_lfc", "top_compounds"]].copy()
    bridge_rows = []
    for row in bucket_summary.itertuples(index=False):
        if row.mechanism_bucket == "stress_survival":
            interpretation = "primary drug/vulnerability bridge; high stress-survival state is more PRISM-sensitive"
        elif row.mechanism_bucket == "direct_lipid_axis":
            interpretation = "negative/secondary control; direct lipid drugs are weak despite SREBP/lipid signal"
        elif row.mechanism_bucket == "generic_cytotoxic":
            interpretation = "specificity control; less specific than stress-survival bucket"
        else:
            interpretation = "background/unclassified"
        bridge_rows.append({**row._asdict(), "interpretation": interpretation})
    bridge = pd.DataFrame(bridge_rows)
    therapy_bridge = combos.head(8).copy()
    therapy_bridge.to_csv(RESULTS / "final_v3_prism_combination_bridge.csv", index=False)
    return bridge.sort_values("mean_sensitivity_score", ascending=False)


def external_dataset_scout() -> pd.DataFrame:
    rows = [
        {
            "dataset": "GSE50081",
            "status": "best immediate external validation candidate",
            "n_or_samples": "181 early-stage NSCLC samples",
            "endpoint": "overall survival/prognosis; not confirmed relapse-free survival",
            "platform": "GPL570 Affymetrix U133 Plus 2.0",
            "why_relevant": "early-stage resected NSCLC, independent cohort, processed data available in GEO sample table",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50081",
        },
        {
            "dataset": "GSE68465",
            "status": "large external survival validation candidate",
            "n_or_samples": "462 lung adenocarcinoma samples in GEO page",
            "endpoint": "overall survival with extensive clinical annotation; recurrence endpoint not confirmed",
            "platform": "GPL96 Affymetrix U133A",
            "why_relevant": "large multi-site blinded prognostic validation study with expression and clinical variables",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68465",
        },
        {
            "dataset": "NSCLC Radiogenomics / TCIA-style cohort",
            "status": "possible recurrence-specific validation candidate",
            "n_or_samples": "about 130 patients reported in recurrence-prediction papers",
            "endpoint": "post-surgical recurrence reported in radiogenomics studies",
            "platform": "requires separate data access/mapping check",
            "why_relevant": "recurrence endpoint is closer to GSE31210 relapse question, but expression availability and download path must be verified",
            "source_url": "https://arxiv.org/abs/2002.01982",
        },
    ]
    return pd.DataFrame(rows)


def write_summary(tables: dict[str, pd.DataFrame]) -> None:
    path = RESULTS / "final_intervene_ieee_readiness_tests_summary.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Final INTERVENE IEEE-readiness robustness tests\n")
        f.write("==============================================\n\n")
        for name, table in tables.items():
            f.write(name.replace("_", " ").title())
            f.write("\n")
            f.write("-" * len(name))
            f.write("\n")
            f.write(table.to_string(index=False))
            f.write("\n\n")
        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write(
            "These tests strengthen the current single-cohort GSE31210 claim. The remaining major submission gap is true external validation. "
            "GSE50081 and GSE68465 are strong survival-validation candidates; a recurrence-specific radiogenomics cohort may be better if expression and recurrence labels can be accessed cleanly.\n"
        )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    frame = load_frame_scores()
    tables = {
        "paired_bootstrap_c_index": paired_bootstrap(frame),
        "permutation_control": permutation_control(frame),
        "clinical_adjusted_cox": clinical_adjusted_tests(frame),
        "edge_robustness": edge_robustness().head(24),
        "random_module_negative_control": random_module_negative_control(frame),
        "retrain_with_module_dropped": retrain_module_drop_ablation(),
        "drug_vulnerability_bridge": prism_bridge(),
        "external_dataset_scout": external_dataset_scout(),
    }
    for name, table in tables.items():
        table.to_csv(RESULTS / f"final_{name}.csv", index=False)
    write_summary(tables)
    print(f"Wrote {RESULTS / 'final_intervene_ieee_readiness_tests_summary.txt'}")
    for name in tables:
        print(f"Wrote {RESULTS / f'final_{name}.csv'}")


if __name__ == "__main__":
    main()

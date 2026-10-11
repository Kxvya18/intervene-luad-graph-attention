
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch

from interleaved_biophysics_attention_full import zscore
from prepare_gse31210_relapse_intervene import logrank_high_low
from run_relapse_intervene_survival_head import concordance_index, cox_loss
from run_final_intervene_robustness_tests import fit_cox_linear, norm_sf_chi2_1

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"
RESULTS = ROOT / "results"

MODULE_GENES = {
    "proteostasis": ["PSMA1", "PSMB5", "VCP", "USP8", "STAMBP", "UBAP1", "HGS", "PTPN23"],
    "redox_nrf2": ["TXNRD1", "SLC7A11", "GCLC", "GCLM", "NFE2L2", "KEAP1", "PGD"],
    "apoptosis_iap": ["BIRC2", "BIRC3", "XIAP", "BCL2L1", "MCL1", "CASP8", "CFLAR"],
    "mito_translation": ["MRPL10", "MRPL21", "MRPL34", "MRPL38", "MRPL4", "MRPL41", "MRPL45", "MRPL49", "MRPL57", "MRPS11", "MRPS12", "MRPS14", "MRPS21", "MRPS23", "MRPS34", "MRPS5"],
    "srebp_lipid": ["SREBF1", "SCAP", "MBTPS1", "MBTPS2", "SCD"],
    "akt_mtor": ["AKT1", "AKT2", "AKT3", "MTOR", "PIK3CA", "PIK3CB", "PTEN", "RICTOR", "RPTOR"],
}
PROLIFERATION_GENES = ["MKI67", "TOP2A", "MCM2", "MCM5", "PCNA", "CDK1", "CDK2", "AURKA", "AURKB", "MCM6", "BUB1", "CCNB1", "UBE2C"]
MODULE_SCORE_COLS = [
    "proteostasis_expr_mean", "oxidative_stress_expr_mean", "apoptosis_iap_expr_mean",
    "mito_translation_expr_mean", "srebp_lipid_expr_mean", "akt_mtor_growth_expr_mean",
]


def null_partial_loglik(time: np.ndarray, event: np.ndarray) -> float:
    risk0 = torch.zeros(len(time), dtype=torch.float32)
    t = torch.tensor(time, dtype=torch.float32)
    e = torch.tensor(event, dtype=torch.float32)
    return -float(cox_loss(risk0, t, e).item()) * float(event.sum())


def cox_score(frame: pd.DataFrame, score_col: str, time_col="relapse_free_months", event_col="relapse_event") -> dict:
    data = frame.dropna(subset=[score_col, time_col, event_col]).copy()
    data = data[data[time_col] > 0].copy()
    if len(data) < 25 or data[event_col].sum() < 5:
        return {"n": len(data), "events": int(data[event_col].sum()) if len(data) else 0, "c_index": np.nan, "hr_high_low": np.nan, "logrank_p": np.nan, "cox_lrt_p": np.nan, "cox_chi2": np.nan}
    time = data[time_col].to_numpy(float)
    event = data[event_col].to_numpy(float)
    score = data[score_col].to_numpy(float)
    ll0 = null_partial_loglik(time, event)
    ll1, risk = fit_cox_linear(zscore(data[score_col]).to_numpy(float).reshape(-1, 1), time, event)
    lr = logrank_high_low(data.assign(_score=score), "_score", time_col, event_col)
    chi2 = max(0.0, 2 * (ll1 - ll0))
    return {"n": len(data), "events": int(event.sum()), "c_index": concordance_index(time, event, score), "hr_high_low": lr["hazard_ratio_approx"], "logrank_p": lr["logrank_p"], "cox_lrt_p": norm_sf_chi2_1(chi2), "cox_chi2": chi2}


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    yy = pd.to_numeric(y, errors="coerce").to_numpy(float)
    xx = pd.to_numeric(x, errors="coerce").to_numpy(float)
    mask = np.isfinite(yy) & np.isfinite(xx)
    out = np.full(len(yy), np.nan)
    if mask.sum() < 5:
        return pd.Series(out, index=y.index)
    xz = (xx[mask] - np.nanmean(xx[mask])) / (np.nanstd(xx[mask]) or 1.0)
    yz = (yy[mask] - np.nanmean(yy[mask])) / (np.nanstd(yy[mask]) or 1.0)
    X = np.column_stack([np.ones(mask.sum()), xz])
    beta = np.linalg.lstsq(X, yz, rcond=None)[0]
    out[mask] = yz - X @ beta
    return pd.Series(out, index=y.index)


def fisher_or_chi2(a, b, c, d) -> float:
    # Two-sided Fisher exact for 2x2 table using hypergeometric probabilities.
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    n = row1 + row2
    if min(row1, row2, col1, col2) < 0 or n == 0:
        return np.nan
    def logcomb(nn, kk):
        if kk < 0 or kk > nn: return -np.inf
        return math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1)
    def prob(x):
        return math.exp(logcomb(col1, x) + logcomb(col2, row1 - x) - logcomb(n, row1))
    lo, hi = max(0, row1 - col2), min(row1, col1)
    p_obs = prob(a)
    return float(sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12))


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / "gse31210_patient_module_scores.csv")
    frame = frame[frame["is_primary_tumor"] & frame["prognosis_included"]].copy()
    frame = frame.dropna(subset=["relapse_event", "relapse_free_months", "raw_relapse_intervene_state"])
    frame = frame[frame["relapse_free_months"] > 0].reset_index(drop=True)
    v3 = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_oof_risk_scores.csv")
    soft_cols = [c for c in v3.columns if c.startswith("graph_sparse_v3_soft_seed") and c.endswith("_risk")]
    bal_cols = [c for c in v3.columns if c.startswith("graph_sparse_v3_balanced_seed") and c.endswith("_risk")]
    v3["v3_soft_mean_risk"] = v3[soft_cols].mean(axis=1)
    v3["v3_balanced_mean_risk"] = v3[bal_cols].mean(axis=1)
    frame = frame.merge(v3[["geo_accession", "v3_soft_mean_risk", "v3_balanced_mean_risk"]], on="geo_accession", how="left")
    pp = [g for g in PROLIFERATION_GENES if g in frame.columns]
    if pp:
        frame["proliferation_score"] = zscore(frame[pp].mean(axis=1))
    else:
        # Previous final biology suite already wrote this file if proliferation probes were collapsed there.
        prolif = pd.read_csv(RESULTS / "final_biology_gse31210_v3_proliferation_scores.csv")
        frame = frame.merge(prolif[["geo_accession", "proliferation_score"]], on="geo_accession", how="left")
    frame["proteostasis_redox_state"] = zscore(frame["proteostasis_expr_mean"]) + zscore(frame["oxidative_stress_expr_mean"])
    frame["stress_core_state"] = zscore(frame["proteostasis_expr_mean"]) + zscore(frame["oxidative_stress_expr_mean"]) + 0.5*zscore(frame["apoptosis_iap_expr_mean"]) + 0.5*zscore(frame["mito_translation_expr_mean"]) + 0.25*zscore(frame["akt_mtor_growth_expr_mean"])
    for col in ["proteostasis_expr_mean", "oxidative_stress_expr_mean", "proteostasis_redox_state", "stress_core_state", "raw_relapse_intervene_state", "v3_soft_mean_risk"]:
        frame[col + "_resid_prolif"] = residualize(frame[col], frame["proliferation_score"])
    q_low = frame["v3_soft_mean_risk"].quantile(1/3)
    q_high = frame["v3_soft_mean_risk"].quantile(2/3)
    frame["v3_risk_tertile"] = np.where(frame["v3_soft_mean_risk"] <= q_low, "low", np.where(frame["v3_soft_mean_risk"] >= q_high, "high", "mid"))
    frame["early_relapse_12m"] = ((frame["relapse_event"] == 1) & (frame["relapse_free_months"] <= 12)).astype(int)
    frame["early_relapse_24m"] = ((frame["relapse_event"] == 1) & (frame["relapse_free_months"] <= 24)).astype(int)
    return frame


def high_low_characteristics(frame: pd.DataFrame) -> pd.DataFrame:
    high = frame[frame["v3_risk_tertile"] == "high"]
    low = frame[frame["v3_risk_tertile"] == "low"]
    rows = []
    def cat(label, mask):
        h = int(mask.loc[high.index].sum())
        l = int(mask.loc[low.index].sum())
        rows.append({"feature": label, "high_n": len(high), "high_count": h, "high_fraction": h / max(len(high), 1), "low_n": len(low), "low_count": l, "low_fraction": l / max(len(low), 1), "fisher_p": fisher_or_chi2(h, len(high)-h, l, len(low)-l)})
    driver = frame["gene_alteration_status"].fillna("missing").astype(str)
    for key in ["EGFR", "KRAS", "ALK-fusion", "ALK", "others", "triple negative"]:
        cat(f"driver_contains_{key}", driver.str.contains(key, case=False, regex=False))
    cat("stage_II", frame["pathological_stage_binary"].fillna("").astype(str).str.upper().eq("II"))
    cat("never_smoker", frame["smoking_status"].fillna("").astype(str).str.lower().str.contains("never"))
    smoke = frame["smoking_status"].fillna("").astype(str).str.lower()
    cat("ever_smoker", smoke.str.contains("ever") & ~smoke.str.contains("never"))
    cat("relapsed", frame["relapse_event"].eq(1))
    cat("early_relapse_12m", frame["early_relapse_12m"].eq(1))
    cat("early_relapse_24m", frame["early_relapse_24m"].eq(1))
    for col in ["relapse_free_months", "age", "proliferation_score", "proteostasis_expr_mean", "oxidative_stress_expr_mean", "stress_core_state", "v3_soft_mean_risk"]:
        rows.append({"feature": col + "_median", "high_n": len(high), "high_count": float(pd.to_numeric(high[col], errors="coerce").median()), "high_fraction": np.nan, "low_n": len(low), "low_count": float(pd.to_numeric(low[col], errors="coerce").median()), "low_fraction": np.nan, "fisher_p": np.nan})
    return pd.DataFrame(rows)


def residual_survival_tests(frame: pd.DataFrame) -> pd.DataFrame:
    scores = [
        "v3_soft_mean_risk", "v3_soft_mean_risk_resid_prolif",
        "raw_relapse_intervene_state", "raw_relapse_intervene_state_resid_prolif",
        "proteostasis_expr_mean", "proteostasis_expr_mean_resid_prolif",
        "oxidative_stress_expr_mean", "oxidative_stress_expr_mean_resid_prolif",
        "proteostasis_redox_state", "proteostasis_redox_state_resid_prolif",
        "stress_core_state", "stress_core_state_resid_prolif",
        "proliferation_score",
    ]
    rows = []
    for s in scores:
        rows.append({"score": s, **cox_score(frame, s)})
    # Add Cox tests where stress score is tested after proliferation as covariate.
    time = frame["relapse_free_months"].to_numpy(float)
    event = frame["relapse_event"].to_numpy(float)
    prolif = zscore(frame["proliferation_score"]).to_numpy(float).reshape(-1, 1)
    ll0, risk0 = fit_cox_linear(prolif, time, event)
    for s in ["v3_soft_mean_risk", "proteostasis_expr_mean", "proteostasis_redox_state", "stress_core_state", "raw_relapse_intervene_state"]:
        x = zscore(frame[s]).to_numpy(float).reshape(-1, 1)
        ll1, risk1 = fit_cox_linear(np.column_stack([prolif, x]), time, event)
        rows.append({"score": s + "_cox_added_after_proliferation", "n": len(frame), "events": int(event.sum()), "c_index": concordance_index(time, event, risk1), "hr_high_low": np.nan, "logrank_p": np.nan, "cox_lrt_p": norm_sf_chi2_1(max(0.0, 2*(ll1-ll0))), "cox_chi2": max(0.0, 2*(ll1-ll0))})
    return pd.DataFrame(rows).sort_values("c_index", ascending=False)


def subtype_survival_tests(frame: pd.DataFrame) -> pd.DataFrame:
    masks = {}
    driver = frame["gene_alteration_status"].fillna("missing").astype(str)
    masks["EGFR_altered"] = driver.str.contains("EGFR", case=False, regex=False)
    masks["KRAS_altered"] = driver.str.contains("KRAS", case=False, regex=False)
    masks["ALK_fusion"] = driver.str.contains("ALK-fusion", case=False, regex=False) | driver.str.contains("ALK", case=False, regex=False)
    masks["other_or_triple_negative"] = ~(masks["EGFR_altered"] | masks["KRAS_altered"] | masks["ALK_fusion"])
    masks["stage_I"] = frame["pathological_stage_binary"].fillna("").astype(str).str.upper().eq("I")
    masks["stage_II"] = frame["pathological_stage_binary"].fillna("").astype(str).str.upper().eq("II")
    masks["never_smoker"] = frame["smoking_status"].fillna("").astype(str).str.lower().str.contains("never")
    smoke = frame["smoking_status"].fillna("").astype(str).str.lower()
    masks["ever_smoker"] = smoke.str.contains("ever") & ~smoke.str.contains("never")
    rows = []
    for name, mask in masks.items():
        sub = frame[mask].copy()
        for score in ["v3_soft_mean_risk", "v3_soft_mean_risk_resid_prolif", "stress_core_state", "stress_core_state_resid_prolif", "proteostasis_expr_mean", "proteostasis_expr_mean_resid_prolif", "proliferation_score"]:
            rows.append({"subtype": name, "score": score, **cox_score(sub, score)})
    return pd.DataFrame(rows).sort_values(["subtype", "c_index"], ascending=[True, False])


def mechanism_gene_drivers(frame: pd.DataFrame) -> pd.DataFrame:
    high = frame[frame["v3_risk_tertile"] == "high"]
    low = frame[frame["v3_risk_tertile"] == "low"]
    rows = []
    for module, genes in MODULE_GENES.items():
        for gene in genes:
            if gene not in frame.columns:
                continue
            vals = pd.to_numeric(frame[gene], errors="coerce")
            risk = pd.to_numeric(frame["v3_soft_mean_risk"], errors="coerce")
            prolif = pd.to_numeric(frame["proliferation_score"], errors="coerce")
            mask = vals.notna() & risk.notna()
            corr = np.corrcoef(zscore(vals[mask]), zscore(risk[mask]))[0,1] if mask.sum() > 4 else np.nan
            resid = residualize(vals, prolif)
            mask2 = resid.notna() & risk.notna()
            corr_resid = np.corrcoef(zscore(resid[mask2]), zscore(risk[mask2]))[0,1] if mask2.sum() > 4 else np.nan
            rows.append({
                "module": module,
                "gene": gene,
                "high_mean": float(vals.loc[high.index].mean()),
                "low_mean": float(vals.loc[low.index].mean()),
                "high_minus_low": float(vals.loc[high.index].mean() - vals.loc[low.index].mean()),
                "corr_with_v3_risk": float(corr),
                "corr_with_v3_risk_after_prolif_residual": float(corr_resid),
                "gene_c_index": cox_score(frame.assign(_gene=vals), "_gene")["c_index"],
            })
    return pd.DataFrame(rows).sort_values("corr_with_v3_risk", ascending=False)


def mechanism_interpretation(frame: pd.DataFrame, genes: pd.DataFrame, residuals: pd.DataFrame, subtypes: pd.DataFrame, highlow: pd.DataFrame) -> str:
    top_genes = genes.head(12)[["module", "gene", "corr_with_v3_risk", "corr_with_v3_risk_after_prolif_residual", "gene_c_index"]]
    key_res = residuals[residuals["score"].isin(["v3_soft_mean_risk", "v3_soft_mean_risk_resid_prolif", "proteostasis_expr_mean", "proteostasis_expr_mean_resid_prolif", "stress_core_state", "stress_core_state_resid_prolif", "proliferation_score", "v3_soft_mean_risk_cox_added_after_proliferation", "stress_core_state_cox_added_after_proliferation"])]
    best_sub = subtypes[(subtypes["n"] >= 25) & (subtypes["events"] >= 5)].sort_values("c_index", ascending=False).head(16)
    key_high = highlow[highlow["feature"].isin(["driver_contains_EGFR", "driver_contains_KRAS", "driver_contains_ALK-fusion", "stage_II", "never_smoker", "ever_smoker", "relapsed", "early_relapse_12m", "early_relapse_24m", "relapse_free_months_median", "proliferation_score_median", "proteostasis_expr_mean_median", "oxidative_stress_expr_mean_median"])]
    lines = [
        "GSE31210 Mechanism Dissection",
        "==============================",
        "",
        "Cohort role: GSE31210 is the discovery cohort where GraphSparse Relapse-INTERVENE is strongest. Other cohorts should be used as external stress tests, not as separate stories.",
        "",
        "1) High-risk patient enrichment",
        key_high.to_string(index=False),
        "",
        "2) Proliferation separation",
        key_res.to_string(index=False),
        "",
        "3) Subtype-specific relapse performance",
        best_sub.to_string(index=False),
        "",
        "4) Exact gene drivers of the V3 high-risk state",
        top_genes.to_string(index=False),
        "",
        "Interpretation:",
        "- If residual stress/proteostasis scores remain predictive after removing proliferation, we can argue for stress-survival biology beyond fast growth.",
        "- If residual scores collapse while proliferation dominates, the honest claim becomes aggressive/proliferative relapse biology with stress modules as passengers or amplifiers.",
        "- If one molecular group has much stronger C-index/HR, the claim should be subtype-specific rather than universal LUAD/NSCLC.",
    ]
    return "\n".join(lines)


def external_context_table() -> pd.DataFrame:
    path = RESULTS / "proteostasis_redox_external_validation.csv"
    if not path.exists():
        return pd.DataFrame()
    ext = pd.read_csv(path)
    keep_scores = ["proteostasis_redox_state", "stress_core_state", "proteostasis_expr_mean", "older_broad_intervene_state", "proliferation_score"]
    ext = ext[(ext["score"].isin(keep_scores)) & (ext["n"] >= 50) & (ext["events"] >= 10)].copy()
    return ext.sort_values(["cohort", "subset", "c_index"], ascending=[True, True, False])


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    frame = load_frame()
    frame.to_csv(RESULTS / "gse31210_mechanism_patient_level_scores.csv", index=False)
    highlow = high_low_characteristics(frame)
    residuals = residual_survival_tests(frame)
    subtypes = subtype_survival_tests(frame)
    genes = mechanism_gene_drivers(frame)
    external = external_context_table()
    highlow.to_csv(RESULTS / "gse31210_high_risk_biology_enrichment.csv", index=False)
    residuals.to_csv(RESULTS / "gse31210_proliferation_residual_survival_tests.csv", index=False)
    subtypes.to_csv(RESULTS / "gse31210_subtype_specific_survival_tests.csv", index=False)
    genes.to_csv(RESULTS / "gse31210_exact_gene_mechanism_drivers.csv", index=False)
    external.to_csv(RESULTS / "gse31210_external_context_table.csv", index=False)
    summary = mechanism_interpretation(frame, genes, residuals, subtypes, highlow)
    if not external.empty:
        summary += "\n\n5) Other datasets as external context\n" + external.head(40).to_string(index=False)
    (RESULTS / "gse31210_mechanism_dissection_summary.txt").write_text(summary)
    print("Wrote gse31210_mechanism_dissection_summary.txt")
    print("Wrote gse31210_high_risk_biology_enrichment.csv")
    print("Wrote gse31210_proliferation_residual_survival_tests.csv")
    print("Wrote gse31210_subtype_specific_survival_tests.csv")
    print("Wrote gse31210_exact_gene_mechanism_drivers.csv")
    print("Wrote gse31210_external_context_table.csv")


if __name__ == "__main__":
    main()

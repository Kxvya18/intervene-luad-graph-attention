
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from interleaved_biophysics_attention_full import zscore
from prepare_gse31210_relapse_intervene import logrank_high_low, read_expression, read_annotation
from run_relapse_intervene_survival_head import concordance_index, cox_loss
from run_final_intervene_robustness_tests import fit_cox_linear, norm_sf_chi2_1
from run_gse31210_mechanism_dissection import residualize

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"
RESULTS = ROOT / "results"

AXES = {
    "proteasome_core": ["PSMA1", "PSMB5"],
    "ubiquitin_vcp_axis": ["VCP", "USP8", "STAMBP", "UBAP1", "HGS", "PTPN23"],
    "upr_chaperone_axis": ["HSPA5", "HSP90AA1", "HSP90AB1", "HSP90B1", "HSPB1", "CALR", "CANX", "XBP1", "ATF4", "ATF6", "ERN1", "EIF2AK3", "DDIT3"],
    "mito_translation_core": ["MRPL10", "MRPL21", "MRPL34", "MRPL38", "MRPL4", "MRPL41", "MRPL45", "MRPL49", "MRPL57", "MRPS11", "MRPS12", "MRPS14", "MRPS21", "MRPS23", "MRPS34", "MRPS5"],
    "mito_top_relapse": ["MRPL21", "MRPS23", "MRPL38", "MRPS34", "MRPS11", "MRPL10"],
    "redox_buffering": ["TXNRD1", "SLC7A11", "GCLC", "GCLM", "NFE2L2", "KEAP1", "PGD"],
}
PROLIFERATION_GENES = ["MKI67", "TOP2A", "MCM2", "MCM5", "PCNA", "CDK1", "CDK2", "AURKA", "AURKB", "MCM6", "BUB1", "CCNB1", "UBE2C"]


def collapse_gene_expression(expr: pd.DataFrame, annot: pd.DataFrame, genes: set[str]) -> pd.DataFrame:
    ann = annot[annot["gene_symbol"].isin(genes)].drop_duplicates("probe_id")
    probe_to_gene = ann.set_index("probe_id")["gene_symbol"]
    keep = [p for p in expr.index if p in probe_to_gene.index]
    sub = expr.loc[keep].copy()
    sub["gene_symbol"] = [probe_to_gene.loc[p] for p in keep]
    return sub.groupby("gene_symbol").median(numeric_only=True).T.reset_index().rename(columns={"index": "geo_accession"})


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


def fit_added_after(frame: pd.DataFrame, base_col: str, add_col: str) -> dict:
    data = frame.dropna(subset=[base_col, add_col, "relapse_free_months", "relapse_event"]).copy()
    time = data["relapse_free_months"].to_numpy(float)
    event = data["relapse_event"].to_numpy(float)
    base = zscore(data[base_col]).to_numpy(float).reshape(-1, 1)
    add = zscore(data[add_col]).to_numpy(float).reshape(-1, 1)
    ll0, risk0 = fit_cox_linear(base, time, event)
    ll1, risk1 = fit_cox_linear(np.column_stack([base, add]), time, event)
    chi2 = max(0.0, 2 * (ll1 - ll0))
    return {"test": f"{add_col}_added_after_{base_col}", "n": len(data), "events": int(event.sum()), "base_c_index": concordance_index(time, event, risk0), "base_plus_add_c_index": concordance_index(time, event, risk1), "delta_c_index": concordance_index(time, event, risk1) - concordance_index(time, event, risk0), "lrt_p_for_added_score": norm_sf_chi2_1(chi2), "lrt_chi2_for_added_score": chi2}


def load_frame() -> pd.DataFrame:
    base = pd.read_csv(PROCESSED / "gse31210_patient_module_scores.csv")
    base = base[base["is_primary_tumor"] & base["prognosis_included"]].copy()
    base = base.dropna(subset=["relapse_event", "relapse_free_months", "raw_relapse_intervene_state"])
    base = base[base["relapse_free_months"] > 0].reset_index(drop=True)
    v3 = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_oof_risk_scores.csv")
    soft_cols = [c for c in v3.columns if c.startswith("graph_sparse_v3_soft_seed") and c.endswith("_risk")]
    v3["v3_soft_mean_risk"] = v3[soft_cols].mean(axis=1)
    base = base.merge(v3[["geo_accession", "v3_soft_mean_risk"]], on="geo_accession", how="left")
    genes = set(PROLIFERATION_GENES)
    for g in AXES.values(): genes.update(g)
    expr = read_expression()
    annot = read_annotation()
    gene_expr = collapse_gene_expression(expr, annot, genes)
    # Add only genes missing from base, to avoid duplicate columns.
    missing_gene_cols = [c for c in gene_expr.columns if c == "geo_accession" or c not in base.columns]
    frame = base.merge(gene_expr[missing_gene_cols], on="geo_accession", how="left")
    pp = [g for g in PROLIFERATION_GENES if g in frame.columns]
    frame["proliferation_score"] = zscore(frame[pp].mean(axis=1))
    for axis, genes in AXES.items():
        present = [g for g in genes if g in frame.columns]
        frame[f"{axis}_score"] = zscore(frame[present].mean(axis=1)) if present else np.nan
        frame[f"{axis}_genes_present"] = len(present)
        frame[f"{axis}_resid_prolif"] = residualize(frame[f"{axis}_score"], frame["proliferation_score"])
    frame["mito_redox_score"] = zscore(frame["mito_top_relapse_score"]) + zscore(frame["redox_buffering_score"])
    frame["mito_redox_resid_prolif"] = residualize(frame["mito_redox_score"], frame["proliferation_score"])
    frame["proteasome_mito_redox_score"] = zscore(frame["proteasome_core_score"]) + zscore(frame["mito_top_relapse_score"]) + zscore(frame["redox_buffering_score"])
    frame["proteasome_mito_redox_resid_prolif"] = residualize(frame["proteasome_mito_redox_score"], frame["proliferation_score"])
    hi_p = frame["proteasome_core_score"] >= frame["proteasome_core_score"].median()
    hi_mr = frame["mito_redox_score"] >= frame["mito_redox_score"].median()
    frame["proteasome_mito_redox_group"] = np.select([hi_p & hi_mr, hi_p & ~hi_mr, ~hi_p & hi_mr], ["high_both", "proteasome_only", "mito_redox_only"], default="low_both")
    # Score-coded group: 3 high both, 2 proteasome only, 1 mito redox only, 0 low both.
    frame["proteasome_mito_redox_group_score"] = frame["proteasome_mito_redox_group"].map({"low_both": 0, "mito_redox_only": 1, "proteasome_only": 2, "high_both": 3}).astype(float)
    frame["mrpl21_resid_prolif"] = residualize(frame["MRPL21"], frame["proliferation_score"])
    return frame


def axis_survival_table(frame: pd.DataFrame) -> pd.DataFrame:
    scores = [
        "proteasome_core_score", "proteasome_core_resid_prolif",
        "ubiquitin_vcp_axis_score", "ubiquitin_vcp_axis_resid_prolif",
        "upr_chaperone_axis_score", "upr_chaperone_axis_resid_prolif",
        "mito_translation_core_score", "mito_translation_core_resid_prolif",
        "mito_top_relapse_score", "mito_top_relapse_resid_prolif",
        "redox_buffering_score", "redox_buffering_resid_prolif",
        "mito_redox_score", "mito_redox_resid_prolif",
        "proteasome_mito_redox_score", "proteasome_mito_redox_resid_prolif",
        "MRPL21", "mrpl21_resid_prolif", "TXNRD1", "PSMA1", "PSMB5", "VCP", "proliferation_score",
        "proteasome_mito_redox_group_score",
    ]
    rows = [{"score": s, **cox_score(frame, s)} for s in scores if s in frame.columns]
    return pd.DataFrame(rows).sort_values("c_index", ascending=False)


def combination_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in frame.groupby("proteasome_mito_redox_group"):
        rows.append({
            "group": group,
            "n": len(sub),
            "events": int(sub["relapse_event"].sum()),
            "event_fraction": float(sub["relapse_event"].mean()),
            "early_24m_events": int(((sub["relapse_event"] == 1) & (sub["relapse_free_months"] <= 24)).sum()),
            "median_rfs_months": float(sub["relapse_free_months"].median()),
            "median_proliferation": float(sub["proliferation_score"].median()),
            "median_v3_risk": float(sub["v3_soft_mean_risk"].median()),
        })
    out = pd.DataFrame(rows)
    order = {"low_both": 0, "mito_redox_only": 1, "proteasome_only": 2, "high_both": 3}
    return out.assign(order=out["group"].map(order)).sort_values("order").drop(columns="order")


def independence_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in ["proteasome_core_score", "mito_top_relapse_score", "redox_buffering_score", "mito_redox_score", "proteasome_mito_redox_score", "MRPL21", "PSMA1", "PSMB5"]:
        rows.append(fit_added_after(frame, "proliferation_score", score))
    for score in ["mito_redox_score", "redox_buffering_score", "MRPL21"]:
        rows.append(fit_added_after(frame, "proteasome_core_score", score))
    return pd.DataFrame(rows).sort_values("lrt_p_for_added_score")


def gene_table(frame: pd.DataFrame) -> pd.DataFrame:
    genes = sorted(set(g for genes in AXES.values() for g in genes if g in frame.columns) | {"PSMA1", "PSMB5", "MRPL21", "TXNRD1", "VCP"})
    rows = []
    for gene in genes:
        vals = pd.to_numeric(frame[gene], errors="coerce")
        risk = pd.to_numeric(frame["v3_soft_mean_risk"], errors="coerce")
        resid = residualize(vals, frame["proliferation_score"])
        mask = vals.notna() & risk.notna()
        maskr = resid.notna() & risk.notna()
        rows.append({
            "gene": gene,
            "corr_v3": float(np.corrcoef(zscore(vals[mask]), zscore(risk[mask]))[0, 1]) if mask.sum() > 4 else np.nan,
            "corr_v3_after_prolif_residual": float(np.corrcoef(zscore(resid[maskr]), zscore(risk[maskr]))[0, 1]) if maskr.sum() > 4 else np.nan,
            "c_index": cox_score(frame.assign(_g=vals), "_g")["c_index"],
            "resid_c_index": cox_score(frame.assign(_g=resid), "_g")["c_index"],
        })
    return pd.DataFrame(rows).sort_values("corr_v3", ascending=False)


def write_summary(axis: pd.DataFrame, combo: pd.DataFrame, indep: pd.DataFrame, genes: pd.DataFrame) -> None:
    lines = [
        "GSE31210 Refined Mechanism Axes",
        "=================================",
        "",
        "Purpose: split broad proteostasis into proteasome core, ubiquitin/VCP, and UPR/chaperone axes; test MRPL21/mitochondrial translation; test combined proteasome + mito/redox relapse state.",
        "",
        "1) Axis survival ranking",
        axis.head(24).to_string(index=False),
        "",
        "2) Proteasome + mito/redox combination groups",
        combo.to_string(index=False),
        "",
        "3) Added-value tests after proliferation or proteasome",
        indep.to_string(index=False),
        "",
        "4) Individual gene ranking",
        genes.head(30).to_string(index=False),
        "",
        "Paper interpretation guide:",
        "- If proteasome_core_resid_prolif is strong, the signal is not only cell-cycle/proliferation.",
        "- If high_both has the highest relapse fraction or earliest relapse, use this as the biological hypothesis: proteasome burden plus mito/redox adaptation marks a relapse-prone state.",
        "- If UPR/chaperone is weak, avoid claiming global ER stress/UPR unless literature supports it separately.",
    ]
    (RESULTS / "gse31210_refined_mechanism_axes_summary.txt").write_text("\n".join(lines))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    frame = load_frame()
    axis = axis_survival_table(frame)
    combo = combination_table(frame)
    indep = independence_table(frame)
    genes = gene_table(frame)
    frame.to_csv(RESULTS / "gse31210_refined_mechanism_patient_scores.csv", index=False)
    axis.to_csv(RESULTS / "gse31210_refined_axis_survival_tests.csv", index=False)
    combo.to_csv(RESULTS / "gse31210_proteasome_mito_redox_combination_groups.csv", index=False)
    indep.to_csv(RESULTS / "gse31210_refined_axis_independence_tests.csv", index=False)
    genes.to_csv(RESULTS / "gse31210_refined_gene_rankings.csv", index=False)
    write_summary(axis, combo, indep, genes)
    print("Wrote gse31210_refined_mechanism_axes_summary.txt")
    print("Wrote gse31210_refined_axis_survival_tests.csv")
    print("Wrote gse31210_proteasome_mito_redox_combination_groups.csv")
    print("Wrote gse31210_refined_axis_independence_tests.csv")
    print("Wrote gse31210_refined_gene_rankings.csv")


if __name__ == "__main__":
    main()

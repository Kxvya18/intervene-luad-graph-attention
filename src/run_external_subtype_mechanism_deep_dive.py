
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
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed" / "relapse"

MODULES = [
    "srebp_lipid",
    "mito_translation",
    "oxidative_stress",
    "proteostasis",
    "apoptosis_iap",
    "akt_mtor_growth",
]
MODULE_COLS = [f"{m}_expr_mean" for m in MODULES]


def clean_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_z_train_apply(train: pd.Series, test: pd.Series) -> tuple[np.ndarray, np.ndarray, float, float]:
    tr = clean_num(train).to_numpy(float)
    te = clean_num(test).to_numpy(float)
    mu = float(np.nanmean(tr))
    sd = float(np.nanstd(tr))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    tr_z = (tr - mu) / sd
    te_z = (te - mu) / sd
    tr_z = np.nan_to_num(tr_z, nan=0.0)
    te_z = np.nan_to_num(te_z, nan=0.0)
    return tr_z, te_z, mu, sd


def null_partial_loglik(time: np.ndarray, event: np.ndarray) -> float:
    risk0 = torch.zeros(len(time), dtype=torch.float32)
    t = torch.tensor(time, dtype=torch.float32)
    e = torch.tensor(event, dtype=torch.float32)
    return -float(cox_loss(risk0, t, e).item()) * float(event.sum())


def cox_lrt_one_score(frame: pd.DataFrame, score_col: str, time_col: str, event_col: str) -> dict:
    data = frame.dropna(subset=[score_col, time_col, event_col]).copy()
    data = data[data[time_col] > 0].copy()
    if len(data) < 30 or data[event_col].sum() < 5:
        return {
            "n": len(data), "events": int(data[event_col].sum()) if len(data) else 0,
            "c_index": np.nan, "hr_high_low": np.nan, "logrank_p": np.nan,
            "cox_lrt_p": np.nan, "cox_chi2": np.nan,
        }
    time = data[time_col].to_numpy(float)
    event = data[event_col].to_numpy(float)
    raw_score = data[score_col].to_numpy(float)
    score = zscore(data[score_col]).to_numpy(float).reshape(-1, 1)
    ll0 = null_partial_loglik(time, event)
    ll1, risk = fit_cox_linear(score, time, event)
    lr = logrank_high_low(data.assign(_score=data[score_col]), "_score", time_col, event_col)
    return {
        "n": len(data),
        "events": int(event.sum()),
        "c_index": concordance_index(time, event, raw_score),
        "hr_high_low": lr["hazard_ratio_approx"],
        "logrank_p": lr["logrank_p"],
        "cox_lrt_p": norm_sf_chi2_1(2 * (ll1 - ll0)),
        "cox_chi2": max(0.0, 2 * (ll1 - ll0)),
    }


def load_gse31210() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / "GSE31210" / "gse31210_patient_module_scores.csv")
    frame = frame[frame["is_primary_tumor"] & frame["prognosis_included"]].copy()
    frame = frame.dropna(subset=["relapse_event", "relapse_free_months", "raw_relapse_intervene_state"])
    frame = frame[frame["relapse_free_months"] > 0].reset_index(drop=True)
    v3 = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_oof_risk_scores.csv")
    cols = [c for c in v3.columns if c.startswith("graph_sparse_v3_soft_seed") and c.endswith("_risk")]
    v3["v3_soft_mean_risk"] = v3[cols].mean(axis=1)
    frame = frame.merge(v3[["geo_accession", "v3_soft_mean_risk"]], on="geo_accession", how="left")
    return frame


def load_gse50081() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / "GSE50081" / "gse50081_intervene_external_scores.csv")
    frame = frame.dropna(subset=["recurrence_event", "disease_free_survival_time"])
    frame = frame[frame["disease_free_survival_time"] > 0].reset_index(drop=True)
    return frame


def train_linear_cox_transfer(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target_score: str | None = None) -> tuple[pd.DataFrame, dict]:
    train_x = []
    test_x = []
    scaling = []
    for col in feature_cols:
        tr_z, te_z, mu, sd = safe_z_train_apply(train[col], test[col])
        train_x.append(tr_z)
        test_x.append(te_z)
        scaling.append({"feature": col, "train_mean": mu, "train_sd": sd})
    x_train = np.column_stack(train_x)
    x_test = np.column_stack(test_x)
    time = train["relapse_free_months"].to_numpy(float)
    event = train["relapse_event"].to_numpy(float)
    ll, train_risk = fit_cox_linear(x_train, time, event)

    # Refit a ridge-like linear mapping from standardized module values to the learned V3 score.
    # This gives a transparent transfer score whose coefficients can be inspected externally.
    if target_score is not None:
        y = zscore(train[target_score]).to_numpy(float)
        lam = 0.5
        xtx = x_train.T @ x_train + lam * np.eye(x_train.shape[1])
        beta = np.linalg.solve(xtx, x_train.T @ y)
        intercept = float(np.nanmean(y) - np.nanmean(x_train, axis=0) @ beta)
        train_risk = intercept + x_train @ beta
        test_risk = intercept + x_test @ beta
        model_kind = f"ridge_projection_to_{target_score}"
    else:
        # Approximate coefficients by least-squares projection of fitted Cox risk.
        beta = np.linalg.lstsq(x_train, train_risk, rcond=None)[0]
        intercept = 0.0
        test_risk = x_test @ beta
        model_kind = "cox_linear_module_transfer"

    coef = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": beta,
        "abs_coefficient": np.abs(beta),
        "model_kind": model_kind,
    }).sort_values("abs_coefficient", ascending=False)
    meta = {
        "model_kind": model_kind,
        "train_c_index": concordance_index(time, event, train_risk),
        "train_events": int(event.sum()),
        "train_n": len(train),
        "intercept": intercept,
    }
    return test.assign(learned_transfer_risk=test_risk), {"coef": coef, "scaling": pd.DataFrame(scaling), "meta": meta}


def subgroup_masks(g: pd.DataFrame) -> dict[str, pd.Series]:
    hist = g.get("histology", pd.Series("", index=g.index)).astype(str).str.lower()
    stage = g.get("stage", pd.Series("", index=g.index)).astype(str).str.lower().str.replace(" ", "", regex=False)
    smoking = g.get("smoking", pd.Series("", index=g.index)).astype(str).str.lower()
    masks = {
        "all_gse50081": pd.Series(True, index=g.index),
        "adenocarcinoma_only": hist.str.contains("adenocarcinoma", na=False),
        "non_adenocarcinoma": ~hist.str.contains("adenocarcinoma", na=False),
        "stage_i_only": stage.str.startswith("1") | stage.str.startswith("i"),
        "stage_ia_only": stage.isin(["1a", "ia"]),
        "stage_ib_only": stage.isin(["1b", "ib"]),
        "adeno_stage_i": hist.str.contains("adenocarcinoma", na=False) & (stage.str.startswith("1") | stage.str.startswith("i")),
        "smoker_current_or_former": smoking.str.contains("smoker|ever|current|former", na=False) & ~smoking.str.contains("never", na=False),
        "never_smoker": smoking.str.contains("never", na=False),
    }
    return masks


def subgroup_validation(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_cols = ["fixed_intervene_state", "learned_transfer_risk", "proliferation_score"] + MODULE_COLS
    for group_name, mask in subgroup_masks(g).items():
        sub = g[mask].copy()
        for score in score_cols:
            if score not in sub.columns:
                continue
            stats = cox_lrt_one_score(sub, score, "disease_free_survival_time", "recurrence_event")
            rows.append({"subgroup": group_name, "score": score, **stats})
    return pd.DataFrame(rows).sort_values(["score", "subgroup"])


def direction_concordance(gse31210: pd.DataFrame, gse50081: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, col in zip(MODULES, MODULE_COLS):
        g1 = cox_lrt_one_score(gse31210, col, "relapse_free_months", "relapse_event")
        g2 = cox_lrt_one_score(gse50081, col, "disease_free_survival_time", "recurrence_event")
        c1 = g1["c_index"]
        c2 = g2["c_index"]
        rows.append({
            "module": m,
            "gse31210_c_index": c1,
            "gse31210_direction": "risk_high" if c1 >= 0.5 else "risk_low",
            "gse31210_p": g1["cox_lrt_p"],
            "gse50081_c_index": c2,
            "gse50081_direction": "risk_high" if c2 >= 0.5 else "risk_low",
            "gse50081_p": g2["cox_lrt_p"],
            "direction_concordant": (c1 >= 0.5 and c2 >= 0.5) or (c1 < 0.5 and c2 < 0.5),
            "external_effect_strength_abs_c_minus_0_5": abs(c2 - 0.5) if np.isfinite(c2) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("external_effect_strength_abs_c_minus_0_5", ascending=False)


def internal_mechanism_table(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in ["raw_relapse_intervene_state", "v3_soft_mean_risk"] + MODULE_COLS:
        stats = cox_lrt_one_score(g, score, "relapse_free_months", "relapse_event")
        rows.append({"cohort": "GSE31210", "score": score, **stats})
    return pd.DataFrame(rows).sort_values("c_index", ascending=False)


def clinical_external_adjustments(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_cols = ["fixed_intervene_state", "learned_transfer_risk"]
    parts = []
    if "age" in g:
        age = pd.to_numeric(g["age"], errors="coerce")
        parts.append(zscore(age.fillna(age.median())).rename("age_z").to_frame())
    for col in ["stage", "histology", "smoking"]:
        if col in g:
            parts.append(pd.get_dummies(g[col].fillna("missing").astype(str), prefix=col, dtype=float))
    if parts:
        cov = pd.concat(parts, axis=1).fillna(0.0)
    else:
        cov = pd.DataFrame(index=g.index)
    time = g["disease_free_survival_time"].to_numpy(float)
    event = g["recurrence_event"].to_numpy(float)
    ll0, risk0 = fit_cox_linear(cov.to_numpy(float), time, event)
    for score in score_cols:
        x = zscore(g[score]).to_numpy(float).reshape(-1, 1)
        ll1, risk1 = fit_cox_linear(np.column_stack([cov.to_numpy(float), x]), time, event)
        rows.append({
            "test": f"GSE50081 {score} adjusted for age/stage/histology/smoking",
            "n": len(g),
            "events": int(event.sum()),
            "clinical_only_c_index": concordance_index(time, event, risk0),
            "clinical_plus_score_c_index": concordance_index(time, event, risk1),
            "delta_c_index": concordance_index(time, event, risk1) - concordance_index(time, event, risk0),
            "lrt_chi2_for_score": 2 * (ll1 - ll0),
            "lrt_p_for_score": norm_sf_chi2_1(2 * (ll1 - ll0)),
        })
    return pd.DataFrame(rows)


def write_summary(internal: pd.DataFrame, transfer_meta: dict, subgroup: pd.DataFrame, concord: pd.DataFrame, clinical: pd.DataFrame) -> None:
    best_sub = subgroup.dropna(subset=["c_index"]).copy()
    best_sub = best_sub[(best_sub["events"] >= 5) & (best_sub["n"] >= 30)].sort_values("c_index", ascending=False).head(12)
    lines = []
    lines.append("Deep External/Subtype Mechanism Upgrade")
    lines.append("========================================")
    lines.append("")
    lines.append("Question: can the relapse-associated stress-survival signal be made more universal or mechanistically sharper?")
    lines.append("")
    lines.append("1) Internal GSE31210 mechanism ranking")
    lines.append(internal.to_string(index=False))
    lines.append("")
    lines.append("2) Learned transfer model")
    lines.append(str(transfer_meta["meta"]))
    lines.append(transfer_meta["coef"].to_string(index=False))
    lines.append("")
    lines.append("3) Best external/subtype signals in GSE50081")
    lines.append(best_sub.to_string(index=False))
    lines.append("")
    lines.append("4) Direction concordance across GSE31210 and GSE50081")
    lines.append(concord.to_string(index=False))
    lines.append("")
    lines.append("5) External clinical-adjusted transfer")
    lines.append(clinical.to_string(index=False))
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- If a score validates only in adenocarcinoma/stage-I subsets, the claim becomes subtype-specific rather than universal.")
    lines.append("- If module directions disagree across cohorts, the biology may be context-dependent or the fixed transfer score may be missing cohort-specific normalization/treatment biology.")
    lines.append("- A breakthrough-grade claim needs repeated external agreement; a single strong internal cohort plus failed external cohort is a strong hypothesis, not proof of causation.")
    (RESULTS / "deep_relapse_mechanism_upgrade_summary.txt").write_text("\n".join(lines))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    gse31210 = load_gse31210()
    gse50081 = load_gse50081()
    gse50081, transfer = train_linear_cox_transfer(gse31210, gse50081, MODULE_COLS, target_score="v3_soft_mean_risk")
    internal = internal_mechanism_table(gse31210)
    subgroup = subgroup_validation(gse50081)
    concord = direction_concordance(gse31210, gse50081)
    clinical = clinical_external_adjustments(gse50081)

    internal.to_csv(RESULTS / "deep_internal_gse31210_mechanism_ranking.csv", index=False)
    gse50081.to_csv(RESULTS / "deep_external_gse50081_scores_with_transfer.csv", index=False)
    transfer["coef"].to_csv(RESULTS / "deep_external_gse50081_transfer_model_coefficients.csv", index=False)
    transfer["scaling"].to_csv(RESULTS / "deep_external_gse50081_transfer_model_scaling.csv", index=False)
    subgroup.to_csv(RESULTS / "deep_external_gse50081_subgroup_validation.csv", index=False)
    concord.to_csv(RESULTS / "deep_external_gse50081_module_concordance.csv", index=False)
    clinical.to_csv(RESULTS / "deep_external_gse50081_clinical_adjusted_transfer.csv", index=False)
    write_summary(internal, transfer, subgroup, concord, clinical)

    print("Wrote deep_internal_gse31210_mechanism_ranking.csv")
    print("Wrote deep_external_gse50081_scores_with_transfer.csv")
    print("Wrote deep_external_gse50081_transfer_model_coefficients.csv")
    print("Wrote deep_external_gse50081_subgroup_validation.csv")
    print("Wrote deep_external_gse50081_module_concordance.csv")
    print("Wrote deep_external_gse50081_clinical_adjusted_transfer.csv")
    print("Wrote deep_relapse_mechanism_upgrade_summary.txt")


if __name__ == "__main__":
    main()

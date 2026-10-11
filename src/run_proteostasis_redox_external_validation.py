
from __future__ import annotations

from pathlib import Path
import csv
import gzip
import re
import numpy as np
import pandas as pd
import torch

from interleaved_biophysics_attention_full import module_specs, zscore
from prepare_gse31210_relapse_intervene import logrank_high_low
from run_relapse_intervene_survival_head import concordance_index, cox_loss
from run_final_intervene_robustness_tests import fit_cox_linear, norm_sf_chi2_1

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "relapse"
PROCESSED = ROOT / "data" / "processed" / "relapse"
RESULTS = ROOT / "results"

MODULES = ["srebp_lipid", "mito_translation", "oxidative_stress", "proteostasis", "apoptosis_iap", "akt_mtor_growth"]
PROLIFERATION_GENES = ["MKI67", "TOP2A", "MCM2", "MCM5", "PCNA", "CDK1", "CDK2", "AURKA", "AURKB", "MCM6", "BUB1", "CCNB1", "UBE2C"]


def normalize_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", k.strip().lower().replace("(", "").replace(")", "")).strip("_")


def parse_value(v):
    if v is None:
        return np.nan
    t = str(v).strip().strip(chr(34))
    return np.nan if t in {"", "--", "not known", "NA", "nan"} else t


def parse_geo_metadata(path: Path) -> pd.DataFrame:
    recs = []
    titles = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_geo_accession"):
                parts = next(csv.reader([line], delimiter="\t"))
                recs = [{"geo_accession": str(parse_value(x))} for x in parts[1:]]
            elif line.startswith("!Sample_title"):
                parts = next(csv.reader([line], delimiter="\t"))
                titles = [str(parse_value(x)) for x in parts[1:]]
            elif line.startswith("!Sample_source_name_ch1"):
                parts = next(csv.reader([line], delimiter="\t"))
                if not recs:
                    recs = [{} for _ in parts[1:]]
                for rec, val in zip(recs, parts[1:]):
                    rec["source_name"] = parse_value(val)
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = next(csv.reader([line], delimiter="\t"))
                vals = [parse_value(x) for x in parts[1:]]
                if not recs:
                    recs = [{} for _ in vals]
                for rec, val in zip(recs, vals):
                    if pd.isna(val) or ":" not in str(val):
                        continue
                    k, raw = str(val).split(":", 1)
                    rec[normalize_key(k)] = parse_value(raw)
    for i, rec in enumerate(recs):
        if i < len(titles):
            rec["sample_title"] = titles[i]
    return pd.DataFrame(recs)


def read_expression(path: Path) -> pd.DataFrame:
    expr = pd.read_csv(path, sep="\t", comment="!", index_col=0)
    expr.index = expr.index.astype(str).str.strip(chr(34))
    expr.columns = expr.columns.astype(str).str.strip(chr(34))
    return expr.apply(pd.to_numeric, errors="coerce")


def read_annotation(path: Path) -> pd.DataFrame:
    rows = []
    header = None
    inside = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                inside = True
                continue
            if line == "!platform_table_end":
                break
            if not inside:
                continue
            parts = next(csv.reader([line], delimiter="\t"))
            if header is None:
                header = parts
            else:
                rows.append(parts)
    annot = pd.DataFrame(rows, columns=header).rename(columns={"ID": "probe_id", "Gene symbol": "gene_symbol"})
    out = []
    for row in annot[["probe_id", "gene_symbol"]].itertuples(index=False):
        for sym in str(row.gene_symbol).split("///"):
            sym = sym.strip()
            if sym and sym != "---":
                out.append({"probe_id": str(row.probe_id), "gene_symbol": sym})
    return pd.DataFrame(out).drop_duplicates()


def collapse_gene_expression(expr: pd.DataFrame, annot: pd.DataFrame, genes: set[str]) -> pd.DataFrame:
    ann = annot[annot["gene_symbol"].isin(genes)].drop_duplicates("probe_id")
    probe_to_gene = ann.set_index("probe_id")["gene_symbol"]
    keep = [p for p in expr.index if p in probe_to_gene.index]
    sub = expr.loc[keep].copy()
    sub["gene_symbol"] = [probe_to_gene.loc[p] for p in keep]
    return sub.groupby("gene_symbol").median(numeric_only=True).T.reset_index().rename(columns={"index": "geo_accession"})


def module_gene_map() -> dict[str, list[str]]:
    specs = {s.token: list(s.genes) for s in module_specs()}
    return {m: specs[m] for m in MODULES}


def add_scores(frame: pd.DataFrame, genes_by_module: dict[str, list[str]]) -> pd.DataFrame:
    out = frame.copy()
    for m, genes in genes_by_module.items():
        present = [g for g in genes if g in out.columns]
        out[f"{m}_expr_mean"] = out[present].mean(axis=1) if present else np.nan
        out[f"{m}_genes_present"] = len(present)
    pp = [g for g in PROLIFERATION_GENES if g in out.columns]
    out["proliferation_score"] = zscore(out[pp].mean(axis=1)) if pp else np.nan
    out["proteostasis_redox_state"] = zscore(out["proteostasis_expr_mean"]) + zscore(out["oxidative_stress_expr_mean"])
    out["proteostasis_redox_apoptosis_state"] = out["proteostasis_redox_state"] + 0.5 * zscore(out["apoptosis_iap_expr_mean"])
    out["stress_core_state"] = (
        zscore(out["proteostasis_expr_mean"])
        + zscore(out["oxidative_stress_expr_mean"])
        + 0.5 * zscore(out["mito_translation_expr_mean"])
        + 0.5 * zscore(out["apoptosis_iap_expr_mean"])
        + 0.25 * zscore(out["akt_mtor_growth_expr_mean"])
    )
    out["older_broad_intervene_state"] = (
        zscore(out["srebp_lipid_expr_mean"])
        + zscore(out["mito_translation_expr_mean"])
        + 0.5 * zscore(out["oxidative_stress_expr_mean"])
        + 0.3 * zscore(out["proteostasis_expr_mean"])
        + 0.3 * zscore(out["apoptosis_iap_expr_mean"])
        + 0.2 * zscore(out["akt_mtor_growth_expr_mean"])
    )
    return out


def null_partial_loglik(time: np.ndarray, event: np.ndarray) -> float:
    risk0 = torch.zeros(len(time), dtype=torch.float32)
    t = torch.tensor(time, dtype=torch.float32)
    e = torch.tensor(event, dtype=torch.float32)
    return -float(cox_loss(risk0, t, e).item()) * float(event.sum())


def score_stats(frame: pd.DataFrame, score_col: str, time_col: str, event_col: str, cohort: str, subset: str) -> dict:
    data = frame.dropna(subset=[score_col, time_col, event_col]).copy()
    data = data[data[time_col] > 0].copy()
    if len(data) < 30 or data[event_col].sum() < 5:
        return {"cohort": cohort, "subset": subset, "score": score_col, "n": len(data), "events": int(data[event_col].sum()) if len(data) else 0, "c_index": np.nan, "hr_high_low": np.nan, "logrank_p": np.nan, "cox_lrt_p": np.nan, "cox_chi2": np.nan}
    time = data[time_col].to_numpy(float)
    event = data[event_col].to_numpy(float)
    raw = data[score_col].to_numpy(float)
    ll0 = null_partial_loglik(time, event)
    ll1, risk = fit_cox_linear(zscore(data[score_col]).to_numpy(float).reshape(-1, 1), time, event)
    lr = logrank_high_low(data.assign(_score=data[score_col]), "_score", time_col, event_col)
    return {
        "cohort": cohort,
        "subset": subset,
        "score": score_col,
        "n": len(data),
        "events": int(event.sum()),
        "c_index": concordance_index(time, event, raw),
        "hr_high_low": lr["hazard_ratio_approx"],
        "logrank_p": lr["logrank_p"],
        "cox_lrt_p": norm_sf_chi2_1(max(0.0, 2 * (ll1 - ll0))),
        "cox_chi2": max(0.0, 2 * (ll1 - ll0)),
    }


def clinical_adjusted(frame: pd.DataFrame, score_col: str, time_col: str, event_col: str, cohort: str) -> dict:
    data = frame.dropna(subset=[score_col, time_col, event_col]).copy()
    data = data[data[time_col] > 0].copy()
    if len(data) < 50 or data[event_col].sum() < 10:
        return {"cohort": cohort, "score": score_col, "n": len(data), "events": int(data[event_col].sum()) if len(data) else 0, "clinical_only_c_index": np.nan, "clinical_plus_score_c_index": np.nan, "delta_c_index": np.nan, "lrt_p_for_score": np.nan}
    parts = []
    for age_col in ["age", "age_at_surgery"]:
        if age_col in data:
            age = pd.to_numeric(data[age_col], errors="coerce")
            parts.append(zscore(age.fillna(age.median())).rename("age_z").to_frame())
            break
    for col in ["stage", "disease_stage", "tumor_stage", "histology", "smoking_history", "adjuvant_treatment", "clinical_treatment_adjuvant_chemo"]:
        if col in data:
            parts.append(pd.get_dummies(data[col].fillna("missing").astype(str), prefix=col, dtype=float))
    cov = pd.concat(parts, axis=1).fillna(0.0) if parts else pd.DataFrame(index=data.index)
    time = data[time_col].to_numpy(float)
    event = data[event_col].to_numpy(float)
    ll0, risk0 = fit_cox_linear(cov.to_numpy(float), time, event)
    x = zscore(data[score_col]).to_numpy(float).reshape(-1, 1)
    ll1, risk1 = fit_cox_linear(np.column_stack([cov.to_numpy(float), x]), time, event)
    return {
        "cohort": cohort,
        "score": score_col,
        "n": len(data),
        "events": int(event.sum()),
        "clinical_only_c_index": concordance_index(time, event, risk0),
        "clinical_plus_score_c_index": concordance_index(time, event, risk1),
        "delta_c_index": concordance_index(time, event, risk1) - concordance_index(time, event, risk0),
        "lrt_chi2_for_score": max(0.0, 2 * (ll1 - ll0)),
        "lrt_p_for_score": norm_sf_chi2_1(max(0.0, 2 * (ll1 - ll0))),
    }


def prepare_gse30219_or_37745(gse: str) -> tuple[pd.DataFrame, str, str]:
    series = RAW / gse / f"{gse}_series_matrix.txt.gz"
    meta = parse_geo_metadata(series)
    meta["relapse_event"] = pd.to_numeric(meta.get("relapse_event_1_no_event_0"), errors="coerce")
    if "recurrence" in meta:
        meta["recurrence_event"] = meta["recurrence"].astype(str).str.lower().map({"yes": 1, "no": 0})
    if "days_to_recurrence_to_last_visit" in meta:
        meta["recurrence_time_months"] = pd.to_numeric(meta["days_to_recurrence_to_last_visit"], errors="coerce") / 30.4375
    if "disease_free_survival_in_months" in meta:
        meta["dfs_months"] = pd.to_numeric(meta["disease_free_survival_in_months"], errors="coerce")
    expr = read_expression(series)
    annot = read_annotation(RAW / "GSE31210" / "GPL570.annot.gz")
    genes = set(PROLIFERATION_GENES)
    for gs in module_gene_map().values(): genes.update(gs)
    frame = add_scores(meta.merge(collapse_gene_expression(expr, annot, genes), on="geo_accession", how="left"), module_gene_map())
    outdir = PROCESSED / gse
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / f"{gse.lower()}_proteostasis_redox_scores.csv", index=False)
    if gse == "GSE30219":
        return frame, "dfs_months", "relapse_event"
    return frame, "recurrence_time_months", "recurrence_event"


def prepare_gse68465() -> tuple[pd.DataFrame, str, str]:
    gse = "GSE68465"
    series = RAW / gse / f"{gse}_series_matrix.txt.gz"
    meta = parse_geo_metadata(series)
    meta["progression_relapse_event"] = meta.get("first_progression_or_relapse").astype(str).str.lower().map({"yes": 1, "no": 0})
    meta["months_to_first_progression_num"] = pd.to_numeric(meta.get("months_to_first_progression"), errors="coerce")
    meta["mths_to_last_clinical_assessment_num"] = pd.to_numeric(meta.get("mths_to_last_clinical_assessment"), errors="coerce")
    # For event-positive cases, survival time must be the observed time to progression/relapse.
    # If that month is missing, exclude the sample from survival validation instead of treating last follow-up as event time.
    meta["pfs_months"] = np.where(meta["progression_relapse_event"] == 1, meta["months_to_first_progression_num"], meta["mths_to_last_clinical_assessment_num"])
    meta["pfs_months"] = pd.to_numeric(meta["pfs_months"], errors="coerce")
    expr = read_expression(series)
    annot = read_annotation(RAW / gse / "GPL96.annot.gz")
    genes = set(PROLIFERATION_GENES)
    for gs in module_gene_map().values(): genes.update(gs)
    frame = add_scores(meta.merge(collapse_gene_expression(expr, annot, genes), on="geo_accession", how="left"), module_gene_map())
    outdir = PROCESSED / gse
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "gse68465_proteostasis_redox_scores.csv", index=False)
    return frame, "pfs_months", "progression_relapse_event"


def subsets(frame: pd.DataFrame, cohort: str) -> dict[str, pd.Series]:
    idx = frame.index
    masks = {"all": pd.Series(True, index=idx)}
    hist_col = "histology" if "histology" in frame else None
    if hist_col:
        hist = frame[hist_col].astype(str).str.lower()
        masks["adenocarcinoma"] = hist.str.contains("adc|adeno|adenocarcinoma", na=False)
        masks["non_adenocarcinoma"] = ~masks["adenocarcinoma"]
    if "disease_stage" in frame:
        st = frame["disease_stage"].astype(str).str.lower()
        masks["node_negative"] = st.str.contains("pn0", na=False)
        masks["t1_t2_node_negative"] = st.str.contains("pn0", na=False) & st.str.contains("pt1|pt2", regex=True, na=False)
    if "tumor_stage" in frame:
        st = frame["tumor_stage"].astype(str).str.lower()
        masks["stage_i"] = st.str.contains("1a|1b", regex=True, na=False)
    if "pt_stage" in frame and "pn_stage" in frame:
        masks["t1n0"] = frame["pt_stage"].astype(str).str.upper().eq("T1") & frame["pn_stage"].astype(str).str.upper().eq("N0")
    return masks


def run_validation() -> None:
    RESULTS.mkdir(exist_ok=True)
    prepared = []
    for gse in ["GSE30219", "GSE37745"]:
        prepared.append((gse, *prepare_gse30219_or_37745(gse)))
    prepared.append(("GSE68465", *prepare_gse68465()))

    scores = [
        "proteostasis_redox_state",
        "proteostasis_redox_apoptosis_state",
        "stress_core_state",
        "proteostasis_expr_mean",
        "oxidative_stress_expr_mean",
        "apoptosis_iap_expr_mean",
        "older_broad_intervene_state",
        "proliferation_score",
    ]
    rows = []
    clinical_rows = []
    for cohort, frame, time_col, event_col in prepared:
        frame = frame.dropna(subset=[time_col, event_col]).copy()
        frame = frame[frame[time_col] > 0].copy()
        for subset_name, mask in subsets(frame, cohort).items():
            sub = frame[mask].copy()
            for score in scores:
                if score in sub:
                    rows.append(score_stats(sub, score, time_col, event_col, cohort, subset_name))
        for score in ["proteostasis_redox_state", "stress_core_state", "older_broad_intervene_state", "proliferation_score"]:
            if score in frame:
                clinical_rows.append(clinical_adjusted(frame, score, time_col, event_col, cohort))

    validation = pd.DataFrame(rows).sort_values(["cohort", "subset", "c_index"], ascending=[True, True, False])
    clinical = pd.DataFrame(clinical_rows).sort_values(["cohort", "score"])
    validation.to_csv(RESULTS / "proteostasis_redox_external_validation.csv", index=False)
    clinical.to_csv(RESULTS / "proteostasis_redox_external_clinical_adjusted.csv", index=False)

    main = validation[(validation["score"].isin(["proteostasis_redox_state", "stress_core_state", "proteostasis_expr_mean", "oxidative_stress_expr_mean", "older_broad_intervene_state", "proliferation_score"])) & (validation["n"] >= 50) & (validation["events"] >= 10)].copy()
    main = main.sort_values(["cohort", "score", "c_index"], ascending=[True, True, False])
    best = validation[(validation["n"] >= 50) & (validation["events"] >= 10)].sort_values("c_index", ascending=False).head(25)
    lines = [
        "Proteostasis-Redox External Validation",
        "======================================",
        "",
        "Main goal: test the tighter proteostasis-redox relapse state externally, instead of the older broad SREBP-heavy state.",
        "",
        "Cohorts:",
        "- GSE68465: lung adenocarcinoma; first progression/relapse plus follow-up time.",
        "- GSE30219: NSCLC; disease-free survival and relapse labels; adenocarcinoma subset available.",
        "- GSE37745: NSCLC; recurrence labels and recurrence/last-visit time; adenocarcinoma subset available.",
        "",
        "Main comparable tests (n>=50, events>=10):",
        main.to_string(index=False),
        "",
        "Best external/subtype signals:",
        best.to_string(index=False),
        "",
        "Clinical-adjusted tests:",
        clinical.to_string(index=False),
        "",
        "Interpretation rules:",
        "- A valid external signal should beat or complement proliferation and should not be weaker than the older broad state.",
        "- GSE68465 is the strongest external LUAD relapse/progression test because it is lung adenocarcinoma-specific and large.",
        "- GSE30219/GSE37745 are useful supporting checks but include mixed NSCLC histologies.",
    ]
    (RESULTS / "proteostasis_redox_external_validation_summary.txt").write_text("\n".join(lines))
    print("Wrote proteostasis_redox_external_validation.csv")
    print("Wrote proteostasis_redox_external_clinical_adjusted.csv")
    print("Wrote proteostasis_redox_external_validation_summary.txt")


if __name__ == "__main__":
    run_validation()

from __future__ import annotations

from pathlib import Path
import csv
import gzip
import math
import re

import numpy as np
import pandas as pd

from interleaved_biophysics_attention_full import module_specs, zscore


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "relapse" / "GSE31210"
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"
RESULTS = ROOT / "results"

SERIES = RAW / "GSE31210_series_matrix.txt.gz"
ANNOT = RAW / "GPL570.annot.gz"

PAPER_MODULES = {
    "srebp_lipid",
    "mito_translation",
    "oxidative_stress",
    "proteostasis",
    "apoptosis_iap",
    "akt_mtor_growth",
}


def parse_value(value: object) -> object:
    if value is None:
        return np.nan
    text = str(value).strip().strip('"')
    if text == "":
        return np.nan
    return text


def normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = key.replace("(", "").replace(")", "")
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key


def parse_sample_metadata() -> pd.DataFrame:
    sample_ids: list[str] = []
    titles: list[str] = []
    accessions: list[str] = []
    records: list[dict[str, object]] = []

    with gzip.open(SERIES, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_geo_accession"):
                parts = next(csv.reader([line], delimiter="\t"))
                accessions = [str(parse_value(x)) for x in parts[1:]]
                records = [{"geo_accession": acc} for acc in accessions]
            elif line.startswith("!Sample_title"):
                parts = next(csv.reader([line], delimiter="\t"))
                titles = [str(parse_value(x)) for x in parts[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = next(csv.reader([line], delimiter="\t"))
                values = [parse_value(x) for x in parts[1:]]
                if not records:
                    records = [{} for _ in values]
                for rec, value in zip(records, values):
                    if pd.isna(value) or ":" not in str(value):
                        continue
                    key, raw = str(value).split(":", 1)
                    norm = normalize_key(key)
                    raw = raw.strip()
                    if norm in rec and not pd.isna(rec[norm]):
                        suffix = 2
                        while f"{norm}_{suffix}" in rec:
                            suffix += 1
                        norm = f"{norm}_{suffix}"
                    rec[norm] = raw

    for i, rec in enumerate(records):
        if i < len(titles):
            rec["sample_title"] = titles[i]
    meta = pd.DataFrame(records)

    rename = {
        "age_years": "age",
        "pstage_iorii": "pathological_stage_binary",
        "months_before_relapse_censor": "relapse_free_months",
        "days_before_relapse_censor": "relapse_free_days",
        "days_before_death_censor": "overall_survival_days",
        "exclude_for_prognosis_analysis_due_to_incomplete_resection_or_adjuvant_therapy": "prognosis_exclusion",
    }
    meta = meta.rename(columns={k: v for k, v in rename.items() if k in meta.columns})
    if meta.columns.duplicated().any():
        meta = meta.T.groupby(level=0).first().T
    for col in ["age", "relapse_free_months", "relapse_free_days", "overall_survival_days"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    if "relapse" in meta.columns:
        meta["relapse_event"] = meta["relapse"].astype(str).str.lower().map(
            {
                "relapse": 1,
                "relapsed": 1,
                "recurrence": 1,
                "yes": 1,
                "1": 1,
                "not relapsed": 0,
                "no relapse": 0,
                "no": 0,
                "0": 0,
                "none": 0,
            }
        )
        missing = meta["relapse_event"].isna()
        meta.loc[missing, "relapse_event"] = meta.loc[missing, "relapse"].astype(str).str.lower().map(
            {"relapsed": 1, "non-relapse": 0, "non relapse": 0, "not_relapsed": 0}
        )
    if "death" in meta.columns:
        meta["death_event"] = meta["death"].astype(str).str.lower().map(
            {"dead": 1, "death": 1, "yes": 1, "1": 1, "alive": 0, "no": 0, "0": 0}
        )
    if "tissue" in meta.columns:
        meta["is_primary_tumor"] = meta["tissue"].astype(str).str.lower().eq("primary lung tumor")
    else:
        meta["is_primary_tumor"] = True
    if "prognosis_exclusion" in meta.columns:
        meta["prognosis_included"] = meta["prognosis_exclusion"].astype(str).str.lower().eq("none")
    else:
        meta["prognosis_included"] = True
    return meta


def read_expression() -> pd.DataFrame:
    expr = pd.read_csv(SERIES, sep="\t", comment="!", index_col=0)
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = expr.columns.astype(str).str.strip('"')
    return expr.apply(pd.to_numeric, errors="coerce")


def read_annotation() -> pd.DataFrame:
    rows = []
    in_table = False
    header = None
    with gzip.open(ANNOT, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                header = None
                continue
            if line == "!platform_table_end":
                break
            if not in_table:
                continue
            parts = next(csv.reader([line], delimiter="\t"))
            if header is None:
                header = parts
                continue
            rows.append(parts)
    annot = pd.DataFrame(rows, columns=header)
    annot = annot.rename(columns={"ID": "probe_id", "Gene symbol": "gene_symbol"})
    annot["probe_id"] = annot["probe_id"].astype(str)
    annot["gene_symbol"] = annot["gene_symbol"].astype(str)
    annot = annot[["probe_id", "gene_symbol"]].copy()
    exploded = []
    for row in annot.itertuples(index=False):
        symbols = [s.strip() for s in str(row.gene_symbol).split("///") if s.strip()]
        for symbol in symbols:
            if symbol and symbol != "---":
                exploded.append({"probe_id": row.probe_id, "gene_symbol": symbol})
    return pd.DataFrame(exploded).drop_duplicates()


def module_gene_set() -> dict[str, list[str]]:
    specs = module_specs()
    out = {}
    for spec in specs:
        if spec.token in PAPER_MODULES:
            out[spec.token] = list(spec.genes)
    return out


def collapse_to_module_genes(expr: pd.DataFrame, annot: pd.DataFrame, genes: set[str]) -> pd.DataFrame:
    ann = annot[annot["gene_symbol"].isin(genes)].copy()
    probe_to_gene = ann.set_index("probe_id")["gene_symbol"]
    keep_probes = [p for p in expr.index if p in probe_to_gene.index]
    sub = expr.loc[keep_probes].copy()
    sub["gene_symbol"] = [probe_to_gene.loc[p] for p in keep_probes]
    values = sub.drop(columns=["gene_symbol"])
    values["gene_symbol"] = sub["gene_symbol"].values
    gene_expr = values.groupby("gene_symbol").median(numeric_only=True)
    return gene_expr.T.reset_index().rename(columns={"index": "geo_accession"})


def build_patient_tokens(meta: pd.DataFrame, gene_expr: pd.DataFrame, modules: dict[str, list[str]]) -> pd.DataFrame:
    patient = meta.merge(gene_expr, on="geo_accession", how="left")
    for module, genes in modules.items():
        present = [g for g in genes if g in patient.columns]
        if present:
            patient[f"{module}_expr_mean"] = patient[present].mean(axis=1)
            patient[f"{module}_expr_median"] = patient[present].median(axis=1)
            patient[f"{module}_expr_dispersion"] = patient[present].std(axis=1)
            patient[f"{module}_genes_present"] = len(present)
        else:
            patient[f"{module}_expr_mean"] = np.nan
            patient[f"{module}_expr_median"] = np.nan
            patient[f"{module}_expr_dispersion"] = np.nan
            patient[f"{module}_genes_present"] = 0

    # Transfer score from the validated DepMap/PRISM biology: lipid + mito + oxidative + stress-survival support.
    patient["raw_relapse_intervene_state"] = (
        zscore(patient["srebp_lipid_expr_mean"])
        + zscore(patient["mito_translation_expr_mean"])
        + 0.50 * zscore(patient["oxidative_stress_expr_mean"])
        + 0.30 * zscore(patient["proteostasis_expr_mean"])
        + 0.30 * zscore(patient["apoptosis_iap_expr_mean"])
        + 0.20 * zscore(patient["akt_mtor_growth_expr_mean"])
    )
    patient["srebp_mito_expression_coupling"] = zscore(patient["srebp_lipid_expr_mean"]) * zscore(
        patient["mito_translation_expr_mean"]
    )
    return patient


def logrank_high_low(frame: pd.DataFrame, score_col: str, time_col: str, event_col: str) -> dict[str, object]:
    data = frame[[score_col, time_col, event_col]].dropna().copy()
    data = data[data[time_col] > 0].copy()
    if len(data) < 20 or data[event_col].sum() < 3:
        return {
            "test": "logrank high vs low",
            "n": len(data),
            "events": int(data[event_col].sum()) if len(data) else 0,
            "high_n": 0,
            "low_n": 0,
            "high_events": 0,
            "low_events": 0,
            "hazard_ratio_approx": np.nan,
            "logrank_chi2": np.nan,
            "logrank_p": np.nan,
        }
    high_cut = data[score_col].quantile(2 / 3)
    low_cut = data[score_col].quantile(1 / 3)
    data = data[(data[score_col] >= high_cut) | (data[score_col] <= low_cut)].copy()
    data["group"] = np.where(data[score_col] >= high_cut, 1, 0)
    times = sorted(data.loc[data[event_col] == 1, time_col].unique())
    observed_high = expected_high = var_high = 0.0
    for t in times:
        at_risk = data[data[time_col] >= t]
        events = data[(data[time_col] == t) & (data[event_col] == 1)]
        n = len(at_risk)
        if n <= 1:
            continue
        n_high = int(at_risk["group"].sum())
        d = len(events)
        d_high = int(events["group"].sum())
        observed_high += d_high
        expected_high += d * n_high / n
        var_high += (n_high / n) * (1 - n_high / n) * d * (n - d) / max(n - 1, 1)
    stat = (observed_high - expected_high) ** 2 / var_high if var_high > 0 else np.nan
    p = float(math.erfc(math.sqrt(stat / 2))) if not pd.isna(stat) else np.nan
    high = data[data["group"] == 1]
    low = data[data["group"] == 0]
    high_rate = (high[event_col].sum() + 0.5) / (high[time_col].sum() + 0.5)
    low_rate = (low[event_col].sum() + 0.5) / (low[time_col].sum() + 0.5)
    fisher_p = two_sided_fisher(
        int(high[event_col].sum()),
        len(high) - int(high[event_col].sum()),
        int(low[event_col].sum()),
        len(low) - int(low[event_col].sum()),
    )
    return {
        "test": "logrank high vs low",
        "n": len(data),
        "events": int(data[event_col].sum()),
        "high_n": len(high),
        "low_n": len(low),
        "high_events": int(high[event_col].sum()),
        "low_events": int(low[event_col].sum()),
        "hazard_ratio_approx": float(high_rate / low_rate) if low_rate > 0 else np.nan,
        "event_fisher_p": float(fisher_p),
        "logrank_chi2": float(stat) if not pd.isna(stat) else np.nan,
        "logrank_p": p,
    }


def hypergeom_prob(a: int, b: int, c: int, d: int) -> float:
    return math.exp(
        math.lgamma(a + b + 1)
        + math.lgamma(c + d + 1)
        + math.lgamma(a + c + 1)
        + math.lgamma(b + d + 1)
        - math.lgamma(a + 1)
        - math.lgamma(b + 1)
        - math.lgamma(c + 1)
        - math.lgamma(d + 1)
        - math.lgamma(a + b + c + d + 1)
    )


def two_sided_fisher(a: int, b: int, c: int, d: int) -> float:
    row1 = a + b
    row2 = c + d
    col1 = a + c
    observed = hypergeom_prob(a, b, c, d)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    p = 0.0
    for x in range(lo, hi + 1):
        y = row1 - x
        z = col1 - x
        w = row2 - z
        prob = hypergeom_prob(x, y, z, w)
        if prob <= observed + 1e-12:
            p += prob
    return min(1.0, p)


def write_summary(meta: pd.DataFrame, patient: pd.DataFrame, gene_expr: pd.DataFrame, survival: pd.DataFrame) -> None:
    path = RESULTS / "relapse_gse31210_preparation_summary.txt"
    tumor = patient[patient["is_primary_tumor"]].copy()
    prognosis = tumor[tumor["prognosis_included"]].copy()
    with path.open("w", encoding="utf-8") as f:
        f.write("GSE31210 Relapse-INTERVENE preparation\n")
        f.write("=====================================\n\n")
        f.write(f"All samples: {len(meta)}\n")
        f.write(f"Primary tumor samples: {int(patient['is_primary_tumor'].sum())}\n")
        f.write(f"Prognosis-included primary tumors: {len(prognosis)}\n")
        f.write(f"Samples with relapse event label: {int(prognosis['relapse_event'].notna().sum())}\n")
        f.write(f"Relapse events: {int(prognosis['relapse_event'].fillna(0).sum())}\n")
        f.write(f"Samples with relapse-free months: {int(prognosis['relapse_free_months'].notna().sum())}\n")
        f.write(f"Gene-level module expression columns: {gene_expr.shape[1] - 1}\n\n")
        f.write("Module gene coverage\n")
        f.write("--------------------\n")
        coverage_cols = [c for c in patient.columns if c.endswith("_genes_present")]
        f.write(patient[coverage_cols].iloc[0].to_string())
        f.write("\n\nInitial relapse-free survival test\n")
        f.write("----------------------------------\n")
        f.write(survival.to_string(index=False))
        f.write("\n\nInterpretation\n")
        f.write("--------------\n")
        f.write(
            "This is a first transfer test using expression-derived module scores, not the final Relapse-INTERVENE AI model. "
            "If the high-state group trends toward more/earlier relapse, the next step is a survival-head model with "
            "IHA-inspired attention and clinical covariate baselines.\n"
        )


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = parse_sample_metadata()
    expr = read_expression()
    annot = read_annotation()
    modules = module_gene_set()
    genes = sorted({g for gs in modules.values() for g in gs})
    gene_expr = collapse_to_module_genes(expr, annot, set(genes))
    patient = build_patient_tokens(meta, gene_expr, modules)

    prognosis = patient[patient["is_primary_tumor"] & patient["prognosis_included"]].copy()
    survival = pd.DataFrame(
        [
            logrank_high_low(
                prognosis,
                "raw_relapse_intervene_state",
                "relapse_free_months",
                "relapse_event",
            )
        ]
    )

    meta.to_csv(PROCESSED / "gse31210_sample_metadata.csv", index=False)
    gene_expr.to_csv(PROCESSED / "gse31210_module_gene_expression.csv", index=False)
    patient.to_csv(PROCESSED / "gse31210_patient_module_scores.csv", index=False)
    survival.to_csv(RESULTS / "relapse_gse31210_initial_survival_test.csv", index=False)
    write_summary(meta, patient, gene_expr, survival)

    print(f"Wrote {PROCESSED / 'gse31210_sample_metadata.csv'}")
    print(f"Wrote {PROCESSED / 'gse31210_module_gene_expression.csv'}")
    print(f"Wrote {PROCESSED / 'gse31210_patient_module_scores.csv'}")
    print(f"Wrote {RESULTS / 'relapse_gse31210_initial_survival_test.csv'}")
    print(f"Wrote {RESULTS / 'relapse_gse31210_preparation_summary.txt'}")


if __name__ == "__main__":
    main()

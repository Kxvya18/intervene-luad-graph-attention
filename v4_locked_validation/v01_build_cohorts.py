"""
v4 step 1 — genome-wide gene-level expression + clinical tables for the five cohorts (PROTOCOL.md, 'Data' and
'Expression processing'). Output: cache/<cohort>.pkl with keys expr (samples x genes, within-cohort z-scores),
clin (time_months, event, stage, primary/LUAD flag, histology) and meta.
"""
import csv
import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw" / "relapse", ROOT / "data" / "processed" / "relapse"
OUT = Path(__file__).resolve().parent / "cache"
ANNOT = {"GPL570": RAW / "GSE31210" / "GPL570.annot.gz", "GPL96": RAW / "GSE68465" / "GPL96.annot.gz"}


def read_annotation(path):
    rows, header, on = [], None, False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                on = True; continue
            if line == "!platform_table_end":
                break
            if not on:
                continue
            parts = next(csv.reader([line], delimiter="\t"))
            if header is None:
                header = parts; continue
            rows.append(parts)
    a = pd.DataFrame(rows, columns=header).rename(columns={"ID": "probe_id", "Gene symbol": "gene_symbol"})
    a = a[["probe_id", "gene_symbol"]].astype(str)
    a = a[(a.gene_symbol != "") & (a.gene_symbol != "nan") & ~a.gene_symbol.str.contains("///")]   # unique-gene probes only
    return a


def read_series(path):
    expr = pd.read_csv(path, sep="\t", comment="!", index_col=0)
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = expr.columns.astype(str).str.strip('"')
    return expr.apply(pd.to_numeric, errors="coerce")


def gene_matrix(cohort, platform):
    expr = read_series(RAW / cohort / f"{cohort}_series_matrix.txt.gz")
    if np.nanmax(expr.to_numpy()) > 100:                       # MAS5 / GPL96 intensities -> log2
        expr = np.log2(expr.clip(lower=0) + 1)
    ann = read_annotation(ANNOT[platform]).drop_duplicates("probe_id").set_index("probe_id")["gene_symbol"]
    keep = expr.index.intersection(ann.index)
    g = expr.loc[keep].groupby(ann.loc[keep].values).median()   # per-gene median across probes
    return g.T                                                  # samples x genes (not yet z-scored)


def clinical(cohort):
    if cohort == "GSE31210":
        d = pd.read_csv(PROC / cohort / "gse31210_patient_module_scores.csv")
        d = d[d.is_primary_tumor & d.prognosis_included]
        c = pd.DataFrame({"geo_accession": d.geo_accession, "time": d.relapse_free_months, "event": d.relapse_event,
                          "stage": d.pathological_stage.astype(str).str.upper().str[:2].map({"IA": 1, "IB": 1, "II": 2}),
                          "histology": "LUAD", "luad": True})
    elif cohort == "GSE30219":
        d = pd.read_csv(PROC / cohort / "gse30219_proteostasis_redox_scores.csv")
        c = pd.DataFrame({"geo_accession": d.geo_accession, "time": d.dfs_months, "event": d.relapse_event,
                          "stage": d.pt_stage.astype(str).str.extract(r"(\d)")[0].astype(float),
                          "histology": d.histology, "luad": d.histology.eq("ADC")})
    elif cohort == "GSE37745":
        d = pd.read_csv(PROC / cohort / "gse37745_proteostasis_redox_scores.csv")
        c = pd.DataFrame({"geo_accession": d.geo_accession, "time": d.recurrence_time_months, "event": d.recurrence_event,
                          "stage": d.tumor_stage.astype(str).str.extract(r"(\d)")[0].astype(float),
                          "histology": d.histology, "luad": d.histology.eq("adeno")})
    elif cohort == "GSE50081":
        d = pd.read_csv(PROC / cohort / "gse50081_intervene_external_scores.csv")
        c = pd.DataFrame({"geo_accession": d.geo_accession, "time": d.disease_free_survival_time * 12.0,
                          "event": d.recurrence_event, "stage": d["stage"].astype(str).str.extract(r"(\d)")[0].astype(float),
                          "histology": d.histology,
                          "luad": d.histology.isin(["adenocarcinoma", "NSCLC-favor adenocarcinoma"])})
    else:   # GSE68465
        d = pd.read_csv(PROC / cohort / "gse68465_proteostasis_redox_scores.csv")
        st = d.disease_stage.astype(str).str.extract(r"p?T(\d)")[0].astype(float)
        c = pd.DataFrame({"geo_accession": d.geo_accession, "time": d.pfs_months, "event": d.progression_relapse_event,
                          "stage": st, "histology": d.disease_state, "luad": d.disease_state.eq("Lung Adenocarcinoma")})
    c = c.dropna(subset=["time", "event"])
    return c[c.time > 0].reset_index(drop=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cohorts = {"GSE31210": "GPL570", "GSE30219": "GPL570", "GSE37745": "GPL570", "GSE50081": "GPL570", "GSE68465": "GPL96"}
    for cohort, plat in cohorts.items():
        g = gene_matrix(cohort, plat)
        c = clinical(cohort)
        c = c[c.geo_accession.isin(g.index)].reset_index(drop=True)
        for subset in ("luad", "all"):
            cc = c[c.luad] if subset == "luad" else c
            e = g.loc[cc.geo_accession]
            z = (e - e.mean()) / e.std(ddof=0).replace(0, np.nan)       # within-cohort, within-population z-scores
            pickle.dump(dict(expr=z.dropna(axis=1, how="all"), clin=cc.reset_index(drop=True), platform=plat),
                        open(OUT / f"{cohort}_{subset}.pkl", "wb"))
        print(f"{cohort} {plat}: genes {g.shape[1]}, LUAD n={int(c.luad.sum())} events={int(c[c.luad].event.sum())}, "
              f"all n={len(c)} events={int(c.event.sum())}, MRPL21 present={'MRPL21' in g.columns}", flush=True)

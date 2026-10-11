"""
INTERVENE-2 data layer: downloads public expression data, builds per-cohort gene matrices (within-cohort z-scores),
LUAD clinical tables, and the pathway token set.

Cohorts
  development : GSE31210, GSE30219, GSE37745, GSE68465   (GEO microarrays)
  sealed test : GSE50081 (GEO), TCGA-LUAD (RNA-seq, UCSC Xena)  - never used for any design choice
Endpoints: relapse / recurrence / disease-free / progression-free time in months (see static/clinical_*.csv;
TCGA uses PFI and PFI.time from the Xena curated survival table).
"""
from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

STATIC = Path(__file__).resolve().parent / "static"
GEO = {
    "GSE31210": ("GPL570", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31210/matrix/GSE31210_series_matrix.txt.gz"),
    "GSE30219": ("GPL570", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE30nnn/GSE30219/matrix/GSE30219_series_matrix.txt.gz"),
    "GSE37745": ("GPL570", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE37nnn/GSE37745/matrix/GSE37745_series_matrix.txt.gz"),
    "GSE68465": ("GPL96", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68465/matrix/GSE68465_series_matrix.txt.gz"),
    "GSE50081": ("GPL570", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50081/matrix/GSE50081_series_matrix.txt.gz"),
}
XENA = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
TCGA_FILES = {"expr": "TCGA.LUAD.sampleMap%2FHiSeqV2.gz", "surv": "survival%2FLUAD_survival.txt",
              "clin": "TCGA.LUAD.sampleMap%2FLUAD_clinicalMatrix"}
HALLMARK_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
DEV = ["GSE31210", "GSE30219", "GSE37745", "GSE68465"]
SEALED = ["GSE50081", "TCGA-LUAD"]

DEPMAP_MODULES = {   # the six cell-line-derived INTERVENE modules (independent of patient data)
    "DM_srebp_lipid": ["SCAP", "SCD", "MBTPS1", "MBTPS2", "SREBF1"],
    "DM_mito_translation": ["MRPL10", "MRPL21", "MRPL34", "MRPL38", "MRPL4", "MRPL41", "MRPL45", "MRPL49", "MRPL57",
                            "MRPS11", "MRPS12", "MRPS14", "MRPS21", "MRPS23", "MRPS34", "MRPS5"],
    "DM_oxidative_stress": ["KEAP1", "NFE2L2", "PGD", "TXNRD1", "GCLC", "GCLM", "SLC7A11"],
    "DM_apoptosis_iap": ["CFLAR", "XIAP", "BIRC2", "BIRC3", "CASP8", "MCL1", "BCL2L1"],
    "DM_proteostasis": ["USP8", "STAMBP", "PTPN23", "UBAP1", "HGS", "VCP", "PSMA1", "PSMB5"],
    "DM_akt_mtor_growth": ["AKT1", "AKT2", "AKT3", "MTOR", "PIK3CA", "PIK3CB", "RPTOR", "RICTOR", "PTEN"],
}
PROLIFERATION = ["MKI67", "TOP2A", "MCM2", "MCM5", "PCNA", "CDK1", "CDK2", "AURKA", "AURKB", "MCM6", "BUB1", "CCNB1", "UBE2C"]


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        req = urllib.request.Request(url, headers={"User-Agent": "intervene2"})
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    return dest


def _series_matrix(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", errors="replace") as f:
        lines = [l for l in f if not l.startswith("!")]
    e = pd.read_csv(io.StringIO("".join(lines)), sep="\t", index_col=0)
    e.index = e.index.astype(str).str.strip('"'); e.columns = e.columns.astype(str).str.strip('"')
    return e.apply(pd.to_numeric, errors="coerce")


def _zscore(g: pd.DataFrame) -> pd.DataFrame:
    sd = g.std(ddof=0).replace(0, np.nan)
    return ((g - g.mean()) / sd).dropna(axis=1, how="any")


def load_geo(cohort: str, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    plat, url = GEO[cohort]
    e = _series_matrix(fetch(url, cache / f"{cohort}_series_matrix.txt.gz"))
    if np.nanmax(e.to_numpy()) > 100:
        e = np.log2(e.clip(lower=0) + 1)
    ann = pd.read_csv(STATIC / f"{plat}_probe_gene.csv.gz", dtype=str).set_index("probe_id")["gene_symbol"]
    keep = e.index.intersection(ann.index)
    g = e.loc[keep].groupby(ann.loc[keep].values).median().T
    clin = pd.read_csv(STATIC / f"clinical_{cohort}.csv")
    clin = clin[clin.geo_accession.isin(g.index)].rename(columns={"geo_accession": "sample"}).reset_index(drop=True)
    return _zscore(g.loc[clin["sample"]]), clin


def load_tcga(cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = pd.read_csv(fetch(XENA + TCGA_FILES["expr"], cache / "TCGA_LUAD_HiSeqV2.gz"), sep="\t", index_col=0)   # log2(RSEM+1)
    s = pd.read_csv(fetch(XENA + TCGA_FILES["surv"], cache / "TCGA_LUAD_survival.txt"), sep="\t")
    c = pd.read_csv(fetch(XENA + TCGA_FILES["clin"], cache / "TCGA_LUAD_clinical.tsv"), sep="\t", index_col=0)
    prim = [x for x in e.columns if x[13:15] == "01"]                      # primary tumours only
    e = e[prim]; e.columns = [x[:15] for x in e.columns]
    s = s.rename(columns={"sample": "sample"}); s["sample"] = s["sample"].str[:15]
    s = s.dropna(subset=["PFI", "PFI.time"]).drop_duplicates("sample")
    stage = c.get("pathologic_stage", pd.Series(dtype=str)).astype(str)
    stage_num = stage.str.extract(r"Stage (IV|III|II|I)")[0].map({"I": 1, "II": 2, "III": 3, "IV": 4})
    stage_num.index = [x[:15] for x in stage_num.index]
    clin = pd.DataFrame({"sample": s["sample"], "time": s["PFI.time"] / 30.4375, "event": s["PFI"].astype(int)})
    clin["stage"] = clin["sample"].map(stage_num.groupby(level=0).first())
    clin = clin[(clin.time > 0) & clin["sample"].isin(e.columns)].reset_index(drop=True)
    g = e[clin["sample"]].T
    return _zscore(g), clin


def load_hallmarks(cache: Path) -> dict[str, list[str]]:
    p = fetch(HALLMARK_URL, cache / "MSigDB_Hallmark_2020.txt")
    sets = {}
    for line in p.read_text().splitlines():
        parts = [x.split(",")[0] for x in line.split("\t") if x]
        if len(parts) > 2:
            sets["HM_" + parts[0].replace(" ", "_")] = parts[1:]
    return sets


def build(cache: Path, min_genes: int = 5) -> dict:
    cohorts = {c: load_geo(c, cache) for c in GEO}
    cohorts["TCGA-LUAD"] = load_tcga(cache)
    common = set.intersection(*[set(x.columns) for x, _ in cohorts.values()])
    tokens = {**load_hallmarks(cache), **DEPMAP_MODULES, "PROLIFERATION": PROLIFERATION}
    tokens = {k: sorted(set(v) & common) for k, v in tokens.items()}
    tokens = {k: v for k, v in tokens.items() if len(v) >= min_genes}
    genes = sorted(set().union(*tokens.values()))
    data = {}
    for c, (x, clin) in cohorts.items():
        prolif = x[[g for g in PROLIFERATION if g in x.columns]].mean(1).to_numpy()
        data[c] = dict(X=x[genes].to_numpy(np.float32), clin=clin, prolif=(prolif - prolif.mean()) / prolif.std())
    return dict(cohorts=data, genes=genes, tokens=tokens,
                token_index={k: [genes.index(g) for g in v] for k, v in tokens.items()})

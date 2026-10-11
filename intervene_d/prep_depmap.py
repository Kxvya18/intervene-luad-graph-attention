"""
INTERVENE-D data preparation (run once locally on the DepMap 24Q4 + PRISM Repurposing 24Q2 downloads; the output
npz is what Colab uses). Data: DepMap Public / PRISM Repurposing, CC BY 4.0 (Broad Institute, depmap.org).

Output intervene_d_data.npz:
  cells, lineage           cell lines with expression, copy number and PRISM response; OncotreeLineage
  genes, tokens, tok_idx   pathway tokens (Hallmark 2020 + filtered KEGG 2021 + 6 DepMap-derived INTERVENE modules)
  expr, cn, mut            cells x genes (expression and CN z-scored across cells; mut = damaging/hotspot 0/1), float16
  Y                        cells x drugs, PRISM log2 fold-change at the primary dose (NaN = not measured), float16
  drugs, drug_name, moa, target
Drug selection (uses response values only, no split information): >= 70% of cells measured; selective response
(5%-80% of measured cells with LFC < -1); one row per compound name (most cells measured); top 800 by response SD.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("/Volumes/Seagate/simulations/ALL_PROJECTS/DepMap/data/raw")
CACHE = Path("/Volumes/Seagate/simulations/ALL_PROJECTS/DepMap/intervene_d_cache")
OUT = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intervene2.data import DEPMAP_MODULES  # noqa: E402

DISEASE = re.compile(r"disease|infection|cancer|carcinoma|leukemia|glioma|melanoma|lymphoma|sarcoma|cardiomyopathy|"
                     r"diabetes|addiction|virus|viral|coronavirus|salmonella|tuberculosis|measles|influenza|hepatitis|"
                     r"malaria|leishmaniasis|toxoplasmosis|amoebiasis|chagas|pertussis|legionellosis|shigellosis|"
                     r"yersinia|escherichia|staphylococcus|asthma|arthritis|lupus|sclerosis|atherosclerosis|"
                     r"graft|allograft|thyroiditis|myocarditis|alcoholism|nephropathy|neurodegeneration|prion", re.I)


def gene_sets(path, prefix):
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = [x for x in line.split("\t") if x]
        if len(parts) > 2:
            out[prefix + parts[0].replace(" ", "_")] = [g.split(",")[0] for g in parts[1:]]
    return out


def main():
    OUT.mkdir(exist_ok=True)
    model = pd.read_csv(RAW / "Model.csv", usecols=["ModelID", "OncotreeLineage"]).set_index("ModelID")
    Yall = pd.read_csv(RAW / "PRISM" / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv", index_col=0)
    comp = pd.read_csv(RAW / "PRISM" / "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv").set_index("IDs")
    expr = pd.read_csv(RAW / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    expr.columns = [c.split(" (")[0] for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    hm = gene_sets(CACHE / "MSigDB_Hallmark_2020.txt", "HM_")
    kg = {k: v for k, v in gene_sets(CACHE / "KEGG_2021_Human.txt", "KEGG_").items() if not DISEASE.search(k)}
    kg = {k: v for k, v in kg.items() if 15 <= len(v) <= 200}
    sets = {**hm, **kg, **DEPMAP_MODULES}
    universe = set(expr.columns)
    sets = {k: sorted(set(v) & universe) for k, v in sets.items()}
    sets = {k: v for k, v in sets.items() if len(v) >= 5}
    genes = sorted(set().union(*sets.values()))
    cn = pd.read_csv(RAW / "OmicsCNGene.csv", index_col=0)
    cn.columns = [c.split(" (")[0] for c in cn.columns]; cn = cn.loc[:, ~cn.columns.duplicated()]
    cells = sorted(set(expr.index) & set(cn.index) & set(Yall.columns) & set(model.index))
    # drugs
    Y = Yall[cells]
    cov = Y.notna().mean(1); frac = (Y < -1).sum(1) / Y.notna().sum(1)
    keep = (cov >= 0.70) & frac.between(0.05, 0.80)
    cand = comp.loc[keep[keep].index].assign(cov=cov[keep], sd=Y[keep].std(1))
    cand = cand.sort_values("cov", ascending=False).drop_duplicates("Drug.Name")
    drugs = cand.sort_values("sd", ascending=False).head(800).index.tolist()
    # mutations: damaging (VEP HIGH impact) or hotspot, restricted to token genes and cells
    mu = pd.read_csv(RAW / "OmicsSomaticMutations.csv", usecols=["ModelID", "HugoSymbol", "VepImpact", "Hotspot"]
                     if "Hotspot" in pd.read_csv(RAW / "OmicsSomaticMutations.csv", nrows=1).columns
                     else ["ModelID", "HugoSymbol", "VepImpact"])
    dam = mu[(mu.VepImpact == "HIGH") | (mu.get("Hotspot", False) == True)]
    dam = dam[dam.HugoSymbol.isin(genes) & dam.ModelID.isin(cells)]
    M = pd.crosstab(dam.ModelID, dam.HugoSymbol).clip(upper=1).reindex(index=cells, columns=genes, fill_value=0)
    z = lambda d: ((d - d.mean()) / d.std(ddof=0).replace(0, 1)).fillna(0)
    E = z(expr.loc[cells, genes]); C = z(cn.reindex(index=cells, columns=genes).fillna(cn.reindex(columns=genes).median()))
    gi = {g: i for i, g in enumerate(genes)}
    tok_names = list(sets); tok_idx = np.array([np.array([gi[g] for g in sets[t]]) for t in tok_names], dtype=object)
    np.savez_compressed(OUT / "intervene_d_data.npz",
                        cells=np.array(cells), lineage=model.loc[cells, "OncotreeLineage"].fillna("Unknown").to_numpy(),
                        genes=np.array(genes), tokens=np.array(tok_names), tok_idx=tok_idx,
                        expr=E.to_numpy(np.float16), cn=C.to_numpy(np.float16), mut=M.to_numpy(np.int8),
                        Y=Y.loc[drugs].T.to_numpy(np.float16), drugs=np.array(drugs),
                        drug_name=comp.loc[drugs, "Drug.Name"].astype(str).to_numpy(),
                        moa=comp.loc[drugs, "MOA"].fillna("").astype(str).to_numpy(),
                        target=comp.loc[drugs, "repurposing_target"].fillna("").astype(str).to_numpy())
    lin = pd.Series(model.loc[cells, "OncotreeLineage"]).value_counts()
    print(f"cells {len(cells)}, lineages {len(lin)}, genes {len(genes)}, tokens {len(tok_names)} "
          f"(HM {sum(t.startswith('HM_') for t in tok_names)}, KEGG {sum(t.startswith('KEGG_') for t in tok_names)}), "
          f"drugs {len(drugs)} (with target {int((comp.loc[drugs, 'repurposing_target'].notna()).sum())}), "
          f"Y observed {np.isfinite(Y.loc[drugs].to_numpy(float)).mean():.2f}, mutated genes/cell {M.sum(1).mean():.1f}")
    print(lin.head(15).to_dict())


if __name__ == "__main__":
    main()

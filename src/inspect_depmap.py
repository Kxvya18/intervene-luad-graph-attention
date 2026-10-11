from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    gene_effect_path = RAW / "CRISPRGeneEffect.csv"
    model_path = RAW / "Model.csv"

    gene_effect_ids = pd.read_csv(gene_effect_path, usecols=[0])
    model_col = gene_effect_ids.columns[0]
    model_ids = set(gene_effect_ids[model_col].astype(str))

    models = pd.read_csv(model_path)
    overlap = models[models["ModelID"].astype(str).isin(model_ids)].copy()

    lineage_counts = overlap["OncotreeLineage"].fillna("NA").value_counts()
    disease_counts = overlap["OncotreePrimaryDisease"].fillna("NA").value_counts()

    summary_path = OUT / "depmap_initial_inventory.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("DepMap initial inventory\n")
        f.write("========================\n\n")
        f.write(f"CRISPR screened models: {len(gene_effect_ids)}\n")
        f.write(f"Metadata overlap: {len(overlap)}\n")
        f.write(f"Gene-effect columns including model id: {len(pd.read_csv(gene_effect_path, nrows=0).columns)}\n\n")
        f.write("Top OncotreeLineage counts\n")
        f.write("--------------------------\n")
        f.write(lineage_counts.head(30).to_string())
        f.write("\n\nTop OncotreePrimaryDisease counts\n")
        f.write("---------------------------------\n")
        f.write(disease_counts.head(30).to_string())
        f.write("\n")

    lineage_counts.to_csv(OUT / "lineage_counts.csv", header=["n_models"])
    disease_counts.to_csv(OUT / "primary_disease_counts.csv", header=["n_models"])
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

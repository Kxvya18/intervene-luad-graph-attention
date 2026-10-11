from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PRISM = RAW / "PRISM"
RESULTS = ROOT / "results"

MIN_MODELS = 20
TOP_N = 40


def corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < MIN_MODELS:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def clean_broad_id(value: object) -> str:
    text = str(value)
    return text.replace(" - QC Failure", "").strip()


def load_axis_table() -> pd.DataFrame:
    axis = pd.read_csv(RESULTS / "nsclc_srebp_multiomics_model_table.csv")
    cols = [
        "ModelID",
        "srebp_axis_dep",
        "srebp_axis_expr",
        "srebp_axis_cn",
        "srebp_axis_integrated",
    ]
    return axis[cols].dropna(subset=["srebp_axis_dep", "srebp_axis_integrated"]).copy()


def load_prism_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_meta = pd.read_csv(PRISM / "Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv")
    cell_meta = cell_meta[["row_id", "depmap_id", "ccle_name", "screen", "culture"]].copy()

    treatment = pd.read_csv(PRISM / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
    treatment["broad_id_clean"] = treatment["broad_id"].map(clean_broad_id)
    treatment = (
        treatment[treatment["perturbation_type"] == "trt_cp"]
        .groupby("broad_id_clean")
        .agg(
            drug_name=("name", lambda x: sorted(set(x.dropna().astype(str)))[0] if len(x.dropna()) else ""),
            dose_um=("dose", "median"),
            screens=("screen", lambda x: ",".join(sorted(set(x.dropna().astype(str))))),
            compound_plates=("compound_plate", lambda x: ",".join(sorted(set(x.dropna().astype(str)))[:6])),
        )
        .reset_index()
    )
    compound_list = PRISM / "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv"
    if compound_list.exists():
        compound = pd.read_csv(compound_list)
        compound["broad_id_clean"] = compound["IDs"].astype(str).str.replace("BRD:", "", regex=False)
        compound = compound.rename(
            columns={
                "Drug.Name": "compound_list_drug_name",
                "repurposing_target": "repurposing_target",
                "MOA": "moa",
                "Synonyms": "synonyms",
            }
        )
        compound = compound[
            [
                "broad_id_clean",
                "compound_list_drug_name",
                "repurposing_target",
                "moa",
                "synonyms",
            ]
        ].drop_duplicates("broad_id_clean")
        treatment = treatment.merge(compound, on="broad_id_clean", how="outer")
        treatment["drug_name"] = treatment["drug_name"].fillna(treatment["compound_list_drug_name"])
    return cell_meta, treatment


def load_nsclc_prism_lfc(axis: pd.DataFrame, cell_meta: pd.DataFrame) -> pd.DataFrame:
    nsclc_ids = set(axis["ModelID"].astype(str))
    row_to_model = cell_meta[cell_meta["depmap_id"].isin(nsclc_ids)][["row_id", "depmap_id"]]
    row_map = dict(zip(row_to_model["row_id"], row_to_model["depmap_id"]))
    keep_rows = set(row_map)

    chunks = []
    usecols = ["row_id", "broad_id", "dose", "compound_plate", "screen", "culture", "LFC"]
    for chunk in pd.read_csv(PRISM / "Repurposing_Public_24Q2_LFC_COLLAPSED.csv", usecols=usecols, chunksize=500_000):
        filtered = chunk[chunk["row_id"].isin(keep_rows)].copy()
        if len(filtered):
            filtered["ModelID"] = filtered["row_id"].map(row_map)
            filtered["broad_id_clean"] = filtered["broad_id"].map(clean_broad_id)
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame()
    lfc = pd.concat(chunks, ignore_index=True)
    lfc = (
        lfc.groupby(["ModelID", "broad_id_clean"], as_index=False)
        .agg(
            median_lfc=("LFC", "median"),
            n_measurements=("LFC", "size"),
            dose_um=("dose", "median"),
            screens=("screen", lambda x: ",".join(sorted(set(x.dropna().astype(str))))),
        )
    )
    return lfc.merge(axis, on="ModelID", how="inner")


def score_compounds(lfc_axis: pd.DataFrame, treatment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for broad_id, group in lfc_axis.groupby("broad_id_clean"):
        group = group.dropna(subset=["median_lfc", "srebp_axis_dep", "srebp_axis_integrated"])
        if group["ModelID"].nunique() < MIN_MODELS:
            continue
        dep_corr = corr(group["srebp_axis_dep"], group["median_lfc"])
        integrated_corr = corr(group["srebp_axis_integrated"], group["median_lfc"])

        hi = group["srebp_axis_integrated"] >= group["srebp_axis_integrated"].quantile(2 / 3)
        lo = group["srebp_axis_integrated"] <= group["srebp_axis_integrated"].quantile(1 / 3)
        high_mean = float(group.loc[hi, "median_lfc"].mean())
        low_mean = float(group.loc[lo, "median_lfc"].mean())

        rows.append(
            {
                "broad_id_clean": broad_id,
                "n_models": int(group["ModelID"].nunique()),
                "mean_lfc": float(group["median_lfc"].mean()),
                "srebp_dep_lfc_corr": dep_corr,
                "srebp_integrated_lfc_corr": integrated_corr,
                "high_srebp_mean_lfc": high_mean,
                "low_srebp_mean_lfc": low_mean,
                "high_minus_low_lfc": high_mean - low_mean,
                "sensitivity_score": -integrated_corr + -(high_mean - low_mean),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.merge(treatment, on="broad_id_clean", how="left")
    return out.sort_values("sensitivity_score", ascending=False)


def keyword_hits(compounds: pd.DataFrame) -> pd.DataFrame:
    keywords = [
        "lipid",
        "cholesterol",
        "sterol",
        "fatty",
        "SCD",
        "FASN",
        "HMG",
        "statin",
        "mTOR",
        "EGFR",
        "PI3K",
        "oxidative",
        "ferroptosis",
    ]
    text_cols = ["drug_name", "broad_id_clean", "repurposing_target", "moa", "synonyms"]
    rows = []
    for _, row in compounds.iterrows():
        text = " ".join(str(row.get(c, "")) for c in text_cols).lower()
        hits = [kw for kw in keywords if kw.lower() in text]
        if hits:
            data = row.to_dict()
            data["keyword_hits"] = ",".join(hits)
            rows.append(data)
    return pd.DataFrame(rows)


def moa_group_scores(compounds: pd.DataFrame) -> pd.DataFrame:
    if "moa" not in compounds or len(compounds) == 0:
        return pd.DataFrame()
    rows = []
    exploded = compounds.copy()
    exploded["moa"] = exploded["moa"].fillna("").astype(str)
    exploded["repurposing_target"] = exploded.get("repurposing_target", "").fillna("").astype(str)
    for _, row in exploded.iterrows():
        labels = []
        for value in [row["moa"], row["repurposing_target"]]:
            labels.extend([x.strip() for x in str(value).replace(";", "|").split("|") if x.strip()])
        if not labels:
            labels = ["unannotated"]
        for label in sorted(set(labels)):
            data = row.to_dict()
            data["mechanism_label"] = label
            rows.append(data)
    labels_df = pd.DataFrame(rows)
    grouped = (
        labels_df.groupby("mechanism_label")
        .agg(
            n_compounds=("broad_id_clean", "nunique"),
            mean_sensitivity_score=("sensitivity_score", "mean"),
            median_sensitivity_score=("sensitivity_score", "median"),
            mean_integrated_corr=("srebp_integrated_lfc_corr", "mean"),
            mean_high_minus_low_lfc=("high_minus_low_lfc", "mean"),
            top_compounds=("drug_name", lambda x: ", ".join(x.dropna().astype(str).head(8))),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n_compounds"] >= 2].sort_values(
        ["mean_sensitivity_score", "n_compounds"], ascending=[False, False]
    )
    return grouped


def write_summary(compounds: pd.DataFrame, keyword_df: pd.DataFrame, lfc_axis: pd.DataFrame) -> None:
    mechanisms = moa_group_scores(compounds)
    summary_path = RESULTS / "prism_srebp_drug_response_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("PRISM 24Q2 drug response validation for SREBP/lipid axis\n")
        f.write("=======================================================\n\n")
        f.write(f"NSCLC PRISM-linked model-treatment rows: {len(lfc_axis)}\n")
        f.write(f"Compounds with at least {MIN_MODELS} NSCLC models: {len(compounds)}\n")
        f.write("\nInterpretation: lower LFC means stronger drug sensitivity. Negative correlation means high SREBP-axis models are more sensitive.\n\n")

        cols = [
            "drug_name",
            "broad_id_clean",
            "repurposing_target",
            "moa",
            "n_models",
            "mean_lfc",
            "srebp_integrated_lfc_corr",
            "srebp_dep_lfc_corr",
            "high_srebp_mean_lfc",
            "low_srebp_mean_lfc",
            "high_minus_low_lfc",
            "sensitivity_score",
        ]
        f.write("Top compounds where high SREBP-axis models are more sensitive\n")
        f.write("-------------------------------------------------------------\n")
        if len(compounds):
            f.write(compounds[cols].head(TOP_N).to_string(index=False))
        else:
            f.write("No compounds passed model-count filter.")
        f.write("\n\nKeyword-filtered compounds of interest\n")
        f.write("--------------------------------------\n")
        if len(keyword_df):
            f.write(keyword_df[cols + ["keyword_hits"]].head(TOP_N).to_string(index=False))
        else:
            f.write("No keyword hits among scored compounds.")
        f.write("\n")
        f.write("\nMechanism-level enrichment summary\n")
        f.write("----------------------------------\n")
        mechanism_cols = [
            "mechanism_label",
            "n_compounds",
            "mean_sensitivity_score",
            "median_sensitivity_score",
            "mean_integrated_corr",
            "mean_high_minus_low_lfc",
            "top_compounds",
        ]
        if len(mechanisms):
            f.write(mechanisms[mechanism_cols].head(TOP_N).to_string(index=False))
        else:
            f.write("No mechanism labels available.")
        f.write("\n")
    print(f"Wrote {summary_path}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    axis = load_axis_table()
    cell_meta, treatment = load_prism_metadata()
    lfc_axis = load_nsclc_prism_lfc(axis, cell_meta)
    compounds = score_compounds(lfc_axis, treatment)
    keyword_df = keyword_hits(compounds) if len(compounds) else pd.DataFrame()
    mechanism_df = moa_group_scores(compounds) if len(compounds) else pd.DataFrame()

    lfc_axis.to_csv(RESULTS / "prism_nsclc_lfc_with_srebp_axis.csv", index=False)
    compounds.to_csv(RESULTS / "prism_srebp_compound_scores.csv", index=False)
    keyword_df.to_csv(RESULTS / "prism_srebp_keyword_compound_hits.csv", index=False)
    mechanism_df.to_csv(RESULTS / "prism_srebp_mechanism_scores.csv", index=False)
    write_summary(compounds, keyword_df, lfc_axis)
    print(f"Wrote {RESULTS / 'prism_srebp_compound_scores.csv'}")


if __name__ == "__main__":
    main()

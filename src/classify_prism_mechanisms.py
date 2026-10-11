from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


BUCKET_RULES = {
    "direct_lipid_axis": [
        "SCD",
        "SREBP",
        "HMGCR",
        "HMG",
        "STATIN",
        "CHOLESTEROL",
        "STEROL",
        "FATTY ACID",
        "FASN",
        "ACETYL-COA CARBOXYLASE",
        "LIPID",
        "LIPOXYGENASE",
        "GLUCOSYLCERAMIDASE",
        "HYPO LIPID",
        "HYPOLIPIDEMIC",
    ],
    "stress_survival": [
        "AKT",
        "PI3K",
        "MTOR",
        "HDAC",
        "DEUBIQUITINASE",
        "PROTEASOME",
        "PROTEOSTASIS",
        "MITOCHONDRIAL",
        "COMPLEX I",
        "XIAP",
        "BIRC",
        "APOPTOSIS",
        "EGFR",
        "FGFR",
        "TYROSINE KINASE",
        "RECEPTOR TYROSINE",
        "NRF2",
        "RETINOID",
        "HIF",
        "HYPOXIA",
    ],
    "generic_cytotoxic": [
        "CDK",
        "AURORA",
        "ANTICANCER",
        "ANTITUMOR",
        "CARDIAC GLYCOSIDE",
        "TOPOISOMERASE",
        "DNA",
        "BMI-1",
        "PURINE",
        "DIHYDROFOLATE",
        "MICROTUBULE",
        "TUBULIN",
        "CHEMOTHERAPY",
        "ANTIVIRAL",
    ],
}


def assign_bucket(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in ["drug_name", "repurposing_target", "moa", "synonyms"]
    ).upper()
    matched = []
    for bucket, terms in BUCKET_RULES.items():
        if any(term in text for term in terms):
            matched.append(bucket)
    if not matched:
        return "unclassified"
    # Prefer the most biologically specific explanation if multiple match.
    for bucket in ["direct_lipid_axis", "stress_survival", "generic_cytotoxic"]:
        if bucket in matched:
            return bucket
    return matched[0]


def matched_terms(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in ["drug_name", "repurposing_target", "moa", "synonyms"]
    ).upper()
    hits = []
    for bucket, terms in BUCKET_RULES.items():
        for term in terms:
            if term in text:
                hits.append(f"{bucket}:{term}")
    return "; ".join(hits)


def summarize(compounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    compounds = compounds.copy()
    compounds["mechanism_bucket"] = compounds.apply(assign_bucket, axis=1)
    compounds["matched_bucket_terms"] = compounds.apply(matched_terms, axis=1)

    bucket_summary = (
        compounds.groupby("mechanism_bucket")
        .agg(
            n_compounds=("broad_id_clean", "nunique"),
            mean_sensitivity_score=("sensitivity_score", "mean"),
            median_sensitivity_score=("sensitivity_score", "median"),
            mean_integrated_corr=("srebp_integrated_lfc_corr", "mean"),
            mean_dep_corr=("srebp_dep_lfc_corr", "mean"),
            mean_high_minus_low_lfc=("high_minus_low_lfc", "mean"),
            top_compounds=("drug_name", lambda x: ", ".join(x.dropna().astype(str).head(12))),
        )
        .reset_index()
        .sort_values("mean_sensitivity_score", ascending=False)
    )
    return compounds, bucket_summary


def paper_facing_tables(compounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = compounds[compounds["mechanism_bucket"].isin(["stress_survival", "direct_lipid_axis"])].copy()
    bucket_order = {"stress_survival": 0, "direct_lipid_axis": 1}
    keep["bucket_order"] = keep["mechanism_bucket"].map(bucket_order)
    keep = keep.sort_values(["bucket_order", "sensitivity_score"], ascending=[True, False])

    primary = keep[keep["mechanism_bucket"] == "stress_survival"].copy()
    secondary = keep[keep["mechanism_bucket"] == "direct_lipid_axis"].copy()
    return primary, secondary


def write_summary(compounds: pd.DataFrame, bucket_summary: pd.DataFrame) -> None:
    primary, secondary = paper_facing_tables(compounds)
    summary_path = RESULTS / "prism_mechanism_bucket_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("PRISM mechanism bucket comparison\n")
        f.write("=================================\n\n")
        f.write("Bucket interpretation\n")
        f.write("---------------------\n")
        f.write("Lower LFC means greater sensitivity. Higher sensitivity_score means high SREBP-axis models are more sensitive.\n\n")
        f.write("Bucket summary\n")
        f.write("--------------\n")
        f.write(bucket_summary.to_string(index=False))
        f.write("\n\n")
        f.write("Paper-facing interpretation\n")
        f.write("---------------------------\n")
        f.write(
            "Keep stress_survival as the main PRISM interpretation. "
            "Keep direct_lipid_axis as secondary/negative evidence. "
            "Generic cytotoxic compounds are excluded from the main claim because they are less specific.\n\n"
        )

        cols = [
            "drug_name",
            "mechanism_bucket",
            "repurposing_target",
            "moa",
            "n_models",
            "srebp_integrated_lfc_corr",
            "srebp_dep_lfc_corr",
            "high_minus_low_lfc",
            "sensitivity_score",
            "matched_bucket_terms",
        ]
        f.write("Primary PRISM evidence: stress_survival\n")
        f.write("---------------------------------------\n")
        if len(primary):
            f.write(primary[cols].head(40).to_string(index=False))
        else:
            f.write("No stress_survival compounds.")
        f.write("\n\n")

        f.write("Secondary/negative PRISM evidence: direct_lipid_axis\n")
        f.write("----------------------------------------------------\n")
        if len(secondary):
            f.write(secondary[cols].head(40).to_string(index=False))
        else:
            f.write("No direct_lipid_axis compounds.")
        f.write("\n\n")

        f.write("Excluded bucket audit\n")
        f.write("---------------------\n")
        f.write(
            "The following buckets are retained in the full CSV for auditability but should not be used as paper-facing claims.\n\n"
        )
        for bucket in ["generic_cytotoxic", "unclassified"]:
            f.write(f"Top compounds: {bucket}\n")
            f.write("-" * (15 + len(bucket)) + "\n")
            sub = compounds[compounds["mechanism_bucket"] == bucket].sort_values(
                "sensitivity_score", ascending=False
            )
            if len(sub):
                f.write(sub[cols].head(30).to_string(index=False))
            else:
                f.write("No compounds in bucket.")
            f.write("\n\n")
    print(f"Wrote {summary_path}")


def main() -> None:
    compounds = pd.read_csv(RESULTS / "prism_srebp_compound_scores.csv")
    compounds, bucket_summary = summarize(compounds)
    primary, secondary = paper_facing_tables(compounds)
    compounds.to_csv(RESULTS / "prism_srebp_compound_scores_bucketed.csv", index=False)
    bucket_summary.to_csv(RESULTS / "prism_mechanism_bucket_scores.csv", index=False)
    primary.to_csv(RESULTS / "prism_primary_stress_survival_hits.csv", index=False)
    secondary.to_csv(RESULTS / "prism_secondary_direct_lipid_axis_hits.csv", index=False)
    write_summary(compounds, bucket_summary)
    print(f"Wrote {RESULTS / 'prism_srebp_compound_scores_bucketed.csv'}")
    print(f"Wrote {RESULTS / 'prism_mechanism_bucket_scores.csv'}")
    print(f"Wrote {RESULTS / 'prism_primary_stress_survival_hits.csv'}")
    print(f"Wrote {RESULTS / 'prism_secondary_direct_lipid_axis_hits.csv'}")


if __name__ == "__main__":
    main()

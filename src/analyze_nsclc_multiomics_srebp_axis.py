from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

PRIMARY_DISEASE = "Non-Small Cell Lung Cancer"
MUTATION_GENES = ["KRAS", "NFE2L2", "KEAP1", "STK11", "EGFR", "TP53"]
SREBP_GENES = ["SREBF1", "SCAP", "SCD", "MBTPS1", "MBTPS2"]
CANDIDATE_GENES = [
    "NFE2L2",
    "KRAS",
    "UBIAD1",
    "UXS1",
    "SMARCA2",
    "TRPM7",
    "CFLAR",
    "PTDSS1",
    "PGD",
    "SREBF1",
    "SCAP",
    "SCD",
    "MBTPS1",
    "MBTPS2",
]


def symbol_to_column(columns: list[str]) -> dict[str, str]:
    mapping = {}
    for col in columns:
        if " (" in col and col.endswith(")"):
            mapping[col.split(" (", 1)[0]] = col
    return mapping


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 4:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def read_wide_subset(path: Path, symbols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    mapping = symbol_to_column(list(header.columns))
    cols = [header.columns[0]] + [mapping[s] for s in symbols if s in mapping]
    df = pd.read_csv(path, usecols=cols)
    df = df.rename(columns={df.columns[0]: "ModelID"})
    df["ModelID"] = df["ModelID"].astype(str)
    return df.rename(columns={mapping[s]: s for s in symbols if s in mapping})


def load_mutation_matrix(nsclc_ids: set[str]) -> pd.DataFrame:
    usecols = [
        "ModelID",
        "HugoSymbol",
        "OncogeneHighImpact",
        "TumorSuppressorHighImpact",
        "LikelyLoF",
        "Hotspot",
        "HessDriver",
    ]
    chunks = []
    for chunk in pd.read_csv(RAW / "OmicsSomaticMutations.csv", usecols=usecols, chunksize=100_000):
        chunk["ModelID"] = chunk["ModelID"].astype(str)
        filtered = chunk[
            chunk["ModelID"].isin(nsclc_ids) & chunk["HugoSymbol"].isin(MUTATION_GENES)
        ].copy()
        if len(filtered):
            chunks.append(filtered)
    mutations = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)

    out = pd.DataFrame({"ModelID": sorted(nsclc_ids)})
    for gene in MUTATION_GENES:
        gene_mut = mutations[mutations["HugoSymbol"] == gene]
        any_ids = set(gene_mut["ModelID"])
        high_ids = set(
            gene_mut[
                gene_mut[
                    [
                        "OncogeneHighImpact",
                        "TumorSuppressorHighImpact",
                        "LikelyLoF",
                        "Hotspot",
                        "HessDriver",
                    ]
                ]
                .fillna(False)
                .astype(bool)
                .any(axis=1)
            ]["ModelID"]
        )
        out[f"{gene}_mut"] = out["ModelID"].isin(any_ids)
        out[f"{gene}_high_impact"] = out["ModelID"].isin(high_ids)
    return out


def build_multiomics_frame() -> pd.DataFrame:
    models = pd.read_csv(RAW / "Model.csv")
    models["ModelID"] = models["ModelID"].astype(str)
    gene_effect_ids = pd.read_csv(RAW / "CRISPRGeneEffect.csv", usecols=[0])
    gene_effect_ids = gene_effect_ids.rename(columns={gene_effect_ids.columns[0]: "ModelID"})
    gene_effect_ids["ModelID"] = gene_effect_ids["ModelID"].astype(str)

    screened_ids = set(gene_effect_ids["ModelID"])
    nsclc_ids = set(
        models.loc[
            (models["OncotreePrimaryDisease"] == PRIMARY_DISEASE)
            & (models["ModelID"].isin(screened_ids)),
            "ModelID",
        ].astype(str)
    )

    symbols = sorted(set(SREBP_GENES + CANDIDATE_GENES + MUTATION_GENES))
    dep = read_wide_subset(RAW / "CRISPRGeneEffect.csv", symbols)
    dep_cols = [c for c in dep.columns if c != "ModelID"]
    dep[dep_cols] = -dep[dep_cols]
    dep = dep.rename(columns={c: f"{c}_dep" for c in dep_cols})

    expr = read_wide_subset(RAW / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", symbols)
    expr_cols = [c for c in expr.columns if c != "ModelID"]
    expr = expr.rename(columns={c: f"{c}_expr" for c in expr_cols})

    cn = read_wide_subset(RAW / "OmicsCNGene.csv", symbols)
    cn_cols = [c for c in cn.columns if c != "ModelID"]
    cn = cn.rename(columns={c: f"{c}_cn" for c in cn_cols})

    mut = load_mutation_matrix(nsclc_ids)
    frame = (
        pd.DataFrame({"ModelID": sorted(nsclc_ids)})
        .merge(dep, on="ModelID", how="left")
        .merge(expr, on="ModelID", how="left")
        .merge(cn, on="ModelID", how="left")
        .merge(mut, on="ModelID", how="left")
    )
    return frame


def add_axis_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for suffix in ["dep", "expr", "cn"]:
        cols = [f"{g}_{suffix}" for g in SREBP_GENES if f"{g}_{suffix}" in out]
        out[f"srebp_axis_{suffix}"] = out[cols].mean(axis=1)
    out["srebp_axis_integrated"] = (
        zscore(out["srebp_axis_dep"])
        + zscore(out["srebp_axis_expr"])
        + zscore(out["srebp_axis_cn"])
    )
    return out


def compare_by_mutation(frame: pd.DataFrame) -> pd.DataFrame:
    axis_cols = [
        "srebp_axis_dep",
        "srebp_axis_expr",
        "srebp_axis_cn",
        "srebp_axis_integrated",
    ]
    rows = []
    for marker in MUTATION_GENES:
        for flag in [f"{marker}_mut", f"{marker}_high_impact"]:
            if flag not in frame:
                continue
            mask = frame[flag].fillna(False).astype(bool)
            n_mut = int(mask.sum())
            n_wt = int((~mask).sum())
            for axis in axis_cols:
                if n_mut < 3 or n_wt < 3:
                    rows.append(
                        {
                            "marker": marker,
                            "flag": flag,
                            "axis": axis,
                            "n_mut": n_mut,
                            "n_wt": n_wt,
                            "mean_mut": np.nan,
                            "mean_wt": np.nan,
                            "difference_mut_minus_wt": np.nan,
                            "note": "insufficient group size",
                        }
                    )
                    continue
                rows.append(
                    {
                        "marker": marker,
                        "flag": flag,
                        "axis": axis,
                        "n_mut": n_mut,
                        "n_wt": n_wt,
                        "mean_mut": float(frame.loc[mask, axis].mean()),
                        "mean_wt": float(frame.loc[~mask, axis].mean()),
                        "difference_mut_minus_wt": float(
                            frame.loc[mask, axis].mean() - frame.loc[~mask, axis].mean()
                        ),
                        "note": "ok",
                    }
                )
    return pd.DataFrame(rows)


def gene_level_multiomics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene in sorted(set(SREBP_GENES + CANDIDATE_GENES)):
        dep_col = f"{gene}_dep"
        expr_col = f"{gene}_expr"
        cn_col = f"{gene}_cn"
        if dep_col not in frame:
            continue
        rows.append(
            {
                "gene_symbol": gene,
                "mean_dependency": float(frame[dep_col].mean()),
                "mean_expression": float(frame[expr_col].mean()) if expr_col in frame else np.nan,
                "mean_copy_number": float(frame[cn_col].mean()) if cn_col in frame else np.nan,
                "corr_dep_expr": corr(frame[dep_col], frame[expr_col]) if expr_col in frame else np.nan,
                "corr_dep_cn": corr(frame[dep_col], frame[cn_col]) if cn_col in frame else np.nan,
                "corr_expr_cn": corr(frame[expr_col], frame[cn_col])
                if expr_col in frame and cn_col in frame
                else np.nan,
                "corr_dep_with_srebp_integrated": corr(frame[dep_col], frame["srebp_axis_integrated"]),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_dep_with_srebp_integrated", ascending=False)


def candidate_by_mutation(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    measures = ["dep", "expr", "cn"]
    for marker in MUTATION_GENES:
        for flag in [f"{marker}_mut", f"{marker}_high_impact"]:
            if flag not in frame:
                continue
            mask = frame[flag].fillna(False).astype(bool)
            n_mut = int(mask.sum())
            n_wt = int((~mask).sum())
            if n_mut < 3 or n_wt < 3:
                continue
            for gene in CANDIDATE_GENES:
                for measure in measures:
                    col = f"{gene}_{measure}"
                    if col not in frame:
                        continue
                    rows.append(
                        {
                            "marker": marker,
                            "flag": flag,
                            "gene_symbol": gene,
                            "measure": measure,
                            "n_mut": n_mut,
                            "n_wt": n_wt,
                            "mean_mut": float(frame.loc[mask, col].mean()),
                            "mean_wt": float(frame.loc[~mask, col].mean()),
                            "difference_mut_minus_wt": float(
                                frame.loc[mask, col].mean() - frame.loc[~mask, col].mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def write_summary(
    mutation_axis: pd.DataFrame,
    gene_multiomics: pd.DataFrame,
    candidate_mutation: pd.DataFrame,
) -> None:
    summary_path = RESULTS / "nsclc_multiomics_srebp_axis_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("NSCLC multi-omics SREBP/lipid axis analysis\n")
        f.write("===========================================\n\n")
        f.write("Mutation-context shifts in SREBP axis\n")
        f.write("-------------------------------------\n")
        axis_focus = mutation_axis[mutation_axis["axis"].isin(["srebp_axis_integrated", "srebp_axis_dep"])]
        f.write(axis_focus.sort_values("difference_mut_minus_wt", ascending=False).to_string(index=False))
        f.write("\n\nGene-level dependency/expression/copy-number coupling\n")
        f.write("----------------------------------------------------\n")
        f.write(gene_multiomics.to_string(index=False))
        f.write("\n\nTop candidate multi-omics mutation shifts\n")
        f.write("-----------------------------------------\n")
        if len(candidate_mutation):
            f.write(
                candidate_mutation.sort_values("difference_mut_minus_wt", ascending=False)
                .head(60)
                .to_string(index=False)
            )
        else:
            f.write("No mutation comparison had enough samples.")
        f.write("\n")
    print(f"Wrote {summary_path}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = add_axis_scores(build_multiomics_frame())
    mutation_axis = compare_by_mutation(frame)
    gene_multiomics = gene_level_multiomics(frame)
    candidate_mutation = candidate_by_mutation(frame)

    frame.to_csv(RESULTS / "nsclc_srebp_multiomics_model_table.csv", index=False)
    mutation_axis.to_csv(RESULTS / "nsclc_srebp_axis_by_mutation_multiomics.csv", index=False)
    gene_multiomics.to_csv(RESULTS / "nsclc_srebp_gene_multiomics_coupling.csv", index=False)
    candidate_mutation.to_csv(RESULTS / "nsclc_candidate_multiomics_by_mutation.csv", index=False)
    write_summary(mutation_axis, gene_multiomics, candidate_mutation)

    print(f"NSCLC model rows: {len(frame)}")
    print(f"Wrote {RESULTS / 'nsclc_multiomics_srebp_axis_summary.txt'}")


if __name__ == "__main__":
    main()

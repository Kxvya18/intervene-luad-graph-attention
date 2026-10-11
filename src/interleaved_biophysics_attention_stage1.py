from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

MIN_NON_NULL = 20
RIDGE_ALPHA = 2.0
N_FOLDS = 5

SREBP_MODULE_GENES = ["SCAP", "SCD", "MBTPS1", "SREBF1", "MBTPS2"]
OXIDATIVE_MARKERS = ["KEAP1", "NFE2L2"]
SURVIVAL_DEP_GENES = ["CFLAR", "PGD", "SMARCA2", "TRPM7", "PTDSS1", "UBIAD1"]


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - s.mean()) / std


def symbol_to_column(columns: list[str]) -> dict[str, str]:
    mapping = {}
    for col in columns:
        if " (" in col and col.endswith(")"):
            mapping[col.split(" (", 1)[0]] = col
    return mapping


def read_gene_effect_subset(model_ids: set[str], symbols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(RAW / "CRISPRGeneEffect.csv", nrows=0)
    mapping = symbol_to_column(list(header.columns))
    found = [s for s in symbols if s in mapping]
    usecols = [header.columns[0]] + [mapping[s] for s in found]
    dep = pd.read_csv(RAW / "CRISPRGeneEffect.csv", usecols=usecols)
    dep = dep.rename(columns={dep.columns[0]: "ModelID"})
    dep["ModelID"] = dep["ModelID"].astype(str)
    dep = dep[dep["ModelID"].isin(model_ids)].copy()
    dep = dep.rename(columns={mapping[s]: s for s in found})
    for gene in found:
        dep[gene] = -pd.to_numeric(dep[gene], errors="coerce")
    return dep


def top_mito_genes(n: int = 16) -> list[str]:
    ranked = pd.read_csv(RESULTS / "nsclc_top_ranked_genes.csv")
    ranked["gene_symbol"] = ranked["gene_symbol"].astype(str)
    mito = ranked[ranked["gene_symbol"].str.match(r"^MRP[SL]", na=False)].copy()
    if len(mito) == 0:
        return []
    mito = mito.sort_values(
        ["dependency_strength", "hidden_nonhub_score"], ascending=[False, True]
    )
    return mito["gene_symbol"].drop_duplicates().head(n).tolist()


def weighted_module_score(dep: pd.DataFrame, weights: pd.Series, prefix: str) -> pd.Series:
    genes = [g for g in weights.index if g in dep.columns]
    if not genes:
        return pd.Series(np.nan, index=dep.index, name=prefix)
    w = weights.loc[genes].abs()
    if float(w.sum()) == 0:
        w = pd.Series(np.ones(len(genes)), index=genes)
    values = dep[genes].apply(zscore)
    return values.mul(w / w.sum(), axis=1).sum(axis=1).rename(prefix)


def load_targets(model_ids: set[str]) -> pd.DataFrame:
    lfc = pd.read_csv(RESULTS / "prism_nsclc_lfc_with_srebp_axis.csv")
    buckets = pd.read_csv(RESULTS / "prism_srebp_compound_scores_bucketed.csv")
    lfc["ModelID"] = lfc["ModelID"].astype(str)
    lfc = lfc[lfc["ModelID"].isin(model_ids)].copy()

    bucket_map = buckets.set_index("broad_id_clean")["mechanism_bucket"].to_dict()
    lfc["mechanism_bucket"] = lfc["broad_id_clean"].map(bucket_map)
    lfc["sensitivity"] = -pd.to_numeric(lfc["median_lfc"], errors="coerce")

    rows = pd.DataFrame({"ModelID": sorted(model_ids)})
    for bucket, col in [
        ("stress_survival", "target_stress_survival_sensitivity"),
        ("generic_cytotoxic", "generic_cytotoxic_sensitivity"),
        ("direct_lipid_axis", "direct_lipid_sensitivity"),
    ]:
        agg = (
            lfc[lfc["mechanism_bucket"] == bucket]
            .groupby("ModelID")["sensitivity"]
            .mean()
            .rename(col)
            .reset_index()
        )
        rows = rows.merge(agg, on="ModelID", how="left")

    rows["target_specific_stress_survival"] = (
        zscore(rows["target_stress_survival_sensitivity"])
        - 0.35 * zscore(rows["generic_cytotoxic_sensitivity"])
    )
    return rows


def make_folds(n: int, k: int = N_FOLDS) -> list[np.ndarray]:
    order = np.arange(n)
    return [order[i::k] for i in range(k)]


def fit_ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y_train.mean()

    xt = (x_train - x_mean) / x_std
    xv = (x_test - x_mean) / x_std
    xt = np.column_stack([np.ones(len(xt)), xt])
    xv = np.column_stack([np.ones(len(xv)), xv])

    penalty = np.eye(xt.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(xt.T @ xt + penalty) @ xt.T @ (y_train - y_mean)
    coef[0] += y_mean
    return xv @ coef


def topk_by_train_corr(
    x_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str], k: int
) -> list[str]:
    rows = []
    for col in feature_cols:
        aligned = pd.concat([x_train[col], y_train], axis=1).dropna()
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) if len(aligned) >= MIN_NON_NULL else np.nan
        rows.append((col, abs(corr) if pd.notna(corr) else -1.0))
    rows = sorted(rows, key=lambda item: item[1], reverse=True)
    return [col for col, _ in rows[: min(k, len(rows))]]


def cv_metrics(
    frame: pd.DataFrame, feature_cols: list[str], target_col: str, select_top_k: int | None = None
) -> dict[str, float]:
    data = frame[["ModelID", target_col] + feature_cols].dropna().copy()
    if len(data) < MIN_NON_NULL or not feature_cols:
        return {
            "n_models": len(data),
            "cv_corr": np.nan,
            "cv_r2": np.nan,
            "cv_mae": np.nan,
        }
    x = data[feature_cols].astype(float).to_numpy()
    y = data[target_col].astype(float).to_numpy()
    preds = np.full(len(data), np.nan)
    selected_counts: dict[str, int] = {}
    for test_idx in make_folds(len(data)):
        train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
        cols = feature_cols
        if select_top_k is not None:
            train_df = data.iloc[train_idx][feature_cols].astype(float)
            train_y = data.iloc[train_idx][target_col].astype(float)
            cols = topk_by_train_corr(train_df, train_y, feature_cols, select_top_k)
            for col in cols:
                selected_counts[col] = selected_counts.get(col, 0) + 1
        col_idx = [feature_cols.index(c) for c in cols]
        preds[test_idx] = fit_ridge_predict(x[train_idx][:, col_idx], y[train_idx], x[test_idx][:, col_idx])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    selected = "; ".join(f"{k}:{v}" for k, v in sorted(selected_counts.items(), key=lambda item: (-item[1], item[0]))[:8])
    return {
        "n_models": len(data),
        "cv_corr": float(pd.Series(y).corr(pd.Series(preds))),
        "cv_r2": float(1.0 - ss_res / ss_tot) if ss_tot else np.nan,
        "cv_mae": float(np.mean(np.abs(y - preds))),
        "selected_features": selected,
    }


def standardized_coefficients(
    frame: pd.DataFrame, feature_cols: list[str], target_col: str
) -> pd.DataFrame:
    data = frame[[target_col] + feature_cols].dropna().copy()
    x = data[feature_cols].astype(float)
    y = data[target_col].astype(float)
    xs = x.apply(zscore).to_numpy()
    ys = zscore(y).to_numpy()
    penalty = np.eye(xs.shape[1]) * RIDGE_ALPHA
    coef = np.linalg.pinv(xs.T @ xs + penalty) @ xs.T @ ys
    out = pd.DataFrame({"feature": feature_cols, "standardized_coef": coef})
    total = out["standardized_coef"].abs().sum()
    out["attention_weight"] = out["standardized_coef"].abs() / total if total else 0.0
    return out.sort_values("attention_weight", ascending=False)


def feature_corr_rows(frame: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        aligned = frame[[col, target_col]].dropna()
        rows.append(
            {
                "feature": col,
                "n_models": len(aligned),
                "target_corr": float(aligned[col].corr(aligned[target_col]))
                if len(aligned) >= MIN_NON_NULL
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_head_frame() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    multi = pd.read_csv(RESULTS / "nsclc_srebp_multiomics_model_table.csv")
    multi["ModelID"] = multi["ModelID"].astype(str)
    model_ids = set(multi["ModelID"])

    dep_symbols = sorted(set(SREBP_MODULE_GENES + SURVIVAL_DEP_GENES + top_mito_genes()))
    dep = read_gene_effect_subset(model_ids, dep_symbols)
    frame = multi.merge(dep, on="ModelID", how="left", suffixes=("", "_raw_dep"))

    residual_genes = pd.read_csv(RESULTS / "selected_srebp_lipid_module_genes.csv")
    residual_weights = residual_genes.set_index("gene_symbol")["thermo_residual_score"]
    frame["head_physics_residual"] = weighted_module_score(
        frame, residual_weights, "head_physics_residual"
    )

    srebp_cols = [f"{g}_dep" for g in SREBP_MODULE_GENES if f"{g}_dep" in frame.columns]
    if srebp_cols:
        frame["head_srebp_lipid"] = frame[srebp_cols].apply(zscore).mean(axis=1)
    else:
        frame["head_srebp_lipid"] = frame["srebp_axis_integrated"]

    mito_genes = [g for g in top_mito_genes() if g in frame.columns]
    frame["head_mito_translation"] = (
        frame[mito_genes].apply(zscore).mean(axis=1) if mito_genes else np.nan
    )

    ox_cols = []
    for marker in OXIDATIVE_MARKERS:
        for suffix, weight in [("_high_impact", 1.0), ("_mut", 0.5)]:
            col = f"{marker}{suffix}"
            if col in frame.columns:
                score_col = f"{col}_weighted"
                frame[score_col] = frame[col].fillna(False).astype(bool).astype(float) * weight
                ox_cols.append(score_col)
    frame["head_oxidative_stress"] = frame[ox_cols].sum(axis=1) if ox_cols else np.nan

    survival_cols = [f"{g}_dep" for g in SURVIVAL_DEP_GENES if f"{g}_dep" in frame.columns]
    frame["head_survival_dependency"] = (
        frame[survival_cols].apply(zscore).mean(axis=1) if survival_cols else np.nan
    )

    targets = load_targets(model_ids)
    frame = frame.merge(targets, on="ModelID", how="left")
    frame["head_generic_cytotoxic_penalty"] = zscore(frame["generic_cytotoxic_sensitivity"])

    base_heads = [
        "head_physics_residual",
        "head_srebp_lipid",
        "head_mito_translation",
        "head_oxidative_stress",
        "head_survival_dependency",
        "head_generic_cytotoxic_penalty",
    ]
    for col in base_heads:
        frame[col] = zscore(frame[col])

    interleaved = []
    for a, b in combinations(base_heads[:5], 2):
        col = f"interleave__{a.replace('head_', '')}__x__{b.replace('head_', '')}"
        frame[col] = zscore(frame[a] * frame[b])
        interleaved.append(col)

    frame["ibpa_stage1_state_score"] = zscore(
        frame[interleaved].mean(axis=1) - 0.35 * frame["head_generic_cytotoxic_penalty"]
    )
    return frame, base_heads, interleaved, mito_genes


def write_summary(
    frame: pd.DataFrame,
    model_scores: pd.DataFrame,
    feature_scores: pd.DataFrame,
    mito_genes: list[str],
) -> None:
    top_features = feature_scores.head(12)
    top_cells = frame.sort_values("ibpa_stage1_state_score", ascending=False).head(12)
    path = RESULTS / "interleaved_biophysics_attention_summary.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Stage 1 Interleaved Biological-Physics Attention prototype\n")
        f.write("=========================================================\n\n")
        f.write("Method note:\n")
        f.write(
            "This is a small interpretable IHA-inspired prototype, not a full transformer. "
            "Biological/physics heads are z-scored per NSCLC cell line. Interleaved pseudo-heads "
            "are pairwise head products between non-target biological/physics heads. "
            "The prediction target is PRISM stress-survival sensitivity, with a specific target "
            "also reported after penalizing generic cytotoxic sensitivity.\n\n"
        )
        f.write(f"NSCLC models with head table: {len(frame)}\n")
        f.write(f"Models with PRISM stress-survival target: {frame['target_stress_survival_sensitivity'].notna().sum()}\n")
        f.write(f"Mitochondrial translation genes used: {', '.join(mito_genes)}\n\n")

        f.write("Cross-validated model comparison\n")
        f.write("--------------------------------\n")
        f.write(model_scores.to_string(index=False))
        f.write("\n\n")

        f.write("Top interleaved/attention features\n")
        f.write("----------------------------------\n")
        f.write(top_features.to_string(index=False))
        f.write("\n\n")

        f.write("Top IBPA Stage 1 cell-line states\n")
        f.write("---------------------------------\n")
        cols = [
            "ModelID",
            "ibpa_stage1_state_score",
            "target_stress_survival_sensitivity",
            "generic_cytotoxic_sensitivity",
            "head_physics_residual",
            "head_srebp_lipid",
            "head_mito_translation",
            "head_oxidative_stress",
        ]
        f.write(top_cells[cols].to_string(index=False))
        f.write("\n\n")

        best = model_scores.sort_values("cv_corr", ascending=False).head(1)
        if len(best):
            row = best.iloc[0]
            f.write("Interpretation\n")
            f.write("--------------\n")
            f.write(
                f"Best cross-validated setting by correlation: {row['model']} "
                f"(cv_corr={row['cv_corr']:.3f}, cv_r2={row['cv_r2']:.3f}). "
                "Keep this extension only if interleaved features beat or clarify the single-head baseline.\n"
            )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    frame, base_heads, interleaved, mito_genes = build_head_frame()

    target_main = "target_stress_survival_sensitivity"
    target_specific = "target_specific_stress_survival"
    paper_heads = [c for c in base_heads if c != "head_generic_cytotoxic_penalty"]
    model_rows = []
    configs = [
        ("single_heads_with_cytotox_to_stress_survival", base_heads, target_main, None),
        ("paper_heads_no_cytotox_to_stress_survival", paper_heads, target_main, None),
        ("interleaved_only_to_stress_survival", interleaved, target_main, None),
        ("paper_heads_plus_interleaved_to_stress_survival", paper_heads + interleaved, target_main, None),
        ("sparse_top3_interleaved_to_stress_survival", interleaved, target_main, 3),
        ("sparse_top5_interleaved_to_stress_survival", interleaved, target_main, 5),
        ("single_heads_with_cytotox_to_specific_stress_survival", base_heads, target_specific, None),
        ("paper_heads_no_cytotox_to_specific_stress_survival", paper_heads, target_specific, None),
        ("interleaved_only_to_specific_stress_survival", interleaved, target_specific, None),
        ("paper_heads_plus_interleaved_to_specific_stress_survival", paper_heads + interleaved, target_specific, None),
        ("sparse_top3_interleaved_to_specific_stress_survival", interleaved, target_specific, 3),
        ("sparse_top5_interleaved_to_specific_stress_survival", interleaved, target_specific, 5),
    ]
    for model, cols, target, top_k in configs:
        row = {"model": model, "target": target, "n_features": len(cols)}
        row.update(cv_metrics(frame, cols, target, select_top_k=top_k))
        model_rows.append(row)
    model_scores = pd.DataFrame(model_rows).sort_values(["target", "cv_corr"], ascending=[True, False])

    feature_corr = feature_corr_rows(frame, base_heads + interleaved, target_main)
    coef = standardized_coefficients(frame, base_heads + interleaved, target_main)
    feature_scores = coef.merge(feature_corr, on="feature", how="left")

    keep_cols = [
        "ModelID",
        "target_stress_survival_sensitivity",
        "generic_cytotoxic_sensitivity",
        "direct_lipid_sensitivity",
        "target_specific_stress_survival",
        "ibpa_stage1_state_score",
    ] + base_heads + interleaved
    frame[keep_cols].to_csv(RESULTS / "interleaved_biophysics_attention_cell_scores.csv", index=False)
    feature_scores.to_csv(RESULTS / "interleaved_biophysics_attention_head_scores.csv", index=False)
    model_scores.to_csv(RESULTS / "interleaved_biophysics_attention_model_comparison.csv", index=False)
    write_summary(frame, model_scores, feature_scores, mito_genes)

    print(f"Wrote {RESULTS / 'interleaved_biophysics_attention_summary.txt'}")
    print(f"Wrote {RESULTS / 'interleaved_biophysics_attention_cell_scores.csv'}")
    print(f"Wrote {RESULTS / 'interleaved_biophysics_attention_head_scores.csv'}")
    print(f"Wrote {RESULTS / 'interleaved_biophysics_attention_model_comparison.csv'}")


if __name__ == "__main__":
    main()

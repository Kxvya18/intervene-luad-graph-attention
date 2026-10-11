from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from interleaved_biophysics_attention_full import high_low_stats, zscore


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def combined_scores() -> pd.DataFrame:
    cells = pd.read_csv(RESULTS / "ibpa_v2_discovery_cell_scores.csv")
    raw = pd.read_csv(RESULTS / "ibpa_v2_raw_prior_comparison.csv")
    cells["ModelID"] = cells["ModelID"].astype(str)

    scores = cells[
        [
            "ModelID",
            "ibpa_v2_state_score",
            "target_stress_survival_sensitivity",
            "generic_cytotoxic_sensitivity",
            "direct_lipid_sensitivity",
            "target_specific_stress_survival",
            "target_cytotoxic_residual_stress_survival",
            "srebp_mito_dependency_coupling",
            "oxidative_context_score",
        ]
    ].copy()

    # Paper 1 baseline: thermodynamic residual alone, before attention interleaving.
    scores["paper1_thermodynamic_residual_score"] = zscore(cells["srebp_lipid_thermo_residual"]) + zscore(
        cells["mito_translation_thermo_residual"]
    )

    # Paper 1 plus pan-cancer penalty: the strongest non-attention extension from the earlier framework.
    scores["paper1_residual_pan_cancer_penalty_score"] = (
        scores["paper1_thermodynamic_residual_score"]
        + 0.35 * zscore(cells["srebp_mito_thermo_coupling"])
        - 0.25 * zscore(cells["generic_cytotoxic_sensitivity"])
    )

    # Raw INTERVENE prior: physics/biology module prior before neural learning.
    raw_prior = raw[raw["score_type"] == "raw_physics_prior"]
    if len(raw_prior):
        # Reconstruct the same prior proxy directly from cell columns so the file is self-contained.
        scores["raw_intervene_physics_prior_score"] = (
            zscore(cells["srebp_lipid_thermo_residual"])
            + zscore(cells["mito_translation_thermo_residual"])
            + 0.35 * zscore(cells["srebp_mito_dependency_coupling"])
            + 0.25 * zscore(cells["oxidative_context_score"])
        )
    else:
        scores["raw_intervene_physics_prior_score"] = np.nan

    scores["learned_intervene_state_score"] = scores["ibpa_v2_state_score"]
    return scores


def evaluate(scores: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        ("paper1_thermodynamic_residual_score", "Paper 1 thermodynamic residual"),
        ("paper1_residual_pan_cancer_penalty_score", "Paper 1 residual + pan-cancer penalty"),
        ("raw_intervene_physics_prior_score", "Raw INTERVENE physics prior"),
        ("learned_intervene_state_score", "Learned INTERVENE state"),
    ]
    value_cols = [
        ("target_stress_survival_sensitivity", "PRISM stress-survival sensitivity"),
        ("target_cytotoxic_residual_stress_survival", "cytotoxic-residual stress-survival score"),
        ("target_specific_stress_survival", "stress-survival minus cytotoxic score"),
        ("generic_cytotoxic_sensitivity", "PRISM generic cytotoxic sensitivity"),
        ("direct_lipid_sensitivity", "PRISM direct lipid sensitivity"),
        ("srebp_mito_dependency_coupling", "SREBP x mitochondrial dependency coupling"),
        ("oxidative_context_score", "KEAP1/NFE2L2 oxidative context"),
    ]
    rows = []
    for score_col, score_label in score_cols:
        for value_col, value_label in value_cols:
            row = high_low_stats(scores, score_col, value_col, value_label, n_perm=3000)
            row["score_label"] = score_label
            row["score_col"] = score_col
            rows.append(row)
    return pd.DataFrame(rows)


def compact_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    keep_tests = [
        "cytotoxic-residual stress-survival score",
        "PRISM stress-survival sensitivity",
        "PRISM generic cytotoxic sensitivity",
        "PRISM direct lipid sensitivity",
        "SREBP x mitochondrial dependency coupling",
        "KEAP1/NFE2L2 oxidative context",
    ]
    compact = evaluation[evaluation["test"].isin(keep_tests)].copy()
    compact = compact[
        [
            "score_label",
            "test",
            "n",
            "n_high",
            "n_low",
            "high_minus_low",
            "effect_size",
            "perm_p_two_sided",
        ]
    ]
    order = {name: i for i, name in enumerate(keep_tests)}
    compact["test_order"] = compact["test"].map(order)
    return compact.sort_values(["score_label", "test_order"]).drop(columns=["test_order"])


def write_summary(compact: pd.DataFrame) -> None:
    path = RESULTS / "paper1_vs_intervene_validation_summary.txt"
    key = compact.set_index(["score_label", "test"])
    with path.open("w", encoding="utf-8") as f:
        f.write("Paper 1 framework versus INTERVENE validation\n")
        f.write("============================================\n\n")
        f.write(
            "Purpose: show that INTERVENE genuinely extends the thermodynamic framework instead of replacing it. "
            "Paper 1 gives the thermodynamic residual baseline; INTERVENE adds module tokens and IHA-inspired "
            "interleaved attention/state learning.\n\n"
        )
        f.write("Compact validation table\n")
        f.write("------------------------\n")
        f.write(compact.to_string(index=False))
        f.write("\n\n")
        f.write("Main reading\n")
        f.write("------------\n")
        for score in [
            "Paper 1 thermodynamic residual",
            "Paper 1 residual + pan-cancer penalty",
            "Raw INTERVENE physics prior",
            "Learned INTERVENE state",
        ]:
            if (score, "cytotoxic-residual stress-survival score") in key.index:
                row = key.loc[(score, "cytotoxic-residual stress-survival score")]
                f.write(
                    f"{score}: residual stress-survival effect={row['high_minus_low']:.3f}, "
                    f"p={row['perm_p_two_sided']:.4f}.\n"
                )
        f.write(
            "\nInterpretation: if the learned INTERVENE state beats or sharpens the Paper 1 residual while generic "
            "cytotoxic and direct lipid controls remain weaker, then the paper can claim an extension from "
            "thermodynamic network scoring to interpretable interleaved module discovery.\n"
        )


def main() -> None:
    scores = combined_scores()
    evaluation = evaluate(scores)
    compact = compact_table(evaluation)
    scores.to_csv(RESULTS / "paper1_vs_intervene_cell_scores.csv", index=False)
    evaluation.to_csv(RESULTS / "paper1_vs_intervene_validation.csv", index=False)
    compact.to_csv(RESULTS / "paper1_vs_intervene_compact_table.csv", index=False)
    write_summary(compact)
    print(f"Wrote {RESULTS / 'paper1_vs_intervene_cell_scores.csv'}")
    print(f"Wrote {RESULTS / 'paper1_vs_intervene_validation.csv'}")
    print(f"Wrote {RESULTS / 'paper1_vs_intervene_compact_table.csv'}")
    print(f"Wrote {RESULTS / 'paper1_vs_intervene_validation_summary.txt'}")


if __name__ == "__main__":
    main()

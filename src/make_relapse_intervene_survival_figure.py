from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


COLORS = {
    "raw_thermodynamic_transfer": "#4C6A92",
    "linear_cox_modules": "#8C8C8C",
    "no_prior_interleaved_attention": "#B65F5F",
    "normal_attention_mha": "#3B7A57",
    "normal_attention_mha_with_prior": "#79A66A",
    "relapse_intervene_prior_0_03": "#7C4D8B",
    "relapse_intervene_prior_0_30": "#A47DB0",
}

LABELS = {
    "raw_thermodynamic_transfer": "Raw transfer",
    "linear_cox_modules": "Linear Cox",
    "no_prior_interleaved_attention": "Interleaved\nno prior",
    "normal_attention_mha": "Normal\nattention",
    "normal_attention_mha_with_prior": "Normal attention\n+ prior",
    "relapse_intervene_prior_0_03": "Relapse-\nINTERVENE",
    "relapse_intervene_prior_0_30": "INTERVENE\nstrong prior",
}


def svg_text(x: float, y: float, text: str, size: int = 14, weight: str = "400", anchor: str = "middle") -> str:
    lines = str(text).split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x:.1f}" y="{y + i * (size + 3):.1f}" text-anchor="{anchor}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="#202020">{line}</text>'
        )
    return "\n".join(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS / "relapse_intervene_survival_model_comparison.csv")
    learned = results[results["seed"].astype(str) != "na"].copy()
    aggregate = (
        learned.groupby("model")
        .agg(
            c_index_mean=("c_index", "mean"),
            c_index_sd=("c_index", "std"),
            hr_mean=("hazard_ratio_approx", "mean"),
            logrank_p_median=("logrank_p", "median"),
            corr_mean=("risk_prior_corr", "mean"),
        )
        .reset_index()
    )
    raw = results[results["model"].eq("raw_thermodynamic_transfer")].iloc[0]
    aggregate = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "model": "raw_thermodynamic_transfer",
                        "c_index_mean": raw["c_index"],
                        "c_index_sd": 0.0,
                        "hr_mean": raw["hazard_ratio_approx"],
                        "logrank_p_median": raw["logrank_p"],
                        "corr_mean": raw["risk_prior_corr"],
                    }
                ]
            ),
            aggregate,
        ],
        ignore_index=True,
    )
    keep = [
        "raw_thermodynamic_transfer",
        "linear_cox_modules",
        "no_prior_interleaved_attention",
        "normal_attention_mha",
        "normal_attention_mha_with_prior",
        "relapse_intervene_prior_0_03",
        "relapse_intervene_prior_0_30",
    ]
    plot = aggregate[aggregate["model"].isin(keep)].copy()
    plot["order"] = plot["model"].map({m: i for i, m in enumerate(keep)})
    plot = plot.sort_values("order")

    width, height = 1180, 720
    left, right = 92, 42
    top, bottom = 96, 138
    chart_w = width - left - right
    chart_h = height - top - bottom
    ymin, ymax = 0.58, 0.69
    x_step = chart_w / len(plot)
    bar_w = x_step * 0.52

    def y_pos(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * chart_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 40, "Relapse-INTERVENE survival validation in GSE31210 NSCLC", 24, "700"),
        svg_text(width / 2, 70, "Out-of-fold relapse-risk prediction; higher C-index means better relapse-free survival ranking", 14),
    ]

    for tick in np.arange(0.58, 0.691, 0.02):
        y = y_pos(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(svg_text(left - 12, y + 5, f"{tick:.2f}", 13, anchor="end"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="1.4"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="1.4"/>')
    parts.append(svg_text(28, top + chart_h / 2, "C-index", 15, "700", anchor="middle").replace("<text", '<text transform="rotate(-90 28 {:.1f})"'.format(top + chart_h / 2), 1))

    for i, row in enumerate(plot.itertuples(index=False)):
        x = left + i * x_step + x_step / 2
        y = y_pos(float(row.c_index_mean))
        base = height - bottom
        color = COLORS.get(row.model, "#777777")
        parts.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{base - y:.1f}" fill="{color}" opacity="0.88"/>')
        if float(row.c_index_sd) > 0:
            y1 = y_pos(float(row.c_index_mean + row.c_index_sd))
            y2 = y_pos(float(row.c_index_mean - row.c_index_sd))
            parts.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="#111" stroke-width="1.5"/>')
            parts.append(f'<line x1="{x - 8:.1f}" y1="{y1:.1f}" x2="{x + 8:.1f}" y2="{y1:.1f}" stroke="#111" stroke-width="1.5"/>')
            parts.append(f'<line x1="{x - 8:.1f}" y1="{y2:.1f}" x2="{x + 8:.1f}" y2="{y2:.1f}" stroke="#111" stroke-width="1.5"/>')
        parts.append(svg_text(x, y - 10, f"{row.c_index_mean:.3f}", 13, "700"))
        p_label = f"HR {row.hr_mean:.2f}\np={row.logrank_p_median:.1e}"
        parts.append(svg_text(x, base + 24, LABELS[row.model], 13, "700"))
        parts.append(svg_text(x, base + 70, p_label, 12))

    best = plot.sort_values("c_index_mean", ascending=False).iloc[0]
    note = (
        "Best current learned model: Relapse-INTERVENE with very weak prior "
        f"(C-index {best.c_index_mean:.3f}, HR {best.hr_mean:.2f}). "
        "The margin over normal attention is small, so the claim should be sharpening plus biological grounding, not a dramatic prediction jump."
    )
    parts.append(f'<rect x="84" y="642" width="1012" height="48" rx="6" fill="#f3f1f5" stroke="#d7d0df"/>')
    parts.append(svg_text(width / 2, 672, note, 14, "400"))
    parts.append("</svg>")

    path = FIGURES / "relapse_intervene_survival_model_comparison.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

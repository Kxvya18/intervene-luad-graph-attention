from __future__ import annotations

from pathlib import Path
import html

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


COLORS = {
    "ink": "#1f2933",
    "muted": "#6b7280",
    "axis": "#c7cdd6",
    "nsclc": "#b2382f",
    "blue": "#2e6f9e",
    "green": "#46785a",
    "gold": "#b8842f",
    "gray": "#8d98a7",
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def write_svg(path: Path, body: str, width: int, height: int) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: {COLORS['ink']}; }}
.title {{ font-size: 24px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: {COLORS['muted']}; }}
.label {{ font-size: 12px; }}
.small {{ font-size: 11px; fill: {COLORS['muted']}; }}
.axis {{ stroke: {COLORS['axis']}; stroke-width: 1; }}
</style>
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def paper1_vs_intervene() -> None:
    compact = pd.read_csv(RESULTS / "paper1_vs_intervene_compact_table.csv")
    test = "cytotoxic-residual stress-survival score"
    data = compact[compact["test"] == test].copy()
    order = [
        "Paper 1 thermodynamic residual",
        "Paper 1 residual + pan-cancer penalty",
        "Raw INTERVENE physics prior",
        "Learned INTERVENE state",
    ]
    data["order"] = data["score_label"].map({name: i for i, name in enumerate(order)})
    data = data.sort_values("order")

    width, height = 960, 560
    left, top, plot_w, plot_h = 285, 105, 560, 330
    max_value = max(1.15, float(data["high_minus_low"].max()) * 1.12)
    rows = []
    rows.append('<text x="36" y="42" class="title">INTERVENE Sharpens The Original Thermodynamic Signal</text>')
    rows.append(
        '<text x="36" y="66" class="subtitle">High-vs-low score difference for PRISM cytotoxic-residual stress-survival response in NSCLC</text>'
    )
    rows.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick / max_value * plot_w
        rows.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="axis" opacity="0.45"/>')
        rows.append(f'<text x="{x - 8:.1f}" y="{top + plot_h + 24}" class="small">{tick:.2g}</text>')

    bar_h = 48
    gap = 31
    for i, row in enumerate(data.itertuples(index=False)):
        y = top + i * (bar_h + gap) + 12
        value = float(row.high_minus_low)
        bar_w = value / max_value * plot_w
        color = COLORS["nsclc"] if row.score_label == "Learned INTERVENE state" else COLORS["blue"]
        rows.append(f'<text x="36" y="{y + 31}" class="label">{esc(row.score_label)}</text>')
        rows.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" fill="{color}" opacity="0.88"/>')
        rows.append(f'<text x="{left + bar_w + 10:.1f}" y="{y + 30}" class="label">{value:.3f}, p={row.perm_p_two_sided:.4f}</text>')

    rows.append(f'<text x="{left}" y="{height - 58}" class="small">Interpretation: Paper 1 finds the residual vulnerability; INTERVENE keeps it and increases the validated residual effect.</text>')
    write_svg(FIGURES / "intervene_exploration_paper1_vs_intervene.svg", "\n".join(rows), width, height)


def cross_cancer_specificity() -> None:
    summary = pd.read_csv(RESULTS / "intervene_cross_cancer_specificity.csv")
    effect_col = "high_minus_low__cytotoxic-residual stress-survival score"
    p_col = "perm_p_two_sided__cytotoxic-residual stress-survival score"
    n_col = "n__cytotoxic-residual stress-survival score"
    data = summary.sort_values(effect_col, ascending=False).head(12).copy()

    width, height = 1080, 650
    left, top, plot_w, plot_h = 360, 105, 585, 420
    min_value = min(-0.4, float(data[effect_col].min()) * 1.15)
    max_value = max(1.25, float(data[effect_col].max()) * 1.08)
    zero_x = left + (0 - min_value) / (max_value - min_value) * plot_w

    rows = []
    rows.append('<text x="36" y="42" class="title">Cross-Cancer Specificity Audit</text>')
    rows.append(
        '<text x="36" y="66" class="subtitle">INTERVENE transfer-state association with cytotoxic-residual stress-survival PRISM response</text>'
    )
    rows.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" stroke="{COLORS["ink"]}" stroke-width="1.2"/>')
    rows.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    for tick in [-0.5, 0, 0.5, 1.0]:
        if tick < min_value or tick > max_value:
            continue
        x = left + (tick - min_value) / (max_value - min_value) * plot_w
        rows.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="axis" opacity="0.45"/>')
        rows.append(f'<text x="{x - 10:.1f}" y="{top + plot_h + 24}" class="small">{tick:.1f}</text>')

    bar_h = 25
    gap = 10
    for i, (_, row) in enumerate(data.iterrows()):
        label = row["lineage_label"]
        value = float(row[effect_col])
        p = float(row[p_col])
        n = int(row[n_col])
        y = top + i * (bar_h + gap)
        x0 = zero_x if value >= 0 else left + (value - min_value) / (max_value - min_value) * plot_w
        x1 = left + (value - min_value) / (max_value - min_value) * plot_w if value >= 0 else zero_x
        color = COLORS["nsclc"] if label == "Non-Small Cell Lung Cancer" else COLORS["green"]
        rows.append(f'<text x="36" y="{y + 18}" class="label">{esc(label)}</text>')
        rows.append(f'<rect x="{x0:.1f}" y="{y}" width="{abs(x1 - x0):.1f}" height="{bar_h}" fill="{color}" opacity="0.88"/>')
        rows.append(f'<text x="{max(x0, x1) + 8:.1f}" y="{y + 18}" class="small">effect={value:.3f}, p={p:.4f}, n={n}</text>')

    rows.append(f'<text x="{left}" y="{height - 72}" class="small">NSCLC is retained as the main context because it has strong residual validation and weak generic/direct-lipid confounding.</text>')
    rows.append(f'<text x="{left}" y="{height - 50}" class="small">Other positive lineages are future validation directions, not current main claims.</text>')
    write_svg(FIGURES / "intervene_exploration_cross_cancer_specificity.svg", "\n".join(rows), width, height)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paper1_vs_intervene()
    cross_cancer_specificity()
    print(f"Wrote {FIGURES / 'intervene_exploration_paper1_vs_intervene.svg'}")
    print(f"Wrote {FIGURES / 'intervene_exploration_cross_cancer_specificity.svg'}")


if __name__ == "__main__":
    main()

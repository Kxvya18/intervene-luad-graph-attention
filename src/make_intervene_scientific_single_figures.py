from __future__ import annotations

from html import escape
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
NOTES = ROOT / "paper_notes"

FG = "#171717"
MUTED = "#555555"
GRID = "#cfcfcf"
BLUE = "#2b6ca3"
TEAL = "#1b8b87"
GREEN = "#3b8d4f"
RED = "#cc4433"
ORANGE = "#d9832e"
PURPLE = "#7357a6"
YELLOW = "#f2c94c"
CELL_FILL = "#dcefa7"
CELL_STROKE = "#5d7f2f"
NUCLEUS = "#8cb5d9"
MITO = "#f6b08a"
LIPID = "#f4d35e"
ROS = "#d33f2f"


def e(s: object) -> str:
    return escape(str(s))


def text(x, y, s, size=14, weight=400, fill=FG, anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{e(s)}</text>'


def multiline(x, y, s, width=36, size=13, weight=400, fill=FG):
    return "\n".join(text(x, y + i * (size + 4), line, size, weight, fill) for i, line in enumerate(wrap(str(s), width)))


def rect(x, y, w, h, fill="white", stroke=FG, rx=5, sw=1.2):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def circle(x, y, r, fill, stroke=FG, sw=1.2):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def ellipse(x, y, rx, ry, fill, stroke=FG, sw=1.2):
    return f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def line(x1, y1, x2, y2, stroke=FG, sw=1.4, arrow=False, dash=None):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{marker}{dash_attr}/>'


def path(d, fill="none", stroke=FG, sw=1.4, arrow=False, dash=None, opacity=1.0):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{marker}{dash_attr}/>'


def svg(w, h, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{FG}"/>
  </marker>
</defs>
<rect width="{w}" height="{h}" fill="white"/>
{body}
</svg>
'''


def organelle_cell(cx, cy, scale=1.0, high=False):
    p = []
    p.append(ellipse(cx, cy, 145 * scale, 82 * scale, CELL_FILL, CELL_STROKE, 2.4))
    # membrane hairs/stress filaments
    for i in range(26):
        ang = 2 * np.pi * i / 26
        x1 = cx + np.cos(ang) * 145 * scale
        y1 = cy + np.sin(ang) * 82 * scale
        x2 = cx + np.cos(ang) * (165 if high else 156) * scale
        y2 = cy + np.sin(ang) * (98 if high else 90) * scale
        p.append(line(x1, y1, x2, y2, CELL_STROKE, 1.5))
    p.append(ellipse(cx - 45 * scale, cy - 5 * scale, 40 * scale, 28 * scale, NUCLEUS, "#4d6f94", 1.4))
    p.append(text(cx - 45 * scale, cy, "N", int(11 * scale), 700, "#304e6b", "middle"))
    # mitochondria
    mito_locs = [(-10, -40, -15), (48, -22, 12), (65, 28, -12), (-8, 42, 22)]
    for dx, dy, rot in mito_locs:
        x, y = cx + dx * scale, cy + dy * scale
        p.append(f'<g transform="rotate({rot} {x:.1f} {y:.1f})">')
        p.append(ellipse(x, y, 26 * scale, 10 * scale, MITO, ORANGE, 1.3))
        p.append(path(f"M{x-14*scale:.1f},{y:.1f} C{x-7*scale:.1f},{y-6*scale:.1f} {x+7*scale:.1f},{y+6*scale:.1f} {x+14*scale:.1f},{y:.1f}", "none", ORANGE, 1))
        p.append("</g>")
    # lipid droplets
    for dx, dy, r in [(90, -30, 12), (106, -5, 8), (84, 16, 10), (58, 0, 7)]:
        p.append(circle(cx + dx * scale, cy + dy * scale, r * scale, LIPID, "#9c7a21", 1))
    if high:
        for dx, dy in [(105, -62), (132, -36), (120, 54), (90, 68)]:
            p.append(text(cx + dx * scale, cy + dy * scale, "ROS", int(10 * scale), 700, ROS, "middle"))
        p.append(circle(cx + 18 * scale, cy + 10 * scale, 14 * scale, "#ffcc80", RED, 1.5))
        for i in range(9):
            ang = 2 * np.pi * i / 9
            p.append(line(cx + 18 * scale, cy + 10 * scale, cx + 18 * scale + np.cos(ang) * 24 * scale, cy + 10 * scale + np.sin(ang) * 24 * scale, RED, 1.2))
    return "\n".join(p)


def fig1_mechanistic_cells():
    p = []
    p.append(text(42, 42, "INTERVENE mechanism model", 22, 700))
    p.append(text(42, 68, "Scientific cartoon: biological module tokens, IHA-style interleaving, and PRISM validation", 13, 400, MUTED))
    centers = [(220, 300), (600, 300), (980, 300)]
    labels = [
        ("A", "Low INTERVENE state", "lower SREBP/mitochondrial stress coupling"),
        ("B", "High INTERVENE state", "SREBP + mitochondrial translation + oxidative stress"),
        ("C", "Validated stress-survival vulnerability", "PRISM stress-survival response after cytotoxic control"),
    ]
    for i, ((cx, cy), (letter, title, subtitle)) in enumerate(zip(centers, labels)):
        p.append(text(cx - 165, 115, letter, 26, 700))
        p.append(text(cx - 95, 115, title, 18, 700))
        p.append(text(cx - 95, 140, subtitle, 12, 400, MUTED))
        p.append(organelle_cell(cx, cy, 1.0, high=i > 0))
    # arrows and annotations
    p.append(line(365, 300, 455, 300, FG, 2, arrow=True))
    p.append(multiline(390, 245, "IHA-style pseudo-head mixing combines module views", 20, 12, 500, FG))
    p.append(line(745, 300, 835, 300, FG, 2, arrow=True))
    p.append(multiline(765, 245, "state tested against PRISM drug response", 21, 12, 500, FG))
    p.append(rect(845, 485, 265, 70, "#fff7ed", ORANGE, 8, 1.4))
    p.append(text(865, 515, "Main validation", 13, 700))
    p.append(text(865, 540, "cytotoxic-residual effect = 1.005", 12, 700, GREEN))
    p.append(text(865, 560, "permutation p = 0.0004", 12, 700, GREEN))
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "intervene_mechanistic_cell_cartoon.svg").write_text(svg(1200, 660, "\n".join(p)), encoding="utf-8")


def fig2_pipeline_apparatus():
    p = []
    p.append(text(45, 40, "INTERVENE analysis pipeline", 22, 700))
    p.append(text(45, 66, "Apparatus-style scientific diagram with data flow, dimensions, and validation checkpoints", 13, 400, MUTED))
    y = 250
    # data tanks
    data_blocks = [
        ("DepMap\nCRISPR", 95, "#e8f1fb"),
        ("Expression", 230, "#e8f1fb"),
        ("Copy\nnumber", 355, "#e8f1fb"),
        ("Mutation", 480, "#e8f1fb"),
        ("PRISM", 1040, "#eaf5ec"),
    ]
    for lab, x, fill in data_blocks:
        p.append(rect(x, y - 65, 88, 130, fill, BLUE if x < 1000 else GREEN, 8, 1.8))
        for i, l in enumerate(lab.split("\n")):
            p.append(text(x + 44, y - 8 + i * 16, l, 13, 700, FG, "middle"))
    p.append(line(185, y, 560, y, FG, 2, arrow=True))
    p.append(line(318, y - 88, 318, y - 15, FG, 1.2, dash="4,4"))
    p.append(text(235, y - 105, "multi-omics feature assembly", 13, 700))
    p.append(rect(575, y - 80, 150, 160, "#f8fbff", "#607d9d", 10, 2))
    p.append(text(650, y - 40, "Module token", 14, 700, FG, "middle"))
    p.append(text(650, y - 15, "matrix X", 14, 700, FG, "middle"))
    p.append(text(650, y + 20, "cell lines ×", 12, 500, MUTED, "middle"))
    p.append(text(650, y + 40, "modules × features", 12, 500, MUTED, "middle"))
    p.append(line(725, y, 790, y, FG, 2, arrow=True))
    p.append(rect(805, y - 80, 165, 160, "#fff7ed", ORANGE, 10, 2))
    p.append(text(887, y - 44, "INTERVENE", 15, 700, FG, "middle"))
    p.append(text(887, y - 20, "IHA-style", 13, 700, FG, "middle"))
    p.append(text(887, y + 4, "pseudo-head", 13, 700, FG, "middle"))
    p.append(text(887, y + 28, "mixing", 13, 700, FG, "middle"))
    # head diagram
    for i, col in enumerate([BLUE, TEAL, ORANGE, PURPLE]):
        p.append(circle(840, y + 58 - i * 18, 6, col, col))
    p.append(circle(930, y + 30, 12, GREEN, GREEN))
    p.append(line(848, y + 58, 918, y + 30, BLUE, 1.2))
    p.append(line(848, y + 40, 918, y + 30, TEAL, 1.2))
    p.append(line(848, y + 22, 918, y + 30, ORANGE, 1.2))
    p.append(line(970, y, 1035, y, FG, 2, arrow=True))
    p.append(text(533, y + 125, "paper-facing tokens: SREBP/lipid, mitochondrial translation, oxidative stress, proteostasis, apoptosis/IAP, AKT/mTOR", 12, 400, MUTED))
    p.append(line(1085, y + 70, 1085, y + 145, GREEN, 1.7, arrow=True))
    p.append(rect(955, y + 155, 260, 80, "#ffffff", GREEN, 8, 1.6))
    p.append(text(985, y + 188, "Stress-survival validation", 13, 700))
    p.append(text(985, y + 212, "direct lipid and cytotoxic controls", 12, 500, MUTED))
    p.append(line(887, y + 80, 887, y + 155, ORANGE, 1.7, arrow=True))
    p.append(rect(760, y + 155, 250, 80, "#ffffff", ORANGE, 8, 1.6))
    p.append(text(790, y + 188, "Vulnerability state score", 13, 700))
    p.append(text(790, y + 212, "physics-guided reconstruction", 12, 500, MUTED))
    (FIGURES / "intervene_pipeline_apparatus.svg").write_text(svg(1260, 560, "\n".join(p)), encoding="utf-8")


def add_axes(p, x, y, w, h, ymin, ymax, ylabel=""):
    zero = y + h - (0 - ymin) / (ymax - ymin) * h
    p.append(line(x, y, x, y + h, FG, 1.1))
    p.append(line(x, y + h, x + w, y + h, FG, 1.1))
    p.append(line(x, zero, x + w, zero, GRID, 1))
    p.append(text(x - 8, y + 4, f"{ymax:.1f}", 10, 400, MUTED, "end"))
    p.append(text(x - 8, zero + 4, "0", 10, 400, MUTED, "end"))
    p.append(text(x - 8, y + h, f"{ymin:.1f}", 10, 400, MUTED, "end"))
    if ylabel:
        p.append(text(x - 42, y + h / 2, ylabel, 11, 400, MUTED, "middle"))
    return zero


def beeswarm(metric, title, outfile, color, ylabel):
    df = pd.read_csv(RESULTS / "ibpa_v2_discovery_cell_scores.csv")
    low, high = df["ibpa_v2_state_score"].quantile([1 / 3, 2 / 3])
    df = df[(df["ibpa_v2_state_score"] <= low) | (df["ibpa_v2_state_score"] >= high)].copy()
    df["group"] = np.where(df["ibpa_v2_state_score"] >= high, "High INTERVENE", "Low INTERVENE")
    vals = df[metric].dropna().astype(float)
    ymin = min(0, vals.min()) - 0.12 * (vals.max() - min(0, vals.min()))
    ymax = vals.max() + 0.12 * (vals.max() - min(0, vals.min()))
    p = [text(48, 42, title, 22, 700), text(48, 66, "Each dot is one NSCLC cell line; boxes show interquartile range", 12, 400, MUTED)]
    x, y, w, h = 120, 115, 560, 360
    add_axes(p, x, y, w, h, ymin, ymax, ylabel)
    rng = np.random.default_rng(123 + len(metric))
    for gi, group in enumerate(["Low INTERVENE", "High INTERVENE"]):
        gvals = df[df["group"] == group][metric].dropna().astype(float).to_numpy()
        cx = x + 170 + gi * 220
        yy = lambda v: y + h - (v - ymin) / (ymax - ymin) * h
        q1, med, q3 = np.quantile(gvals, [0.25, 0.5, 0.75])
        p.append(rect(cx - 38, yy(q3), 76, yy(q1) - yy(q3), "#ffffff", color if gi else "#7e8a96", 3, 1.8))
        p.append(line(cx - 45, yy(med), cx + 45, yy(med), FG, 2))
        for v in gvals:
            p.append(circle(cx + rng.uniform(-48, 48), yy(v), 4, color if gi else "#9aa7b5", "white", 0.5))
        p.append(text(cx, y + h + 30, group, 13, 700, FG, "middle"))
        p.append(text(cx, y + h + 50, f"n={len(gvals)}", 11, 400, MUTED, "middle"))
    diff = df[df["group"] == "High INTERVENE"][metric].mean() - df[df["group"] == "Low INTERVENE"][metric].mean()
    p.append(rect(715, 160, 310, 118, "#ffffff", color, 8, 1.5))
    p.append(text(742, 195, "High - low difference", 14, 700))
    p.append(text(742, 230, f"Δ = {diff:.3f}", 24, 700, color))
    p.append(multiline(742, 260, "This is the paper-facing high-vs-low INTERVENE state comparison.", 34, 12, 400, MUTED))
    (FIGURES / outfile).write_text(svg(1080, 560, "\n".join(p)), encoding="utf-8")


def ablation_plot():
    ab = pd.read_csv(RESULTS / "ibpa_v2_ablation_summary.csv").set_index("ablation")
    order = [
        ("no_prior", "No prior"),
        ("prior_weight_0.05", "Weak prior\n0.05"),
        ("prior_weight_0.22_current", "Current\n0.22"),
        ("prior_weight_0.50", "Strong\n0.50"),
        ("remove_srebp_from_prior", "No\nSREBP"),
        ("remove_mito_from_prior", "No\nmito"),
        ("remove_oxidative_from_prior", "No\noxidative"),
    ]
    vals = [float(ab.loc[k, "cytotoxic_residual_stress_survival_score_effect"]) for k, _ in order]
    pvals = [float(ab.loc[k, "cytotoxic_residual_stress_survival_score_p"]) for k, _ in order]
    p = [text(48, 42, "INTERVENE ablation: weak physics guidance is necessary", 22, 700), text(48, 66, "Outcome: cytotoxic-residual stress-survival high-vs-low effect", 12, 400, MUTED)]
    x, y, w, h = 95, 130, 780, 340
    ymin, ymax = -0.15, 1.25
    zero = add_axes(p, x, y, w, h, ymin, ymax, "effect size")
    bw = w / len(vals) * 0.62
    for i, ((_, lab), v, pv) in enumerate(zip(order, vals, pvals)):
        cx = x + (i + 0.5) * w / len(vals)
        top = y + h - (v - ymin) / (ymax - ymin) * h
        col = GREEN if i == 1 else RED if i == 0 else TEAL if i in [2, 3] else ORANGE
        p.append(rect(cx - bw / 2, min(zero, top), bw, abs(zero - top), col, col, 2, 0))
        p.append(text(cx, top - 10 if v >= 0 else top + 18, f"{v:.2f}", 11, 700, FG, "middle"))
        ptxt = "p<0.001" if pv < 0.001 else f"p={pv:.3f}"
        p.append(text(cx, top - 25 if v >= 0 else top + 34, ptxt, 9, 400, MUTED, "middle"))
        for k, l in enumerate(lab.split("\n")):
            p.append(text(cx, y + h + 24 + k * 13, l, 10, 500, FG, "middle"))
    p.append(rect(910, 165, 250, 130, "#fff7ed", ORANGE, 8, 1.4))
    p.append(text(935, 200, "Key result", 15, 700))
    p.append(multiline(935, 230, "No-prior attention fails. Weak-prior INTERVENE gives the strongest biology-specific signal.", 32, 13, 500))
    (FIGURES / "intervene_plot_ablation.svg").write_text(svg(1220, 570, "\n".join(p)), encoding="utf-8")


def attention_edge_plot():
    edges = pd.read_csv(RESULTS / "ibpa_v2_discovery_attention_edges.csv").head(10)
    p = [text(48, 42, "INTERVENE learned attention edges", 22, 700), text(48, 66, "Top mean attention enrichments from IHA-style pseudo-head mixing", 12, 400, MUTED)]
    x, y, w, h = 115, 120, 720, 360
    vals = edges["attention_enrichment_vs_uniform"].astype(float).to_numpy()
    labels = [f"{r.source_token.replace('_', ' ')} → {r.target_token.replace('_', ' ')}" for r in edges.itertuples()]
    xmax = vals.max() * 1.1
    p.append(line(x, y, x, y + h, FG, 1.1))
    p.append(line(x, y + h, x + w, y + h, FG, 1.1))
    for i, (lab, v) in enumerate(zip(labels, vals)):
        yy = y + i * (h / len(vals)) + 8
        bw = v / xmax * w
        p.append(rect(x, yy, bw, 18, GREEN if "mitochondrial" in lab or "mito" in lab else TEAL, "none", 2, 0))
        p.append(text(x - 8, yy + 14, lab, 10, 500, FG, "end"))
        p.append(text(x + bw + 8, yy + 14, f"{v:.3f}", 10, 700, FG))
    p.append(rect(870, 165, 280, 145, "#ffffff", GREEN, 8, 1.4))
    p.append(text(895, 200, "Biological pattern", 15, 700))
    p.append(multiline(895, 232, "Most strong edges point toward mitochondrial translation, linking SREBP/lipid, proteostasis, apoptosis/IAP, and AKT/mTOR to a stress-survival state.", 34, 13, 500))
    (FIGURES / "intervene_plot_attention_edges.svg").write_text(svg(1200, 560, "\n".join(p)), encoding="utf-8")


def tables_and_notes():
    validation = pd.read_csv(RESULTS / "ibpa_v2_discovery_validation.csv")
    validation[
        validation["test"].isin([
            "PRISM stress-survival sensitivity",
            "cytotoxic-residual stress-survival score",
            "stress-survival minus cytotoxic score",
            "PRISM direct lipid sensitivity",
            "PRISM generic cytotoxic sensitivity",
            "KEAP1/NFE2L2 oxidative context",
            "KEAP1/NFE2L2 high-impact enrichment",
        ])
    ][["test", "n_high", "n_low", "high_mean", "low_mean", "high_minus_low", "effect_size", "perm_p_two_sided"]].to_csv(RESULTS / "intervene_table1_validation_summary.csv", index=False)
    pd.read_csv(RESULTS / "ibpa_v2_ablation_summary.csv")[
        ["ablation", "reconstruction_mse", "state_prior_corr", "cytotoxic_residual_stress_survival_score_effect", "cytotoxic_residual_stress_survival_score_p", "prism_generic_cytotoxic_sensitivity_effect", "prism_direct_lipid_sensitivity_effect"]
    ].to_csv(RESULTS / "intervene_table2_ablation_summary.csv", index=False)
    pd.read_csv(RESULTS / "ibpa_v2_discovery_leads.csv")[
        ["priority", "lead", "paper_role", "evidence"]
    ].to_csv(RESULTS / "intervene_table3_discovery_leads.csv", index=False)
    NOTES.mkdir(exist_ok=True)
    (NOTES / "intervene_figure_drawing_guide.md").write_text(
        """# INTERVENE Figure Drawing Guide

## Figure style
- Use one idea per figure, not many panels crammed together.
- Keep scientific labels on the figure: module names, dataset source, validation endpoint, effect size.
- Use SVG as the editable master. Open in Illustrator, Inkscape, PowerPoint, or BioRender.

## Cell illustration details
- Draw an NSCLC cell with: membrane, nucleus, mitochondria, lipid droplets, ROS/oxidative stress marks.
- Label SREBP/lipid near lipid droplets.
- Label mitochondrial translation near mitochondria.
- Label KEAP1/NFE2L2 or oxidative stress near ROS marks.
- Avoid cartoon-only claims; connect each biological part to a measured module token.

## Quantitative plot rules
- One plot per image when possible.
- Show actual cell-line points, not only bars.
- Always mark low vs high INTERVENE groups and sample size.
- Controls must appear: direct lipid and generic cytotoxic.

## Manuscript figure order
1. Mechanistic cell cartoon: what INTERVENE discovers.
2. Apparatus-style analysis pipeline: data to module tokens to IHA-style mixing to PRISM validation.
3. Cytotoxic-residual stress-survival distribution plot.
4. Direct lipid and generic cytotoxic control plots.
5. Ablation plot.
6. Attention edge plot.
""",
        encoding="utf-8",
    )


def main():
    FIGURES.mkdir(exist_ok=True)
    tables_and_notes()
    fig1_mechanistic_cells()
    fig2_pipeline_apparatus()
    beeswarm("target_cytotoxic_residual_stress_survival", "Cytotoxic-residual stress-survival is enriched in high INTERVENE state", "intervene_plot_cytotoxic_residual.svg", GREEN, "residual score")
    beeswarm("target_stress_survival_sensitivity", "PRISM stress-survival sensitivity is higher in high INTERVENE state", "intervene_plot_stress_survival.svg", BLUE, "drug sensitivity")
    beeswarm("generic_cytotoxic_sensitivity", "Generic cytotoxic response does not explain the INTERVENE state", "intervene_plot_generic_cytotoxic_control.svg", RED, "drug sensitivity")
    beeswarm("direct_lipid_sensitivity", "Direct lipid drugs remain a secondary/weak signal", "intervene_plot_direct_lipid_control.svg", ORANGE, "drug sensitivity")
    ablation_plot()
    attention_edge_plot()
    for path in sorted(FIGURES.glob("intervene_*.svg")):
        print(f"Wrote {path}")
    print(f"Wrote {NOTES / 'intervene_figure_drawing_guide.md'}")


if __name__ == "__main__":
    main()

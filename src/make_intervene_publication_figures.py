from __future__ import annotations

from html import escape
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FG = "#17212b"
MUTED = "#5d6b78"
GRID = "#cfd8e3"
BLUE = "#2c6db2"
TEAL = "#128b86"
GREEN = "#3f8f4f"
RED = "#c74646"
ORANGE = "#d27a2a"
PURPLE = "#7659ad"
CYTO = "#9b5c2e"
PALE = "#f6f8fb"
CELL = "#e8f4f2"
NUCLEUS = "#dbe8ff"
MITO = "#f7d6c7"
LIPID = "#f8e6a0"


def esc(s: object) -> str:
    return escape(str(s))


def text(x: float, y: float, s: object, size: int = 12, weight: int = 400, fill: str = FG, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>'


def multiline(x: float, y: float, s: object, chars: int = 34, size: int = 12, weight: int = 400, fill: str = FG) -> str:
    return "\n".join(text(x, y + i * (size + 4), line, size, weight, fill) for i, line in enumerate(wrap(str(s), chars)))


def rect(x: float, y: float, w: float, h: float, fill: str = "white", stroke: str = GRID, rx: float = 6, sw: float = 1) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = FG, sw: float = 1.5, arrow: bool = False, dash: str | None = None) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{marker}{dash_attr}/>'


def circle(x: float, y: float, r: float, fill: str, stroke: str = "white", sw: float = 1) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def ellipse(x: float, y: float, rx: float, ry: float, fill: str, stroke: str, sw: float = 1.5) -> str:
    return f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def path(d: str, fill: str = "none", stroke: str = FG, sw: float = 1.5, arrow: bool = False, opacity: float = 1.0) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{marker}/>'


def svg(width: int, height: int, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{FG}"/>
  </marker>
</defs>
<rect width="{width}" height="{height}" fill="white"/>
{body}
</svg>
'''


def panel(label: str, title: str, x: float, y: float) -> str:
    return text(x, y, label, 16, 700) + "\n" + text(x + 28, y, title, 16, 700)


def cell_illustration(cx: float, cy: float, scale: float = 1.0, stress: bool = True) -> str:
    p: list[str] = []
    p.append(ellipse(cx, cy, 145 * scale, 92 * scale, CELL, "#6caea9", 2))
    p.append(ellipse(cx - 35 * scale, cy - 8 * scale, 43 * scale, 32 * scale, NUCLEUS, "#6380bf", 1.5))
    p.append(text(cx - 35 * scale, cy - 2 * scale, "nucleus", int(10 * scale), 500, "#486095", "middle"))
    for dx, dy, rot in [(-5, -45, -18), (40, -22, 12), (65, 30, -8), (-15, 45, 20)]:
        x, y = cx + dx * scale, cy + dy * scale
        p.append(f'<g transform="rotate({rot} {x:.1f} {y:.1f})">')
        p.append(ellipse(x, y, 28 * scale, 11 * scale, MITO, ORANGE, 1.4))
        p.append(path(f"M{x-16*scale:.1f},{y:.1f} C{x-8*scale:.1f},{y-7*scale:.1f} {x+8*scale:.1f},{y+7*scale:.1f} {x+16*scale:.1f},{y:.1f}", "none", ORANGE, 1))
        p.append("</g>")
    for dx, dy, r in [(80, -32, 12), (92, -8, 8), (82, 16, 10), (54, 0, 7)]:
        p.append(circle(cx + dx * scale, cy + dy * scale, r * scale, LIPID, "#c89b22", 1.2))
    if stress:
        for dx, dy in [(95, -58), (118, -37), (118, 45), (96, 65)]:
            p.append(text(cx + dx * scale, cy + dy * scale, "ROS", int(11 * scale), 700, RED, "middle"))
        p.append(path(f"M{cx+50*scale:.1f},{cy+58*scale:.1f} C{cx+90*scale:.1f},{cy+100*scale:.1f} {cx+145*scale:.1f},{cy+70*scale:.1f} {cx+170*scale:.1f},{cy+105*scale:.1f}", "none", RED, 1.6, arrow=True))
    return "\n".join(p)


def head_mixing_icon(x: float, y: float) -> str:
    p: list[str] = []
    for i, col in enumerate([BLUE, TEAL, ORANGE, PURPLE]):
        p.append(circle(x, y + i * 28, 11, col, col))
        p.append(text(x, y + i * 28 + 4, f"H{i+1}", 8, 700, "white", "middle"))
    for j, (yy, col) in enumerate([(y + 16, GREEN), (y + 68, RED)]):
        p.append(circle(x + 115, yy, 17, col, col))
        p.append(text(x + 115, yy + 4, f"P{j+1}", 10, 700, "white", "middle"))
    p.append(line(x + 14, y, x + 98, y + 16, BLUE, 1.7, arrow=True))
    p.append(line(x + 14, y + 28, x + 98, y + 16, TEAL, 1.7, arrow=True))
    p.append(line(x + 14, y + 56, x + 98, y + 68, ORANGE, 1.7, arrow=True))
    p.append(line(x + 14, y + 84, x + 98, y + 68, PURPLE, 1.7, arrow=True))
    p.append(text(x - 10, y - 22, "heads", 11, 700, MUTED))
    p.append(text(x + 82, y - 22, "pseudo-heads", 11, 700, MUTED))
    return "\n".join(p)


def quantile_groups(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    low_cut = out[score_col].quantile(1 / 3)
    high_cut = out[score_col].quantile(2 / 3)
    out["intervene_group"] = "middle"
    out.loc[out[score_col] <= low_cut, "intervene_group"] = "low"
    out.loc[out[score_col] >= high_cut, "intervene_group"] = "high"
    return out


def beeswarm_panel(
    df: pd.DataFrame,
    metric: str,
    title: str,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    y_label: str = "sensitivity / score",
) -> str:
    data = df[df["intervene_group"].isin(["low", "high"])][["intervene_group", metric]].dropna().copy()
    vals = data[metric].astype(float)
    ymin = float(min(vals.min(), 0))
    ymax = float(vals.max())
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 1
    ymin -= pad
    ymax += pad
    p: list[str] = [text(x, y - 16, title, 13, 700)]
    p.append(line(x + 42, y, x + 42, y + h, GRID, 1))
    p.append(line(x + 42, y + h, x + w, y + h, GRID, 1))
    p.append(text(x + 4, y + 6, f"{ymax:.1f}", 9, 400, MUTED))
    p.append(text(x + 4, y + h, f"{ymin:.1f}", 9, 400, MUTED))
    p.append(text(x + 2, y + h / 2, y_label, 9, 400, MUTED))
    rng = np.random.default_rng(17 + len(metric))
    for idx, group in enumerate(["low", "high"]):
        gx = x + 110 + idx * 120
        gvals = data.loc[data["intervene_group"] == group, metric].astype(float).to_numpy()
        q1, med, q3 = np.quantile(gvals, [0.25, 0.5, 0.75])
        yy = lambda v: y + h - (v - ymin) / (ymax - ymin) * h
        p.append(rect(gx - 24, yy(q3), 48, yy(q1) - yy(q3), "#ffffff", color if group == "high" else "#7f8b96", rx=2, sw=1.4))
        p.append(line(gx - 28, yy(med), gx + 28, yy(med), FG, 1.5))
        for val in gvals:
            jitter = rng.uniform(-30, 30)
            fill = color if group == "high" else "#9aa7b5"
            p.append(circle(gx + jitter, yy(val), 3.2, fill, "white", 0.4))
        p.append(text(gx, y + h + 18, f"{group} INTERVENE", 10, 500, FG, "middle"))
        p.append(text(gx, y + h + 32, f"n={len(gvals)}", 9, 400, MUTED, "middle"))
    diff = data[data["intervene_group"] == "high"][metric].mean() - data[data["intervene_group"] == "low"][metric].mean()
    p.append(text(x + w - 2, y + 18, f"Δ={diff:.2f}", 11, 700, color, "end"))
    return "\n".join(p)


def figure1_cell_method() -> None:
    p: list[str] = []
    p.append(text(42, 38, "INTERVENE: IHA-inspired attention for physics-guided cancer vulnerability discovery", 20, 700))
    p.append(text(42, 62, "Method schematic with biological module tokens, interleaved pseudo-head mixing, and PRISM validation", 12, 400, MUTED))
    p.append(panel("A", "IHA mechanism adapted from language tokens to module tokens", 42, 105))
    p.append(rect(70, 130, 250, 150, PALE, GRID, rx=8))
    p.append(text(92, 160, "Language IHA", 14, 700))
    for i, tok in enumerate(["The", "cell", "survived", "because", "it"]):
        p.append(rect(95 + i * 42, 190 + (i % 2) * 22, 36, 22, "#ffffff", BLUE, rx=10, sw=1))
        p.append(text(113 + i * 42, 205 + (i % 2) * 22, tok, 8, 500, FG, "middle"))
    p.append(text(95, 255, "tokens = words", 11, 500, MUTED))
    p.append(line(335, 205, 405, 205, FG, 1.8, arrow=True))
    p.append(rect(420, 130, 280, 150, PALE, GRID, rx=8))
    p.append(text(442, 160, "INTERVENE", 14, 700))
    for i, tok in enumerate(["SREBP", "mito", "oxidative", "proteostasis", "IAP"]):
        p.append(rect(445 + i * 47, 190 + (i % 2) * 22, 42, 22, "#ffffff", TEAL, rx=10, sw=1))
        p.append(text(466 + i * 47, 205 + (i % 2) * 22, tok, 8, 500, FG, "middle"))
    p.append(text(445, 255, "tokens = biological modules", 11, 500, MUTED))
    p.append(line(715, 205, 785, 205, FG, 1.8, arrow=True))
    p.append(rect(800, 130, 320, 150, PALE, GRID, rx=8))
    p.append(text(822, 160, "IHA-style pseudo-head mixing", 14, 700))
    p.append(head_mixing_icon(850, 185))

    p.append(panel("B", "Cell-level biological interpretation", 42, 340))
    p.append(cell_illustration(260, 515, 1.1))
    p.append(text(116, 420, "NSCLC cell", 13, 700, TEAL))
    p.append(text(332, 438, "lipid droplets", 11, 500, CYTO))
    p.append(text(357, 505, "mitochondria", 11, 500, ORANGE))
    p.append(text(415, 600, "oxidative stress", 11, 500, RED))
    p.append(rect(555, 395, 245, 170, "#ffffff", "#b7c5d7", rx=10))
    p.append(text(580, 430, "Physics-guided module features", 14, 700))
    for i, item in enumerate(["CRISPR dependency", "expression", "copy number", "mutation context", "thermodynamic residual"]):
        p.append(circle(585, 462 + i * 20, 4, BLUE, BLUE))
        p.append(text(598, 466 + i * 20, item, 11, 400))
    p.append(line(493, 515, 555, 480, FG, 1.6, arrow=True))
    p.append(line(800, 480, 875, 480, FG, 1.6, arrow=True))
    p.append(rect(880, 395, 245, 170, "#ffffff", "#b8d8c0", rx=10))
    p.append(text(905, 430, "Discovered state", 14, 700))
    p.append(multiline(905, 465, "SREBP / mitochondrial / oxidative stress-survival vulnerability", 27, 13, 700, GREEN))
    p.append(text(905, 540, "validated by PRISM", 12, 700, FG))
    (FIGURES / "intervene_fig1_cell_method.svg").write_text(svg(1200, 700, "\n".join(p)), encoding="utf-8")


def figure2_distributions() -> None:
    df = quantile_groups(pd.read_csv(RESULTS / "ibpa_v2_discovery_cell_scores.csv"), "ibpa_v2_state_score")
    p: list[str] = []
    p.append(text(42, 38, "INTERVENE state validation across NSCLC cell lines", 20, 700))
    p.append(text(42, 62, "Each point is one PRISM-linked NSCLC model; boxes show low vs high INTERVENE state tertiles", 12, 400, MUTED))
    p.append(panel("A", "Specific stress-survival signal", 42, 105))
    p.append(beeswarm_panel(df, "target_cytotoxic_residual_stress_survival", "Cytotoxic-residual stress-survival", 70, 145, 330, 260, GREEN, "residual score"))
    p.append(panel("B", "Raw stress-survival sensitivity", 450, 105))
    p.append(beeswarm_panel(df, "target_stress_survival_sensitivity", "PRISM stress-survival sensitivity", 478, 145, 330, 260, BLUE, "drug sensitivity"))
    p.append(panel("C", "Controls are not the main explanation", 850, 105))
    p.append(beeswarm_panel(df, "generic_cytotoxic_sensitivity", "Generic cytotoxic sensitivity", 878, 145, 280, 260, RED, "drug sensitivity"))
    p.append(panel("D", "Direct lipid drugs remain secondary", 42, 490))
    p.append(beeswarm_panel(df, "direct_lipid_sensitivity", "Direct lipid-axis sensitivity", 70, 530, 330, 230, ORANGE, "drug sensitivity"))
    p.append(panel("E", "State distribution", 450, 490))
    x, y, w, h = 500, 530, 580, 225
    vals = df["ibpa_v2_state_score"].to_numpy()
    bins = np.linspace(vals.min(), vals.max(), 16)
    counts, edges = np.histogram(vals, bins=bins)
    maxc = max(counts)
    p.append(line(x, y + h, x + w, y + h, GRID, 1))
    for i, c in enumerate(counts):
        bx = x + i * w / len(counts)
        bw = w / len(counts) - 3
        bh = c / maxc * h
        p.append(rect(bx, y + h - bh, bw, bh, TEAL if edges[i] >= df["ibpa_v2_state_score"].quantile(2/3) else "#aab6c2", "none", rx=1, sw=0))
    p.append(text(x, y + h + 20, "low", 10, 500, MUTED))
    p.append(text(x + w - 20, y + h + 20, "high", 10, 500, MUTED))
    p.append(multiline(x, y - 2, "High INTERVENE state is enriched for stress-survival vulnerability after removing generic cytotoxic contribution.", 70, 13, 500, FG))
    (FIGURES / "intervene_fig2_validation_distributions.svg").write_text(svg(1200, 820, "\n".join(p)), encoding="utf-8")


def small_bar_panel(x: float, y: float, w: float, h: float, labels: list[str], values: list[float], title: str) -> str:
    p = [text(x, y - 18, title, 13, 700)]
    ymin, ymax = min(-0.1, min(values) * 1.15), max(values) * 1.15
    zero = y + h - (0 - ymin) / (ymax - ymin) * h
    p.append(line(x, zero, x + w, zero, GRID, 1))
    bw = w / len(values) * 0.62
    for i, (lab, val) in enumerate(zip(labels, values)):
        cx = x + (i + 0.5) * w / len(values)
        top = y + h - (val - ymin) / (ymax - ymin) * h
        col = GREEN if val > 0.5 else RED if val < 0.05 else TEAL
        p.append(rect(cx - bw/2, min(zero, top), bw, abs(zero-top), col, col, rx=2, sw=0))
        p.append(text(cx, top - 7 if val >= 0 else top + 16, f"{val:.2f}", 10, 700, FG, "middle"))
        for k, l in enumerate(wrap(lab, 10)):
            p.append(text(cx, y + h + 17 + k * 12, l, 9, 400, FG, "middle"))
    return "\n".join(p)


def figure3_ablation_attention() -> None:
    ab = pd.read_csv(RESULTS / "ibpa_v2_ablation_summary.csv").set_index("ablation")
    edges = pd.read_csv(RESULTS / "ibpa_v2_discovery_attention_edges.csv").head(10)
    p: list[str] = []
    p.append(text(42, 38, "Ablation and learned module interactions", 20, 700))
    p.append(text(42, 62, "Weak physics guidance is necessary; attention converges on mitochondrial translation", 12, 400, MUTED))
    p.append(panel("A", "Ablation of physics guidance", 42, 105))
    order = [
        ("no_prior", "No prior"),
        ("prior_weight_0.05", "Weak prior"),
        ("prior_weight_0.22_current", "Current"),
        ("prior_weight_0.50", "Strong"),
        ("remove_srebp_from_prior", "No SREBP"),
        ("remove_mito_from_prior", "No mito"),
        ("remove_oxidative_from_prior", "No oxidative"),
    ]
    vals = [float(ab.loc[k, "cytotoxic_residual_stress_survival_score_effect"]) for k, _ in order]
    p.append(small_bar_panel(78, 150, 560, 250, [v for _, v in order], vals, "Effect on cytotoxic-residual stress-survival"))
    p.append(panel("B", "Attention network in biological cell context", 690, 105))
    p.append(cell_illustration(880, 335, 0.92, stress=True))
    pos = {
        "srebp_lipid": (740, 265),
        "proteostasis": (745, 185),
        "apoptosis_iap": (745, 430),
        "akt_mtor_growth": (862, 185),
        "mito_translation": (915, 335),
        "oxidative_stress": (1045, 335),
    }
    labels = {
        "srebp_lipid": "SREBP/lipid",
        "proteostasis": "proteostasis",
        "apoptosis_iap": "apoptosis/IAP",
        "akt_mtor_growth": "AKT/mTOR",
        "mito_translation": "mitochondrial\ntranslation",
        "oxidative_stress": "oxidative\nstress",
    }
    col = {
        "srebp_lipid": BLUE,
        "proteostasis": PURPLE,
        "apoptosis_iap": ORANGE,
        "akt_mtor_growth": TEAL,
        "mito_translation": GREEN,
        "oxidative_stress": RED,
    }
    used = set()
    for r in edges.itertuples(index=False):
        src, dst = r.source_token, r.target_token
        if src not in pos or dst not in pos or (src, dst) in used:
            continue
        used.add((src, dst))
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        width = min(7, 1.2 + 12 * float(r.attention_enrichment_vs_uniform))
        p.append(line(x1, y1, x2, y2, "#465a69", width, arrow=True))
    for node, (x, y) in pos.items():
        p.append(rect(x - 52, y - 22, 104, 44, "white", col[node], rx=8, sw=1.8))
        for i, lab in enumerate(labels[node].split("\n")):
            p.append(text(x, y - 3 + i * 13, lab, 10, 700, FG, "middle"))
    p.append(panel("C", "Conclusion for manuscript", 42, 500))
    p.append(rect(80, 545, 1030, 120, "#ffffff", "#b8d8c0", rx=10, sw=1.4))
    p.append(multiline(105, 582, "INTERVENE supports an SREBP/mitochondrial/oxidative stress-survival state. No-prior attention fails, weak physics-guided interleaving succeeds, and the signal is not explained by direct lipid drugs or generic cytotoxicity.", 136, 15, 500, FG))
    (FIGURES / "intervene_fig3_ablation_attention_cell.svg").write_text(svg(1200, 720, "\n".join(p)), encoding="utf-8")


def paper_tables() -> None:
    validation = pd.read_csv(RESULTS / "ibpa_v2_discovery_validation.csv")
    keep = [
        "PRISM stress-survival sensitivity",
        "cytotoxic-residual stress-survival score",
        "stress-survival minus cytotoxic score",
        "PRISM direct lipid sensitivity",
        "PRISM generic cytotoxic sensitivity",
        "KEAP1/NFE2L2 oxidative context",
        "KEAP1/NFE2L2 high-impact enrichment",
    ]
    validation[validation["test"].isin(keep)][
        ["test", "n_high", "n_low", "high_mean", "low_mean", "high_minus_low", "effect_size", "perm_p_two_sided"]
    ].to_csv(RESULTS / "intervene_table1_validation_summary.csv", index=False)
    pd.read_csv(RESULTS / "ibpa_v2_ablation_summary.csv")[
        [
            "ablation",
            "reconstruction_mse",
            "state_prior_corr",
            "cytotoxic_residual_stress_survival_score_effect",
            "cytotoxic_residual_stress_survival_score_p",
            "prism_generic_cytotoxic_sensitivity_effect",
            "prism_direct_lipid_sensitivity_effect",
        ]
    ].to_csv(RESULTS / "intervene_table2_ablation_summary.csv", index=False)
    pd.read_csv(RESULTS / "ibpa_v2_discovery_leads.csv")[
        ["priority", "lead", "paper_role", "evidence"]
    ].to_csv(RESULTS / "intervene_table3_discovery_leads.csv", index=False)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    paper_tables()
    figure1_cell_method()
    figure2_distributions()
    figure3_ablation_attention()
    print(f"Wrote {FIGURES / 'intervene_fig1_cell_method.svg'}")
    print(f"Wrote {FIGURES / 'intervene_fig2_validation_distributions.svg'}")
    print(f"Wrote {FIGURES / 'intervene_fig3_ablation_attention_cell.svg'}")
    print(f"Wrote {RESULTS / 'intervene_table1_validation_summary.csv'}")
    print(f"Wrote {RESULTS / 'intervene_table2_ablation_summary.csv'}")
    print(f"Wrote {RESULTS / 'intervene_table3_discovery_leads.csv'}")


if __name__ == "__main__":
    main()

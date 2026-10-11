from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


LABELS = {
    "raw_thermodynamic_transfer": "Raw\ntransfer",
    "graph_sparse_v3_soft": "V3 soft",
    "graph_sparse_v3_balanced": "V3\nbalanced",
    "graph_sparse_v3_strong_edges": "V3 strong\nedges",
}

COLORS = {
    "raw_thermodynamic_transfer": "#4C6A92",
    "graph_sparse_v3_soft": "#7C4D8B",
    "graph_sparse_v3_balanced": "#5E8C61",
    "graph_sparse_v3_strong_edges": "#B65F5F",
}


def text(x: float, y: float, value: str, size: int = 13, weight: str = "400", anchor: str = "middle") -> str:
    out = []
    for i, line in enumerate(str(value).split("\n")):
        out.append(
            f'<text x="{x:.1f}" y="{y + i * (size + 3):.1f}" text-anchor="{anchor}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="#202020">{line}</text>'
        )
    return "\n".join(out)


def panel_bars(rows: pd.DataFrame, metric: str, x0: int, y0: int, w: int, h: int, ymin: float, ymax: float, title: str) -> list[str]:
    parts = [text(x0 + w / 2, y0 - 20, title, 17, "700")]

    def yp(v: float) -> float:
        return y0 + (ymax - v) / (ymax - ymin) * h

    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        val = ymin + frac * (ymax - ymin)
        y = yp(val)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" stroke="#e8e8e8"/>')
        parts.append(text(x0 - 10, y + 4, f"{val:.2f}", 11, anchor="end"))
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#222" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#222" stroke-width="1.2"/>')

    step = w / len(rows)
    bar_w = step * 0.52
    for i, row in enumerate(rows.itertuples(index=False)):
        model = row.model
        value = getattr(row, metric)
        x = x0 + i * step + step / 2
        y = yp(value)
        parts.append(
            f'<rect x="{x - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{y0+h-y:.1f}" '
            f'fill="{COLORS.get(model, "#777")}" opacity="0.9"/>'
        )
        parts.append(text(x, y - 8, f"{value:.3f}", 12, "700"))
        parts.append(text(x, y0 + h + 25, LABELS.get(model, model), 12, "700"))
    return parts


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS / "relapse_intervene_graph_sparse_v3_model_comparison.csv")
    raw = results[results["model"].eq("raw_thermodynamic_transfer")].iloc[0]
    learned = results[results["seed"].astype(str) != "na"]
    agg = (
        learned.groupby("model")
        .agg(c_index=("c_index", "mean"), edge_separation=("edge_separation", "mean"))
        .reset_index()
    )
    rows = pd.concat(
        [
            pd.DataFrame(
                [{"model": "raw_thermodynamic_transfer", "c_index": raw["c_index"], "edge_separation": 0.0}]
            ),
            agg,
        ],
        ignore_index=True,
    )
    order = ["raw_thermodynamic_transfer", "graph_sparse_v3_soft", "graph_sparse_v3_balanced", "graph_sparse_v3_strong_edges"]
    rows["order"] = rows["model"].map({m: i for i, m in enumerate(order)})
    rows = rows.sort_values("order")

    width, height = 1100, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(width / 2, 42, "GraphSparse V3 strengthens both relapse prediction and attention-edge separation", 22, "700"),
        text(width / 2, 70, "GSE31210 NSCLC, five-fold out-of-fold risk prediction across five seeds", 13),
    ]
    parts.extend(panel_bars(rows, "c_index", 90, 135, 430, 330, 0.62, 0.70, "Relapse prediction"))
    parts.extend(panel_bars(rows, "edge_separation", 625, 135, 390, 330, 0.0, 0.32, "Biology-edge separation"))
    note = "V3 turns the earlier near-uniform attention into a sparse edge map while keeping C-index above V2."
    parts.append(f'<rect x="95" y="540" width="910" height="42" rx="6" fill="#f5f3f7" stroke="#ddd4e5"/>')
    parts.append(text(width / 2, 566, note, 14))
    parts.append("</svg>")
    path = FIGURES / "relapse_intervene_graph_sparse_v3_prediction_edges.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

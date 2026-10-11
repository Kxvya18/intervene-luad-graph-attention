from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def km_curve(frame: pd.DataFrame, group_label: str) -> pd.DataFrame:
    data = frame[["relapse_free_months", "relapse_event"]].dropna().copy()
    data = data[data["relapse_free_months"] > 0].sort_values("relapse_free_months")
    n_at_risk = len(data)
    survival = 1.0
    rows = [{"group": group_label, "months": 0.0, "relapse_free_survival": 1.0, "at_risk": n_at_risk}]
    for t, sub in data.groupby("relapse_free_months"):
        events = int(sub["relapse_event"].sum())
        censored = int(len(sub) - events)
        if n_at_risk > 0 and events > 0:
            survival *= 1.0 - events / n_at_risk
            rows.append(
                {
                    "group": group_label,
                    "months": float(t),
                    "relapse_free_survival": survival,
                    "at_risk": n_at_risk,
                }
            )
        n_at_risk -= events + censored
    return pd.DataFrame(rows)


def path_from_curve(curve: pd.DataFrame, x0: int, y0: int, w: int, h: int, max_months: float) -> str:
    points = []
    for row in curve.itertuples(index=False):
        x = x0 + row.months / max_months * w
        y = y0 + (1.0 - row.relapse_free_survival) * h
        points.append((x, y))
    if not points:
        return ""
    parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    last_x, last_y = points[0]
    for x, y in points[1:]:
        parts.append(f"L {x:.1f} {last_y:.1f}")
        parts.append(f"L {x:.1f} {y:.1f}")
        last_x, last_y = x, y
    return " ".join(parts)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    patient = pd.read_csv(PROCESSED / "gse31210_patient_module_scores.csv")
    prognosis = patient[patient["is_primary_tumor"] & patient["prognosis_included"]].copy()
    high_cut = prognosis["raw_relapse_intervene_state"].quantile(2 / 3)
    low_cut = prognosis["raw_relapse_intervene_state"].quantile(1 / 3)
    high = prognosis[prognosis["raw_relapse_intervene_state"] >= high_cut].copy()
    low = prognosis[prognosis["raw_relapse_intervene_state"] <= low_cut].copy()
    high["state_group"] = "high"
    low["state_group"] = "low"
    grouped = pd.concat([high, low], ignore_index=True)
    grouped.to_csv(RESULTS / "relapse_gse31210_high_low_patients.csv", index=False)

    curves = pd.concat([km_curve(high, "high"), km_curve(low, "low")], ignore_index=True)
    curves.to_csv(RESULTS / "relapse_gse31210_km_curve.csv", index=False)

    summary = pd.read_csv(RESULTS / "relapse_gse31210_initial_survival_test.csv").iloc[0]
    max_months = float(grouped["relapse_free_months"].max())
    width, height = 880, 580
    x0, y0, plot_w, plot_h = 95, 95, 680, 360
    high_path = path_from_curve(curves[curves["group"] == "high"], x0, y0, plot_w, plot_h, max_months)
    low_path = path_from_curve(curves[curves["group"] == "low"], x0, y0, plot_w, plot_h, max_months)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:23px;font-weight:700}.small{font-size:12px;fill:#5b6573}.label{font-size:13px}.axis{stroke:#c7cdd6;stroke-width:1}</style>"
    )
    svg.append('<text x="36" y="42" class="title">GSE31210 Relapse Transfer Test</text>')
    svg.append('<text x="36" y="66" class="small">Raw Relapse-INTERVENE state from transferred stress-survival modules; high vs low tertile</text>')
    svg.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" class="axis"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" class="axis"/>')
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 + frac * plot_h
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}" class="axis" opacity="0.35"/>')
        svg.append(f'<text x="48" y="{y + 4:.1f}" class="small">{1-frac:.2f}</text>')
    for month in [0, 24, 48, 72, 96, 120]:
        if month > max_months:
            continue
        x = x0 + month / max_months * plot_w
        svg.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + plot_h}" class="axis" opacity="0.25"/>')
        svg.append(f'<text x="{x - 10:.1f}" y="{y0 + plot_h + 24}" class="small">{month}</text>')
    svg.append(f'<path d="{low_path}" fill="none" stroke="#2e6f9e" stroke-width="3"/>')
    svg.append(f'<path d="{high_path}" fill="none" stroke="#b2382f" stroke-width="3"/>')
    svg.append('<text x="795" y="120" class="label" fill="#b2382f">High state</text>')
    svg.append('<rect x="775" y="111" width="14" height="4" fill="#b2382f"/>')
    svg.append('<text x="795" y="145" class="label" fill="#2e6f9e">Low state</text>')
    svg.append('<rect x="775" y="136" width="14" height="4" fill="#2e6f9e"/>')
    svg.append(f'<text x="95" y="505" class="label">High: {int(summary.high_events)}/{int(summary.high_n)} relapsed; Low: {int(summary.low_events)}/{int(summary.low_n)} relapsed</text>')
    svg.append(f'<text x="95" y="530" class="label">Approx HR={float(summary.hazard_ratio_approx):.2f}; log-rank p={float(summary.logrank_p):.4g}; event Fisher p={float(summary.event_fisher_p):.4g}</text>')
    svg.append('<text x="95" y="555" class="small">Exploratory transfer test only; next step is a survival-head Relapse-INTERVENE model with baseline comparisons.</text>')
    svg.append('</svg>')
    (FIGURES / "relapse_gse31210_km_transfer_test.svg").write_text("\n".join(svg), encoding="utf-8")
    print(f"Wrote {RESULTS / 'relapse_gse31210_high_low_patients.csv'}")
    print(f"Wrote {RESULTS / 'relapse_gse31210_km_curve.csv'}")
    print(f"Wrote {FIGURES / 'relapse_gse31210_km_transfer_test.svg'}")


if __name__ == "__main__":
    main()

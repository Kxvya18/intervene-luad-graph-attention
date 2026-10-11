from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from interleaved_biophysics_attention_full import IBPALayer, PAPER_TOKENS, set_seed, zscore
from prepare_gse31210_relapse_intervene import logrank_high_low


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"
RESULTS = ROOT / "results"

SEEDS = [7, 17, 29, 43, 61]
N_FOLDS = 5
EPOCHS = 700
PATIENCE = 90
LR = 1.5e-3
WEIGHT_DECAY = 2e-3
HIDDEN = 24

MODULES = [
    "srebp_lipid",
    "mito_translation",
    "oxidative_stress",
    "proteostasis",
    "apoptosis_iap",
    "akt_mtor_growth",
]

MODULE_FEATURES = ["expr_mean", "expr_median", "expr_dispersion"]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    attention_mode: str
    prior_weight: float
    use_attention: bool
    description: str


MODEL_SPECS = [
    ModelSpec(
        "linear_cox_modules",
        "linear",
        0.0,
        False,
        "Cox-like linear survival baseline using module expression tokens",
    ),
    ModelSpec(
        "no_prior_interleaved_attention",
        "ibpa",
        0.0,
        True,
        "IHA-inspired interleaved attention trained only on relapse survival",
    ),
    ModelSpec(
        "normal_attention_mha",
        "mha",
        0.0,
        True,
        "standard multi-head attention trained only on relapse survival",
    ),
    ModelSpec(
        "normal_attention_mha_with_prior",
        "mha",
        0.15,
        True,
        "standard multi-head attention with weak DepMap/PRISM biology prior",
    ),
    ModelSpec(
        "relapse_intervene",
        "ibpa",
        0.15,
        True,
        "INTERVENE survival model: interleaved pseudo-head attention plus weak biology prior",
    ),
    ModelSpec(
        "relapse_intervene_prior_0_03",
        "ibpa",
        0.03,
        True,
        "INTERVENE survival model with very weak biology-prior regularization",
    ),
    ModelSpec(
        "relapse_intervene_prior_0_08",
        "ibpa",
        0.08,
        True,
        "INTERVENE survival model with weak biology-prior regularization",
    ),
    ModelSpec(
        "relapse_intervene_prior_0_30",
        "ibpa",
        0.30,
        True,
        "INTERVENE survival model with stronger biology-prior regularization",
    ),
]


class RelapseAttentionModel(nn.Module):
    def __init__(self, n_features: int, mode: str):
        super().__init__()
        self.layer = IBPALayer(n_features, heads=4, pseudoheads=2 if mode == "ibpa" else 1, head_dim=4, mode=mode)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, attn = self.layer(x)
        return self.risk_head(pooled).squeeze(-1), attn


class LinearCoxModel(nn.Module):
    def __init__(self, n_tokens: int, n_features: int):
        super().__init__()
        self.risk = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(n_tokens * n_features),
            nn.Linear(n_tokens * n_features, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.risk(x).squeeze(-1), torch.empty(0)


def load_patient_tensor() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    patient = pd.read_csv(PROCESSED / "gse31210_patient_module_scores.csv")
    frame = patient[patient["is_primary_tumor"] & patient["prognosis_included"]].copy()
    frame = frame.dropna(subset=["relapse_event", "relapse_free_months", "raw_relapse_intervene_state"]).copy()
    frame = frame[frame["relapse_free_months"] > 0].reset_index(drop=True)

    feature_names = []
    x = np.zeros((len(frame), len(MODULES), len(MODULE_FEATURES)), dtype=np.float32)
    for ti, module in enumerate(MODULES):
        for fi, feature in enumerate(MODULE_FEATURES):
            col = f"{module}_{feature}"
            feature_names.append(col) if ti == 0 else None
            values = pd.to_numeric(frame[col], errors="coerce")
            values = values.fillna(values.median() if values.notna().any() else 0.0)
            x[:, ti, fi] = zscore(values).to_numpy(dtype=np.float32)

    time = frame["relapse_free_months"].to_numpy(dtype=np.float32)
    event = frame["relapse_event"].to_numpy(dtype=np.float32)
    prior = zscore(frame["raw_relapse_intervene_state"]).to_numpy(dtype=np.float32)
    return frame, np.nan_to_num(x), time, event, prior, feature_names


def folds(n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.arange(n))
    return [order[i::N_FOLDS] for i in range(N_FOLDS)]


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss_terms = risk - log_cumsum
    event_count = torch.clamp(event.sum(), min=1.0)
    return -torch.sum(loss_terms * event) / event_count


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1
                concordant += 1 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0
            elif event[j] == 1 and time[j] < time[i]:
                comparable += 1
                concordant += 1 if risk[j] > risk[i] else 0.5 if risk[i] == risk[j] else 0
    return float(concordant / comparable) if comparable else np.nan


def train_fold(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    prior: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: ModelSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, nn.Module]:
    set_seed(seed)
    inner = train_idx.copy()
    val_size = max(12, int(round(len(inner) * 0.20)))
    val_idx = inner[:val_size]
    fit_idx = inner[val_size:]

    if spec.use_attention:
        model: nn.Module = RelapseAttentionModel(x.shape[-1], spec.attention_mode)
    else:
        model = LinearCoxModel(x.shape[1], x.shape[2])

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x_fit = torch.tensor(x[fit_idx], dtype=torch.float32)
    t_fit = torch.tensor(time[fit_idx], dtype=torch.float32)
    e_fit = torch.tensor(event[fit_idx], dtype=torch.float32)
    p_fit = torch.tensor(prior[fit_idx], dtype=torch.float32)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32)
    t_val = torch.tensor(time[val_idx], dtype=torch.float32)
    e_val = torch.tensor(event[val_idx], dtype=torch.float32)
    p_val = torch.tensor(prior[val_idx], dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        risk, _ = model(x_fit)
        risk_z = (risk - risk.mean()) / (risk.std(unbiased=False) + 1e-6)
        loss = cox_loss(risk, t_fit, e_fit)
        if spec.prior_weight > 0:
            loss = loss + spec.prior_weight * torch.mean((risk_z - p_fit) ** 2)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_risk, _ = model(x_val)
            val_z = (val_risk - val_risk.mean()) / (val_risk.std(unbiased=False) + 1e-6)
            val_loss = cox_loss(val_risk, t_val, e_val)
            if spec.prior_weight > 0:
                val_loss = val_loss + spec.prior_weight * torch.mean((val_z - p_val) ** 2)
            val_value = float(val_loss.item())
        if val_value < best_val - 1e-5:
            best_val = val_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        risk, attn = model(torch.tensor(x[test_idx], dtype=torch.float32))
    return risk.numpy(), attn.numpy() if spec.use_attention else None, model


def evaluate_score(
    frame: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
    model_name: str,
    description: str,
    seed: int | None = None,
) -> dict[str, object]:
    tmp = frame[["geo_accession", "relapse_free_months", "relapse_event"]].copy()
    tmp["risk_score"] = score
    lr = logrank_high_low(tmp, "risk_score", "relapse_free_months", "relapse_event")
    high_cut = np.quantile(score, 2 / 3)
    low_cut = np.quantile(score, 1 / 3)
    high = score >= high_cut
    low = score <= low_cut
    return {
        "model": model_name,
        "seed": seed if seed is not None else "na",
        "description": description,
        "n": len(frame),
        "events": int(event.sum()),
        "c_index": concordance_index(time, event, score),
        "risk_prior_corr": float(pd.Series(score).corr(pd.Series(frame["raw_relapse_intervene_state"]))),
        "high_n": int(high.sum()),
        "low_n": int(low.sum()),
        "high_events": int(event[high].sum()),
        "low_events": int(event[low].sum()),
        "hazard_ratio_approx": lr["hazard_ratio_approx"],
        "event_fisher_p": lr["event_fisher_p"],
        "logrank_chi2": lr["logrank_chi2"],
        "logrank_p": lr["logrank_p"],
    }


def cv_model(
    frame: pd.DataFrame,
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    prior: np.ndarray,
    spec: ModelSpec,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]], list[nn.Module], list[np.ndarray]]:
    pred = np.full(len(frame), np.nan, dtype=float)
    fold_rows = []
    models = []
    attn_blocks = []
    for fold_id, test_idx in enumerate(folds(len(frame), seed)):
        train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
        risk, attn, model = train_fold(x, time, event, prior, train_idx, test_idx, spec, seed + fold_id * 101)
        pred[test_idx] = risk
        models.append(model)
        if attn is not None and attn.size:
            attn_blocks.append(attn)
        fold_rows.append(
            evaluate_score(
                frame.iloc[test_idx].reset_index(drop=True),
                time[test_idx],
                event[test_idx],
                risk,
                f"{spec.model}_fold{fold_id + 1}",
                spec.description,
                seed,
            )
        )
    return pred, fold_rows, models, attn_blocks


def attention_edges(attention: np.ndarray, model: str, seed: int) -> pd.DataFrame:
    mean_attn = attention.mean(axis=0)
    uniform = 1.0 / len(MODULES)
    rows = []
    for h in range(mean_attn.shape[0]):
        for p in range(mean_attn.shape[1]):
            for source_i, source in enumerate(MODULES):
                for target_i, target in enumerate(MODULES):
                    value = float(mean_attn[h, p, source_i, target_i])
                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "head": h,
                            "pseudohead": p,
                            "source_module": source,
                            "target_module": target,
                            "mean_attention": value,
                            "attention_enrichment_vs_uniform": value - uniform,
                        }
                    )
    return pd.DataFrame(rows)


def write_summary(results: pd.DataFrame, stability: pd.DataFrame) -> None:
    summary_path = RESULTS / "relapse_intervene_survival_head_summary.txt"
    aggregate = (
        results[results["seed"] != "na"]
        .groupby("model")
        .agg(
            c_index_mean=("c_index", "mean"),
            c_index_sd=("c_index", "std"),
            logrank_p_median=("logrank_p", "median"),
            hazard_ratio_mean=("hazard_ratio_approx", "mean"),
            risk_prior_corr_mean=("risk_prior_corr", "mean"),
            high_events_mean=("high_events", "mean"),
            low_events_mean=("low_events", "mean"),
        )
        .reset_index()
        .sort_values(["c_index_mean", "hazard_ratio_mean"], ascending=False)
    )
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Relapse-INTERVENE survival head comparison\n")
        f.write("==========================================\n\n")
        f.write("Dataset: GSE31210 prognosis-included primary NSCLC tumors.\n")
        f.write("Evaluation: five-fold out-of-fold risk scores repeated over five seeds.\n")
        f.write("Endpoint: relapse-free months with relapse event indicator.\n\n")
        f.write("Aggregate model comparison\n")
        f.write("--------------------------\n")
        f.write(aggregate.to_string(index=False))
        f.write("\n\n")
        f.write("Stable top interleaved attention edges\n")
        f.write("--------------------------------------\n")
        if stability.empty:
            f.write("No stability table generated.\n")
        else:
            f.write(stability.head(16).to_string(index=False))
            f.write("\n")
        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write(
            "A good Relapse-INTERVENE result is not only higher C-index. It should also preserve the validated "
            "biology: high-risk patients should be enriched for relapse, and the interleaved attention edges "
            "should repeatedly connect lipid/SREBP, mitochondrial translation, oxidative stress, proteostasis, "
            "apoptosis/IAP, and AKT/mTOR modules.\n"
        )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame, x, time, event, prior, _ = load_patient_tensor()

    all_rows = [
        evaluate_score(
            frame,
            time,
            event,
            prior,
            "raw_thermodynamic_transfer",
            "No learned relapse model; direct transferred INTERVENE biology state",
        )
    ]
    prediction_cols = frame[["geo_accession", "relapse_free_months", "relapse_event", "raw_relapse_intervene_state"]].copy()
    edge_tables = []

    for seed in SEEDS:
        for spec in MODEL_SPECS:
            pred, fold_rows, _, attn_blocks = cv_model(frame, x, time, event, prior, spec, seed)
            all_rows.append(evaluate_score(frame, time, event, pred, spec.model, spec.description, seed))
            prediction_cols[f"{spec.model}_seed{seed}_risk"] = pred
            if spec.use_attention and attn_blocks:
                edge_tables.append(attention_edges(np.concatenate(attn_blocks, axis=0), spec.model, seed))

    results = pd.DataFrame(all_rows)
    edges = pd.concat(edge_tables, ignore_index=True) if edge_tables else pd.DataFrame()
    if not edges.empty:
        top_edges = edges[edges["model"].eq("relapse_intervene")].copy()
        top_edges["edge"] = top_edges["source_module"] + " -> " + top_edges["target_module"]
        stability = (
            top_edges.sort_values("attention_enrichment_vs_uniform", ascending=False)
            .groupby("edge")
            .agg(
                seeds_observed=("seed", "nunique"),
                mean_enrichment=("attention_enrichment_vs_uniform", "mean"),
                mean_attention=("mean_attention", "mean"),
            )
            .reset_index()
            .sort_values(["seeds_observed", "mean_enrichment"], ascending=False)
        )
    else:
        stability = pd.DataFrame()

    results.to_csv(RESULTS / "relapse_intervene_survival_model_comparison.csv", index=False)
    prediction_cols.to_csv(RESULTS / "relapse_intervene_survival_oof_risk_scores.csv", index=False)
    edges.to_csv(RESULTS / "relapse_intervene_survival_attention_edges.csv", index=False)
    stability.to_csv(RESULTS / "relapse_intervene_survival_edge_stability.csv", index=False)
    write_summary(results, stability)

    print(f"Wrote {RESULTS / 'relapse_intervene_survival_model_comparison.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_survival_oof_risk_scores.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_survival_attention_edges.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_survival_edge_stability.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_survival_head_summary.txt'}")


if __name__ == "__main__":
    main()

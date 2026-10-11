from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from interleaved_biophysics_attention_full import set_seed, zscore
from prepare_gse31210_relapse_intervene import logrank_high_low
from run_relapse_intervene_survival_head import cox_loss, concordance_index, folds
from run_relapse_intervene_module_v2 import MODULES, MODULE_FEATURES, BIOLOGY_EDGES, load_data


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEEDS = [7, 17, 29, 43, 61]
EPOCHS = 520
PATIENCE = 65
LR = 1.4e-3
WEIGHT_DECAY = 2e-3
HEADS = 4
PSEUDOHEADS = 2
HEAD_DIM = 4
HIDDEN = 24
TOP_K = 2


@dataclass(frozen=True)
class GraphSparseSpec:
    model: str
    graph_bias_strength: float
    entropy_weight: float
    contrastive_weight: float
    prior_weight: float
    multitask_weight: float
    description: str


SPECS = [
    GraphSparseSpec(
        "graph_sparse_v3_soft",
        graph_bias_strength=0.35,
        entropy_weight=0.02,
        contrastive_weight=0.08,
        prior_weight=0.03,
        multitask_weight=0.08,
        description="Graph-biased sparse INTERVENE with mild contrastive edge sharpening",
    ),
    GraphSparseSpec(
        "graph_sparse_v3_balanced",
        graph_bias_strength=0.65,
        entropy_weight=0.04,
        contrastive_weight=0.18,
        prior_weight=0.03,
        multitask_weight=0.08,
        description="GraphSparse INTERVENE V3 with balanced edge sharpening",
    ),
    GraphSparseSpec(
        "graph_sparse_v3_strong_edges",
        graph_bias_strength=1.05,
        entropy_weight=0.05,
        contrastive_weight=0.35,
        prior_weight=0.03,
        multitask_weight=0.08,
        description="GraphSparse INTERVENE V3 with stronger biology-edge sharpening",
    ),
]


def edge_masks() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    n = len(MODULES)
    positive = np.zeros((n, n), dtype=bool)
    for i, source in enumerate(MODULES):
        for j, target in enumerate(MODULES):
            if source == target:
                positive[i, j] = True
            if (source, target) in BIOLOGY_EDGES or (target, source) in BIOLOGY_EDGES:
                positive[i, j] = True
    negative = ~positive
    graph = np.where(positive, 1.0, -0.35).astype(np.float32)
    return torch.tensor(positive), torch.tensor(negative), graph


class GraphSparseLayer(nn.Module):
    def __init__(self, n_features: int, graph_bias: np.ndarray, graph_bias_strength: float):
        super().__init__()
        self.heads = HEADS
        self.pseudoheads = PSEUDOHEADS
        self.head_dim = HEAD_DIM
        self.q = nn.Linear(n_features, HEADS * HEAD_DIM, bias=False)
        self.k = nn.Linear(n_features, HEADS * HEAD_DIM, bias=False)
        self.v = nn.Linear(n_features, HEADS * HEAD_DIM, bias=False)
        self.alpha_q = nn.Parameter(self.initial_alpha())
        self.alpha_k = nn.Parameter(self.initial_alpha())
        self.alpha_v = nn.Parameter(self.initial_alpha())
        self.collapse_logits = nn.Parameter(torch.zeros(HEADS, PSEUDOHEADS))
        self.temperature_logit = nn.Parameter(torch.tensor(-0.15))
        self.register_buffer("graph_bias", torch.tensor(graph_bias * graph_bias_strength, dtype=torch.float32))

    def initial_alpha(self) -> torch.Tensor:
        alpha = torch.zeros(HEADS, HEADS, PSEUDOHEADS)
        for h in range(HEADS):
            alpha[h, h, 0] = 2.0
            alpha[h, h, 1] = 1.2
            alpha[(h + 1) % HEADS, h, 1] = 0.8
        return alpha

    def sparse_topk(self, attn: torch.Tensor) -> torch.Tensor:
        values, idx = torch.topk(attn, k=TOP_K, dim=-1)
        mask = torch.zeros_like(attn).scatter_(-1, idx, 1.0)
        sparse = attn * mask
        return sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, tokens, _ = x.shape
        q = self.q(x).view(n, tokens, HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k = self.k(x).view(n, tokens, HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        v = self.v(x).view(n, tokens, HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        aq = torch.softmax(self.alpha_q, dim=0)
        ak = torch.softmax(self.alpha_k, dim=0)
        av = torch.softmax(self.alpha_v, dim=0)
        q_t = torch.einsum("mhp,nmtd->nhptd", aq, q)
        k_t = torch.einsum("mhp,nmtd->nhptd", ak, k)
        v_t = torch.einsum("mhp,nmtd->nhptd", av, v)

        temp = torch.nn.functional.softplus(self.temperature_logit) + 0.35
        logits = torch.einsum("nhpid,nhpjd->nhpij", q_t, k_t) / (np.sqrt(HEAD_DIM) * temp)
        logits = logits + self.graph_bias.view(1, 1, 1, tokens, tokens)
        dense_attn = torch.softmax(logits, dim=-1)
        attn = self.sparse_topk(dense_attn)
        out = torch.einsum("nhpij,nhpjd->nhpid", attn, v_t)
        collapse = torch.softmax(self.collapse_logits, dim=-1)
        pooled = torch.einsum("hp,nhptd->nhtd", collapse, out).mean(dim=2).reshape(n, HEADS * HEAD_DIM)
        return pooled, attn, dense_attn


class GraphSparseV3Model(nn.Module):
    def __init__(self, n_features: int, graph_bias: np.ndarray, graph_bias_strength: float):
        super().__init__()
        self.layer = GraphSparseLayer(n_features, graph_bias, graph_bias_strength)
        self.risk = nn.Sequential(
            nn.LayerNorm(HEADS * HEAD_DIM),
            nn.Linear(HEADS * HEAD_DIM, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(HIDDEN, 1),
        )
        self.aux = nn.Sequential(
            nn.LayerNorm(HEADS * HEAD_DIM),
            nn.Linear(HEADS * HEAD_DIM, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled, attn, dense_attn = self.layer(x)
        return self.risk(pooled).squeeze(-1), self.aux(pooled), attn, dense_attn


def attention_metrics(attn: np.ndarray) -> dict[str, float]:
    pos, neg, _ = edge_masks()
    pos_np = pos.numpy()
    neg_np = neg.numpy()
    mean_attn = attn.mean(axis=(0, 1, 2))
    positive_mean = float(mean_attn[pos_np].mean())
    negative_mean = float(mean_attn[neg_np].mean())
    flat = mean_attn.reshape(-1)
    entropy = float(-(flat * np.log(flat + 1e-9)).sum())
    top_edges = float(np.mean(flat >= np.sort(flat)[-8]))
    return {
        "positive_edge_attention": positive_mean,
        "negative_edge_attention": negative_mean,
        "edge_separation": positive_mean - negative_mean,
        "mean_attention_entropy": entropy,
        "top_edge_fraction": top_edges,
    }


def top_edge_table(attn: np.ndarray, model: str, seed: int) -> pd.DataFrame:
    mean_attn = attn.mean(axis=(0, 1, 2))
    rows = []
    for i, source in enumerate(MODULES):
        for j, target in enumerate(MODULES):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "source_module": source,
                    "target_module": target,
                    "edge": f"{source} -> {target}",
                    "mean_attention": float(mean_attn[i, j]),
                    "is_biology_edge": bool(
                        source == target
                        or (source, target) in BIOLOGY_EDGES
                        or (target, source) in BIOLOGY_EDGES
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("mean_attention", ascending=False)


def loss_fn(
    risk: torch.Tensor,
    aux: torch.Tensor,
    attn: torch.Tensor,
    dense_attn: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    raw_prior: torch.Tensor,
    coupling: torch.Tensor,
    spec: GraphSparseSpec,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
) -> torch.Tensor:
    mse = nn.MSELoss()
    risk_z = (risk - risk.mean()) / (risk.std(unbiased=False) + 1e-6)
    loss = cox_loss(risk, time, event)
    loss = loss + spec.prior_weight * mse(risk_z, raw_prior)
    loss = loss + spec.multitask_weight * (mse(aux[:, 0], raw_prior) + mse(aux[:, 1], coupling))

    positive = attn[..., positive_mask].mean()
    negative = attn[..., negative_mask].mean()
    contrastive = torch.relu(torch.tensor(0.075, device=attn.device) - (positive - negative))
    loss = loss + spec.contrastive_weight * contrastive

    dense_entropy = -(dense_attn * torch.log(dense_attn + 1e-8)).sum(dim=-1).mean()
    loss = loss + spec.entropy_weight * dense_entropy
    return loss


def train_fold(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    raw_prior: np.ndarray,
    coupling: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: GraphSparseSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    set_seed(seed)
    positive_mask, negative_mask, graph_bias = edge_masks()
    inner = train_idx.copy()
    val_size = max(12, int(round(len(inner) * 0.20)))
    val_idx = inner[:val_size]
    fit_idx = inner[val_size:]
    model = GraphSparseV3Model(x.shape[-1], graph_bias, spec.graph_bias_strength)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32)

    x_fit, x_val = tensor(x[fit_idx]), tensor(x[val_idx])
    t_fit, t_val = tensor(time[fit_idx]), tensor(time[val_idx])
    e_fit, e_val = tensor(event[fit_idx]), tensor(event[val_idx])
    p_fit, p_val = tensor(raw_prior[fit_idx]), tensor(raw_prior[val_idx])
    c_fit, c_val = tensor(coupling[fit_idx]), tensor(coupling[val_idx])

    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        risk, aux, attn, dense = model(x_fit)
        loss = loss_fn(risk, aux, attn, dense, t_fit, e_fit, p_fit, c_fit, spec, positive_mask, negative_mask)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_risk, val_aux, val_attn, val_dense = model(x_val)
            val_loss = loss_fn(val_risk, val_aux, val_attn, val_dense, t_val, e_val, p_val, c_val, spec, positive_mask, negative_mask)
            value = float(val_loss.item())
        if value < best_val - 1e-5:
            best_val = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    x_test = tensor(x[test_idx])
    with torch.no_grad():
        risk, _, attn, _ = model(x_test)
        occluded = {}
        for module_i, module in enumerate(MODULES):
            x_occ = x_test.clone()
            x_occ[:, module_i, :] = 0.0
            occ_risk, _, _, _ = model(x_occ)
            occluded[module] = occ_risk.numpy()
    return risk.numpy(), attn.numpy(), occluded


def evaluate(frame: pd.DataFrame, time: np.ndarray, event: np.ndarray, score: np.ndarray, model: str, seed: int | str, desc: str, metrics: dict[str, float] | None = None) -> dict[str, object]:
    tmp = frame[["geo_accession", "relapse_free_months", "relapse_event"]].copy()
    tmp["risk_score"] = score
    lr = logrank_high_low(tmp, "risk_score", "relapse_free_months", "relapse_event")
    row = {
        "model": model,
        "seed": seed,
        "description": desc,
        "c_index": concordance_index(time, event, score),
        "hazard_ratio_approx": lr["hazard_ratio_approx"],
        "logrank_p": lr["logrank_p"],
        "risk_prior_corr": float(pd.Series(score).corr(pd.Series(frame["raw_relapse_intervene_state"]))),
        "high_events": lr["high_events"],
        "low_events": lr["low_events"],
    }
    if metrics:
        row.update(metrics)
    return row


def write_summary(results: pd.DataFrame, edges: pd.DataFrame, ablation: pd.DataFrame) -> None:
    learned = results[results["seed"].astype(str) != "na"].copy()
    aggregate = (
        learned.groupby("model")
        .agg(
            c_index_mean=("c_index", "mean"),
            c_index_sd=("c_index", "std"),
            hr_mean=("hazard_ratio_approx", "mean"),
            logrank_p_median=("logrank_p", "median"),
            prior_corr_mean=("risk_prior_corr", "mean"),
            edge_sep_mean=("edge_separation", "mean"),
            entropy_mean=("mean_attention_entropy", "mean"),
            high_events_mean=("high_events", "mean"),
            low_events_mean=("low_events", "mean"),
        )
        .reset_index()
        .sort_values(["edge_sep_mean", "c_index_mean"], ascending=False)
    )
    edge_stability = (
        edges.groupby(["model", "edge", "is_biology_edge"])
        .agg(mean_attention=("mean_attention", "mean"), seeds_observed=("seed", "nunique"))
        .reset_index()
        .sort_values(["model", "mean_attention"], ascending=[True, False])
    )
    path = RESULTS / "relapse_intervene_graph_sparse_v3_summary.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Relapse-INTERVENE GraphSparse V3\n")
        f.write("================================\n\n")
        f.write("Purpose: sharpen module attention edges using graph-biased sparse attention, contrastive biology-edge loss, and leave-one-module-out validation.\n\n")
        f.write("Aggregate model comparison\n")
        f.write("--------------------------\n")
        f.write(aggregate.to_string(index=False))
        f.write("\n\nTop stable edges\n")
        f.write("----------------\n")
        f.write(edge_stability.groupby("model").head(10).to_string(index=False))
        f.write("\n\nLeave-one-module-out risk drop\n")
        f.write("------------------------------\n")
        f.write(ablation.to_string(index=False))
        f.write("\n\nInterpretation\n")
        f.write("--------------\n")
        f.write(
            "If GraphSparse V3 increases edge separation while keeping relapse C-index near the V2 model, "
            "it should be used as the interpretable edge-map model. The primary prediction model can remain "
            "module_intervene_constrained if it has the best C-index.\n"
        )


def main() -> None:
    frame, x, _, time, event, raw_prior, coupling = load_data()
    rows = [evaluate(frame, time, event, raw_prior, "raw_thermodynamic_transfer", "na", "Transferred stress-survival state")]
    predictions = frame[["geo_accession", "relapse_free_months", "relapse_event", "raw_relapse_intervene_state"]].copy()
    edge_tables = []
    ablation_rows = []

    for seed in SEEDS:
        for spec in SPECS:
            pred = np.full(len(frame), np.nan)
            occluded_preds = {module: np.full(len(frame), np.nan) for module in MODULES}
            attn_blocks = []
            for fold_id, test_idx in enumerate(folds(len(frame), seed)):
                train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
                fold_pred, fold_attn, fold_occ = train_fold(
                    x, time, event, raw_prior, coupling, train_idx, test_idx, spec, seed + fold_id * 101
                )
                pred[test_idx] = fold_pred
                attn_blocks.append(fold_attn)
                for module in MODULES:
                    occluded_preds[module][test_idx] = fold_occ[module]
            attn = np.concatenate(attn_blocks, axis=0)
            metrics = attention_metrics(attn)
            rows.append(evaluate(frame, time, event, pred, spec.model, seed, spec.description, metrics))
            predictions[f"{spec.model}_seed{seed}_risk"] = pred
            edge_tables.append(top_edge_table(attn, spec.model, seed))
            for module in MODULES:
                occ = occluded_preds[module]
                ablation_rows.append(
                    {
                        "model": spec.model,
                        "seed": seed,
                        "module_removed": module,
                        "mean_risk_drop_when_removed": float(np.nanmean(pred - occ)),
                        "c_index_after_removal": concordance_index(time, event, occ),
                    }
                )

    results = pd.DataFrame(rows)
    edges = pd.concat(edge_tables, ignore_index=True)
    ablation = pd.DataFrame(ablation_rows)
    results.to_csv(RESULTS / "relapse_intervene_graph_sparse_v3_model_comparison.csv", index=False)
    predictions.to_csv(RESULTS / "relapse_intervene_graph_sparse_v3_oof_risk_scores.csv", index=False)
    edges.to_csv(RESULTS / "relapse_intervene_graph_sparse_v3_attention_edges.csv", index=False)
    ablation.to_csv(RESULTS / "relapse_intervene_graph_sparse_v3_module_ablation.csv", index=False)
    ablation_summary = (
        ablation.groupby(["model", "module_removed"])
        .agg(
            mean_risk_drop_when_removed=("mean_risk_drop_when_removed", "mean"),
            c_index_after_removal=("c_index_after_removal", "mean"),
        )
        .reset_index()
    )
    write_summary(results, edges, ablation_summary)
    print(f"Wrote {RESULTS / 'relapse_intervene_graph_sparse_v3_model_comparison.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_graph_sparse_v3_attention_edges.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_graph_sparse_v3_module_ablation.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_graph_sparse_v3_summary.txt'}")


if __name__ == "__main__":
    main()

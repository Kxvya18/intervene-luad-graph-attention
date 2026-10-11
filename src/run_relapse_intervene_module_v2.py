from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from interleaved_biophysics_attention_full import IBPALayer, set_seed, zscore
from prepare_gse31210_relapse_intervene import logrank_high_low
from run_relapse_intervene_survival_head import cox_loss, concordance_index, folds


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"
RESULTS = ROOT / "results"

SEEDS = [7, 17, 29, 43, 61]
EPOCHS = 520
PATIENCE = 65
LR = 1.4e-3
WEIGHT_DECAY = 2e-3

MODULES = [
    "srebp_lipid",
    "mito_translation",
    "oxidative_stress",
    "proteostasis",
    "apoptosis_iap",
    "akt_mtor_growth",
]
MODULE_FEATURES = ["expr_mean", "expr_median", "expr_dispersion"]

BIOLOGY_EDGES = [
    ("srebp_lipid", "mito_translation"),
    ("mito_translation", "oxidative_stress"),
    ("oxidative_stress", "apoptosis_iap"),
    ("oxidative_stress", "proteostasis"),
    ("proteostasis", "apoptosis_iap"),
    ("akt_mtor_growth", "srebp_lipid"),
    ("akt_mtor_growth", "mito_translation"),
]


@dataclass(frozen=True)
class ModuleV2Spec:
    model: str
    mode: str
    use_clinical: bool
    prior_weight: float
    multitask_weight: float
    attention_prior_weight: float
    description: str


SPECS = [
    ModuleV2Spec("module_mha", "mha", False, 0.0, 0.0, 0.0, "Module-level normal attention"),
    ModuleV2Spec("module_interleaved", "ibpa", False, 0.0, 0.0, 0.0, "Module-level interleaved attention"),
    ModuleV2Spec("module_intervene_multitask", "ibpa", False, 0.03, 0.08, 0.0, "Module-level INTERVENE with multitask biology"),
    ModuleV2Spec("module_intervene_constrained", "ibpa", False, 0.03, 0.08, 0.18, "Module-level INTERVENE with multitask and guided edges"),
    ModuleV2Spec("module_intervene_edge_forced", "ibpa", False, 0.03, 0.08, 1.20, "Module-level INTERVENE with strongly guided biology edges"),
    ModuleV2Spec("module_intervene_clinical", "ibpa", True, 0.03, 0.08, 0.18, "Module-level INTERVENE with clinical covariates and guided edges"),
]


class ModuleV2Model(nn.Module):
    def __init__(self, n_features: int, n_clinical: int, mode: str, use_clinical: bool):
        super().__init__()
        self.use_clinical = use_clinical
        self.layer = IBPALayer(n_features, heads=4, pseudoheads=2 if mode == "ibpa" else 1, head_dim=4, mode=mode)
        self.clinical = (
            nn.Sequential(nn.LayerNorm(n_clinical), nn.Linear(n_clinical, 8), nn.GELU())
            if use_clinical and n_clinical
            else None
        )
        total = 16 + (8 if self.clinical is not None else 0)
        self.risk = nn.Sequential(nn.LayerNorm(total), nn.Linear(total, 24), nn.GELU(), nn.Dropout(0.08), nn.Linear(24, 1))
        self.aux = nn.Sequential(nn.LayerNorm(total), nn.Linear(total, 24), nn.GELU(), nn.Linear(24, 2))

    def forward(self, x: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled, attn = self.layer(x)
        if self.clinical is not None:
            pooled = torch.cat([pooled, self.clinical(clinical)], dim=1)
        return self.risk(pooled).squeeze(-1), self.aux(pooled), attn


def one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    values = series.fillna("missing").astype(str).str.strip().replace({"": "missing", ".": "missing", "nd": "missing"})
    return pd.get_dummies(values, prefix=prefix, dtype=float)


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patient = pd.read_csv(PROCESSED / "gse31210_patient_module_scores.csv")
    frame = patient[patient["is_primary_tumor"] & patient["prognosis_included"]].copy()
    frame = frame.dropna(subset=["relapse_event", "relapse_free_months", "raw_relapse_intervene_state"]).copy()
    frame = frame[frame["relapse_free_months"] > 0].reset_index(drop=True)

    x = np.zeros((len(frame), len(MODULES), len(MODULE_FEATURES)), dtype=np.float32)
    for ti, module in enumerate(MODULES):
        for fi, feature in enumerate(MODULE_FEATURES):
            col = f"{module}_{feature}"
            values = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
            x[:, ti, fi] = zscore(values).to_numpy(dtype=np.float32)

    clinical_parts = []
    age = pd.to_numeric(frame["age"], errors="coerce")
    clinical_parts.append(zscore(age.fillna(age.median())).rename("age_z").to_frame())
    for col, prefix in [
        ("gender", "sex"),
        ("smoking_status", "smoking"),
        ("pathological_stage_binary", "stage"),
        ("gene_alteration_status", "driver"),
        ("myc", "myc"),
    ]:
        clinical_parts.append(one_hot(frame[col], prefix))
    clinical = pd.concat(clinical_parts, axis=1).fillna(0.0).to_numpy(dtype=np.float32)
    time = frame["relapse_free_months"].to_numpy(dtype=np.float32)
    event = frame["relapse_event"].to_numpy(dtype=np.float32)
    raw_prior = zscore(frame["raw_relapse_intervene_state"]).to_numpy(dtype=np.float32)
    coupling = zscore(frame["srebp_mito_expression_coupling"]).to_numpy(dtype=np.float32)
    return frame, x, clinical, time, event, raw_prior, coupling


def attention_prior() -> np.ndarray:
    prior = np.ones((len(MODULES), len(MODULES)), dtype=np.float32) * 0.08
    for i, source in enumerate(MODULES):
        for j, target in enumerate(MODULES):
            if source == target:
                prior[i, j] += 0.20
            if (source, target) in BIOLOGY_EDGES or (target, source) in BIOLOGY_EDGES:
                prior[i, j] += 0.80
    return prior / prior.sum(axis=1, keepdims=True)


def edge_gain(attn: np.ndarray) -> float:
    mean_attn = attn.mean(axis=(0, 1, 2))
    bio = []
    other = []
    for i, source in enumerate(MODULES):
        for j, target in enumerate(MODULES):
            if (source, target) in BIOLOGY_EDGES or (target, source) in BIOLOGY_EDGES:
                bio.append(mean_attn[i, j])
            else:
                other.append(mean_attn[i, j])
    return float(np.mean(bio) - np.mean(other))


def train_fold(
    x: np.ndarray,
    clinical: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    raw_prior: np.ndarray,
    coupling: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: ModuleV2Spec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    inner = train_idx.copy()
    val_size = max(12, int(round(len(inner) * 0.20)))
    val_idx = inner[:val_size]
    fit_idx = inner[val_size:]
    model = ModuleV2Model(x.shape[-1], clinical.shape[1], spec.mode, spec.use_clinical)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    attn_prior = torch.tensor(attention_prior(), dtype=torch.float32)

    tensors = {
        "x_fit": torch.tensor(x[fit_idx], dtype=torch.float32),
        "c_fit": torch.tensor(clinical[fit_idx], dtype=torch.float32),
        "t_fit": torch.tensor(time[fit_idx], dtype=torch.float32),
        "e_fit": torch.tensor(event[fit_idx], dtype=torch.float32),
        "p_fit": torch.tensor(raw_prior[fit_idx], dtype=torch.float32),
        "coupling_fit": torch.tensor(coupling[fit_idx], dtype=torch.float32),
        "x_val": torch.tensor(x[val_idx], dtype=torch.float32),
        "c_val": torch.tensor(clinical[val_idx], dtype=torch.float32),
        "t_val": torch.tensor(time[val_idx], dtype=torch.float32),
        "e_val": torch.tensor(event[val_idx], dtype=torch.float32),
        "p_val": torch.tensor(raw_prior[val_idx], dtype=torch.float32),
        "coupling_val": torch.tensor(coupling[val_idx], dtype=torch.float32),
    }

    def loss_fn(risk: torch.Tensor, aux: torch.Tensor, attn: torch.Tensor, t: torch.Tensor, e: torch.Tensor, p: torch.Tensor, coupling_t: torch.Tensor) -> torch.Tensor:
        risk_z = (risk - risk.mean()) / (risk.std(unbiased=False) + 1e-6)
        loss = cox_loss(risk, t, e)
        if spec.prior_weight:
            loss = loss + spec.prior_weight * mse(risk_z, p)
        if spec.multitask_weight:
            loss = loss + spec.multitask_weight * (mse(aux[:, 0], p) + mse(aux[:, 1], coupling_t))
        if spec.attention_prior_weight:
            loss = loss + spec.attention_prior_weight * mse(attn.mean(dim=(0, 1, 2)), attn_prior)
        return loss

    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        risk, aux, attn = model(tensors["x_fit"], tensors["c_fit"])
        loss = loss_fn(risk, aux, attn, tensors["t_fit"], tensors["e_fit"], tensors["p_fit"], tensors["coupling_fit"])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_risk, val_aux, val_attn = model(tensors["x_val"], tensors["c_val"])
            value = float(loss_fn(val_risk, val_aux, val_attn, tensors["t_val"], tensors["e_val"], tensors["p_val"], tensors["coupling_val"]).item())
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
    with torch.no_grad():
        risk, _, attn = model(torch.tensor(x[test_idx], dtype=torch.float32), torch.tensor(clinical[test_idx], dtype=torch.float32))
    return risk.numpy(), attn.numpy()


def evaluate(frame: pd.DataFrame, time: np.ndarray, event: np.ndarray, score: np.ndarray, model: str, seed: int | str, desc: str, gain: float = np.nan) -> dict[str, object]:
    tmp = frame[["geo_accession", "relapse_free_months", "relapse_event"]].copy()
    tmp["risk_score"] = score
    lr = logrank_high_low(tmp, "risk_score", "relapse_free_months", "relapse_event")
    return {
        "model": model,
        "seed": seed,
        "description": desc,
        "c_index": concordance_index(time, event, score),
        "hazard_ratio_approx": lr["hazard_ratio_approx"],
        "logrank_p": lr["logrank_p"],
        "risk_prior_corr": float(pd.Series(score).corr(pd.Series(frame["raw_relapse_intervene_state"]))),
        "biology_edge_attention_gain": gain,
        "high_events": lr["high_events"],
        "low_events": lr["low_events"],
    }


def write_summary(results: pd.DataFrame) -> None:
    learned = results[results["seed"].astype(str) != "na"]
    aggregate = (
        learned.groupby("model")
        .agg(
            c_index_mean=("c_index", "mean"),
            c_index_sd=("c_index", "std"),
            hr_mean=("hazard_ratio_approx", "mean"),
            logrank_p_median=("logrank_p", "median"),
            prior_corr_mean=("risk_prior_corr", "mean"),
            edge_gain_mean=("biology_edge_attention_gain", "mean"),
            high_events_mean=("high_events", "mean"),
            low_events_mean=("low_events", "mean"),
        )
        .reset_index()
        .sort_values(["c_index_mean", "edge_gain_mean"], ascending=False)
    )
    path = RESULTS / "relapse_intervene_module_v2_summary.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Relapse-INTERVENE module V2\n")
        f.write("===========================\n\n")
        f.write("This experiment keeps compact biological module tokens, then adds multitask biology, guided module edges, and clinical covariates.\n\n")
        f.write(aggregate.to_string(index=False))
        f.write("\n")


def main() -> None:
    frame, x, clinical, time, event, raw_prior, coupling = load_data()
    rows = [evaluate(frame, time, event, raw_prior, "raw_thermodynamic_transfer", "na", "Transferred stress-survival state")]
    predictions = frame[["geo_accession", "relapse_free_months", "relapse_event", "raw_relapse_intervene_state"]].copy()
    for seed in SEEDS:
        for spec in SPECS:
            pred = np.full(len(frame), np.nan)
            attn_blocks = []
            for fold_id, test_idx in enumerate(folds(len(frame), seed)):
                train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
                fold_pred, fold_attn = train_fold(x, clinical, time, event, raw_prior, coupling, train_idx, test_idx, spec, seed + fold_id * 101)
                pred[test_idx] = fold_pred
                attn_blocks.append(fold_attn)
            rows.append(evaluate(frame, time, event, pred, spec.model, seed, spec.description, edge_gain(np.concatenate(attn_blocks, axis=0))))
            predictions[f"{spec.model}_seed{seed}_risk"] = pred
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "relapse_intervene_module_v2_model_comparison.csv", index=False)
    predictions.to_csv(RESULTS / "relapse_intervene_module_v2_oof_risk_scores.csv", index=False)
    pd.DataFrame(attention_prior(), index=MODULES, columns=MODULES).to_csv(RESULTS / "relapse_intervene_module_v2_attention_prior.csv")
    write_summary(results)
    print(f"Wrote {RESULTS / 'relapse_intervene_module_v2_model_comparison.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_module_v2_oof_risk_scores.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_module_v2_summary.txt'}")


if __name__ == "__main__":
    main()

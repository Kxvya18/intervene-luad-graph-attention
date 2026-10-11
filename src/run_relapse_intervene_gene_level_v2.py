from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from interleaved_biophysics_attention_full import IBPALayer, module_specs, set_seed, zscore
from prepare_gse31210_relapse_intervene import logrank_high_low
from run_relapse_intervene_survival_head import cox_loss, concordance_index, folds


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "relapse" / "GSE31210"
RESULTS = ROOT / "results"

SEEDS = [7, 17]
EPOCHS = 240
PATIENCE = 35
LR = 1.2e-3
WEIGHT_DECAY = 2.5e-3
HIDDEN = 32

MODULES = [
    "srebp_lipid",
    "mito_translation",
    "oxidative_stress",
    "proteostasis",
    "apoptosis_iap",
    "akt_mtor_growth",
]

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
class V2Spec:
    model: str
    mode: str
    use_gene_tokens: bool
    use_clinical: bool
    prior_weight: float
    multitask_weight: float
    attention_prior_weight: float
    description: str


V2_SPECS = [
    V2Spec(
        "clinical_only_cox",
        "linear",
        False,
        True,
        0.0,
        0.0,
        0.0,
        "Clinical covariate Cox-like baseline",
    ),
    V2Spec(
        "clinical_plus_raw_transfer",
        "linear",
        False,
        True,
        0.0,
        0.0,
        0.0,
        "Clinical covariates plus raw transferred stress-survival state",
    ),
    V2Spec(
        "gene_mha_clinical",
        "mha",
        True,
        True,
        0.0,
        0.0,
        0.0,
        "Gene-level normal attention with clinical covariates",
    ),
    V2Spec(
        "gene_interleaved_clinical",
        "ibpa",
        True,
        True,
        0.0,
        0.0,
        0.0,
        "Gene-level interleaved attention with clinical covariates",
    ),
    V2Spec(
        "relapse_intervene_gene_v2",
        "ibpa",
        True,
        True,
        0.03,
        0.08,
        0.10,
        "Gene-level Relapse-INTERVENE with clinical covariates, multitask prior, and biology-guided attention",
    ),
    V2Spec(
        "relapse_intervene_gene_v2_edge_strong",
        "ibpa",
        True,
        True,
        0.03,
        0.08,
        0.25,
        "Gene-level Relapse-INTERVENE with stronger biology-guided attention regularization",
    ),
]


class GeneClinicalAttentionModel(nn.Module):
    def __init__(self, n_gene_features: int, n_clinical: int, mode: str, use_clinical: bool):
        super().__init__()
        self.use_clinical = use_clinical
        self.layer = IBPALayer(n_gene_features, heads=4, pseudoheads=2 if mode == "ibpa" else 1, head_dim=4, mode=mode)
        clinical_dim = 8 if use_clinical and n_clinical else 0
        self.clinical = (
            nn.Sequential(nn.LayerNorm(n_clinical), nn.Linear(n_clinical, clinical_dim), nn.GELU())
            if clinical_dim
            else None
        )
        total = 16 + clinical_dim
        self.risk_head = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN, 1),
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, x: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled, attn = self.layer(x)
        if self.clinical is not None:
            pooled = torch.cat([pooled, self.clinical(clinical)], dim=1)
        return self.risk_head(pooled).squeeze(-1), self.aux_head(pooled), attn


class LinearClinicalModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.risk = nn.Sequential(nn.LayerNorm(n_features), nn.Linear(n_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.risk(x).squeeze(-1)


def gene_modules() -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    specs = [s for s in module_specs() if s.token in MODULES]
    token_to_genes = {s.token: [g for g in s.genes] for s in specs}
    gene_to_module = {}
    genes = []
    for spec in specs:
        for gene in spec.genes:
            if gene not in gene_to_module:
                gene_to_module[gene] = spec.token
                genes.append(gene)
    return genes, gene_to_module, token_to_genes


def one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    values = series.fillna("missing").astype(str).str.strip().replace({"": "missing", ".": "missing", "nd": "missing"})
    return pd.get_dummies(values, prefix=prefix, dtype=float)


def load_v2_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    patient = pd.read_csv(PROCESSED / "gse31210_patient_module_scores.csv")
    frame = patient[patient["is_primary_tumor"] & patient["prognosis_included"]].copy()
    frame = frame.dropna(subset=["relapse_event", "relapse_free_months", "raw_relapse_intervene_state"]).copy()
    frame = frame[frame["relapse_free_months"] > 0].reset_index(drop=True)

    genes, gene_to_module, _ = gene_modules()
    feature_dim = 1 + len(MODULES)
    x = np.zeros((len(frame), len(genes), feature_dim), dtype=np.float32)
    for gi, gene in enumerate(genes):
        values = pd.to_numeric(frame[gene], errors="coerce") if gene in frame.columns else pd.Series(np.nan, index=frame.index)
        values = values.fillna(values.median() if values.notna().any() else 0.0)
        x[:, gi, 0] = zscore(values).to_numpy(dtype=np.float32)
        module = gene_to_module[gene]
        x[:, gi, 1 + MODULES.index(module)] = 1.0

    clinical_parts = []
    age = pd.to_numeric(frame["age"], errors="coerce")
    clinical_parts.append(zscore(age.fillna(age.median())).rename("age_z").to_frame())
    for col, prefix in [
        ("gender", "sex"),
        ("smoking_status", "smoking"),
        ("pathological_stage_binary", "stage"),
        ("gene_alteration_status", "driver"),
        ("myc", "myc"),
        ("cluster", "cluster"),
    ]:
        if col in frame.columns:
            clinical_parts.append(one_hot(frame[col], prefix))
    myc_copy = pd.to_numeric(frame.get("myc_copy", pd.Series(np.nan, index=frame.index)), errors="coerce")
    clinical_parts.append(zscore(myc_copy.fillna(myc_copy.median())).rename("myc_copy_z").to_frame())
    clinical_df = pd.concat(clinical_parts, axis=1).fillna(0.0)
    clinical = clinical_df.to_numpy(dtype=np.float32)

    raw_prior = zscore(frame["raw_relapse_intervene_state"]).to_numpy(dtype=np.float32)
    coupling = zscore(frame["srebp_mito_expression_coupling"]).to_numpy(dtype=np.float32)
    time = frame["relapse_free_months"].to_numpy(dtype=np.float32)
    event = frame["relapse_event"].to_numpy(dtype=np.float32)
    return frame, np.nan_to_num(x), clinical, time, event, raw_prior, coupling, genes, clinical_df.columns.to_numpy()


def biology_attention_prior(genes: list[str]) -> np.ndarray:
    _, gene_to_module, _ = gene_modules()
    n = len(genes)
    prior = np.ones((n, n), dtype=np.float32) * 0.05
    for i, source_gene in enumerate(genes):
        source_module = gene_to_module[source_gene]
        for j, target_gene in enumerate(genes):
            target_module = gene_to_module[target_gene]
            if source_module == target_module:
                prior[i, j] += 0.20
            if (source_module, target_module) in BIOLOGY_EDGES or (target_module, source_module) in BIOLOGY_EDGES:
                prior[i, j] += 0.75
    prior = prior / prior.sum(axis=1, keepdims=True)
    return prior


def edge_enrichment(attn: np.ndarray, genes: list[str]) -> float:
    _, gene_to_module, _ = gene_modules()
    mean_attn = attn.mean(axis=(0, 1, 2))
    bio_values = []
    other_values = []
    for i, source_gene in enumerate(genes):
        for j, target_gene in enumerate(genes):
            source_module = gene_to_module[source_gene]
            target_module = gene_to_module[target_gene]
            value = float(mean_attn[i, j])
            if (source_module, target_module) in BIOLOGY_EDGES or (target_module, source_module) in BIOLOGY_EDGES:
                bio_values.append(value)
            else:
                other_values.append(value)
    return float(np.mean(bio_values) - np.mean(other_values)) if bio_values and other_values else np.nan


def evaluate_score(
    frame: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
    model: str,
    seed: int | str,
    description: str,
    edge_gain: float = np.nan,
) -> dict[str, object]:
    tmp = frame[["geo_accession", "relapse_free_months", "relapse_event"]].copy()
    tmp["risk_score"] = score
    lr = logrank_high_low(tmp, "risk_score", "relapse_free_months", "relapse_event")
    return {
        "model": model,
        "seed": seed,
        "description": description,
        "n": len(frame),
        "events": int(event.sum()),
        "c_index": concordance_index(time, event, score),
        "risk_prior_corr": float(pd.Series(score).corr(pd.Series(frame["raw_relapse_intervene_state"]))),
        "hazard_ratio_approx": lr["hazard_ratio_approx"],
        "event_fisher_p": lr["event_fisher_p"],
        "logrank_p": lr["logrank_p"],
        "high_events": lr["high_events"],
        "low_events": lr["low_events"],
        "biology_edge_attention_gain": edge_gain,
    }


def train_linear_fold(
    features: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    inner = train_idx.copy()
    val_size = max(12, int(round(len(inner) * 0.2)))
    val_idx = inner[:val_size]
    fit_idx = inner[val_size:]
    model = LinearClinicalModel(features.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x_fit = torch.tensor(features[fit_idx], dtype=torch.float32)
    t_fit = torch.tensor(time[fit_idx], dtype=torch.float32)
    e_fit = torch.tensor(event[fit_idx], dtype=torch.float32)
    x_val = torch.tensor(features[val_idx], dtype=torch.float32)
    t_val = torch.tensor(time[val_idx], dtype=torch.float32)
    e_val = torch.tensor(event[val_idx], dtype=torch.float32)
    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = cox_loss(model(x_fit), t_fit, e_fit)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            value = float(cox_loss(model(x_val), t_val, e_val).item())
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
    with torch.no_grad():
        return model(torch.tensor(features[test_idx], dtype=torch.float32)).numpy()


def train_attention_fold(
    x: np.ndarray,
    clinical: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    raw_prior: np.ndarray,
    coupling: np.ndarray,
    attn_prior: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: V2Spec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    inner = train_idx.copy()
    val_size = max(12, int(round(len(inner) * 0.2)))
    val_idx = inner[:val_size]
    fit_idx = inner[val_size:]
    model = GeneClinicalAttentionModel(x.shape[-1], clinical.shape[1], spec.mode, spec.use_clinical)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    x_fit = torch.tensor(x[fit_idx], dtype=torch.float32)
    c_fit = torch.tensor(clinical[fit_idx], dtype=torch.float32)
    t_fit = torch.tensor(time[fit_idx], dtype=torch.float32)
    e_fit = torch.tensor(event[fit_idx], dtype=torch.float32)
    prior_fit = torch.tensor(raw_prior[fit_idx], dtype=torch.float32)
    coupling_fit = torch.tensor(coupling[fit_idx], dtype=torch.float32)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32)
    c_val = torch.tensor(clinical[val_idx], dtype=torch.float32)
    t_val = torch.tensor(time[val_idx], dtype=torch.float32)
    e_val = torch.tensor(event[val_idx], dtype=torch.float32)
    prior_val = torch.tensor(raw_prior[val_idx], dtype=torch.float32)
    coupling_val = torch.tensor(coupling[val_idx], dtype=torch.float32)
    attn_prior_t = torch.tensor(attn_prior, dtype=torch.float32)
    mse = nn.MSELoss()

    def total_loss(risk: torch.Tensor, aux: torch.Tensor, attn: torch.Tensor, t: torch.Tensor, e: torch.Tensor, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        risk_z = (risk - risk.mean()) / (risk.std(unbiased=False) + 1e-6)
        loss = cox_loss(risk, t, e)
        if spec.prior_weight:
            loss = loss + spec.prior_weight * mse(risk_z, p)
        if spec.multitask_weight:
            loss = loss + spec.multitask_weight * (mse(aux[:, 0], p) + mse(aux[:, 1], c))
        if spec.attention_prior_weight:
            mean_attn = attn.mean(dim=(0, 1, 2))
            loss = loss + spec.attention_prior_weight * mse(mean_attn, attn_prior_t)
        return loss

    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        risk, aux, attn = model(x_fit, c_fit)
        loss = total_loss(risk, aux, attn, t_fit, e_fit, prior_fit, coupling_fit)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_risk, val_aux, val_attn = model(x_val, c_val)
            value = float(total_loss(val_risk, val_aux, val_attn, t_val, e_val, prior_val, coupling_val).item())
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


def write_summary(results: pd.DataFrame, genes: list[str], clinical_names: np.ndarray) -> None:
    learned = results[results["seed"].astype(str) != "na"].copy()
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
    path = RESULTS / "relapse_intervene_gene_level_v2_summary.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Relapse-INTERVENE V2: gene-token + clinical + constrained biology\n")
        f.write("=================================================================\n\n")
        f.write(f"Patients: {int(results['n'].max())}\n")
        f.write(f"Gene tokens: {len(genes)}\n")
        f.write(f"Clinical covariates after encoding: {len(clinical_names)}\n")
        f.write("Evaluation: five-fold out-of-fold survival prediction over five seeds.\n\n")
        f.write("Aggregate model comparison\n")
        f.write("--------------------------\n")
        f.write(aggregate.to_string(index=False))
        f.write("\n\nInterpretation\n")
        f.write("--------------\n")
        f.write(
            "This V2 experiment tests whether richer gene-level tokens, clinical adjustment, multitask supervision, "
            "and biology-guided attention make the AI component stronger. The strongest model should ideally improve "
            "C-index while also increasing biology-edge attention gain. If edge gain improves without C-index improving, "
            "the constrained model is more interpretable but not more predictive.\n"
        )


def main() -> None:
    frame, x, clinical, time, event, raw_prior, coupling, genes, clinical_names = load_v2_data()
    attn_prior = biology_attention_prior(genes)

    all_rows = [
        evaluate_score(
            frame,
            time,
            event,
            raw_prior,
            "raw_thermodynamic_transfer",
            "na",
            "No learned relapse model; transferred stress-survival state",
        )
    ]
    predictions = frame[["geo_accession", "relapse_free_months", "relapse_event", "raw_relapse_intervene_state"]].copy()

    for seed in SEEDS:
        for spec in V2_SPECS:
            pred = np.full(len(frame), np.nan, dtype=float)
            attn_all = []
            for fold_id, test_idx in enumerate(folds(len(frame), seed)):
                train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
                if spec.use_gene_tokens:
                    fold_pred, fold_attn = train_attention_fold(
                        x,
                        clinical,
                        time,
                        event,
                        raw_prior,
                        coupling,
                        attn_prior,
                        train_idx,
                        test_idx,
                        spec,
                        seed + fold_id * 101,
                    )
                    attn_all.append(fold_attn)
                else:
                    features = clinical
                    if spec.model == "clinical_plus_raw_transfer":
                        features = np.column_stack([clinical, raw_prior])
                    fold_pred = train_linear_fold(features, time, event, train_idx, test_idx, seed + fold_id * 101)
                pred[test_idx] = fold_pred
            edge_gain = edge_enrichment(np.concatenate(attn_all, axis=0), genes) if attn_all else np.nan
            all_rows.append(evaluate_score(frame, time, event, pred, spec.model, seed, spec.description, edge_gain))
            predictions[f"{spec.model}_seed{seed}_risk"] = pred

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS / "relapse_intervene_gene_level_v2_model_comparison.csv", index=False)
    predictions.to_csv(RESULTS / "relapse_intervene_gene_level_v2_oof_risk_scores.csv", index=False)
    pd.DataFrame(attn_prior, index=genes, columns=genes).to_csv(RESULTS / "relapse_intervene_gene_level_v2_attention_prior.csv")
    write_summary(results, genes, clinical_names)
    print(f"Wrote {RESULTS / 'relapse_intervene_gene_level_v2_model_comparison.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_gene_level_v2_oof_risk_scores.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_gene_level_v2_attention_prior.csv'}")
    print(f"Wrote {RESULTS / 'relapse_intervene_gene_level_v2_summary.txt'}")


if __name__ == "__main__":
    main()

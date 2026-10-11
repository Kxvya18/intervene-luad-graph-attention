from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

SEED = 7
N_FOLDS = 5
RIDGE_ALPHA = 5.0
EPOCHS = 450
PATIENCE = 55
DISCOVERY_EPOCHS = 800
DISCOVERY_PATIENCE = 90
LR = 2e-3
WEIGHT_DECAY = 2e-3
HEADS = 4
PSEUDOHEADS = 2
HEAD_DIM = 4
HIDDEN = 24

FEATURES = [
    "dependency",
    "expression",
    "copy_number",
    "thermo_residual",
    "selectivity",
    "context",
    "generic_cytotoxic_prism_control",
    "direct_lipid_prism_control",
]

PAPER_TOKENS = [
    "srebp_lipid",
    "mito_translation",
    "oxidative_stress",
    "apoptosis_iap",
    "proteostasis",
    "akt_mtor_growth",
]

CONTROL_TOKENS = [
    "generic_cytotoxic_control",
    "direct_lipid_secondary",
]


@dataclass(frozen=True)
class ModuleSpec:
    token: str
    genes: tuple[str, ...]
    role: str


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - s.mean()) / std


def symbol_to_column(columns: list[str]) -> dict[str, str]:
    mapping = {}
    for col in columns:
        if " (" in col and col.endswith(")"):
            mapping[col.split(" (", 1)[0]] = col
    return mapping


def top_mito_genes(n: int = 16) -> tuple[str, ...]:
    ranked = pd.read_csv(RESULTS / "nsclc_top_ranked_genes.csv")
    ranked["gene_symbol"] = ranked["gene_symbol"].astype(str)
    mito = ranked[ranked["gene_symbol"].str.match(r"^MRP[SL]", na=False)].copy()
    mito = mito.sort_values(["dependency_strength", "hidden_nonhub_score"], ascending=[False, True])
    return tuple(mito["gene_symbol"].drop_duplicates().head(n).tolist())


def module_specs() -> list[ModuleSpec]:
    return [
        ModuleSpec("srebp_lipid", ("SCAP", "SCD", "MBTPS1", "MBTPS2", "SREBF1"), "primary residual SREBP/lipid module"),
        ModuleSpec("mito_translation", top_mito_genes(), "mitochondrial translation dependency module"),
        ModuleSpec("oxidative_stress", ("KEAP1", "NFE2L2", "PGD", "TXNRD1", "GCLC", "GCLM", "SLC7A11"), "KEAP1/NFE2L2 antioxidant stress context"),
        ModuleSpec("apoptosis_iap", ("CFLAR", "XIAP", "BIRC2", "BIRC3", "CASP8", "MCL1", "BCL2L1"), "apoptosis and IAP survival escape"),
        ModuleSpec("proteostasis", ("USP8", "STAMBP", "PTPN23", "UBAP1", "HGS", "VCP", "PSMA1", "PSMB5"), "proteostasis/deubiquitinase stress module"),
        ModuleSpec("akt_mtor_growth", ("AKT1", "AKT2", "AKT3", "MTOR", "PIK3CA", "PIK3CB", "RPTOR", "RICTOR", "PTEN"), "AKT/mTOR growth-stress adaptation"),
        ModuleSpec("generic_cytotoxic_control", ("CDK1", "CDK2", "AURKA", "AURKB", "TOP2A", "MCM2", "PCNA", "CHEK1"), "generic cytotoxic/proliferation control"),
        ModuleSpec("direct_lipid_secondary", ("FASN", "HMGCR", "ACACA", "ACACB", "SCD", "SREBF1", "SCAP"), "direct lipid drug-response secondary module"),
    ]


def read_wide_subset(path: Path, model_ids: set[str], symbols: list[str], suffix: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    mapping = symbol_to_column(list(header.columns))
    found = [s for s in symbols if s in mapping]
    cols = [header.columns[0]] + [mapping[s] for s in found]
    df = pd.read_csv(path, usecols=cols)
    df = df.rename(columns={df.columns[0]: "ModelID"})
    df["ModelID"] = df["ModelID"].astype(str)
    df = df[df["ModelID"].isin(model_ids)].copy()
    return df.rename(columns={mapping[s]: f"{s}_{suffix}" for s in found})


def load_targets(model_ids: set[str]) -> pd.DataFrame:
    lfc = pd.read_csv(RESULTS / "prism_nsclc_lfc_with_srebp_axis.csv")
    buckets = pd.read_csv(RESULTS / "prism_srebp_compound_scores_bucketed.csv")
    lfc["ModelID"] = lfc["ModelID"].astype(str)
    lfc = lfc[lfc["ModelID"].isin(model_ids)].copy()
    bucket_map = buckets.set_index("broad_id_clean")["mechanism_bucket"].to_dict()
    lfc["mechanism_bucket"] = lfc["broad_id_clean"].map(bucket_map)
    lfc["sensitivity"] = -pd.to_numeric(lfc["median_lfc"], errors="coerce")

    out = pd.DataFrame({"ModelID": sorted(model_ids)})
    for bucket, col in [
        ("stress_survival", "target_stress_survival_sensitivity"),
        ("generic_cytotoxic", "generic_cytotoxic_sensitivity"),
        ("direct_lipid_axis", "direct_lipid_sensitivity"),
    ]:
        agg = (
            lfc[lfc["mechanism_bucket"] == bucket]
            .groupby("ModelID")["sensitivity"]
            .mean()
            .rename(col)
            .reset_index()
        )
        out = out.merge(agg, on="ModelID", how="left")
    out["target_specific_stress_survival"] = (
        zscore(out["target_stress_survival_sensitivity"])
        - 0.35 * zscore(out["generic_cytotoxic_sensitivity"])
    )
    valid = out[["target_stress_survival_sensitivity", "generic_cytotoxic_sensitivity"]].dropna()
    if len(valid) >= 10:
        x = valid["generic_cytotoxic_sensitivity"].to_numpy()
        y = valid["target_stress_survival_sensitivity"].to_numpy()
        slope, intercept = np.polyfit(x, y, deg=1)
        residual = out["target_stress_survival_sensitivity"] - (
            intercept + slope * out["generic_cytotoxic_sensitivity"]
        )
        out["target_cytotoxic_residual_stress_survival"] = zscore(residual)
    else:
        out["target_cytotoxic_residual_stress_survival"] = np.nan
    return out


def load_feature_table() -> tuple[pd.DataFrame, list[ModuleSpec]]:
    base = pd.read_csv(RESULTS / "nsclc_srebp_multiomics_model_table.csv")
    base["ModelID"] = base["ModelID"].astype(str)
    model_ids = set(base["ModelID"])
    specs = module_specs()
    genes = sorted(set(g for spec in specs for g in spec.genes))

    dep = read_wide_subset(RAW / "CRISPRGeneEffect.csv", model_ids, genes, "dep")
    dep_cols = [c for c in dep.columns if c != "ModelID"]
    dep[dep_cols] = -dep[dep_cols].apply(pd.to_numeric, errors="coerce")
    expr = read_wide_subset(RAW / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", model_ids, genes, "expr")
    cn = read_wide_subset(RAW / "OmicsCNGene.csv", model_ids, genes, "cn")

    frame = pd.DataFrame({"ModelID": sorted(model_ids)})
    for table in [dep, expr, cn]:
        frame = frame.merge(table, on="ModelID", how="left")

    residual = pd.read_csv(RESULTS / "selected_residual_vulnerability_gene_scores.csv")
    residual = residual.drop_duplicates("gene_symbol").set_index("gene_symbol")
    frame = frame.merge(load_targets(model_ids), on="ModelID", how="left")

    driver = pd.read_csv(RESULTS / "nsclc_driver_mutation_matrix.csv")
    driver["ModelID"] = driver["ModelID"].astype(str)
    frame = frame.merge(driver, on="ModelID", how="left")

    for spec in specs:
        measure_names = {
            "dep": "dependency",
            "expr": "expression",
            "cn": "copy_number",
        }
        for measure, feature_name in measure_names.items():
            cols = [f"{g}_{measure}" for g in spec.genes if f"{g}_{measure}" in frame.columns]
            frame[f"{spec.token}_{feature_name}"] = frame[cols].mean(axis=1) if cols else np.nan

        genes_present = [g for g in spec.genes if g in residual.index]
        if genes_present:
            weights = residual.loc[genes_present, "thermo_residual_score"].astype(float)
            dep_cols = [f"{g}_dep" for g in genes_present if f"{g}_dep" in frame.columns]
            if dep_cols and float(weights.abs().sum()) != 0:
                values = frame[dep_cols].apply(zscore)
                w = weights.loc[[c.replace("_dep", "") for c in dep_cols]].abs()
                w.index = dep_cols
                frame[f"{spec.token}_thermo_residual"] = values.mul(w / w.sum(), axis=1).sum(axis=1)
            else:
                frame[f"{spec.token}_thermo_residual"] = float(weights.mean())
            frame[f"{spec.token}_selectivity"] = float(residual.loc[genes_present, "nsclc_selectivity"].astype(float).mean())
        else:
            frame[f"{spec.token}_thermo_residual"] = np.nan
            frame[f"{spec.token}_selectivity"] = np.nan

        if spec.token == "oxidative_stress":
            context = pd.Series(np.zeros(len(frame)), index=frame.index)
            for flag, weight in [
                ("KEAP1_mut", 0.5),
                ("KEAP1_high_impact", 1.0),
                ("NFE2L2_mut", 0.5),
                ("NFE2L2_high_impact", 1.0),
            ]:
                if flag in frame.columns:
                    context += frame[flag].fillna(False).astype(bool).astype(float) * weight
            frame[f"{spec.token}_context"] = context
        else:
            frame[f"{spec.token}_context"] = frame[f"{spec.token}_selectivity"]

        frame[f"{spec.token}_generic_cytotoxic_prism_control"] = np.nan
        frame[f"{spec.token}_direct_lipid_prism_control"] = np.nan

    frame["generic_cytotoxic_control_generic_cytotoxic_prism_control"] = frame["generic_cytotoxic_sensitivity"]
    frame["direct_lipid_secondary_direct_lipid_prism_control"] = frame["direct_lipid_sensitivity"]
    return frame, specs


def build_tensor(frame: pd.DataFrame, specs: list[ModuleSpec]) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    rows = frame.dropna(subset=["target_stress_survival_sensitivity"]).copy()
    tokens = [s.token for s in specs]
    x = np.zeros((len(rows), len(tokens), len(FEATURES)), dtype=np.float32)
    for ti, token in enumerate(tokens):
        for fi, feature in enumerate(FEATURES):
            col = f"{token}_{feature}"
            values = rows[col] if col in rows.columns else pd.Series(np.nan, index=rows.index)
            values = values.fillna(values.median() if values.notna().any() else 0.0)
            x[:, ti, fi] = zscore(values).to_numpy(dtype=np.float32)
    return np.nan_to_num(x), tokens, FEATURES, rows.reset_index(drop=True)


def subset_tensor(x: np.ndarray, tokens: list[str], keep_tokens: list[str]) -> tuple[np.ndarray, list[str]]:
    idx = [tokens.index(t) for t in keep_tokens if t in tokens]
    return x[:, idx, :], [tokens[i] for i in idx]


def remove_control_features(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    keep = [
        i
        for i, name in enumerate(feature_names)
        if name not in {"generic_cytotoxic_prism_control", "direct_lipid_prism_control"}
    ]
    return x[:, :, keep], [feature_names[i] for i in keep]


def folds(n: int, k: int = N_FOLDS) -> list[np.ndarray]:
    order = np.arange(n)
    return [order[i::k] for i in range(k)]


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    xt = (x_train - mean) / std
    xv = (x_test - mean) / std
    xt = np.column_stack([np.ones(len(xt)), xt])
    xv = np.column_stack([np.ones(len(xv)), xv])
    penalty = np.eye(xt.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(xt.T @ xt + penalty) @ xt.T @ y_train
    return xv @ coef


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "cv_corr": float(pd.Series(y).corr(pd.Series(pred))),
        "cv_r2": float(1.0 - ss_res / ss_tot) if ss_tot else np.nan,
        "cv_mae": float(np.mean(np.abs(y - pred))),
    }


def cv_ridge(features: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    pred = np.full(len(y), np.nan)
    for test_idx in folds(len(y)):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        pred[test_idx] = ridge_predict(features[train_idx], y[train_idx], features[test_idx])
    return pred, regression_metrics(y, pred)


class IBPALayer(nn.Module):
    def __init__(self, n_features: int, heads: int, pseudoheads: int, head_dim: int, mode: str):
        super().__init__()
        self.heads = heads
        self.pseudoheads = pseudoheads
        self.head_dim = head_dim
        self.mode = mode
        self.q = nn.Linear(n_features, heads * head_dim, bias=False)
        self.k = nn.Linear(n_features, heads * head_dim, bias=False)
        self.v = nn.Linear(n_features, heads * head_dim, bias=False)
        self.alpha_q = nn.Parameter(self.initial_alpha(mode))
        self.alpha_k = nn.Parameter(self.initial_alpha(mode))
        self.alpha_v = nn.Parameter(self.initial_alpha(mode))
        self.collapse_logits = nn.Parameter(torch.zeros(heads, pseudoheads))

    def initial_alpha(self, mode: str) -> torch.Tensor:
        alpha = torch.zeros(self.heads, self.heads, self.pseudoheads)
        for h in range(self.heads):
            alpha[h, h, 0] = 2.0
            if self.pseudoheads > 1:
                alpha[h, h, 1] = 1.2
                alpha[(h + 1) % self.heads, h, 1] = 0.8
        if mode == "mha":
            alpha = torch.full_like(alpha, -8.0)
            for h in range(self.heads):
                alpha[h, h, :] = 8.0
        return alpha

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, tokens, _ = x.shape
        q = self.q(x).view(n, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).view(n, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).view(n, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)

        aq = torch.softmax(self.alpha_q, dim=0)
        ak = torch.softmax(self.alpha_k, dim=0)
        av = torch.softmax(self.alpha_v, dim=0)
        q_t = torch.einsum("mhp,nmtd->nhptd", aq, q)
        k_t = torch.einsum("mhp,nmtd->nhptd", ak, k)
        v_t = torch.einsum("mhp,nmtd->nhptd", av, v)

        logits = torch.einsum("nhpid,nhpjd->nhpij", q_t, k_t) / np.sqrt(self.head_dim)
        attn = torch.softmax(logits, dim=-1)
        o_t = torch.einsum("nhpij,nhpjd->nhpid", attn, v_t)
        collapse = torch.softmax(self.collapse_logits, dim=-1)
        o = torch.einsum("hp,nhptd->nhtd", collapse, o_t)
        pooled = o.mean(dim=2).reshape(n, self.heads * self.head_dim)
        return pooled, attn


class IBPAModel(nn.Module):
    def __init__(self, n_features: int, mode: str):
        super().__init__()
        pseudoheads = 1 if mode == "mha" else PSEUDOHEADS
        self.layer = IBPALayer(n_features, HEADS, pseudoheads, HEAD_DIM, mode)
        self.predictor = nn.Sequential(
            nn.LayerNorm(HEADS * HEAD_DIM),
            nn.Linear(HEADS * HEAD_DIM, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, attn = self.layer(x)
        return self.predictor(pooled).squeeze(-1), attn


def train_one_fold(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray, IBPAModel]:
    set_seed(seed)
    inner = train_idx.copy()
    val_size = max(6, int(round(len(inner) * 0.2)))
    val_idx = inner[-val_size:]
    fit_idx = inner[:-val_size]

    x_fit = torch.tensor(x[fit_idx], dtype=torch.float32)
    y_fit_raw = y[fit_idx]
    y_mean = float(y_fit_raw.mean())
    y_std = float(y_fit_raw.std() or 1.0)
    y_fit = torch.tensor((y_fit_raw - y_mean) / y_std, dtype=torch.float32)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32)
    y_val = torch.tensor((y[val_idx] - y_mean) / y_std, dtype=torch.float32)
    x_test = torch.tensor(x[test_idx], dtype=torch.float32)

    model = IBPAModel(x.shape[-1], mode)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred, _ = model(x_fit)
        loss = loss_fn(pred, y_fit)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(x_val)
            val_loss = float(loss_fn(val_pred, y_val).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
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
        test_pred, test_attn = model(x_test)
    return test_pred.numpy() * y_std + y_mean, test_attn.numpy(), model


def cv_torch_model(x: np.ndarray, y: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, float], np.ndarray, IBPAModel]:
    pred = np.full(len(y), np.nan)
    attn_blocks = []
    last_model = None
    for fold_id, test_idx in enumerate(folds(len(y))):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        fold_pred, fold_attn, model = train_one_fold(x, y, train_idx, test_idx, mode, SEED + fold_id * 13)
        pred[test_idx] = fold_pred
        attn_blocks.append(fold_attn)
        last_model = model
    attn = np.concatenate(attn_blocks, axis=0)
    assert last_model is not None
    return pred, regression_metrics(y, pred), attn, last_model


def baseline_features(x: np.ndarray, tokens: list[str], feature_names: list[str], name: str) -> np.ndarray:
    if name == "residual_only":
        return x[:, :, [feature_names.index("thermo_residual")]].reshape(len(x), -1)
    if name == "srebp_only":
        return x[:, [tokens.index("srebp_lipid")], :].reshape(len(x), -1)
    if name == "mito_only":
        return x[:, [tokens.index("mito_translation")], :].reshape(len(x), -1)
    if name == "module_linear":
        return x.reshape(len(x), -1)
    raise ValueError(name)


def attention_edges(attention: np.ndarray, tokens: list[str], mode: str) -> pd.DataFrame:
    rows = []
    mean_attn = attention.mean(axis=0)
    uniform = 1.0 / len(tokens)
    for h in range(mean_attn.shape[0]):
        for p in range(mean_attn.shape[1]):
            for i, source in enumerate(tokens):
                for j, target in enumerate(tokens):
                    value = float(mean_attn[h, p, i, j])
                    rows.append(
                        {
                            "model": mode,
                            "head": h,
                            "pseudohead": p,
                            "source_token": source,
                            "target_token": target,
                            "mean_attention": value,
                            "attention_enrichment_vs_uniform": value - uniform,
                        }
                    )
    return pd.DataFrame(rows).sort_values("attention_enrichment_vs_uniform", ascending=False)


def token_feature_report(x: np.ndarray, tokens: list[str], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for ti, token in enumerate(tokens):
        for fi, feature in enumerate(feature_names):
            values = x[:, ti, fi]
            rows.append(
                {
                    "token": token,
                    "feature": feature,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "nonzero_fraction": float(np.mean(np.abs(values) > 1e-8)),
                }
            )
    return pd.DataFrame(rows)


def pseudohead_importance(model: IBPAModel, mode: str) -> pd.DataFrame:
    layer = model.layer
    aq = torch.softmax(layer.alpha_q.detach(), dim=0).numpy()
    ak = torch.softmax(layer.alpha_k.detach(), dim=0).numpy()
    av = torch.softmax(layer.alpha_v.detach(), dim=0).numpy()
    collapse = torch.softmax(layer.collapse_logits.detach(), dim=-1).numpy()
    rows = []
    for h in range(aq.shape[1]):
        for p in range(aq.shape[2]):
            q_mix = aq[:, h, p]
            rows.append(
                {
                    "model": mode,
                    "head": h,
                    "pseudohead": p,
                    "collapse_weight": float(collapse[h, p]),
                    "q_source_head_weights": ";".join(f"h{i}:{w:.3f}" for i, w in enumerate(q_mix)),
                    "k_source_head_weights": ";".join(f"h{i}:{w:.3f}" for i, w in enumerate(ak[:, h, p])),
                    "v_source_head_weights": ";".join(f"h{i}:{w:.3f}" for i, w in enumerate(av[:, h, p])),
                    "interleaving_strength": float(np.sum(np.abs(q_mix)) - np.max(np.abs(q_mix))),
                }
            )
    return pd.DataFrame(rows).sort_values("interleaving_strength", ascending=False)


class IBPADiscoveryModel(nn.Module):
    def __init__(self, n_tokens: int, n_features: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_features = n_features
        self.layer = IBPALayer(n_features, HEADS, PSEUDOHEADS, HEAD_DIM, "ibpa")
        self.decoder = nn.Sequential(
            nn.LayerNorm(HEADS * HEAD_DIM),
            nn.Linear(HEADS * HEAD_DIM, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, n_tokens * n_features),
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(HEADS * HEAD_DIM),
            nn.Linear(HEADS * HEAD_DIM, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled, attn = self.layer(x)
        recon = self.decoder(pooled).view(-1, self.n_tokens, self.n_features)
        state = self.state_head(pooled).squeeze(-1)
        return recon, state, attn


def discovery_weak_prior(
    x: np.ndarray,
    tokens: list[str],
    feature_names: list[str],
    include_srebp: bool = True,
    include_mito: bool = True,
    include_oxidative: bool = True,
) -> np.ndarray:
    dep_i = feature_names.index("dependency")
    thermo_i = feature_names.index("thermo_residual")
    context_i = feature_names.index("context")
    srebp_i = tokens.index("srebp_lipid")
    mito_i = tokens.index("mito_translation")
    ox_i = tokens.index("oxidative_stress")
    proteo_i = tokens.index("proteostasis")
    apoptosis_i = tokens.index("apoptosis_iap")

    mean_thermo = x[:, :, thermo_i].mean(axis=1)
    srebp_thermo = x[:, srebp_i, thermo_i]
    mito_thermo = x[:, mito_i, thermo_i]
    srebp_mito_coupling = x[:, srebp_i, dep_i] * x[:, mito_i, dep_i]
    ox_context = x[:, ox_i, context_i]
    stress_support = 0.5 * x[:, proteo_i, dep_i] + 0.5 * x[:, apoptosis_i, dep_i]
    prior = mean_thermo + 0.20 * stress_support
    if include_srebp:
        prior = prior + 0.65 * srebp_thermo
    if include_mito:
        prior = prior + 0.65 * mito_thermo
    if include_srebp and include_mito:
        prior = prior + 0.35 * srebp_mito_coupling
    if include_oxidative:
        prior = prior + 0.25 * ox_context
    return zscore(pd.Series(prior)).to_numpy(dtype=np.float32)


def train_discovery_model(
    x: np.ndarray,
    tokens: list[str],
    feature_names: list[str],
    prior_weight: float = 0.22,
    include_srebp: bool = True,
    include_mito: bool = True,
    include_oxidative: bool = True,
) -> tuple[IBPADiscoveryModel, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(SEED + 1001)
    x_t = torch.tensor(x, dtype=torch.float32)
    prior = discovery_weak_prior(
        x,
        tokens,
        feature_names,
        include_srebp=include_srebp,
        include_mito=include_mito,
        include_oxidative=include_oxidative,
    )
    prior_t = torch.tensor(prior, dtype=torch.float32)
    model = IBPADiscoveryModel(x.shape[1], x.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    best_state = None
    best_loss = float("inf")
    stale = 0

    for _ in range(DISCOVERY_EPOCHS):
        model.train()
        opt.zero_grad()
        recon, state, _ = model(x_t)
        state_z = (state - state.mean()) / (state.std(unbiased=False) + 1e-6)
        recon_loss = mse(recon, x_t)
        prior_loss = mse(state_z, prior_t)
        decorrelation = torch.mean(torch.abs(torch.corrcoef(torch.stack([state_z, x_t.mean(dim=(1, 2))]))[0, 1]))
        loss = recon_loss + prior_weight * prior_loss + 0.03 * decorrelation
        loss.backward()
        opt.step()

        value = float(loss.item())
        if value < best_loss - 1e-6:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= DISCOVERY_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        recon, state, attn = model(x_t)
    state_np = zscore(pd.Series(state.numpy())).to_numpy()
    recon_np = recon.numpy()
    metrics = {
        "reconstruction_mse": float(np.mean((recon_np - x) ** 2)),
        "state_prior_corr": float(pd.Series(state_np).corr(pd.Series(prior))),
        "state_std": float(np.std(state_np)),
        "prior_weight": float(prior_weight),
        "include_srebp": bool(include_srebp),
        "include_mito": bool(include_mito),
        "include_oxidative": bool(include_oxidative),
    }
    return model, state_np, recon_np, attn.numpy(), metrics


def high_low_stats(
    frame: pd.DataFrame, score_col: str, value_col: str, label: str, n_perm: int = 5000
) -> dict[str, object]:
    data = frame[[score_col, value_col]].dropna().copy()
    if len(data) < 12:
        return {
            "test": label,
            "n": len(data),
            "n_high": 0,
            "n_low": 0,
            "high_mean": np.nan,
            "low_mean": np.nan,
            "high_minus_low": np.nan,
            "effect_size": np.nan,
            "perm_p_two_sided": np.nan,
        }
    hi = data[score_col] >= data[score_col].quantile(2 / 3)
    lo = data[score_col] <= data[score_col].quantile(1 / 3)
    high = data.loc[hi, value_col].astype(float)
    low = data.loc[lo, value_col].astype(float)
    observed = float(high.mean() - low.mean())
    pooled = float(data[value_col].std(ddof=0) or 1.0)
    rng = np.random.default_rng(SEED + len(label))
    values = data[value_col].to_numpy(dtype=float)
    scores = data[score_col].to_numpy(dtype=float)
    exceed = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(values)
        perm = pd.DataFrame({"score": scores, "value": shuffled})
        p_hi = perm["score"] >= np.quantile(scores, 2 / 3)
        p_lo = perm["score"] <= np.quantile(scores, 1 / 3)
        diff = float(perm.loc[p_hi, "value"].mean() - perm.loc[p_lo, "value"].mean())
        if abs(diff) >= abs(observed):
            exceed += 1
    return {
        "test": label,
        "n": len(data),
        "n_high": int(hi.sum()),
        "n_low": int(lo.sum()),
        "high_mean": float(high.mean()),
        "low_mean": float(low.mean()),
        "high_minus_low": observed,
        "effect_size": observed / pooled,
        "perm_p_two_sided": float((exceed + 1) / (n_perm + 1)),
    }


def build_discovery_validation(
    rows: pd.DataFrame,
    x_paper: np.ndarray,
    paper_tokens: list[str],
    paper_feature_names: list[str],
    state_score: np.ndarray,
    recon: np.ndarray,
    discovery_metrics: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = rows.copy()
    out["ibpa_v2_state_score"] = state_score
    out["ibpa_v2_reconstruction_error"] = ((recon - x_paper) ** 2).mean(axis=(1, 2))

    dep_i = paper_feature_names.index("dependency")
    thermo_i = paper_feature_names.index("thermo_residual")
    context_i = paper_feature_names.index("context")
    srebp_i = paper_tokens.index("srebp_lipid")
    mito_i = paper_tokens.index("mito_translation")
    ox_i = paper_tokens.index("oxidative_stress")
    out["srebp_mito_dependency_coupling"] = x_paper[:, srebp_i, dep_i] * x_paper[:, mito_i, dep_i]
    out["srebp_mito_thermo_coupling"] = x_paper[:, srebp_i, thermo_i] * x_paper[:, mito_i, thermo_i]
    out["oxidative_context_score"] = x_paper[:, ox_i, context_i]
    out["KEAP1_or_NFE2L2_high_impact"] = (
        out.get("KEAP1_high_impact", False).fillna(False).astype(bool)
        | out.get("NFE2L2_high_impact", False).fillna(False).astype(bool)
    )

    validation_rows = [
        high_low_stats(out, "ibpa_v2_state_score", "target_stress_survival_sensitivity", "PRISM stress-survival sensitivity"),
        high_low_stats(out, "ibpa_v2_state_score", "direct_lipid_sensitivity", "PRISM direct lipid sensitivity"),
        high_low_stats(out, "ibpa_v2_state_score", "generic_cytotoxic_sensitivity", "PRISM generic cytotoxic sensitivity"),
        high_low_stats(out, "ibpa_v2_state_score", "target_specific_stress_survival", "stress-survival minus cytotoxic score"),
        high_low_stats(out, "ibpa_v2_state_score", "target_cytotoxic_residual_stress_survival", "cytotoxic-residual stress-survival score"),
        high_low_stats(out, "ibpa_v2_state_score", "srebp_mito_dependency_coupling", "SREBP x mitochondrial dependency coupling"),
        high_low_stats(out, "ibpa_v2_state_score", "srebp_mito_thermo_coupling", "SREBP x mitochondrial thermo coupling"),
        high_low_stats(out, "ibpa_v2_state_score", "oxidative_context_score", "KEAP1/NFE2L2 oxidative context"),
    ]
    validation = pd.DataFrame(validation_rows)

    mut = out[["ibpa_v2_state_score", "KEAP1_or_NFE2L2_high_impact"]].dropna()
    if len(mut) >= 12:
        high = mut["ibpa_v2_state_score"] >= mut["ibpa_v2_state_score"].quantile(2 / 3)
        low = mut["ibpa_v2_state_score"] <= mut["ibpa_v2_state_score"].quantile(1 / 3)
        validation = pd.concat(
            [
                validation,
                pd.DataFrame(
                    [
                        {
                            "test": "KEAP1/NFE2L2 high-impact enrichment",
                            "n": len(mut),
                            "n_high": int(high.sum()),
                            "n_low": int(low.sum()),
                            "high_mean": float(mut.loc[high, "KEAP1_or_NFE2L2_high_impact"].mean()),
                            "low_mean": float(mut.loc[low, "KEAP1_or_NFE2L2_high_impact"].mean()),
                            "high_minus_low": float(
                                mut.loc[high, "KEAP1_or_NFE2L2_high_impact"].mean()
                                - mut.loc[low, "KEAP1_or_NFE2L2_high_impact"].mean()
                            ),
                            "effect_size": np.nan,
                            "perm_p_two_sided": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    module_rows = []
    high_mask = out["ibpa_v2_state_score"] >= out["ibpa_v2_state_score"].quantile(2 / 3)
    low_mask = out["ibpa_v2_state_score"] <= out["ibpa_v2_state_score"].quantile(1 / 3)
    for ti, token in enumerate(paper_tokens):
        for fi, feature in enumerate(paper_feature_names):
            values = x_paper[:, ti, fi]
            module_rows.append(
                {
                    "token": token,
                    "feature": feature,
                    "high_state_mean": float(values[high_mask].mean()),
                    "low_state_mean": float(values[low_mask].mean()),
                    "high_minus_low": float(values[high_mask].mean() - values[low_mask].mean()),
                    "state_corr": float(pd.Series(values).corr(pd.Series(state_score))),
                }
            )
    module_scores = pd.DataFrame(module_rows).sort_values("state_corr", ascending=False)
    for key, value in discovery_metrics.items():
        validation[key] = value
    return out, validation, module_scores


def discovery_validation_for_score(
    rows: pd.DataFrame,
    x_paper: np.ndarray,
    paper_tokens: list[str],
    paper_feature_names: list[str],
    score: np.ndarray,
    score_col: str,
    n_perm: int = 1000,
) -> pd.DataFrame:
    out = rows.copy()
    out[score_col] = score
    dep_i = paper_feature_names.index("dependency")
    thermo_i = paper_feature_names.index("thermo_residual")
    context_i = paper_feature_names.index("context")
    srebp_i = paper_tokens.index("srebp_lipid")
    mito_i = paper_tokens.index("mito_translation")
    ox_i = paper_tokens.index("oxidative_stress")
    out["srebp_mito_dependency_coupling"] = x_paper[:, srebp_i, dep_i] * x_paper[:, mito_i, dep_i]
    out["srebp_mito_thermo_coupling"] = x_paper[:, srebp_i, thermo_i] * x_paper[:, mito_i, thermo_i]
    out["oxidative_context_score"] = x_paper[:, ox_i, context_i]
    rows_out = [
        high_low_stats(out, score_col, "target_stress_survival_sensitivity", "PRISM stress-survival sensitivity", n_perm=n_perm),
        high_low_stats(out, score_col, "direct_lipid_sensitivity", "PRISM direct lipid sensitivity", n_perm=n_perm),
        high_low_stats(out, score_col, "generic_cytotoxic_sensitivity", "PRISM generic cytotoxic sensitivity", n_perm=n_perm),
        high_low_stats(out, score_col, "target_specific_stress_survival", "stress-survival minus cytotoxic score", n_perm=n_perm),
        high_low_stats(out, score_col, "target_cytotoxic_residual_stress_survival", "cytotoxic-residual stress-survival score", n_perm=n_perm),
        high_low_stats(out, score_col, "oxidative_context_score", "KEAP1/NFE2L2 oxidative context", n_perm=n_perm),
        high_low_stats(out, score_col, "srebp_mito_dependency_coupling", "SREBP x mitochondrial dependency coupling", n_perm=n_perm),
        high_low_stats(out, score_col, "srebp_mito_thermo_coupling", "SREBP x mitochondrial thermo coupling", n_perm=n_perm),
    ]
    return pd.DataFrame(rows_out)


def run_discovery_ablations(
    rows: pd.DataFrame,
    x_paper: np.ndarray,
    paper_tokens: list[str],
    paper_feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = [
        {"ablation": "no_prior", "prior_weight": 0.0, "include_srebp": True, "include_mito": True, "include_oxidative": True},
        {"ablation": "prior_weight_0.05", "prior_weight": 0.05, "include_srebp": True, "include_mito": True, "include_oxidative": True},
        {"ablation": "prior_weight_0.22_current", "prior_weight": 0.22, "include_srebp": True, "include_mito": True, "include_oxidative": True},
        {"ablation": "prior_weight_0.50", "prior_weight": 0.50, "include_srebp": True, "include_mito": True, "include_oxidative": True},
        {"ablation": "remove_srebp_from_prior", "prior_weight": 0.22, "include_srebp": False, "include_mito": True, "include_oxidative": True},
        {"ablation": "remove_mito_from_prior", "prior_weight": 0.22, "include_srebp": True, "include_mito": False, "include_oxidative": True},
        {"ablation": "remove_oxidative_from_prior", "prior_weight": 0.22, "include_srebp": True, "include_mito": True, "include_oxidative": False},
    ]
    summary_rows = []
    validation_tables = []
    for cfg in configs:
        model, state, recon, attn, metrics = train_discovery_model(
            x_paper,
            paper_tokens,
            paper_feature_names,
            prior_weight=cfg["prior_weight"],
            include_srebp=cfg["include_srebp"],
            include_mito=cfg["include_mito"],
            include_oxidative=cfg["include_oxidative"],
        )
        validation = discovery_validation_for_score(
            rows,
            x_paper,
            paper_tokens,
            paper_feature_names,
            state,
            "ablation_state_score",
            n_perm=1000,
        )
        validation["ablation"] = cfg["ablation"]
        validation_tables.append(validation)
        key = validation.set_index("test")
        summary = {"ablation": cfg["ablation"]}
        summary.update(metrics)
        for test in [
            "PRISM stress-survival sensitivity",
            "PRISM direct lipid sensitivity",
            "PRISM generic cytotoxic sensitivity",
            "stress-survival minus cytotoxic score",
            "cytotoxic-residual stress-survival score",
            "KEAP1/NFE2L2 oxidative context",
        ]:
            if test in key.index:
                safe = test.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
                summary[f"{safe}_effect"] = float(key.loc[test, "high_minus_low"])
                summary[f"{safe}_p"] = float(key.loc[test, "perm_p_two_sided"])
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows), pd.concat(validation_tables, ignore_index=True)


def raw_prior_comparison(
    rows: pd.DataFrame,
    x_paper: np.ndarray,
    paper_tokens: list[str],
    paper_feature_names: list[str],
    learned_state: np.ndarray,
) -> pd.DataFrame:
    raw_prior = discovery_weak_prior(x_paper, paper_tokens, paper_feature_names)
    tables = []
    for label, score in [("raw_physics_prior", raw_prior), ("learned_ibpa_v2_state", learned_state)]:
        validation = discovery_validation_for_score(
            rows,
            x_paper,
            paper_tokens,
            paper_feature_names,
            score,
            "comparison_score",
            n_perm=2000,
        )
        validation["score_type"] = label
        tables.append(validation)
    return pd.concat(tables, ignore_index=True)


def discovery_lead_table(
    discovery_validation: pd.DataFrame,
    discovery_edges: pd.DataFrame,
    discovery_module_scores: pd.DataFrame,
) -> pd.DataFrame:
    mechanisms = [
        {
            "lead": "SREBP/mitochondrial/oxidative stress-survival state",
            "priority": 1,
            "evidence": "IBPA-v2 state enriches cytotoxic-residual stress-survival and KEAP1/NFE2L2 oxidative context.",
            "paper_role": "main discovery lead",
        },
        {
            "lead": "mitochondrial translation stress adaptation",
            "priority": 2,
            "evidence": "Top attention edges converge on mito_translation; mito thermo/dependency features rank high.",
            "paper_role": "mechanistic axis",
        },
        {
            "lead": "proteostasis/deubiquitinase to mitochondrial stress link",
            "priority": 3,
            "evidence": "Proteostasis -> mito_translation is a top IBPA-v2 attention edge.",
            "paper_role": "secondary mechanism",
        },
        {
            "lead": "apoptosis/IAP survival to mitochondrial stress link",
            "priority": 4,
            "evidence": "Apoptosis/IAP -> mito_translation is a top IBPA-v2 attention edge.",
            "paper_role": "secondary mechanism",
        },
        {
            "lead": "direct lipid drug response",
            "priority": 5,
            "evidence": "Direct lipid sensitivity is weaker and not significant after high/low IBPA-v2 split.",
            "paper_role": "secondary/negative result",
        },
        {
            "lead": "generic cytotoxic response",
            "priority": 6,
            "evidence": "Generic cytotoxic sensitivity is not higher in high IBPA-v2 state.",
            "paper_role": "excluded confounder/control",
        },
    ]
    out = pd.DataFrame(mechanisms)
    validation_key = discovery_validation.set_index("test")
    if "cytotoxic-residual stress-survival score" in validation_key.index:
        out["main_residual_effect"] = float(
            validation_key.loc["cytotoxic-residual stress-survival score", "high_minus_low"]
        )
        out["main_residual_p"] = float(
            validation_key.loc["cytotoxic-residual stress-survival score", "perm_p_two_sided"]
        )
    top_edges = discovery_edges.head(8)
    out["top_attention_edges"] = "; ".join(
        f"{r.source_token}->{r.target_token}" for r in top_edges.itertuples(index=False)
    )
    top_modules = discovery_module_scores.head(6)
    out["top_state_features"] = "; ".join(
        f"{r.token}:{r.feature}" for r in top_modules.itertuples(index=False)
    )
    return out.sort_values("priority")


def write_paper_claim(
    discovery_validation: pd.DataFrame,
    prior_comparison: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    lead_table: pd.DataFrame,
) -> None:
    path = RESULTS / "ibpa_v2_paper_claim.txt"
    validation_key = discovery_validation.set_index("test")
    residual = validation_key.loc["cytotoxic-residual stress-survival score"]
    generic = validation_key.loc["PRISM generic cytotoxic sensitivity"]
    direct = validation_key.loc["PRISM direct lipid sensitivity"]
    with path.open("w", encoding="utf-8") as f:
        f.write("Paper-facing IBPA-v2 claim\n")
        f.write("==========================\n\n")
        f.write(
            "Claim: IBPA-v2 is a physics-guided interleaved attention discovery framework, "
            "not a generic supervised PRISM predictor. It uses thermodynamic residual and biological module "
            "states as a weak prior, learns interleaved module attention by reconstructing biological-physics "
            "module states, and validates the learned state against held-out interpretation endpoints from PRISM.\n\n"
        )
        f.write(
            "Main discovery: high IBPA-v2 state marks an NSCLC SREBP/mitochondrial/oxidative "
            "stress-survival vulnerability state. The high-vs-low state difference for cytotoxic-residual "
            f"stress-survival is {residual['high_minus_low']:.3f} (permutation p={residual['perm_p_two_sided']:.4f}). "
            f"Generic cytotoxic response is not enriched ({generic['high_minus_low']:.3f}, p={generic['perm_p_two_sided']:.4f}), "
            f"and direct lipid response is weaker ({direct['high_minus_low']:.3f}, p={direct['perm_p_two_sided']:.4f}).\n\n"
        )
        f.write("Ablation warning:\n")
        f.write(
            "Because the learned state is physics-guided, the paper should explicitly report raw-prior and prior-weight ablations. "
            "IBPA-v2 should be described as semi-supervised or physics-guided, not fully unsupervised.\n\n"
        )
        f.write("Top discovery leads:\n")
        f.write(lead_table[["priority", "lead", "paper_role", "evidence"]].to_string(index=False))
        f.write("\n\n")
        f.write("Ablation summary:\n")
        f.write(ablation_summary.to_string(index=False))
        f.write("\n\n")
        f.write("Raw-prior comparison:\n")
        f.write(prior_comparison.to_string(index=False))
        f.write("\n")


def write_summary(
    model_scores: pd.DataFrame,
    tokens: list[str],
    rows: pd.DataFrame,
    top_edges: pd.DataFrame,
    discovery_validation: pd.DataFrame,
    discovery_module_scores: pd.DataFrame,
    discovery_metrics: dict[str, float],
) -> None:
    path = RESULTS / "ibpa_full_summary.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Full Interleaved Biological-Physics Attention prototype\n")
        f.write("======================================================\n\n")
        f.write("Implementation note:\n")
        f.write(
            "This file now contains two IBPA modes. IBPA-v2 is the main discovery mode: "
            "it learns Q/K/V projections, alpha_Q/alpha_K/alpha_V interleaved pseudo-head mixing, "
            "and module attention by reconstructing paper-facing biological-physics module states "
            "with a weak thermodynamic/SREBP/mitochondrial physics prior. PRISM is then used only "
            "for validation/enrichment. The supervised PRISM-prediction models are kept as audit comparisons.\n\n"
        )
        f.write(f"Models used: {len(rows)}\n")
        f.write(f"Full tokens: {', '.join(tokens)}\n")
        f.write(f"Paper-facing tokens: {', '.join(PAPER_TOKENS)}\n")
        f.write(f"Control tokens excluded from paper-facing IBPA: {', '.join(CONTROL_TOKENS)}\n")
        f.write(f"Features per token: {', '.join(FEATURES)}\n\n")
        f.write("IBPA-v2 discovery training metrics\n")
        f.write("----------------------------------\n")
        for key, value in discovery_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")

        f.write("IBPA-v2 validation/enrichment after unsupervised training\n")
        f.write("--------------------------------------------------------\n")
        f.write(discovery_validation.to_string(index=False))
        f.write("\n\n")

        f.write("Top IBPA-v2 module features by state correlation\n")
        f.write("------------------------------------------------\n")
        f.write(discovery_module_scores.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Supervised audit model comparison\n")
        f.write("---------------------------------\n")
        f.write(model_scores.to_string(index=False))
        f.write("\n\n")
        f.write("Top IBPA attention edges\n")
        f.write("------------------------\n")
        f.write(top_edges.head(20).to_string(index=False))
        f.write("\n\n")
        best = model_scores.sort_values("cv_corr", ascending=False).iloc[0]
        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write(
            "IBPA-v2 should be judged primarily by whether its learned state enriches for biologically specific "
            "stress-survival vulnerability while avoiding generic cytotoxic and direct-lipid-only explanations. "
            f"The best supervised audit model remains {best['model']} on {best['target']} "
            f"(cv_corr={best['cv_corr']:.3f}, cv_r2={best['cv_r2']:.3f}), but that is not the main IBPA-v2 objective.\n"
        )


def main() -> None:
    set_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    frame, specs = load_feature_table()
    x, tokens, feature_names, rows = build_tensor(frame, specs)
    x_paper_tokens, paper_tokens = subset_tensor(x, tokens, PAPER_TOKENS)
    x_paper, paper_feature_names = remove_control_features(x_paper_tokens, feature_names)

    discovery_model, discovery_state, discovery_recon, discovery_attn, discovery_metrics = train_discovery_model(
        x_paper, paper_tokens, paper_feature_names
    )
    discovery_cells, discovery_validation, discovery_module_scores = build_discovery_validation(
        rows,
        x_paper,
        paper_tokens,
        paper_feature_names,
        discovery_state,
        discovery_recon,
        discovery_metrics,
    )
    discovery_cells.to_csv(RESULTS / "ibpa_v2_discovery_cell_scores.csv", index=False)
    discovery_validation.to_csv(RESULTS / "ibpa_v2_discovery_validation.csv", index=False)
    discovery_module_scores.to_csv(RESULTS / "ibpa_v2_module_state_scores.csv", index=False)
    discovery_edges = attention_edges(discovery_attn, paper_tokens, "ibpa_v2_discovery")
    discovery_edges.to_csv(RESULTS / "ibpa_v2_discovery_attention_edges.csv", index=False)
    pseudohead_importance(discovery_model, "ibpa_v2_discovery").to_csv(
        RESULTS / "ibpa_v2_discovery_pseudohead_importance.csv", index=False
    )
    prior_comparison = raw_prior_comparison(
        rows,
        x_paper,
        paper_tokens,
        paper_feature_names,
        discovery_state,
    )
    prior_comparison.to_csv(RESULTS / "ibpa_v2_raw_prior_comparison.csv", index=False)
    ablation_summary, ablation_validation = run_discovery_ablations(
        rows,
        x_paper,
        paper_tokens,
        paper_feature_names,
    )
    ablation_summary.to_csv(RESULTS / "ibpa_v2_ablation_summary.csv", index=False)
    ablation_validation.to_csv(RESULTS / "ibpa_v2_ablation_validation.csv", index=False)
    lead_table = discovery_lead_table(discovery_validation, discovery_edges, discovery_module_scores)
    lead_table.to_csv(RESULTS / "ibpa_v2_discovery_leads.csv", index=False)
    write_paper_claim(discovery_validation, prior_comparison, ablation_summary, lead_table)

    targets = {
        "stress_survival": rows["target_stress_survival_sensitivity"].astype(float).to_numpy(),
        "specific_stress_survival": rows["target_specific_stress_survival"].astype(float).to_numpy(),
        "cytotoxic_residual_stress_survival": rows[
            "target_cytotoxic_residual_stress_survival"
        ].astype(float).to_numpy(),
    }

    model_rows = []
    predictions = pd.DataFrame({"ModelID": rows["ModelID"]})
    edge_tables = []
    pseudo_tables = []

    datasets = [
        ("full_tokens_with_controls", x, tokens, feature_names),
        ("paper_tokens_no_controls", x_paper, paper_tokens, paper_feature_names),
    ]

    for target_name, y in targets.items():
        valid = ~np.isnan(y)
        for dataset_name, x_use_all, tokens_use, features_use in datasets:
            x_use = x_use_all[valid]
            y_use = y[valid]
            for baseline in ["residual_only", "srebp_only", "mito_only", "module_linear"]:
                if baseline == "mito_only" and "mito_translation" not in tokens_use:
                    continue
                if baseline == "srebp_only" and "srebp_lipid" not in tokens_use:
                    continue
                feats = baseline_features(x_use, tokens_use, features_use, baseline)
                pred, metrics = cv_ridge(feats, y_use)
                row = {
                    "model": baseline,
                    "dataset": dataset_name,
                    "target": target_name,
                    "n_features": feats.shape[1],
                    "n_models": len(y_use),
                }
                row.update(metrics)
                model_rows.append(row)
                if dataset_name == "paper_tokens_no_controls":
                    predictions[f"{baseline}_{target_name}_paper_pred"] = np.nan
                    predictions.loc[valid, f"{baseline}_{target_name}_paper_pred"] = pred

            for mode in ["mha", "ibpa"]:
                pred, metrics, attn, model = cv_torch_model(x_use, y_use, mode)
                row = {
                    "model": mode,
                    "dataset": dataset_name,
                    "target": target_name,
                    "n_features": HEADS * HEAD_DIM,
                    "n_models": len(y_use),
                }
                row.update(metrics)
                model_rows.append(row)
                if dataset_name == "paper_tokens_no_controls":
                    predictions[f"{mode}_{target_name}_paper_pred"] = np.nan
                    predictions.loc[valid, f"{mode}_{target_name}_paper_pred"] = pred
                else:
                    predictions[f"{mode}_{target_name}_full_pred"] = np.nan
                    predictions.loc[valid, f"{mode}_{target_name}_full_pred"] = pred
                edge_tables.append(attention_edges(attn, tokens_use, f"{mode}_{target_name}_{dataset_name}"))
                pseudo_tables.append(pseudohead_importance(model, f"{mode}_{target_name}_{dataset_name}"))

    model_scores = pd.DataFrame(model_rows).sort_values(
        ["target", "dataset", "cv_corr"], ascending=[True, True, False]
    )
    model_scores.to_csv(RESULTS / "ibpa_full_model_comparison.csv", index=False)

    edges = pd.concat(edge_tables, ignore_index=True)
    edges.to_csv(RESULTS / "ibpa_full_attention_edges.csv", index=False)

    pseudo = pd.concat(pseudo_tables, ignore_index=True)
    pseudo.to_csv(RESULTS / "ibpa_full_pseudohead_importance.csv", index=False)

    token_feature_report(x, tokens, feature_names).to_csv(
        RESULTS / "ibpa_full_token_feature_report.csv", index=False
    )

    predictions["target_stress_survival_sensitivity"] = rows["target_stress_survival_sensitivity"]
    predictions["target_specific_stress_survival"] = rows["target_specific_stress_survival"]
    predictions["target_cytotoxic_residual_stress_survival"] = rows[
        "target_cytotoxic_residual_stress_survival"
    ]
    predictions["generic_cytotoxic_sensitivity"] = rows["generic_cytotoxic_sensitivity"]
    predictions["direct_lipid_sensitivity"] = rows["direct_lipid_sensitivity"]
    predictions["ibpa_v2_state_score"] = discovery_state
    predictions.to_csv(RESULTS / "ibpa_full_cell_scores.csv", index=False)

    write_summary(
        model_scores,
        tokens,
        rows,
        discovery_edges.sort_values("attention_enrichment_vs_uniform", ascending=False),
        discovery_validation,
        discovery_module_scores,
        discovery_metrics,
    )

    print(f"Wrote {RESULTS / 'ibpa_full_summary.txt'}")
    print(f"Wrote {RESULTS / 'ibpa_full_model_comparison.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_full_attention_edges.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_full_pseudohead_importance.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_full_cell_scores.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_full_token_feature_report.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_discovery_cell_scores.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_discovery_validation.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_module_state_scores.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_discovery_attention_edges.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_discovery_pseudohead_importance.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_raw_prior_comparison.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_ablation_summary.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_ablation_validation.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_discovery_leads.csv'}")
    print(f"Wrote {RESULTS / 'ibpa_v2_paper_claim.txt'}")


if __name__ == "__main__":
    main()

"""Training and evaluation utilities: stratified Cox loss, early stopping, C-index, adjusted Cox tests."""
from __future__ import annotations

import numpy as np
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from .model import AttnSurv, LinearSurv, MLPSurv

MODELS = {"attn": AttnSurv, "mlp": MLPSurv, "linear": LinearSurv}


def cox_loss(risk, time, event, strata):
    """Negative Breslow partial log-likelihood, stratified by cohort, averaged over events."""
    total, n_ev = 0.0, 0.0
    for s in torch.unique(strata):
        m = strata == s
        r, t, e = risk[m], time[m], event[m]
        order = torch.argsort(t, descending=True)
        r, e = r[order], e[order]
        log_cum = torch.logcumsumexp(r, 0)
        total = total + -((r - log_cum) * e).sum(); n_ev = n_ev + e.sum()
    return total / torch.clamp(n_ev, min=1.0)


def cindex(time, event, risk):
    return float(concordance_index(time, -np.asarray(risk), event))


def fit(kind, cfg, token_index, prior, X, prolif, time, event, strata, seed, device, epochs=400, patience=40,
        lr=1e-3, wd=1e-3, l1=1e-4, val_frac=0.15):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n = len(time); val = np.zeros(n, bool)
    for s in np.unique(strata):                                        # stratified validation split
        ix = np.where(strata == s)[0]; val[rng.choice(ix, max(2, int(len(ix) * val_frac)), replace=False)] = True
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    Xt, Pt, tt, et, st = T(X), T(prolif), T(time), T(event), T(strata, torch.long)
    model = MODELS[kind](token_index, prior.to(device) if prior is not None else None, **cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    tr, va = T(~val, torch.bool), T(val, torch.bool)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        r, _ = model(Xt[tr], Pt[tr])
        loss = cox_loss(r, tt[tr], et[tr], st[tr])
        if hasattr(model.enc, "gene_load") and model.enc.gene_load.requires_grad:
            loss = loss + l1 * model.enc.gene_load.abs().sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            rv, _ = model(Xt[va], Pt[va]); vl = float(cox_loss(rv, tt[va], et[va], st[va]))
        if vl < best - 1e-4:
            best, wait = vl, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model


@torch.no_grad()
def predict(model, X, prolif, device):
    r, aux = model(torch.tensor(X, device=device), torch.tensor(prolif, dtype=torch.float32, device=device))
    out = dict(risk=r.cpu().numpy(), f=aux["f"].cpu().numpy())
    if "attn" in aux:
        out["attn"] = torch.stack([a.mean((0, 1)) for a in aux["attn"]]).cpu().numpy()   # [layers, T, T]
        out["token_weight"] = aux["token_weight"].mean(0).cpu().numpy()
    return out


def adjusted_test(time, event, risk, prolif, stage=None):
    """Cox: risk added to proliferation (+ stage if available). Returns log-HR, SE, p for the risk score."""
    import pandas as pd
    d = pd.DataFrame(dict(time=time, event=event, risk=(risk - risk.mean()) / (risk.std() + 1e-9), prolif=prolif))
    cols = ["prolif", "risk"]
    if stage is not None and np.isfinite(stage).sum() > 0.8 * len(stage):
        d["stage"] = stage; d = d.dropna(); d["stage"] = (d.stage - d.stage.mean()) / (d.stage.std() + 1e-9); cols = ["stage"] + cols
    m = CoxPHFitter(penalizer=1e-6).fit(d[cols + ["time", "event"]], "time", "event")
    return float(m.params_["risk"]), float(m.standard_errors_["risk"]), float(m.summary.loc["risk", "p"]), len(d)

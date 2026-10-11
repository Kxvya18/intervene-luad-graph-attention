"""
INTERVENE-2 models.

Genes -> pathway tokens (GeneToToken) -> interleaved-head attention over tokens (IHA blocks, Meta-IHA-inspired head
mixing through pseudo-heads) with a soft biological graph bias -> attention-pooled risk.  Proliferation can enter as
an explicit Cox covariate (prolif_offset), so the network is pushed to model relapse biology beyond proliferation.

Comparators share every component except the one being tested:
  mode="iha"  interleaved heads (INTERVENE-2)          mode="mha"  standard multi-head attention (no head mixing)
  graph_bias  on / off                                   MLPSurv: same token encoder, no attention
  LinearSurv: Cox on token mean scores (classical)       risk = prolif only: handled in train.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GeneToToken(nn.Module):
    """Per-token learned gene weighting. Token state = identity embedding + projection of
    [plain mean, attention-weighted mean, weighted SD, signed sparse loading] of the token's genes."""

    def __init__(self, token_index: list[list[int]], d: int):
        super().__init__()
        T, L = len(token_index), max(len(t) for t in token_index)
        idx = torch.zeros(T, L, dtype=torch.long); mask = torch.zeros(T, L, dtype=torch.bool)
        for t, g in enumerate(token_index):
            idx[t, :len(g)] = torch.tensor(g); mask[t, :len(g)] = True
        self.register_buffer("idx", idx); self.register_buffer("mask", mask)
        self.gene_logit = nn.Parameter(torch.zeros(T, L))           # within-token gene attention
        self.gene_load = nn.Parameter(torch.zeros(T, L))            # signed loadings (L1-regularised in training)
        self.embed = nn.Parameter(torch.randn(T, d) * 0.02)
        self.proj = nn.Linear(4, d)

    def gene_weights(self):
        return torch.softmax(self.gene_logit.masked_fill(~self.mask, -1e9), dim=-1)

    def forward(self, x):                                            # x [n, genes]
        g = x[:, self.idx] * self.mask                                # [n, T, L]
        m = self.mask.float(); cnt = m.sum(-1)
        mean = g.sum(-1) / cnt
        w = self.gene_weights()
        wmean = (g * w).sum(-1)
        wsd = torch.sqrt(((g - wmean.unsqueeze(-1)) ** 2 * w).sum(-1) + 1e-6)
        load = (g * self.gene_load).sum(-1) / torch.sqrt(cnt)
        feats = torch.stack([mean, wmean, wsd, load], -1)            # [n, T, 4]
        return self.embed + self.proj(feats), mean                   # token states [n,T,d], token mean scores [n,T]


class IHABlock(nn.Module):
    def __init__(self, d: int, heads: int, head_dim: int, pseudoheads: int, mode: str, prior: torch.Tensor | None,
                 graph_bias: float, dropout: float):
        super().__init__()
        self.h, self.hd, self.mode = heads, head_dim, mode
        self.p = pseudoheads if mode == "iha" else 1
        self.q, self.k, self.v = (nn.Linear(d, heads * head_dim, bias=False) for _ in range(3))
        if self.mode == "iha":
            init = torch.full((heads, heads, self.p), -2.0)
            for h in range(heads):
                init[h, h, :] = 2.0
                if self.p > 1:
                    init[(h + 1) % heads, h, 1:] = 0.5
            self.alpha_q, self.alpha_k, self.alpha_v = (nn.Parameter(init.clone()) for _ in range(3))
            self.collapse = nn.Parameter(torch.zeros(heads, self.p))
        self.out = nn.Linear(heads * head_dim, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)
        if prior is not None and graph_bias > 0:
            self.register_buffer("prior", prior); self.gb = nn.Parameter(torch.tensor(float(graph_bias)))
        else:
            self.prior = None

    def forward(self, s):
        n, T, _ = s.shape
        x = self.ln1(s)
        q = self.q(x).view(n, T, self.h, self.hd).transpose(1, 2)          # [n,h,T,hd]
        k = self.k(x).view(n, T, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(n, T, self.h, self.hd).transpose(1, 2)
        if self.mode == "iha":                                               # interleave heads via pseudo-heads
            mix = lambda a, z: torch.einsum("mhp,nmtd->nhptd", torch.softmax(a, 0), z)
            q, k, v = mix(self.alpha_q, q), mix(self.alpha_k, k), mix(self.alpha_v, v)   # [n,h,p,T,hd]
        else:
            q, k, v = q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)
        logits = torch.einsum("nhpid,nhpjd->nhpij", q, k) / math.sqrt(self.hd)
        if self.prior is not None:
            logits = logits + self.gb * self.prior
        attn = torch.softmax(logits, -1)
        o = torch.einsum("nhpij,nhpjd->nhpid", self.drop(attn), v)
        if self.mode == "iha":
            o = torch.einsum("hp,nhptd->nhtd", torch.softmax(self.collapse, -1), o)
        else:
            o = o.squeeze(2)
        s = s + self.drop(self.out(o.transpose(1, 2).reshape(n, T, self.h * self.hd)))
        s = s + self.drop(self.ff(self.ln2(s)))
        return s, attn.mean(2)                                              # attention [n,h,T,T]


class AttnSurv(nn.Module):
    def __init__(self, token_index, prior, d=32, heads=4, head_dim=8, pseudoheads=2, layers=2, mode="iha",
                 graph_bias=0.5, dropout=0.1, prolif_offset=True):
        super().__init__()
        self.enc = GeneToToken(token_index, d)
        self.blocks = nn.ModuleList([IHABlock(d, heads, head_dim, pseudoheads, mode, prior, graph_bias, dropout)
                                     for _ in range(layers)])
        self.pool = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.prolif_offset = prolif_offset
        self.beta_p = nn.Parameter(torch.zeros(1))

    def forward(self, x, prolif):
        s, _ = self.enc(x); attns = []
        for b in self.blocks:
            s, a = b(s); attns.append(a)
        w = torch.softmax(self.pool(s).squeeze(-1), -1)                     # token importance for this patient
        f = self.head((s * w.unsqueeze(-1)).sum(1)).squeeze(-1)
        risk = f + (self.beta_p * prolif if self.prolif_offset else 0.0)
        return risk, dict(attn=attns, token_weight=w, f=f)


class MLPSurv(nn.Module):
    def __init__(self, token_index, prior=None, d=32, dropout=0.1, prolif_offset=True, **_):
        super().__init__()
        self.enc = GeneToToken(token_index, d); T = len(token_index)
        self.mlp = nn.Sequential(nn.Linear(T * d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, 1))
        self.prolif_offset = prolif_offset; self.beta_p = nn.Parameter(torch.zeros(1))

    def forward(self, x, prolif):
        s, _ = self.enc(x)
        f = self.mlp(s.flatten(1)).squeeze(-1)
        return f + (self.beta_p * prolif if self.prolif_offset else 0.0), dict(f=f)


class LinearSurv(nn.Module):
    """Classical comparator: Cox regression on the token mean scores (+ proliferation)."""

    def __init__(self, token_index, prior=None, prolif_offset=True, **_):
        super().__init__()
        self.enc = GeneToToken(token_index, 4); self.lin = nn.Linear(len(token_index), 1, bias=False)
        self.prolif_offset = prolif_offset; self.beta_p = nn.Parameter(torch.zeros(1))
        for p in self.enc.parameters():
            p.requires_grad_(False)

    def forward(self, x, prolif):
        _, mean = self.enc(x)
        f = self.lin(mean).squeeze(-1)
        return f + (self.beta_p * prolif if self.prolif_offset else 0.0), dict(f=f)


def token_prior(token_index: list[list[int]]) -> torch.Tensor:
    """Biological graph prior: gene-overlap coefficient |A & B| / min(|A|, |B|) between tokens (pathway crosstalk;
    diagonal 1), row-centred. Uses prior knowledge only (gene-set membership), never patient labels."""
    sets = [set(t) for t in token_index]; T = len(sets)
    P = torch.zeros(T, T)
    for i in range(T):
        for j in range(T):
            P[i, j] = 1.0 if i == j else len(sets[i] & sets[j]) / max(1, min(len(sets[i]), len(sets[j])))
    return P - P.mean(1, keepdim=True)

"""
INTERVENE-D: interleaved-head attention for drug response with drug-as-query mechanism attention.

cell omics (expression, copy number, damaging mutations) -> pathway tokens (MultiOmicTokens)
-> interleaved-head self-attention over pathways (IHABlock, shared with INTERVENE-2; Meta-IHA-inspired head mixing)
-> drug cross-attention: every drug owns a learned query that attends over the cell's pathway tokens (interleaved
   heads in "iha" mode); the attention a[cell, drug, pathway] is the model's mechanism map for that drug
-> per-drug readout + a shared cell-level sensitivity term scaled per drug.
Comparators: mode="mha" (no head interleaving anywhere), MLPDrug (same tokens, no attention).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intervene2.model import IHABlock  # noqa: E402


class MultiOmicTokens(nn.Module):
    """Token state = identity embedding + projection of 8 per-token summaries from its genes:
    expression [mean, attention-weighted mean, weighted SD, sparse signed loading], copy number [mean, weighted mean,
    signed loading], damaging-mutation fraction."""

    def __init__(self, token_index, d):
        super().__init__()
        T, L = len(token_index), max(len(t) for t in token_index)
        idx = torch.zeros(T, L, dtype=torch.long); mask = torch.zeros(T, L, dtype=torch.bool)
        for t, g in enumerate(token_index):
            idx[t, :len(g)] = torch.as_tensor(g); mask[t, :len(g)] = True
        self.register_buffer("idx", idx); self.register_buffer("mask", mask)
        self.gene_logit = nn.Parameter(torch.zeros(T, L))
        self.load_e = nn.Parameter(torch.zeros(T, L)); self.load_c = nn.Parameter(torch.zeros(T, L))
        self.embed = nn.Parameter(torch.randn(T, d) * 0.02); self.proj = nn.Linear(8, d)

    def gene_weights(self):
        return torch.softmax(self.gene_logit.masked_fill(~self.mask, -1e9), -1)

    def forward(self, e, c, m):
        msk = self.mask.float(); cnt = msk.sum(-1); w = self.gene_weights()
        ge, gc, gm = e[:, self.idx] * msk, c[:, self.idx] * msk, m[:, self.idx] * msk
        e_mean, e_w = ge.sum(-1) / cnt, (ge * w).sum(-1)
        e_sd = torch.sqrt(((ge - e_w.unsqueeze(-1)) ** 2 * w).sum(-1) + 1e-6)
        e_ld = (ge * self.load_e).sum(-1) / torch.sqrt(cnt)
        c_mean, c_w, c_ld = gc.sum(-1) / cnt, (gc * w).sum(-1), (gc * self.load_c).sum(-1) / torch.sqrt(cnt)
        m_frac = gm.sum(-1) / cnt
        f = torch.stack([e_mean, e_w, e_sd, e_ld, c_mean, c_w, c_ld, m_frac], -1)
        return self.embed + self.proj(f)


class DrugCrossAttention(nn.Module):
    def __init__(self, n_drugs, d, heads, head_dim, pseudoheads, mode):
        super().__init__()
        self.h, self.hd, self.mode = heads, head_dim, mode
        self.p = pseudoheads if mode == "iha" else 1
        self.q = nn.Parameter(torch.randn(n_drugs, heads, head_dim) * 0.1)        # one learned query per drug and head
        self.k = nn.Linear(d, heads * head_dim, bias=False); self.v = nn.Linear(d, heads * head_dim, bias=False)
        if mode == "iha":
            init = torch.full((heads, heads, self.p), -2.0)
            for h in range(heads):
                init[h, h, :] = 2.0
                if self.p > 1:
                    init[(h + 1) % heads, h, 1:] = 0.5
            self.alpha_k, self.alpha_v = nn.Parameter(init.clone()), nn.Parameter(init.clone())
            self.collapse = nn.Parameter(torch.zeros(heads, self.p))
        self.w = nn.Parameter(torch.zeros(n_drugs, heads * head_dim)); self.b = nn.Parameter(torch.zeros(n_drugs))

    def forward(self, s):                                                         # s [n, T, d]
        n, T, _ = s.shape
        k = self.k(s).view(n, T, self.h, self.hd); v = self.v(s).view(n, T, self.h, self.hd)
        if self.mode == "iha":                                                    # interleave key/value heads
            k = torch.einsum("mhp,ntmd->ntphd", torch.softmax(self.alpha_k, 0), k)
            v = torch.einsum("mhp,ntmd->ntphd", torch.softmax(self.alpha_v, 0), v)
        else:
            k, v = k.unsqueeze(2), v.unsqueeze(2)                                 # [n,T,p,h,hd]
        logits = torch.einsum("Dhd,ntphd->nDpht", self.q, k) / math.sqrt(self.hd)
        a = torch.softmax(logits, -1)                                             # [n, D, p, h, T]
        o = torch.einsum("nDpht,ntphd->nDphd", a, v)
        if self.mode == "iha":
            o = torch.einsum("hp,nDphd->nDhd", torch.softmax(self.collapse, -1), o)
        else:
            o = o.squeeze(2)
        y = (o.reshape(n, o.shape[1], -1) * self.w).sum(-1) + self.b              # [n, D]
        return y, a.mean((2, 3))                                                  # mechanism map [n, D, T]


class InterveneD(nn.Module):
    def __init__(self, token_index, n_drugs, prior=None, d=32, heads=4, head_dim=8, pseudoheads=2, layers=1,
                 mode="iha", graph_bias=0.5, dropout=0.1):
        super().__init__()
        self.enc = MultiOmicTokens(token_index, d)
        self.blocks = nn.ModuleList([IHABlock(d, heads, head_dim, pseudoheads, mode, prior, graph_bias, dropout)
                                     for _ in range(layers)])
        self.cross = DrugCrossAttention(n_drugs, d, heads, head_dim, pseudoheads, mode)
        self.sens = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.sens_scale = nn.Parameter(torch.zeros(n_drugs))

    def forward(self, e, c, m):
        s = self.enc(e, c, m); attn = []
        for b in self.blocks:
            s, a = b(s); attn.append(a)
        y, mech = self.cross(s)
        y = y + self.sens(s.mean(1)) * self.sens_scale                         # shared cell sensitivity, per-drug scale
        return y, dict(mech=mech, attn=attn)


class MLPDrug(nn.Module):
    def __init__(self, token_index, n_drugs, prior=None, d=32, dropout=0.1, **_):
        super().__init__()
        self.enc = MultiOmicTokens(token_index, d); T = len(token_index)
        self.mlp = nn.Sequential(nn.Linear(T * d, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, n_drugs))

    def forward(self, e, c, m):
        return self.mlp(self.enc(e, c, m).flatten(1)), {}

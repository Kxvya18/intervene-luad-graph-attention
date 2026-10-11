"""
INTERVENE-2 end-to-end pipeline (PROTOCOL_v5.md). Resumable: every unit of work writes a JSON under --out and is
skipped if present, so a Colab disconnect only loses the fit in progress.

Stages
  dev      leave-one-cohort-out over the 4 development cohorts, every config x seed
  select   automatic, pre-specified choice of the best IHA config and best MHA config (same grid)
  final    train selected models on all development cohorts (10 seeds), evaluate ONCE on the sealed cohorts
  discover interaction / gene discovery from the final INTERVENE-2 ensemble, tested on the sealed cohorts
Usage: python -m intervene2.run --out /content/drive/MyDrive/intervene2 --stage all
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, spearmanr

from .data import DEV, SEALED, build
from .model import token_prior
from .train import adjusted_test, cindex, fit, predict

GRID = {"layers": [1, 2], "graph_bias": [0.0, 0.5]}
# proliferation programs are excluded from interaction discovery (the claim is relapse biology beyond proliferation)
PROLIF_PROGRAMS = {"PROLIFERATION", "HM_E2F_Targets", "HM_G2-M_Checkpoint", "HM_Mitotic_Spindle"}


def configs():
    out = {}
    for mode in ("iha", "mha"):
        for L, gb in itertools.product(GRID["layers"], GRID["graph_bias"]):
            out[f"{mode}_L{L}_gb{gb}"] = ("attn", dict(mode=mode, layers=L, graph_bias=gb, d=32, heads=4, head_dim=8,
                                                        pseudoheads=2, dropout=0.1, prolif_offset=True))
    out["mlp"] = ("mlp", dict(d=32, dropout=0.1, prolif_offset=True))
    out["linear"] = ("linear", dict(prolif_offset=True))
    return out


def stack(D, cohorts):
    X = np.concatenate([D["cohorts"][c]["X"] for c in cohorts]); P = np.concatenate([D["cohorts"][c]["prolif"] for c in cohorts])
    cl = pd.concat([D["cohorts"][c]["clin"].assign(cohort=c) for c in cohorts], ignore_index=True)
    strata = np.concatenate([[i] * len(D["cohorts"][c]["clin"]) for i, c in enumerate(cohorts)])
    return X, P, cl.time.to_numpy(float), cl.event.to_numpy(float), strata


def evaluate(D, c, pred):
    cl = D["cohorts"][c]["clin"]; t, e = cl.time.to_numpy(float), cl.event.to_numpy(int); p = D["cohorts"][c]["prolif"]
    b, se, pv, n = adjusted_test(t, e, pred["f"], p, cl["stage"].to_numpy(float) if "stage" in cl else None)
    return dict(c_risk=cindex(t, e, pred["risk"]), c_f=cindex(t, e, pred["f"]), c_prolif=cindex(t, e, p),
                adj_loghr=b, adj_se=se, adj_p=pv, adj_z=b / se, n=int(len(t)), events=int(e.sum()))


def run_dev(D, out, seeds, device, quick):
    prior = token_prior(list(D["token_index"].values())); tidx = list(D["token_index"].values())
    for name, (kind, cfg) in configs().items():
        for h in DEV:
            for s in seeds:
                f = out / "dev" / f"{name}__{h}__s{s}.json"
                if f.exists():
                    continue
                tr = [c for c in DEV if c != h]
                X, P, t, e, st = stack(D, tr)
                m = fit(kind, cfg, tidx, prior, X, P, t, e, st, s, device, epochs=30 if quick else 400)
                r = evaluate(D, h, predict(m, D["cohorts"][h]["X"], D["cohorts"][h]["prolif"], device))
                f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(dict(config=name, held_out=h, seed=s, **r)))
                print(f"dev {name} {h} s{s}: C={r['c_risk']:.3f} adj_z={r['adj_z']:.2f}", flush=True)


def select(out):
    rows = [json.loads(p.read_text()) for p in (out / "dev").glob("*.json") if not p.name.startswith("._")]
    df = pd.DataFrame(rows)
    summ = df.groupby("config").agg(c_risk=("c_risk", "mean"), adj_z=("adj_z", "mean"), n=("seed", "size")).reset_index()
    pick = {fam: summ[summ.config.str.startswith(fam + "_")].sort_values(["adj_z", "c_risk"], ascending=False).iloc[0].config
            for fam in ("iha", "mha")}
    summ.to_csv(out / "dev_summary.csv", index=False)
    (out / "selection.json").write_text(json.dumps(pick))
    print(summ.round(3).to_string(index=False)); print("selected:", pick)
    return pick


def ensemble(D, out, name, kind, cfg, seeds, device, quick, tag):
    prior = token_prior(list(D["token_index"].values())); tidx = list(D["token_index"].values())
    X, P, t, e, st = stack(D, DEV); preds = {c: [] for c in SEALED + DEV}; mats = []
    for s in seeds:
        f = out / "final" / f"{tag}__s{s}.pkl"
        if f.exists():
            res = pickle.load(open(f, "rb"))
        else:
            m = fit(kind, cfg, tidx, prior, X, P, t, e, st, 100 + s, device, epochs=30 if quick else 400)
            res = {c: predict(m, D["cohorts"][c]["X"], D["cohorts"][c]["prolif"], device) for c in SEALED + DEV}
            if hasattr(m, "enc"):
                res["gene_load"] = m.enc.gene_load.detach().cpu().numpy(); res["gene_w"] = m.enc.gene_weights().detach().cpu().numpy()
            f.parent.mkdir(parents=True, exist_ok=True); pickle.dump(res, open(f, "wb"))
        for c in SEALED + DEV:
            preds[c].append(res[c])
        if "attn" in res[DEV[0]]:
            mats.append(np.mean([res[c]["attn"].mean(0) for c in DEV], 0))
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    ens = {c: dict(risk=np.mean([z(p["risk"]) for p in v], 0), f=np.mean([z(p["f"]) for p in v], 0)) for c, v in preds.items()}
    return ens, mats


def paired_boot(t, e, r1, r2, n=1000, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(t), len(t))
        if e[i].sum() > 1:
            d.append(cindex(t[i], e[i], r1[i]) - cindex(t[i], e[i], r2[i]))
    d = np.array(d); return float(d.mean()), float(d.std()), float((d <= 0).mean())


def run_final(D, out, pick, seeds, device, quick):
    cf = configs(); models = {"INTERVENE2": pick["iha"], "MHA": pick["mha"], "MLP": "mlp", "LINEAR": "linear"}
    ens, mats = {}, {}
    for label, name in models.items():
        kind, cfg = cf[name]; ens[label], mats[label] = ensemble(D, out, name, kind, cfg, seeds, device, quick, label)
    rows, comps = [], []
    for c in SEALED:
        cl = D["cohorts"][c]["clin"]; t, e = cl.time.to_numpy(float), cl.event.to_numpy(int); p = D["cohorts"][c]["prolif"]
        for label in models:
            r = evaluate(D, c, ens[label][c]); rows.append(dict(cohort=c, model=label, **r))
        rows.append(dict(cohort=c, model="PROLIFERATION_ONLY", c_risk=cindex(t, e, p), n=len(t), events=int(e.sum())))
        for comp in ("MHA", "MLP", "LINEAR"):
            m_, s_, p_ = paired_boot(t, e, ens["INTERVENE2"][c]["risk"], ens[comp][c]["risk"], seed=hash(c + comp) % 2 ** 31)
            comps.append(dict(cohort=c, comparison=f"INTERVENE2-{comp}", delta_c=m_, se=s_, p_one_sided=p_))
        m_, s_, p_ = paired_boot(t, e, ens["INTERVENE2"][c]["risk"], p, seed=7)
        comps.append(dict(cohort=c, comparison="INTERVENE2-PROLIFERATION", delta_c=m_, se=s_, p_one_sided=p_))
    res, cmp_ = pd.DataFrame(rows), pd.DataFrame(comps)
    pooled = []
    for comp, g in cmp_.groupby("comparison"):
        w = 1 / g.se ** 2; est = float((w * g.delta_c).sum() / w.sum()); se = float(np.sqrt(1 / w.sum()))
        pooled.append(dict(comparison=comp, pooled_delta_c=est, se=se, p_one_sided=float(norm.sf(est / se))))
    a = res[res.model == "INTERVENE2"]; w = 1 / a.adj_se ** 2
    adj = dict(pooled_adj_loghr=float((w * a.adj_loghr).sum() / w.sum()), se=float(np.sqrt(1 / w.sum())))
    adj["p_two_sided"] = float(2 * norm.sf(abs(adj["pooled_adj_loghr"] / adj["se"])))
    res.to_csv(out / "sealed_results.csv", index=False); cmp_.to_csv(out / "sealed_comparisons.csv", index=False)
    pd.DataFrame(pooled).to_csv(out / "sealed_pooled.csv", index=False); (out / "sealed_adjusted.json").write_text(json.dumps(adj))
    stab = {k: (float(np.mean([spearmanr(x.ravel(), y.ravel())[0] for x, y in itertools.combinations(v, 2)])) if len(v) > 1 else None)
            for k, v in mats.items() if v}
    (out / "attention_stability.json").write_text(json.dumps(stab))
    print(res.round(3).to_string(index=False)); print(pd.DataFrame(pooled).round(4).to_string(index=False)); print("adjusted:", adj, "stability:", stab)
    return mats


def run_discover(D, out, mats, top=10):
    names = list(D["token_index"].keys()); M = np.mean(mats["INTERVENE2"], 0); S = (M + M.T) / 2
    pairs = [(S[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
             if not ({names[i], names[j]} & PROLIF_PROGRAMS)]
    pairs = sorted(pairs, reverse=True)[:top]
    rows = []
    from lifelines import CoxPHFitter
    for score, a, b in pairs:
        zs, ws, per = [], [], {}
        for c in SEALED:
            co = D["cohorts"][c]; cl = co["clin"]; X = co["X"]
            ma = X[:, D["token_index"][a]].mean(1); mb = X[:, D["token_index"][b]].mean(1)
            ma, mb = (ma - ma.mean()) / ma.std(), (mb - mb.mean()) / mb.std()
            d = pd.DataFrame(dict(time=cl.time, event=cl.event, prolif=co["prolif"], A=ma, B=mb, AxB=ma * mb))
            cols = ["prolif", "A", "B", "AxB"]
            if cl["stage"].notna().mean() > 0.8:
                d["stage"] = cl["stage"].to_numpy(); d = d.dropna(); cols = ["stage"] + cols
            m = CoxPHFitter(penalizer=1e-6).fit(d[cols + ["time", "event"]], "time", "event")
            bb, se = m.params_["AxB"], m.standard_errors_["AxB"]; per[c] = (float(bb), float(m.summary.loc["AxB", "p"]))
            zs.append(bb); ws.append(1 / se ** 2)
        est = float(np.sum(np.array(ws) * np.array(zs)) / np.sum(ws)); se = float(np.sqrt(1 / np.sum(ws)))
        rows.append(dict(token_a=a, token_b=b, attention=float(score), pooled_interaction_loghr=est, se=se,
                         p=float(2 * norm.sf(abs(est / se))), **{f"{c}_loghr_p": per[c] for c in SEALED}))
    df = pd.DataFrame(rows).sort_values("p")
    m = len(df); df["bh_q"] = np.minimum.accumulate((df.p * m / np.arange(1, m + 1))[::-1])[::-1].clip(upper=1)
    df.to_csv(out / "discovery_interactions_sealed.csv", index=False); print(df.round(4).to_string(index=False))
    gl = [pickle.load(open(p, "rb")) for p in sorted((out / "final").glob("INTERVENE2__s*.pkl"))]
    if gl and "gene_load" in gl[0]:
        load = np.mean([g["gene_load"] for g in gl], 0); genes = D["genes"]; rows = []
        for t, (tok, idx) in enumerate(D["token_index"].items()):
            for k, gi in enumerate(idx):
                rows.append(dict(token=tok, gene=genes[gi], loading=float(load[t, k])))
        g = pd.DataFrame(rows); g["abs"] = g.loading.abs()
        g.sort_values("abs", ascending=False).head(200).to_csv(out / "discovery_genes.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--cache", default=None)
    ap.add_argument("--stage", default="all", choices=["data", "dev", "select", "final", "discover", "all"])
    ap.add_argument("--quick", action="store_true", help="smoke test: few seeds and epochs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); cache = Path(a.cache or out / "cache")
    device = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", device, flush=True)
    dp = out / "data.pkl"
    if dp.exists():
        D = pickle.load(open(dp, "rb"))
    else:
        D = build(cache); pickle.dump(D, open(dp, "wb"))
    print({c: (len(v["clin"]), int(v["clin"].event.sum())) for c, v in D["cohorts"].items()}, "tokens:", len(D["tokens"]),
          "genes:", len(D["genes"]), flush=True)
    dev_seeds, fin_seeds = ([0] if a.quick else [0, 1, 2, 3, 4]), ([0] if a.quick else list(range(10)))
    if a.quick:   # smoke test must never touch the sealed cohorts: stand-in with two development cohorts
        globals()["SEALED"] = ["GSE31210", "GSE30219"]
        print("QUICK MODE: sealed cohorts replaced by development stand-ins; results are not meaningful", flush=True)
    if a.stage in ("dev", "all"):
        run_dev(D, out, dev_seeds, device, a.quick)
    if a.stage in ("select", "final", "discover", "all"):
        pick = json.loads((out / "selection.json").read_text()) if (out / "selection.json").exists() and a.stage != "select" else select(out)
    if a.stage in ("final", "discover", "all"):
        mats = run_final(D, out, pick, fin_seeds, device, a.quick)
        run_discover(D, out, mats)

"""
v4 step 2 — the locked tests H1-H5 of PROTOCOL.md, run once. Writes results/*.csv and results/SUMMARY.md.
Usage: python v02_locked_tests.py [--subset luad|all]   (luad = primary, all = secondary all-histology)
"""
import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
CACHE, RES = HERE / "cache", HERE / "results"
EXTERNAL = ["GSE30219", "GSE37745", "GSE50081", "GSE68465"]
S4 = ["MRPL21", "PSMA1", "PSMB5", "TXNRD1"]
P13 = ["MKI67", "TOP2A", "MCM2", "MCM5", "PCNA", "CDK1", "CDK2", "AURKA", "AURKB", "MCM6", "BUB1", "CCNB1", "UBE2C"]
ROOT = HERE.parents[0]


def load(c, subset):
    return pickle.load(open(CACHE / f"{c}_{subset}.pkl", "rb"))


def score(expr, genes):
    g = [x for x in genes if x in expr.columns]
    return expr[g].mean(axis=1).to_numpy(), g


def cidx(time, event, risk):
    return concordance_index(time, -risk, event)


def cox(df, cols):
    m = CoxPHFitter(penalizer=1e-6).fit(df[cols + ["time", "event"]], "time", "event")
    return m


def boot_c(time, event, risk, n=1000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(time), len(time))
        if event[i].sum() > 1:
            out.append(cidx(time[i], event[i], risk[i]))
    return np.array(out)


def dersimonian_laird(est, se):
    w = 1 / se ** 2; fe = np.sum(w * est) / w.sum()
    q = np.sum(w * (est - fe) ** 2); k = len(est)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (se ** 2 + tau2); re = np.sum(wr * est) / wr.sum(); se_re = np.sqrt(1 / wr.sum())
    return re, se_re, tau2


def score_test_z(X, time, event):
    """Vectorised Cox score-test z at beta = 0 for every column of X (Breslow ties)."""
    order = np.argsort(-time, kind="stable"); X, time, event = X[order], time[order], event[order]
    cs = np.cumsum(X, axis=0); cs2 = np.cumsum(X ** 2, axis=0); n_at = np.arange(1, len(time) + 1)[:, None]
    # risk set of subject i (sorted by descending time) = rows 0..last index with time >= time_i
    last = np.searchsorted(-time, -time, side="right") - 1
    mean = cs[last] / n_at[last]; var = cs2[last] / n_at[last] - mean ** 2
    ev = event.astype(bool)
    U = (X[ev] - mean[ev]).sum(0); I = var[ev].sum(0)
    return U / np.sqrt(np.maximum(I, 1e-12))


def h1_nested(rng_seed=20261011, folds=5, repeats=50):
    d = load("GSE31210", "luad"); X, cl = d["expr"], d["clin"]
    pool = [c for c in pd.read_csv(ROOT / "data" / "processed" / "relapse" / "GSE31210" /
                                   "gse31210_module_gene_expression.csv", nrows=1).columns if c in X.columns]
    time, event = cl.time.to_numpy(float), cl.event.to_numpy(int)
    Xp = X[pool].to_numpy(float)
    rng = np.random.default_rng(rng_seed); cs, picks = [], []
    for r in range(repeats):
        fold = np.empty(len(time), int)
        for e in (0, 1):                                    # stratified by event
            ix = rng.permutation(np.where(event == e)[0]); fold[ix] = np.arange(len(ix)) % folds
        risk = np.empty(len(time))
        for f in range(folds):
            tr, te = fold != f, fold == f
            z = score_test_z(Xp[tr], time[tr], event[tr])
            top = np.argsort(-z)[:4]; picks += [pool[i] for i in top]
            risk[te] = Xp[te][:, top].mean(1)
        cs.append(cidx(time, event, risk))
    cs = np.array(cs); sel = pd.Series(picks).value_counts() / (folds * repeats)
    return dict(pool_size=len(pool), mean=cs.mean(), lo=np.percentile(cs, 2.5), hi=np.percentile(cs, 97.5)), sel


def external_tables(subset):
    rows, ext = [], {}
    for c in ["GSE31210"] + EXTERNAL:
        d = load(c, subset); X, cl = d["expr"], d["clin"]
        s4, g4 = score(X, S4); p13, gp = score(X, P13)
        df = cl.assign(S4=(s4 - s4.mean()) / s4.std(), P13=(p13 - p13.mean()) / p13.std())
        t, e = df.time.to_numpy(float), df.event.to_numpy(int)
        b = boot_c(t, e, df.S4.to_numpy(), seed=hash(c) % 2 ** 32)
        m_adj = cox(df, ["P13", "S4"]); m_s4 = cox(df, ["S4"])
        r = dict(cohort=c, n=len(df), events=int(e.sum()), s4_genes=";".join(g4), p13_genes=len(gp),
                 c_S4=cidx(t, e, df.S4.to_numpy()), c_S4_lo=np.percentile(b, 2.5), c_S4_hi=np.percentile(b, 97.5),
                 c_S4_se=b.std(), c_P13=cidx(t, e, df.P13.to_numpy()),
                 hr_S4=float(np.exp(m_s4.params_["S4"])), p_S4=float(m_s4.summary.loc["S4", "p"]),
                 loghr_S4_adj=float(m_adj.params_["S4"]), se_S4_adj=float(m_adj.standard_errors_["S4"]),
                 p_S4_adj=float(m_adj.summary.loc["S4", "p"]), loghr_P13_adj=float(m_adj.params_["P13"]),
                 p_P13_adj=float(m_adj.summary.loc["P13", "p"]))
        st = df.dropna(subset=["stage"])
        if st.stage.nunique() > 1 and len(st) > 30:
            st = st.assign(stage=(st.stage - st.stage.mean()) / st.stage.std())
            ms = cox(st, ["stage", "S4"]); msp = cox(st, ["stage", "P13", "S4"])
            r.update(n_stage=len(st), p_S4_given_stage=float(ms.summary.loc["S4", "p"]),
                     loghr_S4_given_stage=float(ms.params_["S4"]),
                     p_S4_given_stage_P13=float(msp.summary.loc["S4", "p"]),
                     loghr_S4_given_stage_P13=float(msp.params_["S4"]))
        rows.append(r); ext[c] = (X, cl)
    return pd.DataFrame(rows), ext


def h4_random(ext, n_sets=1000, seed=20261011):
    common = set.intersection(*[set(ext[c][0].columns) for c in EXTERNAL])
    common = sorted(common - set(S4)); rng = np.random.default_rng(seed)
    out = []
    pre = {c: (ext[c][1].time.to_numpy(float), ext[c][1].event.to_numpy(int),
               ext[c][0][[g for g in P13 if g in ext[c][0].columns]].mean(1).to_numpy()) for c in EXTERNAL}
    for k in range(n_sets):
        genes = list(rng.choice(common, 4, replace=False)); cs, lh, ws = [], [], []
        for c in EXTERNAL:
            X = ext[c][0]; t, e, p = pre[c]
            s = X[genes].mean(1).to_numpy(); s = (s - s.mean()) / s.std(); pz = (p - p.mean()) / p.std()
            cs.append(cidx(t, e, s))
            m = cox(pd.DataFrame(dict(time=t, event=e, P13=pz, S4=s)), ["P13", "S4"])
            b, se = m.params_["S4"], m.standard_errors_["S4"]; lh.append(b); ws.append(1 / se ** 2)
        out.append(dict(set=";".join(genes), mean_c=np.mean(cs), pooled_loghr_adj=np.sum(np.array(ws) * np.array(lh)) / np.sum(ws)))
    return pd.DataFrame(out), len(common)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", default="luad"); a = ap.parse_args()
    RES.mkdir(exist_ok=True); tag = a.subset
    lines = [f"# v4 locked results ({tag})\n"]
    if tag == "luad":
        h1, sel = h1_nested(); sel.to_csv(RES / "H1_gene_selection_frequency.csv")
        lines += [f"## H1 nested internal (GSE31210, pool {h1['pool_size']} module genes, 5x50)",
                  f"mean C = {h1['mean']:.3f} [{h1['lo']:.3f}, {h1['hi']:.3f}]  -> PASS={h1['mean'] > 0.60 and h1['lo'] > 0.55}",
                  "selection frequency of S4 genes: " + ", ".join(f"{g} {sel.get(g, 0):.2f}" for g in S4),
                  "top-8 selected: " + ", ".join(f"{g} {v:.2f}" for g, v in sel.head(8).items()), ""]
    tab, ext = external_tables(tag); tab.to_csv(RES / f"H2_H3_H5_cohort_table_{tag}.csv", index=False)
    e = tab[tab.cohort.isin(EXTERNAL)]
    re, se_re, tau2 = dersimonian_laird(e.c_S4.to_numpy(), e.c_S4_se.to_numpy())
    w = 1 / e.se_S4_adj.to_numpy() ** 2; fe = np.sum(w * e.loghr_S4_adj) / w.sum(); fe_se = np.sqrt(1 / w.sum())
    from scipy.stats import norm
    p_fe = 2 * norm.sf(abs(fe / fe_se))
    re3, se3, tau3 = dersimonian_laird(e.loghr_S4_adj.to_numpy(), e.se_S4_adj.to_numpy())
    pd.set_option("display.width", 250)
    lines += ["## Per-cohort (S4 = MRPL21+PSMA1+PSMB5+TXNRD1; P13 = proliferation)", "```",
              tab[["cohort", "n", "events", "s4_genes", "c_S4", "c_S4_lo", "c_S4_hi", "c_P13", "hr_S4", "p_S4",
                   "loghr_S4_adj", "p_S4_adj", "p_P13_adj"]].round(3).to_string(index=False), "```",
              f"## H2 pooled external C(S4) (random effects) = {re:.3f} [{re - 1.96 * se_re:.3f}, {re + 1.96 * se_re:.3f}], tau2={tau2:.4f}"
              f"  -> PASS={re - 1.96 * se_re > 0.5}",
              f"## H3 pooled S4 log-HR given P13: fixed {fe:.3f} (SE {fe_se:.3f}, p={p_fe:.4f}); random {re3:.3f} (SE {se3:.3f})"
              f"  -> PASS={fe > 0 and p_fe < 0.05}"]
    rnd, ncommon = h4_random(ext); rnd.to_csv(RES / f"H4_random_signatures_{tag}.csv", index=False)
    obs_c = e.c_S4.mean(); obs_h = fe
    pc, ph = (rnd.mean_c < obs_c).mean(), (rnd.pooled_loghr_adj < obs_h).mean()
    lines += [f"## H4 random 4-gene signatures (n=1000 from {ncommon} genes in all cohorts)",
              f"S4 mean external C = {obs_c:.3f}: percentile {100 * pc:.1f} (random median {rnd.mean_c.median():.3f}, 95th {rnd.mean_c.quantile(.95):.3f})",
              f"S4 pooled adjusted log-HR = {obs_h:.3f}: percentile {100 * ph:.1f} (random median {rnd.pooled_loghr_adj.median():.3f}, 95th {rnd.pooled_loghr_adj.quantile(.95):.3f})",
              f"  -> PASS={pc > 0.95 and ph > 0.95}"]
    stg = tab.dropna(subset=["p_S4_given_stage"]) if "p_S4_given_stage" in tab else tab.iloc[0:0]
    if len(stg):
        lines += ["## H5 stage-adjusted (secondary)", "```", stg[["cohort", "n_stage", "loghr_S4_given_stage", "p_S4_given_stage",
                  "loghr_S4_given_stage_P13", "p_S4_given_stage_P13"]].round(4).to_string(index=False), "```"]
    (RES / f"SUMMARY_{tag}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

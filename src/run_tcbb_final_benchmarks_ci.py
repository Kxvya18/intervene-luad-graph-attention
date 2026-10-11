
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


def zscore(series):
    arr=np.asarray(series,dtype=float)
    mu=np.nanmean(arr); sd=np.nanstd(arr)
    if not np.isfinite(sd) or sd==0: sd=1.0
    return pd.Series((arr-mu)/sd, index=getattr(series, 'index', None))

def concordance_index(time, event, risk):
    time=np.asarray(time,float); event=np.asarray(event,float); risk=np.asarray(risk,float)
    num=0.0; den=0.0
    n=len(time)
    for i in range(n):
        if event[i] != 1: continue
        for j in range(n):
            if time[j] <= time[i]: continue
            den += 1
            if risk[i] > risk[j]: num += 1
            elif risk[i] == risk[j]: num += 0.5
    return float(num/den) if den else np.nan

def logrank_high_low(frame, score_col, time_col, event_col):
    d=frame.dropna(subset=[score_col,time_col,event_col]).copy()
    hi=d[score_col] >= d[score_col].median()
    groups=[hi, ~hi]
    times=np.sort(d.loc[d[event_col].eq(1), time_col].unique())
    O=[0.0,0.0]; E=[0.0,0.0]; V=0.0
    for t in times:
        at_risk=[float(((d[time_col]>=t)&g).sum()) for g in groups]
        events=[float(((d[time_col]==t)&d[event_col].eq(1)&g).sum()) for g in groups]
        n=sum(at_risk); m=sum(events)
        if n<=1 or m==0: continue
        O[0]+=events[0]; O[1]+=events[1]
        E[0]+=m*at_risk[0]/n; E[1]+=m*at_risk[1]/n
        V += at_risk[0]*at_risk[1]*m*(n-m)/(n*n*(n-1))
    chi=((O[0]-E[0])**2/V) if V>0 else np.nan
    p=math.erfc(math.sqrt(max(chi,0)/2)) if np.isfinite(chi) else np.nan
    high_events=int((hi & d[event_col].eq(1)).sum()); low_events=int(((~hi) & d[event_col].eq(1)).sum())
    high_n=int(hi.sum()); low_n=int((~hi).sum())
    hr=((high_events+0.5)/(high_n-high_events+0.5))/((low_events+0.5)/(low_n-low_events+0.5)) if high_n and low_n else np.nan
    return {'n':len(d),'events':int(d[event_col].sum()),'hazard_ratio_approx':float(hr),'logrank_p':float(p),'high_events':high_events,'low_events':low_events}

def norm_sf_chi2_1(x):
    return float(math.erfc(math.sqrt(max(x,0.0)/2.0)))

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
PROCESSED=ROOT/'data'/'processed'/'relapse'
B=120
SEED=2707

MODULE_FEATURES=['proteostasis_expr_mean','mito_translation_expr_mean','oxidative_stress_expr_mean','apoptosis_iap_expr_mean','srebp_lipid_expr_mean','akt_mtor_growth_expr_mean']
REFINED_FEATURES=['MRPL21','PSMA1','PSMB5','TXNRD1']
REFINED_PLUS_PROLIF=['MRPL21','PSMA1','PSMB5','TXNRD1','proliferation_score']


def load_internal():
    f=pd.read_csv(RESULTS/'gse31210_refined_mechanism_patient_scores.csv')
    v3=pd.read_csv(RESULTS/'relapse_intervene_graph_sparse_v3_oof_risk_scores.csv')
    soft=[c for c in v3.columns if c.startswith('graph_sparse_v3_soft_seed') and c.endswith('_risk')]
    v3['intervene_v3_oof_mean_risk']=v3[soft].mean(axis=1)
    f=f.merge(v3[['geo_accession','intervene_v3_oof_mean_risk']],on='geo_accession',how='left')
    if 'raw_module_score' not in f:
        f['raw_module_score']=zscore(f['proteostasis_expr_mean'])+zscore(f['mito_translation_expr_mean'])+zscore(f['oxidative_stress_expr_mean'])
    if 'intervene_refined_gene_score' not in f:
        f['intervene_refined_gene_score']=zscore(f['MRPL21'])+zscore(f['PSMA1'])+zscore(f['PSMB5'])+0.5*zscore(f['TXNRD1'])
    if 'mrpl21_proteasome_score' not in f:
        f['mrpl21_proteasome_score']=zscore(f['MRPL21'])+zscore(f['proteasome_core_score'])
    if 'mrpl21_proteasome_prolif_score' not in f:
        f['mrpl21_proteasome_prolif_score']=f['mrpl21_proteasome_score']+zscore(f['proliferation_score'])
    return f.dropna(subset=['relapse_free_months','relapse_event']).copy()


def load_external_frames():
    frames=[]
    specs={
        'GSE30219':(PROCESSED/'GSE30219'/'gse30219_proteostasis_redox_scores.csv','dfs_months','relapse_event'),
        'GSE37745':(PROCESSED/'GSE37745'/'gse37745_proteostasis_redox_scores.csv','recurrence_time_months','recurrence_event'),
        'GSE68465':(PROCESSED/'GSE68465'/'gse68465_proteostasis_redox_scores.csv','pfs_months','progression_relapse_event'),
    }
    for cohort,(path,time,event) in specs.items():
        if not path.exists(): continue
        f=pd.read_csv(path)
        if {'PSMA1','PSMB5'}.issubset(f.columns):
            f['proteasome_core_score']=zscore(f[['PSMA1','PSMB5']].mean(axis=1))
        if 'MRPL21' in f.columns:
            f['mrpl21_proteasome_score']=zscore(f['MRPL21'])+zscore(f['proteasome_core_score'])
            f['mrpl21_proteasome_prolif_score']=f['mrpl21_proteasome_score']+zscore(f['proliferation_score'])
            f['intervene_refined_gene_score']=zscore(f['MRPL21'])+zscore(f['PSMA1'])+zscore(f['PSMB5'])+0.5*zscore(f['TXNRD1'])
        f['_cohort']=cohort; f['_time_col']=time; f['_event_col']=event
        frames.append(f)
    return frames


def stat_for_score(df, score, time_col, event_col):
    d=df[[score,time_col,event_col]].dropna().copy(); d=d[d[time_col]>0]
    if len(d)<20 or d[event_col].sum()<3:
        return {'n':len(d),'events':int(d[event_col].sum()) if len(d) else 0,'c_index':np.nan,'hr':np.nan,'logrank_p':np.nan}
    t=d[time_col].to_numpy(float); e=d[event_col].to_numpy(float); r=d[score].to_numpy(float)
    lr=logrank_high_low(d.assign(_score=r),'_score',time_col,event_col)
    return {'n':len(d),'events':int(e.sum()),'c_index':concordance_index(t,e,r),'hr':lr['hazard_ratio_approx'],'logrank_p':lr['logrank_p']}


def hr_approx_arrays(score, event):
    score=np.asarray(score,float); event=np.asarray(event,float)
    med=np.nanmedian(score)
    hi=score>=med; lo=~hi
    high_events=float(event[hi].sum()); low_events=float(event[lo].sum())
    high_n=float(hi.sum()); low_n=float(lo.sum())
    if high_n == 0 or low_n == 0:
        return np.nan
    return float(((high_events+0.5)/(high_n-high_events+0.5))/((low_events+0.5)/(low_n-low_events+0.5)))

def bootstrap_ci(df, score, time_col, event_col, b=B, seed=SEED):
    d=df[[score,time_col,event_col]].dropna().copy(); d=d[d[time_col]>0].reset_index(drop=True)
    obs=stat_for_score(d,score,time_col,event_col)
    if len(d)<20 or obs['events']<3:
        return {**obs,'c_index_ci_low':np.nan,'c_index_ci_high':np.nan,'hr_ci_low':np.nan,'hr_ci_high':np.nan}
    rng=np.random.default_rng(seed)
    cs=[]; hrs=[]; n=len(d)
    score_arr=d[score].to_numpy(float); time_arr=d[time_col].to_numpy(float); event_arr=d[event_col].to_numpy(float)
    for _ in range(b):
        idx=rng.integers(0,n,n)
        c=concordance_index(time_arr[idx], event_arr[idx], score_arr[idx])
        hr=hr_approx_arrays(score_arr[idx], event_arr[idx])
        if np.isfinite(c): cs.append(c)
        if np.isfinite(hr): hrs.append(hr)
    return {**obs,
            'bootstrap_n': int(b),
            'c_index_ci_low':float(np.nanpercentile(cs,2.5)) if cs else np.nan,
            'c_index_ci_high':float(np.nanpercentile(cs,97.5)) if cs else np.nan,
            'hr_ci_low':float(np.nanpercentile(hrs,2.5)) if hrs else np.nan,
            'hr_ci_high':float(np.nanpercentile(hrs,97.5)) if hrs else np.nan}


def fit_penalized_cox(x, time, event, l1=0.0, l2=1e-3, seed=7):
    rng=np.random.default_rng(seed)
    x=np.asarray(x,float); time=np.asarray(time,float); event=np.asarray(event,float)
    mu=np.nanmean(x,axis=0); sd=np.nanstd(x,axis=0); sd[sd==0]=1.0
    x=np.nan_to_num((x-mu)/sd,nan=0.0)
    n,p=x.shape
    beta=np.zeros(p,dtype=float)
    order=np.argsort(-time)
    lr=0.03
    for it in range(900):
        risk=x@beta
        exp_r=np.exp(np.clip(risk,-30,30))
        loss=0.0; grad=np.zeros(p,dtype=float)
        event_idx=np.where(event==1)[0]
        for i in event_idx:
            at=time>=time[i]
            denom=exp_r[at].sum()
            if denom<=0: continue
            weighted=(x[at]*exp_r[at,None]).sum(axis=0)/denom
            loss -= risk[i]-np.log(denom)
            grad -= x[i]-weighted
        grad += 2*l2*beta + l1*np.sign(beta)
        beta -= lr*grad/max(len(event_idx),1)
        if it in (300,650): lr *= 0.45
    risk=x@beta
    # approximate partial loglik without penalties
    ll=0.0
    exp_r=np.exp(np.clip(risk,-30,30))
    for i in np.where(event==1)[0]:
        denom=exp_r[time>=time[i]].sum()
        if denom>0: ll += risk[i]-np.log(denom)
    return float(ll),risk,beta

def classical_baselines(frame):
    d=frame.dropna(subset=['relapse_free_months','relapse_event']).copy(); d=d[d['relapse_free_months']>0]
    t=d['relapse_free_months'].to_numpy(float); e=d['relapse_event'].to_numpy(float)
    baselines=[]
    feature_sets={
        'ridge_cox_modules':MODULE_FEATURES,
        'lasso_cox_modules':MODULE_FEATURES,
        'ridge_cox_refined_genes':REFINED_FEATURES,
        'lasso_cox_refined_genes':REFINED_FEATURES,
        'ridge_cox_refined_genes_plus_proliferation':REFINED_PLUS_PROLIF,
        'lasso_cox_refined_genes_plus_proliferation':REFINED_PLUS_PROLIF,
    }
    for name,feats in feature_sets.items():
        present=[f for f in feats if f in d.columns]
        x=d[present].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        l1=0.02 if name.startswith('lasso') else 0.0
        l2=1e-3 if name.startswith('ridge') else 1e-4
        ll,risk,coef=fit_penalized_cox(x,t,e,l1=l1,l2=l2,seed=17)
        tmp=d.assign(_risk=risk)
        st=bootstrap_ci(tmp,'_risk','relapse_free_months','relapse_event',b=80,seed=SEED+11)
        baselines.append({'model':name,'features':';'.join(present),'n_features':len(present),'n':st['n'],'events':st['events'],'c_index':st['c_index'],'c_index_ci_low':st['c_index_ci_low'],'c_index_ci_high':st['c_index_ci_high'],'hr':st['hr'],'hr_ci_low':st['hr_ci_low'],'hr_ci_high':st['hr_ci_high'],'logrank_p':st['logrank_p'],'nonzero_coef_count':int(np.sum(np.abs(coef)>1e-3))})
    return pd.DataFrame(baselines).sort_values('c_index',ascending=False)


def internal_ci(frame):
    scores=['MRPL21','proteasome_core_score','mrpl21_proteasome_score','mrpl21_proteasome_prolif_score','intervene_refined_gene_score','intervene_v3_oof_mean_risk','raw_module_score','proliferation_score','proteasome_mito_redox_score','proteasome_mito_redox_resid_prolif']
    rows=[]
    for s in scores:
        if s in frame.columns:
            rows.append({'cohort':'GSE31210','score':s,**bootstrap_ci(frame,s,'relapse_free_months','relapse_event')})
    return pd.DataFrame(rows).sort_values('c_index',ascending=False)


def external_ci(frames):
    rows=[]
    scores=['MRPL21','proteasome_core_score','mrpl21_proteasome_score','mrpl21_proteasome_prolif_score','intervene_refined_gene_score','proteostasis_redox_state','stress_core_state','older_broad_intervene_state','proliferation_score']
    for f in frames:
        cohort=f['_cohort'].iloc[0]; time=f['_time_col'].iloc[0]; event=f['_event_col'].iloc[0]
        for s in scores:
            if s in f.columns:
                rows.append({'cohort':cohort,'score':s,**bootstrap_ci(f,s,time,event,b=80,seed=SEED+len(rows))})
    return pd.DataFrame(rows).sort_values(['cohort','c_index'],ascending=[True,False])


def write_summary(internal, external, baselines):
    lines=['TCBB Final Benchmarks and Confidence Intervals','================================================','',
           'Internal GSE31210 score confidence intervals:',internal.to_string(index=False),'',
           'Classical Cox baselines:',baselines.to_string(index=False),'',
           'External validation score confidence intervals:',external.to_string(index=False),'',
           'Interpretation:',
           '- Ridge/L1 Cox baselines provide classical survival comparators for TCBB-style benchmarking.',
           '- Bootstrap CIs quantify uncertainty for MRPL21, proteasome-core, INTERVENE-refined scores, V3 risk, and external cohorts.',
           '- External results remain partial: GSE30219 supports MRPL21/refined axes, GSE37745 is weak, and GSE68465 lacks MRPL21 mapping under current processing.']
    (RESULTS/'tcbb_final_benchmark_ci_summary.txt').write_text('\n'.join(lines))


def main():
    RESULTS.mkdir(exist_ok=True)
    frame=load_internal(); ext_frames=load_external_frames()
    internal=internal_ci(frame); external=external_ci(ext_frames); baselines=classical_baselines(frame)
    internal.to_csv(RESULTS/'tcbb_final_internal_score_confidence_intervals.csv',index=False)
    external.to_csv(RESULTS/'tcbb_final_external_score_confidence_intervals.csv',index=False)
    baselines.to_csv(RESULTS/'tcbb_final_classical_survival_baselines.csv',index=False)
    write_summary(internal,external,baselines)
    print('Wrote tcbb_final_internal_score_confidence_intervals.csv')
    print('Wrote tcbb_final_external_score_confidence_intervals.csv')
    print('Wrote tcbb_final_classical_survival_baselines.csv')
    print('Wrote tcbb_final_benchmark_ci_summary.txt')

if __name__=='__main__': main()

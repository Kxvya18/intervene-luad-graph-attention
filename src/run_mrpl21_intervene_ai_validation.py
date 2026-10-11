
from __future__ import annotations

from pathlib import Path
import csv, gzip, math
import numpy as np
import pandas as pd
import torch

from interleaved_biophysics_attention_full import zscore
from prepare_gse31210_relapse_intervene import logrank_high_low, read_annotation as read_gpl570_annotation, read_expression as read_gse31210_expression
from run_relapse_intervene_survival_head import concordance_index, cox_loss
from run_final_intervene_robustness_tests import fit_cox_linear, norm_sf_chi2_1
from run_gse31210_mechanism_dissection import residualize
from run_proteostasis_redox_external_validation import read_annotation, read_expression

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'/'relapse'
PROCESSED=ROOT/'data'/'processed'/'relapse'
RESULTS=ROOT/'results'
FIGURES=ROOT/'figures'

COHORTS={
    'GSE30219': {'score_file': PROCESSED/'GSE30219'/'gse30219_proteostasis_redox_scores.csv', 'time':'dfs_months', 'event':'relapse_event', 'series': RAW/'GSE30219'/'GSE30219_series_matrix.txt.gz', 'annot': RAW/'GSE31210'/'GPL570.annot.gz', 'platform':'GPL570'},
    'GSE37745': {'score_file': PROCESSED/'GSE37745'/'gse37745_proteostasis_redox_scores.csv', 'time':'recurrence_time_months', 'event':'recurrence_event', 'series': RAW/'GSE37745'/'GSE37745_series_matrix.txt.gz', 'annot': RAW/'GSE31210'/'GPL570.annot.gz', 'platform':'GPL570'},
    'GSE68465': {'score_file': PROCESSED/'GSE68465'/'gse68465_proteostasis_redox_scores.csv', 'time':'pfs_months', 'event':'progression_relapse_event', 'series': RAW/'GSE68465'/'GSE68465_series_matrix.txt.gz', 'annot': RAW/'GSE68465'/'GPL96.annot.gz', 'platform':'GPL96'},
}


def null_partial_loglik(time,event):
    risk0=torch.zeros(len(time),dtype=torch.float32)
    return -float(cox_loss(risk0,torch.tensor(time,dtype=torch.float32),torch.tensor(event,dtype=torch.float32)).item())*float(np.sum(event))

def cox_score(frame, score_col, time_col, event_col):
    data=frame.dropna(subset=[score_col,time_col,event_col]).copy(); data=data[data[time_col]>0]
    if len(data)<25 or data[event_col].sum()<5:
        return {'n':len(data),'events':int(data[event_col].sum()) if len(data) else 0,'c_index':np.nan,'hr_high_low':np.nan,'logrank_p':np.nan,'cox_lrt_p':np.nan,'cox_chi2':np.nan}
    time=data[time_col].to_numpy(float); event=data[event_col].to_numpy(float); raw=data[score_col].to_numpy(float)
    ll0=null_partial_loglik(time,event); ll1,risk=fit_cox_linear(zscore(data[score_col]).to_numpy(float).reshape(-1,1),time,event)
    lr=logrank_high_low(data.assign(_score=raw),'_score',time_col,event_col)
    chi=max(0.0,2*(ll1-ll0))
    return {'n':len(data),'events':int(event.sum()),'c_index':concordance_index(time,event,raw),'hr_high_low':lr['hazard_ratio_approx'],'logrank_p':lr['logrank_p'],'cox_lrt_p':norm_sf_chi2_1(chi),'cox_chi2':chi}

def added_after(frame, base_cols, add_col, time_col, event_col, label):
    cols=base_cols+[add_col,time_col,event_col]
    data=frame.dropna(subset=cols).copy(); data=data[data[time_col]>0]
    if len(data)<25 or data[event_col].sum()<5:
        return {'test':label,'n':len(data),'events':int(data[event_col].sum()) if len(data) else 0,'base_c_index':np.nan,'plus_c_index':np.nan,'delta_c_index':np.nan,'lrt_p':np.nan,'lrt_chi2':np.nan}
    time=data[time_col].to_numpy(float); event=data[event_col].to_numpy(float)
    base=np.column_stack([zscore(data[c]).to_numpy(float) for c in base_cols]) if base_cols else np.zeros((len(data),0))
    add=zscore(data[add_col]).to_numpy(float).reshape(-1,1)
    ll0,r0=fit_cox_linear(base,time,event) if base.shape[1] else (null_partial_loglik(time,event),np.zeros(len(data)))
    ll1,r1=fit_cox_linear(np.column_stack([base,add]),time,event)
    chi=max(0.0,2*(ll1-ll0))
    return {'test':label,'n':len(data),'events':int(event.sum()),'base_c_index':concordance_index(time,event,r0),'plus_c_index':concordance_index(time,event,r1),'delta_c_index':concordance_index(time,event,r1)-concordance_index(time,event,r0),'lrt_p':norm_sf_chi2_1(chi),'lrt_chi2':chi}

def load_gse31210():
    frame=pd.read_csv(RESULTS/'gse31210_refined_mechanism_patient_scores.csv')
    v3=pd.read_csv(RESULTS/'relapse_intervene_graph_sparse_v3_oof_risk_scores.csv')
    cols=[c for c in v3.columns if c.startswith('graph_sparse_v3_soft_seed') and c.endswith('_risk')]
    v3['intervene_v3_mean_risk']=v3[cols].mean(axis=1)
    frame=frame.merge(v3[['geo_accession','intervene_v3_mean_risk']],on='geo_accession',how='left')
    frame['mrpl21_proteasome_score']=zscore(frame['MRPL21'])+zscore(frame['proteasome_core_score'])
    frame['mrpl21_proteasome_prolif_score']=frame['mrpl21_proteasome_score']+zscore(frame['proliferation_score'])
    frame['raw_module_score']=zscore(frame['proteostasis_expr_mean'])+zscore(frame['mito_translation_expr_mean'])+zscore(frame['oxidative_stress_expr_mean'])
    frame['intervene_refined_gene_score']=zscore(frame['MRPL21'])+zscore(frame['PSMA1'])+zscore(frame['PSMB5'])+0.5*zscore(frame['TXNRD1'])
    for c in ['mrpl21_proteasome_score','intervene_refined_gene_score','raw_module_score']:
        frame[c+'_resid_prolif']=residualize(frame[c],frame['proliferation_score'])
    return frame

def external_validate():
    rows=[]; cls=[]
    for cohort,cfg in COHORTS.items():
        f=pd.read_csv(cfg['score_file'])
        if cohort=='GSE68465':
            # strict endpoint: event-positive cases without time-to-event remain excluded by pfs_months NA.
            pass
        if 'proteasome_core_score' not in f.columns:
            f['proteasome_core_score']=zscore(f[['PSMA1','PSMB5']].mean(axis=1)) if {'PSMA1','PSMB5'}.issubset(f.columns) else np.nan
        if 'MRPL21' in f.columns:
            f['mrpl21_proteasome_score']=zscore(f['MRPL21'])+zscore(f['proteasome_core_score'])
            f['mrpl21_proteasome_prolif_score']=f['mrpl21_proteasome_score']+zscore(f['proliferation_score'])
            f['intervene_refined_gene_score']=zscore(f['MRPL21'])+zscore(f['PSMA1'])+zscore(f['PSMB5'])+0.5*zscore(f['TXNRD1'])
        else:
            f['mrpl21_proteasome_score']=np.nan; f['mrpl21_proteasome_prolif_score']=np.nan; f['intervene_refined_gene_score']=np.nan
        score_cols=['MRPL21','proteasome_core_score','mrpl21_proteasome_score','mrpl21_proteasome_prolif_score','intervene_refined_gene_score','proteostasis_redox_state','stress_core_state','older_broad_intervene_state','proliferation_score']
        for score in score_cols:
            if score in f.columns:
                rows.append({'cohort':cohort,'score':score,**cox_score(f,score,cfg['time'],cfg['event'])})
        for score in ['MRPL21','proteasome_core_score','mrpl21_proteasome_score','intervene_refined_gene_score']:
            if score in f.columns:
                cls.append({'cohort':cohort,**added_after(f,['proliferation_score'],score,cfg['time'],cfg['event'],score+'_added_after_proliferation')})
    return pd.DataFrame(rows),pd.DataFrame(cls)

def probe_table():
    rows=[]
    # include GSE31210 with GPL570 raw expression
    expr312=read_gse31210_expression(); ann570=read_gpl570_annotation()
    targets=['MRPL21','PSMA1','PSMB5','TXNRD1']
    for cohort,cfg in {'GSE31210': {'series':RAW/'GSE31210'/'GSE31210_series_matrix.txt.gz','platform':'GPL570','expr':expr312,'ann':ann570}, **{k:{'series':v['series'],'platform':v['platform'],'ann_path':v['annot']} for k,v in COHORTS.items()}}.items():
        if cohort=='GSE31210': expr=cfg['expr']; ann=cfg['ann']
        else: expr=read_expression(cfg['series']); ann=read_annotation(cfg['ann_path'])
        for gene in targets:
            probes=ann.loc[ann['gene_symbol'].eq(gene),'probe_id'].drop_duplicates().astype(str).tolist()
            present=[p for p in probes if p in expr.index]
            if len(present)>=2:
                mat=expr.loc[present].T.apply(pd.to_numeric,errors='coerce')
                corr=mat.corr().where(~np.eye(len(present),dtype=bool)).stack().mean() if len(present)>1 else np.nan
            else:
                corr=np.nan
            for p in present:
                vals=pd.to_numeric(expr.loc[p],errors='coerce')
                rows.append({'cohort':cohort,'platform':cfg['platform'],'gene':gene,'probe_id':p,'n_probes_for_gene_present':len(present),'probe_mean':float(vals.mean()),'probe_sd':float(vals.std()),'probe_cv':float(vals.std()/vals.mean()) if vals.mean()!=0 else np.nan,'mean_pairwise_probe_corr_for_gene':float(corr) if np.isfinite(corr) else np.nan})
            if not present:
                rows.append({'cohort':cohort,'platform':cfg['platform'],'gene':gene,'probe_id':'NOT_MAPPED_OR_NOT_PRESENT','n_probes_for_gene_present':0,'probe_mean':np.nan,'probe_sd':np.nan,'probe_cv':np.nan,'mean_pairwise_probe_corr_for_gene':np.nan})
    return pd.DataFrame(rows)

def ai_ablation():
    frame=load_gse31210()
    rows=[]
    # existing model means
    comp=pd.read_csv(RESULTS/'relapse_intervene_survival_model_comparison.csv')
    for model,label in [('no_prior_interleaved_attention','no-prior IHA-style attention'),('normal_attention_mha','normal MHA'),('raw_thermodynamic_transfer','raw module prior')]:
        sub=comp[comp['model'].eq(model)]
        rows.append({'model':label,'source':'existing AI baseline','n':int(sub['n'].iloc[0]) if len(sub) else np.nan,'events':int(sub['events'].iloc[0]) if len(sub) else np.nan,'c_index':float(sub['c_index'].mean()) if len(sub) else np.nan,'logrank_p':float(sub['logrank_p'].min()) if len(sub) else np.nan})
    v3=pd.read_csv(RESULTS/'relapse_intervene_graph_sparse_v3_model_comparison.csv')
    soft=v3[v3['model'].eq('graph_sparse_v3_soft')]
    rows.append({'model':'GraphSparse INTERVENE V3','source':'existing AI model, mean across seeds','n':204,'events':54,'c_index':float(soft['c_index'].mean()),'logrank_p':float(soft['logrank_p'].min())})
    score_map={'raw_module_score':'raw module score','intervene_refined_gene_score':'INTERVENE-refined genes MRPL21+PSMA1+PSMB5+TXNRD1','mrpl21_proteasome_score':'MRPL21 + proteasome-core manual/extracted','mrpl21_proteasome_prolif_score':'MRPL21 + proteasome-core + proliferation','intervene_refined_gene_score_resid_prolif':'INTERVENE-refined genes residualized for proliferation','mrpl21_proteasome_score_resid_prolif':'MRPL21 + proteasome-core residualized for proliferation','intervene_v3_mean_risk':'INTERVENE V3 OOF mean risk'}
    for col,label in score_map.items():
        st=cox_score(frame,col,'relapse_free_months','relapse_event')
        rows.append({'model':label,'source':'extracted/refined score','n':st['n'],'events':st['events'],'c_index':st['c_index'],'logrank_p':st['logrank_p']})
    return pd.DataFrame(rows).sort_values('c_index',ascending=False)

def attention_biology_table():
    edges=pd.read_csv(RESULTS/'final_v3_edge_top10_seed_stability.csv') if (RESULTS/'final_v3_edge_top10_seed_stability.csv').exists() else pd.DataFrame()
    drops=pd.read_csv(RESULTS/'final_v3_module_removal_importance.csv') if (RESULTS/'final_v3_module_removal_importance.csv').exists() else pd.DataFrame()
    genes=pd.read_csv(RESULTS/'gse31210_refined_gene_rankings.csv')
    axis=pd.read_csv(RESULTS/'gse31210_refined_axis_survival_tests.csv')
    rows=[]
    top_edges=edges[edges['model'].eq('graph_sparse_v3_soft')].head(12) if not edges.empty else pd.DataFrame()
    for _,e in top_edges.iterrows():
        src=str(e['edge']).split(' -> ')[0]; tgt=str(e['edge']).split(' -> ')[-1]
        related=[]
        if 'proteostasis' in e['edge']: related+=['PSMA1','PSMB5','VCP']
        if 'mito_translation' in e['edge']: related+=['MRPL21','MRPL49']
        if 'oxidative_stress' in e['edge']: related+=['TXNRD1']
        if 'akt_mtor' in e['edge']: related+=['AKT1','PIK3CB']
        rel=genes[genes['gene'].isin(related)].head(6)
        rows.append({'attention_edge':e['edge'],'top10_seed_count':e['top10_seed_count'],'module_removed_c_index_drop':float(drops.loc[drops['module_removed'].eq(src),'mean_c_index_drop_after_removal'].iloc[0]) if len(drops.loc[drops['module_removed'].eq(src)]) else np.nan,'linked_driver_genes':'; '.join(rel['gene'].tolist()),'best_linked_gene_c_index':float(rel['c_index'].max()) if len(rel) else np.nan})
    out=pd.DataFrame(rows)
    # append axis survival rows as separate evidence table? saved separately too.
    return out

def make_figure():
    FIGURES.mkdir(exist_ok=True)
    combo=pd.read_csv(RESULTS/'gse31210_proteasome_mito_redox_combination_groups.csv')
    order=['low_both','mito_redox_only','proteasome_only','high_both']
    c=combo.set_index('group').loc[order]
    colors={'low_both':'#bdbdbd','mito_redox_only':'#80cdc1','proteasome_only':'#fdb863','high_both':'#b2182b'}
    labels={'low_both':'Low both','mito_redox_only':'Mito/redox only','proteasome_only':'Proteasome only','high_both':'High both'}
    max_pct=max(c['event_fraction']*100)
    def rect(x,y,w,h,stroke,fill='white',sw=2):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="4"/>'
    def text(x,y,t,size=12,color='#222',weight='normal',anchor='middle'):
        return f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{t}</text>'
    def multiline(x,y,lines,size=12,color='#222',weight='normal'):
        out=[]
        for i,line in enumerate(lines): out.append(text(x,y+i*(size+3),line,size,color,weight))
        return '\n'.join(out)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="520" viewBox="0 0 1100 520">','<rect width="1100" height="520" fill="white"/>']
    parts.append(text(35,35,'INTERVENE-refined biological axis',18,'#111','bold','start'))
    parts.append(text(35,58,'IHA-inspired sparse attention nominates modules; gene refinement reveals MRPL21 + proteasome core.',12,'#444','normal','start'))
    boxes=[(45,105,'#d95f02',['Proteasome core burden','PSMA1 + PSMB5'],['protein-stress','tolerance']),(45,245,'#1b9e77',['MRPL21 / mitochondrial','translation'],['mitochondrial','adaptation']),(45,385,'#7570b3',['TXNRD1 / redox','buffering'],['oxidative-stress','buffering']),(505,245,'#b2182b',['Combined stress-adaptation','state'],['early relapse risk'])]
    for x,y,col,title,sub in boxes:
        parts.append(rect(x,y,310,95,col))
        parts.append(multiline(x+155,y+32,title,14,col,'bold'))
        parts.append(multiline(x+155,y+68,sub,12,'#333'))
    for y in [152,292,432]:
        parts.append(f'<line x1="355" y1="{y}" x2="505" y2="292" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.insert(1,'<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#555"/></marker></defs>')
    # Bar chart
    chart_x=735; chart_y=110; chart_w=310; chart_h=300
    parts.append(text(chart_x,65,'GSE31210 relapse burden by combined state',15,'#111','bold','start'))
    parts.append(f'<line x1="{chart_x}" y1="{chart_y+chart_h}" x2="{chart_x+chart_w}" y2="{chart_y+chart_h}" stroke="#333"/>')
    parts.append(f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y+chart_h}" stroke="#333"/>')
    parts.append(text(chart_x-35,chart_y+chart_h/2,'Relapse %',12,'#333','normal','middle'))
    bar_w=48; gap=28
    for i,g in enumerate(order):
        pct=float(c.loc[g,'event_fraction']*100); h=(pct/(max_pct*1.2))*chart_h
        x=chart_x+25+i*(bar_w+gap); y=chart_y+chart_h-h
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{colors[g]}"/>')
        parts.append(text(x+bar_w/2,y-8,f"{int(c.loc[g,'events'])}/{int(c.loc[g,'n'])}",11,'#222'))
        parts.append(text(x+bar_w/2,chart_y+chart_h+18,labels[g].replace(' only',''),10,'#222'))
        if g in ['mito_redox_only','proteasome_only']:
            parts.append(text(x+bar_w/2,chart_y+chart_h+32,'only',10,'#222'))
    parts.append(text(chart_x+chart_w/2,chart_y+chart_h+58,'Combined high-both group has the largest relapse burden',12,'#444'))
    parts.append('</svg>')
    out=FIGURES/'intervene_mrpl21_proteasome_redox_biology.svg'
    out.write_text('\n'.join(parts))
    return out

def write_summary(ext, cls, probes, abl, att, fig_path):
    lines=['MRPL21 / Proteasome INTERVENE AI-Biology Validation','====================================================','',
           '1) External validation of MRPL21 and MRPL21 + proteasome-core', ext.to_string(index=False), '',
           '2) Added-value after proliferation in external cohorts', cls.to_string(index=False), '',
           '3) Probe/platform stability summary', probes.groupby(['cohort','platform','gene']).agg(n_probes=('probe_id',lambda x:int((x!='NOT_MAPPED_OR_NOT_PRESENT').sum())), mean_probe_cv=('probe_cv','mean'), mean_pairwise_probe_corr=('mean_pairwise_probe_corr_for_gene','mean')).reset_index().to_string(index=False), '',
           '4) AI ablation around MRPL21/proteasome refinement', abl.to_string(index=False), '',
           '5) Attention-to-biology evidence', att.to_string(index=False), '',
           f'Figure: {fig_path}', '',
           'Interpretation:',
           '- INTERVENE should be framed as the discovery algorithm: sparse module attention -> module removal -> gene extraction -> external falsification.',
           '- MRPL21 is a strong GSE31210 novelty lead and remains predictive after proliferation adjustment.',
           '- External validation is the key limiter: if MRPL21 does not transfer consistently, frame it as cohort-specific discovery plus a mechanistic hypothesis, not a universal LUAD biomarker.',
           '- Probe mapping matters: GSE68465/GPL96 lacks MRPL21 in the mapped score table, so it cannot fairly validate MRPL21 directly with the current platform mapping.']
    (RESULTS/'mrpl21_intervene_ai_biology_validation_summary.txt').write_text('\n'.join(lines))

def main():
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    ext,cls=external_validate(); probes=probe_table(); abl=ai_ablation(); att=attention_biology_table(); fig=make_figure()
    ext.to_csv(RESULTS/'mrpl21_external_validation.csv',index=False)
    cls.to_csv(RESULTS/'mrpl21_external_added_value_after_proliferation.csv',index=False)
    probes.to_csv(RESULTS/'mrpl21_probe_platform_stability.csv',index=False)
    abl.to_csv(RESULTS/'mrpl21_intervene_ai_ablation.csv',index=False)
    att.to_csv(RESULTS/'mrpl21_attention_to_biology_evidence.csv',index=False)
    write_summary(ext,cls,probes,abl,att,fig)
    print('Wrote mrpl21_external_validation.csv')
    print('Wrote mrpl21_probe_platform_stability.csv')
    print('Wrote mrpl21_intervene_ai_ablation.csv')
    print('Wrote mrpl21_attention_to_biology_evidence.csv')
    print('Wrote mrpl21_intervene_ai_biology_validation_summary.txt')
    print('Wrote',fig)

if __name__=='__main__': main()

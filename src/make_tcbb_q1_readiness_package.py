
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
NOTES = ROOT / 'paper_notes'


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(x, digits=3):
    if pd.isna(x): return 'NA'
    if isinstance(x, (int, np.integer)): return str(int(x))
    try:
        return f'{float(x):.{digits}g}'
    except Exception:
        return str(x)


def evidence_rows() -> list[dict[str, str]]:
    rows=[]
    abl=read_csv('mrpl21_intervene_ai_ablation.csv')
    ext=read_csv('mrpl21_external_validation.csv')
    perm=read_csv('final_permutation_control.csv')
    clin=read_csv('final_clinical_adjusted_cox.csv')
    boot=read_csv('final_paired_bootstrap_c_index.csv')
    edge=read_csv('final_v3_edge_top10_seed_stability.csv')
    drop=read_csv('final_v3_module_removal_importance.csv')
    probe=read_csv('mrpl21_probe_platform_stability.csv')
    combo=read_csv('gse31210_proteasome_mito_redox_combination_groups.csv')
    internal_ci=read_csv('tcbb_final_internal_score_confidence_intervals.csv')
    external_ci=read_csv('tcbb_final_external_score_confidence_intervals.csv')
    cox_base=read_csv('tcbb_final_classical_survival_baselines.csv')

    def add(area, item, status, evidence, file):
        rows.append({'area':area,'item':item,'status':status,'evidence':evidence,'source_file':file})

    if not abl.empty:
        best=abl.sort_values('c_index',ascending=False).iloc[0]
        v3=abl[abl['model'].astype(str).str.contains('INTERVENE V3 OOF',regex=False)]
        raw=abl[abl['model'].astype(str).eq('raw module score')]
        add('AI benchmark','Refined gene axis outperforms raw broad module score','complete',f"best={best['model']} C-index={fmt(best['c_index'])}; V3 OOF={fmt(v3['c_index'].iloc[0]) if len(v3) else 'NA'}; raw module={fmt(raw['c_index'].iloc[0]) if len(raw) else 'NA'}",'results/mrpl21_intervene_ai_ablation.csv')
    else:
        add('AI benchmark','Refined gene axis benchmark','missing','mrpl21_intervene_ai_ablation.csv not found','')

    if not perm.empty:
        r=perm.iloc[0]
        add('Statistics','Permutation-label control','complete',f"observed C={fmt(r['observed_c_index'])}, null mean={fmt(r['perm_mean_c_index'])}, p={fmt(r['perm_p_c_index_ge_observed'])}",'results/final_permutation_control.csv')

    if not clin.empty:
        best=clin.sort_values('cox_lrt_p_for_v3').iloc[0]
        add('Statistics','Clinical-adjusted survival value','complete',f"best adjusted model={best['clinical_model']}, delta C={fmt(best['delta_c_index'])}, p={fmt(best['cox_lrt_p_for_v3'])}",'results/final_clinical_adjusted_cox.csv')

    if not boot.empty:
        raw=boot[boot['comparison'].astype(str).str.contains('raw_transfer')]
        mha=boot[boot['comparison'].astype(str).str.contains('normal_attention')]
        add('Statistics','Paired bootstrap model improvement','partial',f"V3 vs raw delta={fmt(raw['observed_delta_c_index'].iloc[0]) if len(raw) else 'NA'}, p={fmt(raw['bootstrap_p_delta_le_0'].iloc[0]) if len(raw) else 'NA'}; V3 vs MHA delta={fmt(mha['observed_delta_c_index'].iloc[0]) if len(mha) else 'NA'}",'results/final_paired_bootstrap_c_index.csv')

    if not edge.empty:
        stable=edge[(edge['model'].eq('graph_sparse_v3_soft')) & (edge['top10_seed_count']>=5)]
        add('Interpretability','Stable sparse attention edges','complete',f"{len(stable)} soft-model edges appear in top10 across all 5 seeds",'results/final_v3_edge_top10_seed_stability.csv')

    if not drop.empty:
        top=drop.sort_values('mean_c_index_drop_after_removal',ascending=False).iloc[0]
        add('Interpretability','Module-removal importance','complete',f"top removed module={top['module_removed']}, C-index drop={fmt(top['mean_c_index_drop_after_removal'])}",'results/final_v3_module_removal_importance.csv')

    if not combo.empty:
        hi=combo[combo['group'].eq('high_both')].iloc[0]
        lo=combo[combo['group'].eq('low_both')].iloc[0]
        add('Biology','Combined proteasome + mito/redox state','complete',f"high-both relapse={int(hi['events'])}/{int(hi['n'])} ({fmt(hi['event_fraction']*100)}%); low-both={int(lo['events'])}/{int(lo['n'])} ({fmt(lo['event_fraction']*100)}%)",'results/gse31210_proteasome_mito_redox_combination_groups.csv')

    ext_source = external_ci if not external_ci.empty else ext
    if not ext_source.empty:
        gse302=ext_source[(ext_source['cohort'].eq('GSE30219')) & (ext_source['score'].eq('MRPL21'))]
        gse377=ext_source[(ext_source['cohort'].eq('GSE37745')) & (ext_source['score'].eq('MRPL21'))]
        add('External validation','MRPL21 external validation','partial',f"GSE30219 MRPL21 C={fmt(gse302['c_index'].iloc[0]) if len(gse302) else 'NA'}, p={fmt(gse302['logrank_p'].iloc[0]) if len(gse302) else 'NA'}; GSE37745 C={fmt(gse377['c_index'].iloc[0]) if len(gse377) else 'NA'}, p={fmt(gse377['logrank_p'].iloc[0]) if len(gse377) else 'NA'}",'results/tcbb_final_external_score_confidence_intervals.csv')

    if not internal_ci.empty:
        refined=internal_ci[internal_ci['score'].eq('intervene_refined_gene_score')]
        v3ci=internal_ci[internal_ci['score'].eq('intervene_v3_oof_mean_risk')]
        add('Statistics','Bootstrap confidence intervals for internal scores','complete',f"refined C={fmt(refined['c_index'].iloc[0])} CI={fmt(refined['c_index_ci_low'].iloc[0])}-{fmt(refined['c_index_ci_high'].iloc[0])}; V3 C={fmt(v3ci['c_index'].iloc[0])} CI={fmt(v3ci['c_index_ci_low'].iloc[0])}-{fmt(v3ci['c_index_ci_high'].iloc[0])}",'results/tcbb_final_internal_score_confidence_intervals.csv')

    if not cox_base.empty:
        best=cox_base.sort_values('c_index',ascending=False).iloc[0]
        add('AI benchmark','Classical Cox survival baselines','complete',f"best={best['model']} C-index={fmt(best['c_index'])}; broad-module ridge/L1 also included",'results/tcbb_final_classical_survival_baselines.csv')

    if not probe.empty:
        mr=probe[probe['gene'].eq('MRPL21')]
        gpl570=mr[mr['platform'].eq('GPL570')]
        gpl96=mr[mr['platform'].eq('GPL96')]
        add('Data quality','MRPL21 probe/platform audit','limitation',f"GPL570 MRPL21 probes per cohort={'; '.join(gpl570['cohort'].astype(str)+':'+gpl570['n_probes_for_gene_present'].astype(str))}; GPL96={'; '.join(gpl96['cohort'].astype(str)+':'+gpl96['n_probes_for_gene_present'].astype(str))}",'results/mrpl21_probe_platform_stability.csv')

    add('Submission readiness','Wet-lab/causal validation','missing','No perturbation experiment; frame as computational discovery lead, not causal proof','')
    add('Submission readiness','Final manuscript figures','complete','Final SVG panels generated for architecture, AI ablation, biology mechanism, and external validation/falsification','figures/tcbb_figure1_intervene_architecture.svg; figures/tcbb_figure2_ai_ablation_attention.svg; figures/intervene_mrpl21_proteasome_redox_biology.svg; figures/tcbb_figure4_external_validation_falsification.svg')
    add('Submission readiness','One-command reproducibility','complete','Final runner and reproducibility manifest created','src/run_tcbb_final_pipeline.sh; paper_notes/tcbb_reproducibility_manifest.md')
    add('Submission readiness','Manuscript draft','complete','Draft Methods, Results, Discussion, limitations, and novelty comparison created','paper_notes/tcbb_manuscript_draft.md')
    return rows


def table_registry() -> pd.DataFrame:
    entries = [
        ('Table 1','Dataset/cohort summary','final_external_dataset_scout.csv; mrpl21_external_validation.csv'),
        ('Table 2','AI ablation and benchmark','mrpl21_intervene_ai_ablation.csv; tcbb_final_classical_survival_baselines.csv; relapse_intervene_survival_model_comparison.csv'),
        ('Table 3','Robustness statistics and confidence intervals','tcbb_final_internal_score_confidence_intervals.csv; tcbb_final_external_score_confidence_intervals.csv; final_paired_bootstrap_c_index.csv; final_permutation_control.csv; final_clinical_adjusted_cox.csv'),
        ('Table 4','Attention-to-biology evidence','mrpl21_attention_to_biology_evidence.csv; final_v3_edge_top10_seed_stability.csv'),
        ('Table 5','External validation and probe audit','mrpl21_external_validation.csv; mrpl21_probe_platform_stability.csv'),
    ]
    return pd.DataFrame(entries, columns=['table','purpose','source_files'])


def figure_registry() -> pd.DataFrame:
    entries = [
        ('Figure 1','INTERVENE architecture','exists','figures/tcbb_figure1_intervene_architecture.svg'),
        ('Figure 2','AI ablation and sparse attention stability','exists','figures/tcbb_figure2_ai_ablation_attention.svg'),
        ('Figure 3','MRPL21/proteasome biological mechanism','exists','figures/intervene_mrpl21_proteasome_redox_biology.svg'),
        ('Figure 4','External validation/falsification','exists','figures/tcbb_figure4_external_validation_falsification.svg'),
    ]
    return pd.DataFrame(entries, columns=['figure','purpose','status','source_files'])




def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '_No rows._'
    cols=list(df.columns)
    def cell(v):
        text='' if pd.isna(v) else str(v)
        return text.replace('|','\\|').replace('\n','<br>')
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,row in df.iterrows():
        lines.append('| '+' | '.join(cell(row[c]) for c in cols)+' |')
    return '\n'.join(lines)

def write_dashboard(evidence: pd.DataFrame, tables: pd.DataFrame, figures: pd.DataFrame) -> None:
    counts=evidence['status'].value_counts().to_dict()
    lines=[]
    lines.append('# TCBB / Q1 Evidence Dashboard')
    lines.append('')
    lines.append('## Current Verdict')
    lines.append('')
    lines.append('The project is Q1-plausible as a method-led computational biology paper, but not yet ready as a pure biological breakthrough paper. The central contribution should be INTERVENE as an interpretable physics-guided attention framework; MRPL21/proteasome is the case-study discovery output.')
    lines.append('')
    lines.append('## Readiness Counts')
    lines.append('')
    for k in ['complete','partial','limitation','missing']:
        lines.append(f'- {k}: {counts.get(k,0)}')
    lines.append('')
    lines.append('## Evidence Checklist')
    lines.append('')
    lines.append(md_table(evidence))
    lines.append('')
    lines.append('## Final Table Registry')
    lines.append('')
    lines.append(md_table(tables))
    lines.append('')
    lines.append('## Final Figure Registry')
    lines.append('')
    lines.append(md_table(figures))
    lines.append('')
    lines.append('## Immediate Work Needed For TCBB/Q1')
    lines.append('')
    lines.append('1. Increase bootstrap iterations to 1000 before final submission tables.')
    lines.append('2. Add one stronger external RNA-seq relapse cohort with direct MRPL21 measurement if available.')
    lines.append('3. Add nested cross-validation if model tuning choices are presented as optimized.')
    lines.append('4. Keep the manuscript method-led and avoid claiming MRPL21 as a proven causal relapse driver.')
    lines.append('5. Convert SVG figures to journal-resolution PDF/TIFF after final content decisions.')
    (RESULTS/'tcbb_q1_evidence_dashboard.md').write_text('\n'.join(lines))


def main():
    evidence=pd.DataFrame(evidence_rows())
    tables=table_registry()
    figures=figure_registry()
    evidence.to_csv(RESULTS/'tcbb_q1_evidence_checklist.csv',index=False)
    tables.to_csv(RESULTS/'tcbb_final_table_registry.csv',index=False)
    figures.to_csv(RESULTS/'tcbb_final_figure_registry.csv',index=False)
    write_dashboard(evidence,tables,figures)
    print('Wrote results/tcbb_q1_evidence_dashboard.md')
    print('Wrote results/tcbb_q1_evidence_checklist.csv')
    print('Wrote results/tcbb_final_table_registry.csv')
    print('Wrote results/tcbb_final_figure_registry.csv')

if __name__ == '__main__':
    main()

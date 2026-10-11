from __future__ import annotations

from pathlib import Path
import math
import textwrap
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

BG = '#ffffff'
INK = '#1b1b1b'
MUTED = '#5d6670'
BLUE = '#2f6db3'
TEAL = '#0f8b8d'
RED = '#c94736'
ORANGE = '#d98b28'
GREEN = '#438b4a'
PURPLE = '#6f5aa7'
GRAY = '#d8dde3'
LIGHT = '#f5f7fa'


def esc(s: object) -> str:
    return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')


def text(x, y, s, size=18, weight='normal', fill=INK, anchor='start'):
    return f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>'


def wrapped(x, y, s, width=34, size=15, line_h=20, fill=MUTED, weight='normal'):
    out=[]
    for i,line in enumerate(textwrap.wrap(str(s), width=width)):
        out.append(text(x,y+i*line_h,line,size=size,weight=weight,fill=fill))
    return '\n'.join(out)


def box(x,y,w,h,label,sub='',fill=LIGHT,stroke=GRAY,color=INK):
    parts=[f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
           text(x+16,y+30,label,18,'700',color)]
    if sub:
        parts.append(wrapped(x+16,y+58,sub,width=max(18,int(w/8)),size=13,line_h=17,fill=MUTED))
    return '\n'.join(parts)


def arrow(x1,y1,x2,y2,color=MUTED,w=2.4):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" marker-end="url(#arrow)"/>'


def svg_wrap(w,h,body):
    defs='<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#5d6670"/></marker></defs>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="{BG}"/>{defs}\n{body}\n</svg>\n'


def fig1_architecture():
    body=[]
    body += [text(42,48,'Figure 1. INTERVENE architecture',24,'700')]
    body += [text(42,78,'IHA-inspired interleaved head mixing is adapted from language tokens to biological module tokens.',15,'normal',MUTED)]
    xs=[42,280,518,756,994]
    labels=[
        ('Thermodynamic prior','DepMap dependency, expression, CN and PRISM define stress-survival module pressure.'),
        ('Module tokens','Proteostasis, redox, mitochondrial translation, apoptosis/IAP, SREBP/lipid and AKT/mTOR become tokens.'),
        ('Interleaved attention','Pseudo-heads mix module information before survival prediction.'),
        ('GraphSparse layer','Biology-guided sparse edges stabilize interpretable module couplings.'),
        ('Outputs','Relapse risk, stable module edges and gene-level drivers are extracted for validation.')]
    colors=[BLUE,TEAL,PURPLE,ORANGE,GREEN]
    for i,(x,(lab,sub)) in enumerate(zip(xs,labels)):
        body += [box(x,130,190,145,lab,sub,fill='#fbfcfe',stroke=colors[i],color=colors[i])]
        if i < len(xs)-1:
            body += [arrow(x+190,202,xs[i+1]-16,202)]
    body += [text(518,340,'Interleaved pseudo-head mixing',18,'700')]
    y0=375
    modules=['P','R','M','A','L','G']
    cols=[BLUE,RED,TEAL,ORANGE,GREEN,PURPLE]
    for row in range(3):
        body += [text(440,y0+row*48+25,f'head {row+1}',13,'700',MUTED)]
        for j,m in enumerate(modules):
            x=520+j*58+(row%2)*18
            body += [f'<circle cx="{x}" cy="{y0+row*48+18}" r="18" fill="{cols[j]}" opacity="0.86"/>', text(x,y0+row*48+24,m,15,'700','#fff','middle')]
        for j in range(5):
            body += [arrow(538+j*58+(row%2)*18,y0+row*48+18,560+j*58+(row%2)*18,y0+row*48+18,color='#b8c0ca',w=1.4)]
    body += [wrapped(42,350,'Special case: if attention mixing is identity and graph sparsity is zero, INTERVENE reduces to a raw thermodynamic/module score. With sparse interleaving, it learns cross-module interaction terms.',width=44,size=15,line_h=21,fill=INK)]
    body += [wrapped(994,350,'Primary use in this project: discover relapse-associated stress-survival structure, then extract a compact MRPL21/proteasome/redox gene axis for external testing.',width=40,size=15,line_h=21,fill=INK)]
    (FIGURES/'tcbb_figure1_intervene_architecture.svg').write_text(svg_wrap(1240,560,'\n'.join(body)))


def bar_chart(rows, x, y, w, h, label_col, value_col, color=BLUE, max_val=None, min_val=0.5):
    vals=[float(r[value_col]) for r in rows]
    if max_val is None: max_val=max(vals) if vals else 1
    scale=lambda v: (v-min_val)/(max_val-min_val) if max_val>min_val else 0
    out=[]
    bar_h=min(28, h/max(len(rows),1)-7)
    for i,r in enumerate(rows):
        yy=y+i*(bar_h+8)
        bw=max(1,scale(float(r[value_col]))*w)
        out.append(text(x,yy+bar_h*0.72,str(r[label_col])[:44],12,'normal',INK))
        out.append(f'<rect x="{x+290}" y="{yy}" width="{w}" height="{bar_h}" fill="#edf1f5"/>')
        out.append(f'<rect x="{x+290}" y="{yy}" width="{bw}" height="{bar_h}" fill="{color}"/>')
        out.append(text(x+300+bw,yy+bar_h*0.72,f"{float(r[value_col]):.3f}",12,'700',INK))
    return '\n'.join(out)


def fig2_ablation():
    abl=pd.read_csv(RESULTS/'mrpl21_intervene_ai_ablation.csv').head(11)
    edges=pd.read_csv(RESULTS/'final_v3_edge_top10_seed_stability.csv')
    rem=pd.read_csv(RESULTS/'final_v3_module_removal_importance.csv')
    body=[text(42,48,'Figure 2. AI ablation and interpretability evidence',24,'700'),
          text(42,78,'INTERVENE is evaluated as both a survival model and a mechanism extraction engine.',15,'normal',MUTED)]
    body.append(text(42,122,'A. Survival ablation (C-index)',18,'700'))
    body.append(bar_chart(abl.to_dict('records'),42,150,280,360,'model','c_index',color=BLUE,max_val=0.78,min_val=0.64))
    body.append(text(650,122,'B. Stable sparse attention edges',18,'700'))
    top=edges[(edges['model']=='graph_sparse_v3_soft')].head(8).copy()
    for i,r in enumerate(top.to_dict('records')):
        y=154+i*36
        c=GREEN if r['top10_seed_count']>=5 else ORANGE
        body.append(f'<rect x="650" y="{y-18}" width="390" height="26" fill="#f7f9fb" stroke="#e0e5ea"/>')
        body.append(text(662,y, r['edge'],13,'normal',INK))
        body.append(f'<rect x="1055" y="{y-18}" width="120" height="26" fill="#edf1f5"/>')
        body.append(f'<rect x="1055" y="{y-18}" width="{24*float(r["top10_seed_count"])}" height="26" fill="{c}"/>')
        body.append(text(1184,y, f"{int(r['top10_seed_count'])}/5",13,'700',INK))
    body.append(text(650,480,'C. Module removal importance',18,'700'))
    for i,r in enumerate(rem.head(6).to_dict('records')):
        x=650+i*88
        val=max(0,float(r['mean_c_index_drop_after_removal']))
        bh=val/0.05*112
        body.append(f'<rect x="{x}" y="{610-bh}" width="48" height="{bh}" fill="{RED if i<2 else TEAL}"/>')
        body.append(text(x+24,628,str(r['module_removed']).replace('_',' ')[:13],11,'normal',INK,'middle'))
        body.append(text(x+24,600-bh,f"{val:.3f}",11,'700',INK,'middle'))
    body.append(wrapped(42,535,'Interpretation: V3 does not dominate all classical scores, but it finds stable biology-guided couplings. The extracted MRPL21/proteasome/redox gene axis is the strongest compact signal.',width=72,size=15,line_h=21,fill=INK))
    (FIGURES/'tcbb_figure2_ai_ablation_attention.svg').write_text(svg_wrap(1240,700,'\n'.join(body)))


def fig4_external():
    ext=pd.read_csv(RESULTS/'tcbb_final_external_score_confidence_intervals.csv')
    focus=ext[ext['score'].isin(['MRPL21','intervene_refined_gene_score','mrpl21_proteasome_score','proliferation_score','proteasome_core_score'])].copy()
    cohorts=['GSE30219','GSE37745','GSE68465']
    colors={'MRPL21':BLUE,'intervene_refined_gene_score':PURPLE,'mrpl21_proteasome_score':TEAL,'proliferation_score':RED,'proteasome_core_score':ORANGE}
    body=[text(42,48,'Figure 4. External validation and falsification',24,'700'),
          text(42,78,'External cohorts support a method-led claim, not a universal MRPL21 proof.',15,'normal',MUTED)]
    x0=82; y0=150; plot_w=980; plot_h=360
    body.append(f'<line x1="{x0}" y1="{y0+plot_h}" x2="{x0+plot_w}" y2="{y0+plot_h}" stroke="{INK}"/>')
    body.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+plot_h}" stroke="{INK}"/>')
    for tick in [0.5,0.55,0.60,0.65,0.70]:
        yy=y0+plot_h-(tick-0.5)/0.22*plot_h
        body.append(f'<line x1="{x0-5}" y1="{yy}" x2="{x0+plot_w}" y2="{yy}" stroke="#eef1f4"/>')
        body.append(text(x0-12,yy+4,f'{tick:.2f}',12,'normal',MUTED,'end'))
    body.append(text(30,330,'C-index',13,'700',INK,'middle'))
    group_w=plot_w/len(cohorts)
    legend_y=560
    legend_items=list(colors.items())
    for j,(score,c) in enumerate(legend_items):
        lx=82+j*220
        body.append(f'<rect x="{lx}" y="{legend_y}" width="18" height="18" fill="{c}"/>')
        body.append(text(lx+26,legend_y+14,score.replace('_',' '),12,'normal',INK))
    for ci,cohort in enumerate(cohorts):
        sub=focus[focus['cohort']==cohort]
        body.append(text(x0+group_w*ci+group_w/2,y0+plot_h+36,cohort,15,'700',INK,'middle'))
        for k,(_,r) in enumerate(sub.iterrows()):
            v=float(r['c_index'])
            bh=max(0,(v-0.5)/0.22*plot_h)
            bx=x0+ci*group_w+38+k*42
            by=y0+plot_h-bh
            body.append(f'<rect x="{bx}" y="{by}" width="28" height="{bh}" fill="{colors.get(r["score"],GRAY)}" opacity="0.9"/>')
            if math.isfinite(float(r.get('c_index_ci_low',float('nan')))):
                lo=y0+plot_h-(float(r['c_index_ci_low'])-0.5)/0.22*plot_h
                hi=y0+plot_h-(float(r['c_index_ci_high'])-0.5)/0.22*plot_h
                body.append(f'<line x1="{bx+14}" y1="{hi}" x2="{bx+14}" y2="{lo}" stroke="{INK}" stroke-width="1"/>')
    body.append(wrapped(82,620,'Takeaway: GSE30219 supports MRPL21/refined stress-survival scores, GSE37745 is weak, and GSE68465 mainly shows proliferation because MRPL21 is absent in the current GPL96 mapping. This is evidence for a falsifiable computational discovery framework, not final causal proof.',width=116,size=15,line_h=21,fill=INK))
    (FIGURES/'tcbb_figure4_external_validation_falsification.svg').write_text(svg_wrap(1240,720,'\n'.join(body)))


def main():
    fig1_architecture()
    fig2_ablation()
    fig4_external()
    print('Wrote tcbb_figure1_intervene_architecture.svg')
    print('Wrote tcbb_figure2_ai_ablation_attention.svg')
    print('Wrote tcbb_figure4_external_validation_falsification.svg')


if __name__ == '__main__':
    main()

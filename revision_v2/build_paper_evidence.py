"""Generate manuscript tables and readable diagrams from verified revision results.

No model fitting or model selection is performed here. All numerical cells are
drawn from the completed result bundle; raw records and weights are not exported.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = argparse.ArgumentParser()
P.add_argument('--evidence', type=Path, required=True)
P.add_argument('--paper', type=Path, required=True)
A = P.parse_args()
R, O = A.evidence, A.paper
gate = json.loads((R/'results_ready_for_paper.json').read_text())
for name, digest in gate['file_sha256'].items():
    assert hashlib.sha256((R/name).read_bytes()).hexdigest() == digest, name
T = O/'tables'; T.mkdir(exist_ok=True)
F = O/'figures'; F.mkdir(exist_ok=True)
DS = ['OnlineRetailII','TaFeng','Instacart']
DN = {'OnlineRetailII':'Online Retail II','TaFeng':'Ta-Feng','Instacart':'Instacart'}
MN = {'selected':'GTS, selected inputs','fixed':'GTS, fixed inputs',
      'encoder':'Graph-temporal encoder','lstm':'LSTM','gru':'GRU','boosting':'Boosting',
      'deepar':'DeepAR adaptation','weibull':'Neural Weibull AFT',
      'mlp_mae':'Small NN, MAE','mlp_mse':'Small NN, MSE',
      'window_mean':'Window mean','window_median':'Window median',
      'last_gap':'Last gap','training_median':'Training median',
      'tuned_window_mean':'Tuned window mean','tuned_window_median':'Tuned window median',
      'selected_without_encoder':'Without encoder','selected_nonrecurrent':'Boosting + mean',
      'selected_direct_only':'Direct predictions only','GTS':'GTS',
      'transformer_direct':'Transformer, direct','transformer_recursive':'Transformer, recursive'}
cells=[]
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def number(v,bold=False,digits=3):
    s=f'{v:.{digits}f}'
    return r'\textbf{'+s+'}' if bold else s
def table(name,caption,label,heads,rows,fmt=None,size='small'):
    text=[r'\begin{table}[htbp]','\\centering\\'+size,
          r'\caption{'+caption+'}',r'\label{'+label+'}',
          r'\begin{tabular}{'+(fmt or 'l'+'r'*(len(heads)-1))+'}',r'\toprule',
          ' & '.join(heads)+r'\\\midrule']
    text += [' & '.join(map(str,row))+r'\\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (T/(name+'.tex')).write_text('\n'.join(text)+'\n',encoding='utf-8')
def metric_rows(frame,models,cols=('MAE','RMSE','MedAE')):
    d=frame.set_index('model').loc[models]
    minima=d[list(cols)].min()
    out=[]
    for model,row in d.iterrows():
        values=[number(row[c],np.isclose(row[c],minima[c],rtol=0,atol=1e-9)) for c in cols]
        out.append([MN.get(model,esc(model)),*values])
        for c in cols:cells.append({'table_source':'metrics','model':model,'metric':c,'value':float(row[c])})
    return out
M=pd.read_csv(R/'metrics.csv')
primary=['selected','fixed','boosting','encoder','lstm','gru','deepar','weibull','mlp_mae','mlp_mse',
         'training_median','tuned_window_mean','tuned_window_median','window_mean','window_median','last_gap']
for ds in DS:
    table('main_'+ds,f'{DN[ds]}: errors of five-seed mean predictions on the same completed test outcomes. Lower is better; bold marks the minimum in each metric among the rows shown.',
          'tab:main'+ds,['Model','MAE','RMSE','MedAE'],metric_rows(M[M.dataset==ds],primary))
man={ds:json.loads((R.parent/'prepared'/ds/'manifest.json').read_text()) for ds in DS}
rows=[]
for ds in DS:
    m=man[ds];pc=m['phase_counts']
    rows.append([DN[ds],f"{m['series']:,}",f"{pc['0']['observed']:,}",f"{pc['0']['origins']-pc['0']['observed']:,}",f"{pc['4']['observed']:,}",f"{pc['4']['origins']-pc['4']['observed']:,}"])
table('cohorts','Prepared series and forecast origins. Incomplete means that the successor was not observed before that phase ended. It does not mean that the customer never purchased again.','tab:cohorts',
      ['Dataset','Series','Train complete','Train incomplete','Test complete','Test incomplete'],rows,size='footnotesize')
rows=[]
for ds in DS:
    for phase in range(5):
        pc=man[ds]['phase_counts'][str(phase)]
        rows.append([DN[ds],['Training','Early stop','Input selection','Stack fit','Test'][phase],f"{pc['origins']:,}",f"{pc['observed']:,}"])
table('phases','All prepared origins and completed outcomes by phase. Distributional models use incomplete training origins; common point metrics use completed outcomes.','tab:phases',
      ['Dataset','Phase','Origins','Complete'],rows,fmt='llrr')
S=pd.read_csv(R/'input_selections.csv')
rows=[[DN[r.dataset],int(r.seed),int(r.selected_input_channels),int(r.L),int(r.K),int(r.evaluated_configurations)] for r in S.itertuples()]
table('selections','Selected numerical channels (including availability flags), event-window length L, group count K, and number of distinct tested configurations. Padding and group indicators are additional channels.','tab:selections',
      ['Dataset','Seed','Channels','L','K','Trials'],rows,fmt='lrrrrr')
seed=pd.read_csv(R/'seed_metrics.csv');rows=[]
for ds in DS:
    for model in ['fixed','selected','boosting','deepar']:
        v=seed[(seed.dataset==ds)&(seed.model==model)].MAE
        rows.append([DN[ds],MN[model],number(v.mean()),number(v.std(ddof=1))])
table('seed_stability','Mean and sample standard deviation of the five individual-run MAEs. These are not the errors of the five-seed mean predictions in the primary tables.','tab:stability',
      ['Dataset','Model','Mean MAE','SD'],rows,fmt='llrr')
B=pd.read_csv(R/'ablation_metrics.csv');rows=[]
labels={'zero_graph':'Zero graph','random_graph':'Random item vectors','no_attention':'No attention','last_step_pool':'Last-step pooling','no_groups':'No group indicators'}
for ds in DS:
    m=M[(M.dataset==ds)&(M.model=='selected')].iloc[0]
    rows.append([DN[ds],'Full GTS',m.MAE,m.RMSE,m.MedAE])
    for v,l in labels.items():
        r=B[(B.dataset==ds)&(B.variant==v)&(B.scope=='full_system')].iloc[0]
        rows.append([DN[ds],l,r.MAE,r.RMSE,r.MedAE])
    for model in ['selected_without_encoder','selected_nonrecurrent','selected_direct_only']:
        r=M[(M.dataset==ds)&(M.model==model)].iloc[0]
        rows.append([DN[ds],MN[model],r.MAE,r.RMSE,r.MedAE])
for ds in DS:
    subset=[x for x in rows if x[0]==DN[ds]]
    for col in [2,3,4]:
        low=min(x[col] for x in subset)
        for x in subset:x[col]=number(x[col],abs(x[col]-low)<1e-9)
table('ablations','Structural controls: five-seed mean predictions. The no-group control refits all learned members; graph, attention and pooling controls replace the encoder and refit stack weights. Bold marks minima within each dataset among these controls.','tab:ablations',
      ['Dataset','System','MAE','RMSE','MedAE'],rows,fmt='llrrr',size='footnotesize')
for scope in ['full_system','standalone_encoder']:
    rows=[]
    for ds in DS:
        sub=B[(B.dataset==ds)&(B.scope==scope)&B.variant.str.startswith('graph_')]
        for r in sub.itertuples():
            p,d=r.variant.replace('graph_P','').split('_d')
            rows.append([DN[ds],p,d,number(r.MAE,np.isclose(r.MAE,sub.MAE.min(),rtol=0,atol=1e-9)),number(r.RMSE,np.isclose(r.RMSE,sub.RMSE.min(),rtol=0,atol=1e-9)),number(r.MedAE,np.isclose(r.MedAE,sub.MedAE.min(),rtol=0,atol=1e-9))])
    # Three separate compact tables avoid a multi-page float.
    for ds in DS:
        table('graph_'+scope+'_'+ds,f'{DN[ds]} graph diagnostic, seed 42, {scope.replace("_"," ")}. P is the product limit and d is the vector size. Bold marks genuine minima across this grid; tied rounded values need not be equal.',
              'tab:graph'+scope+ds,['P','d','MAE','RMSE','MedAE'],[x[1:] for x in rows if x[0]==DN[ds]])
rows=[]
for ds in DS:
    sub=B[(B.dataset==ds)&(B.scope=='full_system')&B.variant.str.startswith('logloss_')]
    for r in sub.itertuples():
        tau,cap=r.variant.replace('logloss_tau','').split('_cap')
        rows.append([DN[ds],tau,'None' if cap=='None' else '99th percentile',number(r.MAE,np.isclose(r.MAE,sub.MAE.min(),rtol=0,atol=1e-9)),number(r.RMSE,np.isclose(r.RMSE,sub.RMSE.min(),rtol=0,atol=1e-9)),number(r.MedAE,np.isclose(r.MedAE,sub.MedAE.min(),rtol=0,atol=1e-9))])
table('loss_grid','Log-pinball diagnostics, seed 42, full system after replacing the encoder. All test gaps remain uncapped. Bold marks minima within each dataset across this diagnostic grid; no grid winner replaces the primary model.','tab:lossgrid',
      ['Dataset',r'$\tau$','Training cap','MAE','RMSE','MedAE'],rows,fmt='lllrrr',size='footnotesize')
I=pd.read_csv(R/'paired_inference.csv');rows=[]
for ds in DS:
    for ref in ['fixed','boosting','deepar','selected_without_encoder']:
        r=I[(I.dataset==ds)&(I.reference==ref)].iloc[0]
        rows.append([DN[ds],MN[ref],number(r.MAE_reference_minus_selected),f'[{r.MAE_low:.3f}, {r.MAE_high:.3f}]',number(r.MAE_p_Holm,False,4)])
table('inference','Paired MAE comparisons. Difference = reference minus selected GTS, so positive favors GTS. Intervals are 95 percent customer-bootstrap intervals; p values use the declared 108-test Holm family.','tab:inference',
      ['Dataset','Reference','Difference',r'95\% interval',r'$p_{\rm Holm}$'],rows,fmt='llrrr',size='footnotesize')
rows=[]
for ds in DS:
    for ref in ['fixed','boosting','deepar']:
        r=I[(I.dataset==ds)&(I.reference==ref)].iloc[0]
        rows.append([DN[ds],MN[ref],f'{r.RMSE_reference_minus_selected:.3f} [{r.RMSE_low:.3f}, {r.RMSE_high:.3f}]',f'{r.MedAE_reference_minus_selected:.3f} [{r.MedAE_low:.3f}, {r.MedAE_high:.3f}]'])
table('metric_intervals','Paired differences and 95 percent bootstrap intervals for RMSE and MedAE. Positive differences favor selected GTS. Intervals describe original-customer resampling conditional on the fitted five-seed predictions.','tab:metricintervals',
      ['Dataset','Reference','RMSE difference [interval]','MedAE difference [interval]'],rows,fmt='llrr',size='scriptsize')
H=pd.read_csv(R/'horizon_metrics.csv')
for h in range(1,6):
    sub=H[H.horizon==h]
    table('horizon'+str(h),f'Five-purchase-day task, horizon {h}: five-seed mean predictions on 69 matching origins. Errors are calendar-day gaps, not the date of the fifth purchase. Bold marks minima across these rows.',
          'tab:horizon'+str(h),['Model','MAE','RMSE','MedAE'],metric_rows(sub,['GTS','boosting','encoder','lstm','gru','transformer_direct','transformer_recursive','window_mean','window_median']))
rows=[]
for h in range(1,6):
    models=['GTS','transformer_direct','transformer_recursive','boosting','lstm','gru']
    sub=H[H.horizon==h].set_index('model')
    low=sub.loc[models].MAE.min()
    rows.append([h,*[number(sub.loc[m].MAE,np.isclose(sub.loc[m].MAE,low,atol=1e-9,rtol=0)) for m in models]])
table('horizons','MAE at each forecast horizon on 69 common Online Retail II origins. Bold identifies minima among these six models for each horizon; the complete three-metric tables are in the appendix.','tab:horizons',
      ['h','GTS','Direct Tr.','Recursive Tr.','Boosting','LSTM','GRU'],rows)
BC=pd.read_csv(R/'inference_benchmarks.csv');rows=[]
for ds in DS:
    for model in ['GTS_5seed','boosting_1seed','boosting_5seed','LSTM_1seed','LSTM_5seed']:
        r=BC[(BC.dataset==ds)&(BC.model==model)].iloc[0]
        rows.append([DN[ds],esc(model.replace('_',' ')),int(r.batch_origins),number(r.median_batch_seconds,False,4),number(r.median_ms_per_origin,False,4),number(r.incremental_forward_cuda_peak_bytes/2**20,False,1)])
table('inference_cost','Warm batched model evaluation: median of five repetitions on prepared histories. These are measured batch costs, not online service latency. Memory is additional forward-pass CUDA allocation, not total process memory.','tab:inferencecost',
      ['Dataset','Model','Batch','Seconds','ms/origin','MiB'],rows,fmt='llrrrr',size='footnotesize')
rows=[]
for ds in DS:
    runs=R.parent/'runs'/ds;fit_seconds=0.;stack_seconds=0.
    for seed in [42,7,1400,2026,31337]:
        folder=runs/f'seed_{seed}'
        info=json.loads((folder/'selected_complete.json').read_text())
        configs=[]
        for kind in ['encoder','lstm','gru']:
            configs += [v['config'] for v in info['members'][kind]['fits'].values()]
        keys={hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16] for c in configs}
        fit_seconds+=sum(json.loads((folder/'fits'/f'{key}.json').read_text())['training_seconds'] for key in keys)
        fit_seconds+=info['members']['boosting']['training_seconds']
        stack_seconds+=json.loads((folder/'selected.json').read_text())['stack_fit_seconds']
    deploy=json.loads((R/f'{ds}_deployed_artifacts.json').read_text())
    rows.append([DN[ds],number(man[ds]['seconds']),number(S[S.dataset==ds].selection_seconds.sum()/60),number(fit_seconds/60),number(stack_seconds/60),number(deploy['unpruned_five_seed_model_artifact_bytes']/2**20),f"{deploy['unpruned_deployed_neural_parameters']:,}"])
table('fit_cost','Recorded costs: preparation seconds (including graph construction), search minutes, selected-arm member-fit minutes (both direct/residual candidates), primary-stack minutes, deployed model MiB and neural parameter count. Five seeds are totaled; other comparison/ablation fits and auxiliary combiners are excluded. Timing includes observed machine load.','tab:fitcost',
      ['Dataset','Prep. s','Search min','Fit min','Stack min','MiB','Parameters'],rows,size='scriptsize')
rows=[]
for ds in DS:
    orig=pd.read_csv(R.parent/'prepared'/ds/'graph_coverage.csv')
    x=orig[(orig.phase==4)&(orig.max_products==600)].iloc[0]
    q=pd.read_csv(R/f'{ds}_forecast_graph_coverage.csv')
    q=q[(q.phase==4)&q.observed_successor_only&(q.max_products==600)&(q.dimensions==16)].iloc[0]
    rows.append([DN[ds],f'{int(x.baskets):,}',number(100*x.no_represented_item_fraction,False,2),f'{int(q.forecast_origins):,}',number(100*q.zero_graph_vector_fraction,False,2)])
table('coverage','Graph coverage at P=600, d=16. Empty source baskets and zero forecast-origin vectors use different denominators. An Instacart order can generate several item-pair forecasts.','tab:coverage',
      ['Dataset','Source test baskets',r'No top item (\%)','Complete test origins',r'Zero vector (\%)'],rows,size='scriptsize')
# Preserve all six independently checked result-figure families.
for source in (R/'figures').iterdir():
    if source.suffix in {'.pdf','.png','.svg'}:shutil.copy2(source,F/source.name)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
def diagram(name,kind):
    fig,ax=plt.subplots(figsize=(11,6.1 if kind=='overview' else 5.3));ax.set(xlim=(0,11),ylim=(0,6.1 if kind=='overview' else 5.3));ax.axis('off')
    def box(x,y,w,h,title,body='',color='#EAF2F6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.08',lw=1,edgecolor='#668291',facecolor=color))
        ax.text(x+w/2,y+h-.24,title,ha='center',va='top',weight='bold',fontsize=11)
        if body:ax.text(x+w/2,y+.18,body,ha='center',va='bottom',fontsize=9.3,linespacing=1.45)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.2,color='#435C69'))
    if kind=='overview':
        box(.15,4.75,2.3,1.0,'Public purchases','Positive records\nKnown customer IDs')
        box(3.0,4.75,3.0,1.0,'Five separate phases','Fit → early stop → select\n→ stack weights → test')
        box(6.6,4.75,4.05,1.0,'Training-only representations','Feature ranking • customer profiles\nFixed co-purchase item graph')
        arrow((2.5,5.25),(2.95,5.25));arrow((6.05,5.25),(6.55,5.25))
        box(.15,2.65,3.3,1.15,'Bounded input search','Number of channels • window length\nOptional number of groups')
        box(4.15,2.65,6.5,1.15,'Five complementary members','Graph encoder • LSTM • GRU • Boosting • Window mean\nRecurrent members: select direct / residual / average')
        arrow((4.5,4.67),(1.8,3.88));arrow((8.6,4.67),(7.4,3.88));arrow((3.55,3.2),(4.05,3.2))
        box(2.5,.5,6.0,1.15,'Fit weights, then evaluate once','Nonnegative weights that sum to one\nFive-seed mean prediction • matched comparisons')
        arrow((7.4,2.57),(5.65,1.73))
        ax.text(5.5,.02,'The test phase supplies outcomes only for final evaluation.',ha='center',fontsize=10,color='#435C69')
    else:
        box(.2,3.85,3.0,1.0,'Observed event window','Numeric channels + valid-row flag\nOptional customer-group indicators')
        box(.2,1.8,3.0,1.0,'Basket graph vector','Mean of represented item vectors\nTraining graph kept fixed')
        box(4.0,3.85,2.7,1.0,'Join + project + LSTM','32 projected features\n64 recurrent units')
        box(7.5,3.85,3.1,1.0,'Four-head attention','Padding keys masked\nResidual connection + normalization')
        box(7.5,1.8,3.1,1.0,'Pool valid rows','Mean pooling\n64-unit prediction layer')
        box(4.0,1.8,2.7,1.0,'Nonnegative gap','Direct or median residual\nSelect on a separate phase')
        arrow((3.3,4.35),(3.9,4.35));arrow((3.28,2.85),(4.05,3.77));arrow((6.8,4.35),(7.4,4.35))
        arrow((9.05,3.77),(9.05,2.88));arrow((7.4,2.3),(6.8,2.3))
        ax.text(5.5,.65,'Every context row is already observed at the forecast origin.\nAttention does not update the item graph.',ha='center',fontsize=10,linespacing=1.6,color='#435C69')
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');fig.savefig(F/(name+'.png'),bbox_inches='tight',dpi=180);plt.close(fig)
diagram('revision_overview','overview');diagram('revision_encoder','encoder')
# An editable diagrams.net overview, importable into Lucidchart.
nodes=[('Public purchases','Positive records; known customer IDs'),('Five phases','Train / early stop / select / stack / test'),('Training representations','Ranked channels; customer profiles; fixed graph'),('Input search','Channel count; window length; optional groups'),('Model pool','Encoder; LSTM; GRU; boosting; window mean'),('Fit independent stack weights','Nonnegative weights sum to one'),('Evaluate','Five-seed mean; matched outcomes; uncertainty')]
import xml.etree.ElementTree as ET
mx=ET.Element('mxfile',host='app.diagrams.net');dia=ET.SubElement(mx,'diagram',name='GTS revised workflow');gm=ET.SubElement(dia,'mxGraphModel',dx='1200',dy='900',grid='1',page='1',pageWidth='1200',pageHeight='900');root=ET.SubElement(gm,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
for i,(title,body) in enumerate(nodes):
    c=ET.SubElement(root,'mxCell',id=str(i+2),value=f'{title}\n{body}',style='rounded=1;whiteSpace=wrap;html=0;fillColor=#EAF2F6;strokeColor=#668291;fontSize=15;',vertex='1',parent='1');ET.SubElement(c,'mxGeometry',x=str(90+(i%2)*540),y=str(40+(i//2)*190),width='440',height='120',attrib={'as':'geometry'})
    if i:
        edge=ET.SubElement(root,'mxCell',id='e'+str(i),edge='1',parent='1',source=str(i+1),target=str(i+2),style='endArrow=block;');ET.SubElement(edge,'mxGeometry',relative='1',attrib={'as':'geometry'})
ET.ElementTree(mx).write(F/'revision_overview.drawio',encoding='utf-8',xml_declaration=True)
# Improve contrast of the manuscript copy of the checked graph heatmap.
fig,axes=plt.subplots(1,3,figsize=(11,3.7))
for ax,ds in zip(axes,DS):
    sub=B[(B.dataset==ds)&(B.scope=='full_system')&B.variant.str.startswith('graph_')]
    base=float(sub[sub.variant=='graph_P600_d16'].MAE.iloc[0]); values=np.zeros((4,3))
    for i,p in enumerate([100,300,600,1000]):
        for j,d in enumerate([8,16,32]):values[i,j]=float(sub[sub.variant==f'graph_P{p}_d{d}'].MAE.iloc[0])-base
    lim=max(.001,np.abs(values).max());im=ax.imshow(values,cmap='PuOr_r',vmin=-lim,vmax=lim)
    for i in range(4):
        for j in range(3):ax.text(j,i,f'{values[i,j]:+.3f}',ha='center',va='center',color='white' if abs(values[i,j])>.65*lim else '#111111',fontsize=9)
    ax.set(xticks=range(3),xticklabels=[8,16,32],yticks=range(4),yticklabels=[100,300,600,1000],xlabel='Vector dimensions',title=DN[ds]);ax.set_ylabel('Represented product limit');fig.colorbar(im,ax=ax,fraction=.035,pad=.04)
fig.tight_layout()
for ext in ['pdf','svg','png']:fig.savefig(F/('revision_graph_sensitivity_readable.'+ext),bbox_inches='tight',dpi=200)
plt.close(fig)
(O/'paper_numeric_cells.json').write_text(json.dumps(cells,indent=2),encoding='utf-8')
print('Verified gate hashes; generated tables and 8 figure families, plus editable workflow.')

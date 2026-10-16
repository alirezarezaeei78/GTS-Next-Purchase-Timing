"""Score complete five-seed fixed/adaptive runs, grouped by original customer."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--benchmark-root',type=Path,default=ROOT/'work')
p.add_argument('--experiment-root',type=Path,default=ROOT/'work/adaptive/instacart')
p.add_argument('--output-root',type=Path,default=ROOT)
a=p.parse_args();data=a.output_root/'data';data.mkdir(exist_ok=True,parents=True)
seeds=[42,7,1400,2026,31337]
def metrics(y,p):
    e=np.abs(y-p)
    return {'MAE':float(e.mean()),'RMSE':float(np.sqrt((e**2).mean())),
        **{f'within_{k}_pct':float(100*(e<k).mean()) for k in [1,2,3]}}
base=a.benchmark_root/'prepared/instacart_pairs'
split=np.load(base/'split.npy');y=np.load(base/'y.npy')[split==2,0]
customer=np.load(base/'customer.npy')[split==2];means={};records={};files={}
out=pd.read_parquet(base/'keys.parquet').loc[split==2].reset_index(drop=True)
out['actual']=y
for arm in ['fixed','adaptive']:
    runs=[];seed_metrics=[]
    for seed in seeds:
        f=a.experiment_root/arm/f'seed_{seed}/predictions.npz'
        pred=np.load(f)['test'];assert len(pred)==len(y) and np.isfinite(pred).all() and (pred>=0).all()
        runs.append(pred);seed_metrics.append({'seed':seed,**metrics(y,pred)})
        files[f'{arm}/{seed}']=hashlib.sha256(f.read_bytes()).hexdigest()
        out[f'{arm}_s{seed}']=pred
    means[arm]=np.mean(runs,axis=0);out[arm]=means[arm]
    scores=[row['MAE'] for row in seed_metrics]
    records[arm]={'ensemble':metrics(y,means[arm]),'seed_mean_MAE':float(np.mean(scores)),
        'seed_sd_MAE':float(np.std(scores,ddof=1)),'seed_metrics':seed_metrics}
delta=np.abs(y-means['fixed'])-np.abs(y-means['adaptive'])
u,inv=np.unique(customer,return_inverse=True);totals=np.bincount(inv,weights=delta);counts=np.bincount(inv)
rng=np.random.default_rng(719);values=[]
for start in range(0,2000,50):
    sample=rng.integers(0,len(u),(50,len(u)))
    values.extend(totals[sample].sum(1)/counts[sample].sum(1))
selection=json.loads((a.experiment_root/'selection_report.json').read_text())
summary={'experiment':'Fresh matched PyTorch fixed/adaptive GTS, full Instacart same-item cohort',
    'n_pairs':len(y),'n_customers':len(u),'seeds':seeds,'models':records,
    'selection':selection['selection'],'selected_features':selection['selected_features'],
    'bootstrap':{'difference_fixed_minus_adaptive':float(delta.mean()),'low':float(np.quantile(values,.025)),
        'high':float(np.quantile(values,.975)),'resamples':2000,'unit':'original customer'},
    'relative_MAE_reduction_pct':float(100*delta.mean()/records['fixed']['ensemble']['MAE']),
    'input_sha256':files,'training_runs':10,'probe_configurations':len(selection['trials']),
    'runtime':json.loads((a.experiment_root/'runtime.json').read_text())}
out.to_parquet(data/'adaptive_instacart_predictions.parquet',index=False)
(data/'adaptive_instacart_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
pd.DataFrame([{'model':arm,**row['ensemble'],'seed_mean_MAE':row['seed_mean_MAE'],
    'seed_sd_MAE':row['seed_sd_MAE']} for arm,row in records.items()]).to_csv(data/'adaptive_instacart_metrics.csv',index=False)
pd.DataFrame(selection['trials']).to_csv(data/'input_selection_trials.csv',index=False)
shutil.copy2(a.experiment_root/'selection_report.json',data/'input_selection_report.json')
shutil.copy2(a.experiment_root/'preparation_audit.json',data/'adaptive_preparation_audit.json')
print(json.dumps(summary,indent=2))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.7),layout='constrained')
    trials=pd.DataFrame(selection['trials']);scan=trials[(trials.context==5)&(trials.groups==0)].sort_values('features')
    axes[0].plot(scan.features,scan.validation_MAE,'o-',color='#176B92',markersize=3)
    best=selection['selection'];axes[0].scatter([best['features']],[best['validation_MAE']],s=65,color='#B66A22',zorder=4)
    axes[0].annotate(f"Selected: {best['features']} variables",(best['features'],best['validation_MAE']),
        xytext=(.45,.82),textcoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#B66A22'},fontsize=10)
    axes[0].set(xlabel='Retained ranked variables',ylabel='Probe validation MAE (days)',title='Selecting input count')
    for i,seed in enumerate(seeds):
        vals=[records[arm]['seed_metrics'][i]['MAE'] for arm in ['fixed','adaptive']]
        axes[1].plot([0,1],vals,'o-',color='#82929B',alpha=.7,markersize=4)
    axes[1].scatter([0,1],[records[arm]['seed_mean_MAE'] for arm in ['fixed','adaptive']],marker='D',s=55,color='#176B92',zorder=4)
    axes[1].set(xticks=[0,1],xticklabels=['Fixed inputs','Selected inputs'],ylabel='Single-run test MAE (days)',title='Matched final-model fits')
    axes[1].margins(x=.35,y=.2)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    figures=a.output_root/'figures';figures.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(figures/f'adaptive_instacart.{ext}',bbox_inches='tight',dpi=250)
    plt.close(fig)
except ImportError:
    print('Metrics saved. Install matplotlib to regenerate the figure.')

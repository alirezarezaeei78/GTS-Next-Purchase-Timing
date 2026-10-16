"""Verify repository prediction evidence without source data or model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
def metrics(y,p):
    e=np.asarray(y)-np.asarray(p)
    return float(np.abs(e).mean()),float(np.sqrt((e**2).mean()))
checks=0
s=json.loads((ROOT/'data/verified_results.json').read_text())
names={'Training median':'naive_global_median','Last gap':'naive_last_gap','Window mean':'naive_customer_mean_gap',
    'Window median':'naive_customer_median_gap','Boosting':'gbm_flat','LSTM':'lstm_plain','GRU':'gru_plain','GTS':'dgt_full'}
for dataset,n in [('OnlineRetailII',1678),('TaFeng',5796)]:
    d=pd.read_csv(ROOT/'data'/f'{dataset}_paired_predictions.csv');assert len(d)==n and d.customer_id.notna().all()
    for row in s['daily']:
        if row['dataset']==dataset:
            m,r=metrics(d.actual_days,d[names[row['model']]])
            assert abs(m-row['MAE'])<1e-8 and abs(r-row['RMSE'])<1e-8;checks+=2
for task,h in [('instacart_pairs',1),('retail_411',5)]:
    d=pd.read_parquet(ROOT/'data'/f'{task}_paired_predictions.parquet')
    y=np.column_stack([d[f'actual_gap_h{i}'] for i in range(1,h+1)])
    for model,row in s['public_tasks'][task]['models'].items():
        pred=np.column_stack([d[f'{model}_h{i}'] for i in range(1,h+1)])
        m,r=metrics(y,pred)
        assert abs(m-row['metrics']['mae_days'])<1e-8 and abs(r-row['metrics']['rmse_days'])<1e-8;checks+=2
fresh=json.loads((ROOT/'data/adaptive_instacart_results.json').read_text())
d=pd.read_parquet(ROOT/'data/adaptive_instacart_predictions.parquet')
assert len(d)==103855 and d.customer_id.nunique()==25508
for arm,row in fresh['models'].items():
    pred=d[[f'{arm}_s{seed}' for seed in fresh['seeds']]].mean(axis=1)
    assert np.allclose(pred,d[arm],atol=1e-12)
    m,r=metrics(d.actual,pred)
    assert abs(m-row['ensemble']['MAE'])<1e-8 and abs(r-row['ensemble']['RMSE'])<1e-8;checks+=3
selection=json.loads((ROOT/'data/input_selection_report.json').read_text())
best=min(selection['trials'],key=lambda r:(r['validation_MAE'],r['features'],r['groups'],r['context']))
assert best==selection['selection']==fresh['selection']
assert len(selection['selected_features'])==best['features'];checks+=2
print(f'PASS: {checks} numerical checks, matched cohorts, ensemble averaging, and validation selection.')

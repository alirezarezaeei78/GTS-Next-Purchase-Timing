"""Verify the public aggregate package without private/raw records or weights."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent
manifest=json.loads((root/'public_evidence_manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
r=root/'results';m=pd.read_csv(r/'metrics.csv')
assert set(m.dataset)=={'OnlineRetailII','TaFeng','Instacart'}
assert m[['MAE','RMSE','MedAE']].notna().all().all()
assert (m[['MAE','RMSE','MedAE']]>=0).all().all()
assert (m.RMSE+1e-10>=m.MAE).all()
for ds,n in [('OnlineRetailII',627),('TaFeng',1808),('Instacart',204118)]:
    assert set(m[m.dataset==ds].origins)=={n}
w=pd.read_csv(r/'combiner_weights.csv')
np.testing.assert_allclose(w.groupby(['dataset','seed','arm']).weight.sum(),1,atol=1e-8)
assert w.weight.min()>=0
selection=pd.read_csv(r/'input_selections.csv')
assert len(selection)==15 and selection.evaluated_configurations.between(14,15).all()
a=pd.read_csv(r/'ablation_metrics.csv')
assert set(a[a.seeds==5].variant)=={'zero_graph','random_graph','no_attention','last_step_pool','no_groups'}
assert set(a[a.seeds==1].scope)=={'full_system','standalone_encoder'}
h=pd.read_csv(r/'horizon_metrics.csv');assert set(h.horizon)==set(range(1,6)) and set(h.origins)=={69}
b=pd.read_csv(r/'inference_benchmarks.csv');assert len(b)==15 and b.live_predictions_match_saved.all()
for file in ['paired_inference.csv','horizon_inference.csv']:
    d=pd.read_csv(r/file)
    assert d.bootstrap_resamples.eq(2000).all() and d.permutation_resamples.eq(9999).all()
    for metric in ['MAE','RMSE','MedAE']:assert (d[metric+'_low']<=d[metric+'_high']).all()
print('Public hashes and aggregate consistency passed. Full prediction-level verification requires locally reproduced raw-data preparation and saved models; use verify_results.py for that check.')

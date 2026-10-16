"""Run the historical fixed-input daily pipeline on the two public sources.

The default drops empty IDs before string conversion. A clean refit is a NEW
experiment and must not be equated to the manuscript's archived predictions.
"""
from pathlib import Path
import argparse,tempfile,os,json
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import pandas as pd
import tensorflow as tf
import npe_core,npe_data,run_comparison

p=argparse.ArgumentParser()
p.add_argument('--dataset',choices=['online-retail-ii','ta-feng'],required=True)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--epochs',type=int,default=30)
p.add_argument('--legacy-pooled-ids',action='store_true',help='Recreate old missing-ID grouping for audit only')
a=p.parse_args();npe_core.RANDOM_SEED=a.seed;tf.keras.utils.set_random_seed(a.seed)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
a.output.mkdir(parents=True,exist_ok=True)
def loader():
    if a.dataset=='ta-feng':return npe_data.load_ta_feng(a.input)
    if a.legacy_pooled_ids:return npe_data.load_generic(a.input)
    raw=npe_data._read_any(a.input,None)
    customer=npe_data._first_existing(raw.columns,npe_data.CUSTOMER_CANDIDATES)
    assert customer is not None
    raw=raw.loc[raw[customer].notna()].copy()
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'identified.parquet';raw.to_parquet(path,index=False)
        return npe_data.load_generic(path)
cfg=npe_core.Config(epochs=a.epochs,make_plots=False,timing_loss='pinball',pinball_tau=.65,anchor_stat='median')
name='OnlineRetailII' if a.dataset=='online-retail-ii' else 'TaFeng'
run_comparison.compare_dataset(name,loader,cfg,a.output/name)
(a.output/'run_protocol.json').write_text(json.dumps({'dataset':name,'seed':a.seed,'epochs':a.epochs,
    'drop_missing_id_before_grouping':not a.legacy_pooled_ids,'archived_model_policy':'16 mandatory + five ranked + four groups + five events',
    'note':'Fresh refits need new scoring; archived scores are provided separately.'},indent=2))

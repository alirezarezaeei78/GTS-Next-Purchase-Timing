"""Benchmark saved models on prepared histories and verify live inference.

Timing is batched model evaluation with feature-window assembly, not an online
retail service latency or a marketing-profit measurement. Raw-file preparation
and feature/graph fitting are reported separately.
"""
from pathlib import Path
import argparse, gc, json, sys, time
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared', type=Path, required=True)
parser.add_argument('--runs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cuda-runtime', type=Path)
parser.add_argument('--datasets', nargs='+', default=['OnlineRetailII', 'TaFeng', 'Instacart'])
parser.add_argument('--repetitions', type=int, default=5)
args = parser.parse_args()
if args.cuda_runtime:
    sys.path.insert(0, str(args.cuda_runtime))
import numpy as np
import pandas as pd
import joblib, torch
from engine import Data, Network, DEVICE, SEEDS, VERSION, identity, dump
from threadpoolctl import threadpool_limits


def load_seed(data, folder):
    rec = json.loads((folder / 'selected_complete.json').read_text())
    cfg = rec['config']; models = {}; size = 0; parameters = 0
    for kind in ['encoder', 'lstm', 'gru']:
        info = rec['members'][kind]
        modes = ['direct', 'residual'] if info['choice'] == 'average' else [info['choice']]
        models[kind] = []
        for mode in modes:
            fit_info = info['fits'][mode]
            path = folder / 'fits' / (identity(fit_info['config']) + '.pt')
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            mc = checkpoint['config']
            model = Network(len(mc['columns']) + 1 + mc['K'], mc, data.scale_y,
                            checkpoint['center'], checkpoint['spread']).to(DEVICE)
            model.load_state_dict(checkpoint['state_dict']); model.eval()
            models[kind].append(model)
            size += path.stat().st_size; parameters += sum(p.numel() for p in model.parameters())
    boost_path = folder / 'fits' / (identity({**cfg, 'kind': 'boost', 'version': VERSION}) + '.joblib')
    models['boosting'] = joblib.load(boost_path); size += boost_path.stat().st_size
    meta = json.loads((folder / 'selected.json').read_text())
    return {'config': cfg, 'models': models, 'weights': meta['weights'], 'bytes': size, 'parameters': parameters}


def predict_seed(data, record, rows, component=None):
    cfg, models = record['config'], record['models']
    x, g, valid, anchor = data.batch(rows, cfg)
    pool = {}
    wanted = [component] if component else ['encoder', 'lstm', 'gru', 'boosting', 'window_mean']
    with torch.no_grad(), threadpool_limits(limits=4):
        if any(k in wanted for k in ['encoder', 'lstm', 'gru']):
            tx, tg, tv, ta = [torch.as_tensor(a, device=DEVICE) for a in [x, g, valid, anchor]]
            for kind in ['encoder', 'lstm', 'gru']:
                if kind in wanted:
                    values = [model.point(model(tx, tg, tv), ta).cpu().numpy() for model in models[kind]]
                    pool[kind] = np.mean(values, axis=0)
        if 'boosting' in wanted:
            pool['boosting'] = np.maximum(0, models['boosting'].predict(x.reshape(len(x), -1)))
        if 'window_mean' in wanted:
            ix, available = data.indices(rows, cfg['L'])
            available &= ix > data.starts[ix]
            pool['window_mean'] = np.nanmean(np.where(available, data.gaps[ix], np.nan), axis=1)
    if component:
        return pool[component]
    return sum(record['weights'][kind] * values for kind, values in pool.items())


args.output.mkdir(parents=True, exist_ok=True)
records = []
for dataset in args.datasets:
    data = Data(args.prepared / dataset)
    rows = data.test[:1000]
    trained = [load_seed(data, args.runs / dataset / f'seed_{seed}') for seed in SEEDS]
    primary = np.mean([predict_seed(data, rec, rows) for rec in trained], axis=0)
    saved = []
    for seed in SEEDS:
        frame = pd.read_parquet(args.runs / dataset / f'seed_{seed}/selected.parquet').set_index('sample_row')
        saved.append(frame.prediction.loc[rows].to_numpy())
    reference = np.mean(saved, axis=0)
    np.testing.assert_allclose(primary, reference, atol=2e-4, rtol=2e-5)
    cases = [('GTS_5seed', None, trained), ('boosting_1seed', 'boosting', trained[:1]),
             ('boosting_5seed', 'boosting', trained), ('LSTM_1seed', 'lstm', trained[:1]), ('LSTM_5seed', 'lstm', trained)]
    for name, component, chosen in cases:
        # One warm-up before timed repetitions; cached arrays/labels are shared.
        np.mean([predict_seed(data, rec, rows, component) for rec in chosen], axis=0)
        times = []; peaks = []
        for repeat in range(args.repetitions):
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
            started = time.perf_counter()
            output = np.mean([predict_seed(data, rec, rows, component) for rec in chosen], axis=0)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
                peaks.append(max(0, torch.cuda.max_memory_allocated() - baseline))
            times.append(time.perf_counter() - started)
        records.append({'dataset': dataset, 'model': name, 'batch_origins': len(rows),
            'median_batch_seconds': float(np.median(times)),
            'median_ms_per_origin': float(1000 * np.median(times) / len(rows)),
            'repetitions': args.repetitions, 'incremental_forward_cuda_peak_bytes': max(peaks, default=0),
            'live_predictions_match_saved': True,
            'scope': 'prepared-history model evaluation; excludes raw feature/graph construction; all GTS members evaluated before combination'})
    dump(args.output / f'{dataset}_deployed_artifacts.json', {
        'unpruned_five_seed_model_artifact_bytes': sum(r['bytes'] for r in trained),
        'unpruned_deployed_neural_parameters': sum(r['parameters'] for r in trained),
        'includes': 'selected direct/residual/average neural members and boosting models',
        'excludes': 'raw transactions, prepared feature arrays, shared graph products/embeddings'})
    del trained, data; gc.collect(); torch.cuda.empty_cache()
pd.DataFrame(records).to_csv(args.output / 'inference_benchmarks.csv', index=False)
dump(args.output / 'benchmark_complete.json', {'datasets': args.datasets, 'seeds': SEEDS,
     'torch': torch.__version__, 'device': str(DEVICE), 'live_inference_verified_against_saved_predictions': True})
print('Measured batched inference and verified reconstructed model predictions against saved outputs.')

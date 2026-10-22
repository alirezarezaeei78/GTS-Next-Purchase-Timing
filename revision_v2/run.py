"""Run the predeclared clean five-seed study; resume completed fit artifacts.

The manuscript is deliberately untouched. Final test results are aggregated by
a separate script only after this run has completed.
"""
from pathlib import Path
import argparse, gc, hashlib, itertools, json, os, sys, time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cuda-runtime', type=Path)
parser.add_argument('--cache-root', type=Path, help='Large temporary drive for on-disk histogram-boosting inputs')
parser.add_argument('--datasets', nargs='+', default=['OnlineRetailII', 'TaFeng', 'Instacart'])
parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 1400, 2026, 31337])
parser.add_argument('--stage', choices=['main', 'ablations', 'all'], default='all')
parser.add_argument('--selection-epochs', type=int, default=3)
parser.add_argument('--selection-trials', type=int, default=12)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()
if args.cache_root:
    os.environ['GTS_CACHE_ROOT'] = str(args.cache_root.resolve())
if args.cuda_runtime:
    sys.path.insert(0, str(args.cuda_runtime))
import numpy as np
import pandas as pd
import torch
from engine import Data, fit, boost, simplex_mae, log, dump, identity, DEVICE


def config(columns, seed, **changes):
    return {'columns': list(map(int, columns)), 'L': 5, 'K': 0, 'P': 600, 'd': 16,
            'kind': 'encoder', 'graph': 'spectral', 'mode': 'direct', 'pool': 'mean',
            'attention': True, 'loss': 'day_mae', 'batch_size': 4096, 'seed': int(seed), **changes}


def add_availability(data, columns):
    return list(dict.fromkeys(list(columns) + [i for i, name in enumerate(data.names) if name.endswith('_available')]))


def fixed_columns(data, ranking):
    # Corrected static comparator: the original feature families plus five
    # training-ranked extras; availability channels are structural safeguards.
    base = ['last_gap', 'events_seen', 'gap_mean', 'gap_sd', 'amount_sum', 'amount_mean',
            'last_amount', 'relative_amount', 'tenure', 'item_count', 'quantity',
            'sin_dow', 'cos_dow', 'sin_month', 'cos_month', 'global_last_gap']
    columns = [data.names.index(name) for name in base if name in data.names]
    extras = [c for c in ranking['indices'] if c not in columns and not data.names[c].endswith('_available')][:5]
    return add_availability(data, columns + extras)


def select_inputs(data, seed, folder, ranking):
    path = folder / f'selection_s{seed}.json'
    if path.exists():
        return json.loads(path.read_text())
    started = time.perf_counter()
    rank = [c for c in ranking['indices'] if not data.names[c].endswith('_available')]
    sizes = sorted(set([min(8, len(rank)), min(16, len(rank)), min(24, len(rank)), len(rank)]))
    grid = list(itertools.product(sizes, [1, 3, 5, 10], [0, 2, 4, 8]))
    rng = np.random.default_rng(seed)
    selected = [grid[i] for i in rng.permutation(len(grid))[:args.selection_trials]]
    # Mandatory anchors prevent random search from omitting the simple alternatives.
    selected += [(len(rank), 5, 0), (min(16, len(rank)), 5, 0)]
    candidates = [config(fixed_columns(data, ranking), seed, K=4)]
    candidates += [config(add_availability(data, rank[:n]), seed, L=length, K=k) for n, length, k in selected]
    trials, seen = [], set()
    for cfg in candidates:
        key = identity(cfg)
        if key in seen:
            continue
        seen.add(key)
        _, info = fit(data, cfg, folder / 'selection_fits', epochs=args.selection_epochs, patience=2)
        trials.append({'config': cfg, 'selection_MAE': info['selection_MAE'],
                       'training_seconds': info['training_seconds']})
        dump(folder / f'selection_s{seed}_progress.json', trials)
    best = min(trials, key=lambda r: (r['selection_MAE'], len(r['config']['columns']), r['config']['L'], r['config']['K']))
    result = {'seed': seed, 'selected': best['config'], 'trials': trials,
              'selection_seconds': time.perf_counter() - started,
              'fit_phase': 0, 'early_stop_phase': 1, 'configuration_score_phase': 2,
              'stack_phase_not_used': 3, 'test_phase_not_used': 4,
              'scope': 'bounded actual temporal-encoder search; no global-optimality claim'}
    dump(path, result)
    return result


def recurrent_pair(data, cfg, kind, folder):
    predictions, information = {}, {}
    for mode in ['direct', 'residual']:
        member_cfg = {**cfg, 'kind': kind, 'mode': mode}
        if kind != 'encoder':
            member_cfg['pool'] = 'last'
        predictions[mode], information[mode] = fit(data, member_cfg, folder, epochs=args.epochs)
    predictions['average'] = (predictions['direct'] + predictions['residual']) / 2
    scores = {name: float(np.abs(p[data.select] - data.y[data.select]).mean()) for name, p in predictions.items()}
    choice = min(scores, key=lambda name: (scores[name], name))
    return predictions[choice], {'choice': choice, 'selection_scores': scores, 'fits': information}, predictions


def combine(data, pool, label, folder, member_info=None):
    names = list(pool)
    matrix = np.column_stack([pool[name] for name in names])
    started = time.perf_counter()
    weights = simplex_mae(matrix[data.stack], data.y[data.stack])
    prediction = matrix @ weights
    result = pd.DataFrame({'sample_row': data.predict_rows, 'prediction': prediction[data.predict_rows]})
    result.to_parquet(folder / f'{label}.parquet', index=False)
    dump(folder / f'{label}.json', {'members': names, 'weights': dict(zip(names, weights.tolist())),
        'stack_fit_phase': 3, 'stack_origins': len(data.stack), 'solver': 'scipy HiGHS global simplex MAE LP',
        'stack_fit_seconds': time.perf_counter() - started, 'member_information': member_info or {}})
    return prediction


def save_members(data, pool, folder, label):
    frame = pd.DataFrame({'sample_row': data.predict_rows})
    for name, p in pool.items():
        frame[name] = p[data.predict_rows]
    frame.to_parquet(folder / f'{label}_members.parquet', index=False)


def arm(data, cfg, folder, label):
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / f'{label}_complete.json'
    if complete.exists():
        frame = pd.read_parquet(folder / f'{label}_members.parquet')
        pool = {}
        for name in frame.columns:
            if name != 'sample_row':
                p = np.full(len(data.y), np.nan); p[frame.sample_row] = frame[name]; pool[name] = p
        return pool, json.loads(complete.read_text())
    pool, information = {}, {}
    all_modes = {}
    for kind in ['encoder', 'lstm', 'gru']:
        pool[kind], information[kind], all_modes[kind] = recurrent_pair(data, cfg, kind, folder / 'fits')
    pool['boosting'], information['boosting'] = boost(data, cfg, folder / 'fits')
    pool['window_mean'] = data.naive(cfg['L'])
    combine(data, pool, label, folder, information)
    combine(data, {k: v for k, v in pool.items() if k != 'encoder'}, label + '_without_encoder', folder)
    combine(data, {k: pool[k] for k in ['boosting', 'window_mean']}, label + '_nonrecurrent', folder)
    direct_pool = {**pool, **{kind: all_modes[kind]['direct'] for kind in all_modes}}
    combine(data, direct_pool, label + '_direct_only', folder)
    save_members(data, pool, folder, label)
    for kind in all_modes:
        save_members(data, all_modes[kind], folder, label + '_' + kind + '_modes')
    dump(complete, {'config': cfg, 'members': information, 'complete': True})
    return pool, json.loads(complete.read_text())


def baselines(data, ranking, seed, folder):
    complete = folder / 'baselines_complete.json'
    if complete.exists():
        return
    cfg = config(add_availability(data, ranking['indices']), seed)
    pool, info = {}, {}
    for kind in ['deepar', 'weibull']:
        choices = []
        for length in [5, 10]:
            p, rec = fit(data, {**cfg, 'kind': kind, 'L': length, 'pool': 'last'}, folder / 'baseline_fits', epochs=args.epochs)
            choices.append((rec['selection_MAE'], length, p, rec))
        _, _, pool[kind], info[kind] = min(choices, key=lambda t: t[:2])
    base12 = ['last_gap', 'gap_lag2', 'gap_lag3', 'gap_lag4', 'gap_lag5', 'gap_mean',
              'gap_sd', 'events_seen', 'last_amount', 'relative_amount', 'tenure', 'global_last_gap']
    cols = [data.names.index(name) for name in base12]
    for kind in ['mlp_mae', 'mlp_mse']:
        pool[kind], info[kind] = fit(data, config(add_availability(data, cols), seed,
            kind=kind, L=1, pool='last'), folder / 'baseline_fits', epochs=args.epochs)
    pool['window_median'] = data.naive(5, 'median')
    pool['last_gap'] = data.naive(1)
    save_members(data, pool, folder, 'baselines')
    dump(complete, {'models': info, 'note': 'Matched eight-unit NN adaptation with MAE and MSE; no original-paper score reproduction claimed.'})


def ablations(data, cfg, pool, folder):
    # Group indicators change every learned member's inputs, so this ablation
    # refits the complete system rather than replacing the encoder alone.
    arm(data, {**cfg, 'K': 0}, folder, 'no_groups')
    variants = [('zero_graph', {'graph': 'zero'}), ('random_graph', {'graph': 'random'}),
                ('no_attention', {'attention': False}), ('last_step_pool', {'pool': 'last'})]
    # Broad sensitivity grids are explicitly one-seed diagnostics; the main
    # comparisons and four structural ablations use all five independent seeds.
    if cfg['seed'] == 42:
        variants += [(f'graph_P{p}_d{d}', {'P': p, 'd': d}) for p, d in itertools.product([100, 300, 600, 1000], [8, 16, 32])]
        variants += [(f'logloss_tau{tau:.2f}_cap{cap}', {'loss': 'log_pinball', 'tau': tau, 'cap': cap})
                     for tau, cap in itertools.product([.50, .55, .60, .65, .70], [None, .99])]
    for label, changes in variants:
        complete = folder / f'{label}_complete.json'
        if complete.exists():
            continue
        encoder, information, modes = recurrent_pair(data, {**cfg, **changes}, 'encoder', folder / 'fits')
        variant_pool = {**pool, 'encoder': encoder}
        combine(data, variant_pool, label, folder, {'changed_encoder': information,
                'scope': 'Full ensemble recombined after replacing only the encoder; other four fitted members held fixed.'})
        save_members(data, {'encoder': encoder, **{f'encoder_{name}': p for name, p in modes.items()}}, folder, label)
        dump(complete, {'config': {**cfg, **changes}, 'encoder': information})


args.output.mkdir(parents=True, exist_ok=True)
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
dump(args.output / f'run_manifest_{args.stage}.json', {'started_date': '2026-10-19', 'seeds': args.seeds,
    'source_sha256': source_hashes, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    'torch': torch.__version__, 'device': str(DEVICE), 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU',
    'status': 'running; no manuscript changes authorized before aggregation'})
for dataset in args.datasets:
    data = Data(args.prepared / dataset)
    folder = args.output / dataset; folder.mkdir(exist_ok=True)
    ranking = data.rank(folder)
    for seed in args.seeds:
        run_folder = folder / f'seed_{seed}'; run_folder.mkdir(exist_ok=True)
        selected = select_inputs(data, seed, run_folder, ranking)
        fixed = config(fixed_columns(data, ranking), seed, K=4)
        selected_cfg = selected['selected']
        fixed_pool, fixed_info = arm(data, fixed, run_folder, 'fixed')
        selected_pool, selected_info = arm(data, selected_cfg, run_folder, 'selected')
        baselines(data, ranking, seed, run_folder)
        if args.stage in ['ablations', 'all']:
            ablations(data, selected_cfg, selected_pool, run_folder)
        dump(args.output / 'progress.json', {'dataset': dataset, 'seed': seed, 'stage': args.stage,
             'state': 'seed completed', 'all_datasets': args.datasets, 'all_seeds': args.seeds})
    del data
    gc.collect()
dump(args.output / f'completed_{args.stage}.json', {'datasets': args.datasets, 'seeds': args.seeds, 'complete': True})
log('Completed requested clean revision stage', args.stage)

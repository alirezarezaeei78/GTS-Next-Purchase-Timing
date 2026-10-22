"""Score completed revision runs, with customer-level paired uncertainty.

This program refuses an incomplete main study by default. It does not select
models by test scores and does not modify the manuscript.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

SEEDS = [42, 7, 1400, 2026, 31337]
DATASETS = ['OnlineRetailII', 'TaFeng', 'Instacart']


def metrics(y, prediction):
    error = np.asarray(prediction) - np.asarray(y)
    ae = np.abs(error)
    return {'MAE': float(ae.mean()), 'RMSE': float(np.sqrt((error ** 2).mean())),
            'MedAE': float(np.median(ae)), 'mean_signed_error': float(error.mean()),
            'observed_below_prediction_pct': float(100 * (y <= prediction).mean()),
            **{f'within_{d}_pct': float(100 * (ae < d).mean()) for d in [1, 2, 3]}}


def holm(pvalues):
    values = np.asarray(pvalues); order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order]))
    result = np.empty(len(values)); result[order] = adjusted
    return result


def paired_inference(y, predictions, customers, bootstrap_count=2000, permutation_count=9999):
    """Pairs stay together; Holm is applied later across all dataset comparisons.

    MAE and MSE sign-flip tests exchange entire original-customer prediction
    vectors. MedAE has a paired cluster-bootstrap interval rather than an
    inappropriate time-series DM test on unordered customer observations.
    """
    names = list(predictions)
    p = np.column_stack([predictions[name] for name in names])
    errors = p - y[:, None]; ae = np.abs(errors); sq = errors ** 2
    unique, inverse = np.unique(customers, return_inverse=True)
    n = np.bincount(inverse)
    sums = np.column_stack([np.bincount(inverse, weights=ae[:, c]) for c in range(len(names))])
    squares = np.column_stack([np.bincount(inverse, weights=sq[:, c]) for c in range(len(names))])
    rng = np.random.default_rng(719)
    boot = np.empty((bootstrap_count, len(names), 3))
    sorted_rows = [np.argsort(ae[:, c], kind='stable') for c in range(len(names))]
    for b in range(bootstrap_count):
        sampled = rng.integers(0, len(unique), len(unique))
        weights = np.bincount(sampled, minlength=len(unique))
        denominator = weights @ n
        boot[b, :, 0] = weights @ sums / denominator
        boot[b, :, 1] = np.sqrt(weights @ squares / denominator)
        row_weights = weights[inverse]
        # Weighted resample median, with the same even-N convention as NumPy.
        for c, order in enumerate(sorted_rows):
            cumulative = np.cumsum(row_weights[order])
            lo = np.searchsorted(cumulative, (denominator - 1) // 2 + 1)
            hi = np.searchsorted(cumulative, denominator // 2 + 1)
            boot[b, c, 2] = (ae[order[lo], c] + ae[order[hi], c]) / 2
    primary = names.index('selected')
    result = []
    for c, name in enumerate(names):
        if c == primary:
            continue
        differences = boot[:, c] - boot[:, primary]
        entry = {'reference': name, 'unit': 'original customer', 'customers': len(unique),
                 'bootstrap_resamples': bootstrap_count}
        for j, metric in enumerate(['MAE', 'RMSE', 'MedAE']):
            entry[f'{metric}_reference_minus_selected'] = float(metrics(y, p[:, c])[metric] - metrics(y, p[:, primary])[metric])
            entry[f'{metric}_low'] = float(np.quantile(differences[:, j], .025))
            entry[f'{metric}_high'] = float(np.quantile(differences[:, j], .975))
        delta = np.column_stack([sums[:, c] - sums[:, primary], squares[:, c] - squares[:, primary]])
        statistic = np.abs(delta.sum(axis=0))
        rng = np.random.default_rng(20261019)
        exceed = np.zeros(2, dtype='int64')
        for start in range(0, permutation_count, 32):
            signs = rng.choice([-1., 1.], size=(min(32, permutation_count - start), len(unique)))
            exceed += (np.abs(signs @ delta) >= statistic[None, :] - 1e-12).sum(axis=0)
        entry['MAE_p_raw'] = float((exceed[0] + 1) / (permutation_count + 1))
        entry['MSE_p_raw'] = float((exceed[1] + 1) / (permutation_count + 1))
        entry['permutation_resamples'] = permutation_count
        result.append(entry)
    return result


def read_pool(folder, samples):
    pool = {}
    for label in ['fixed', 'selected', 'baselines']:
        path = folder / f'{label}_members.parquet'
        frame = pd.read_parquet(path).set_index('sample_row')
        for name in frame.columns:
            if label == 'fixed':
                target = 'fixed_' + name
            else:
                target = name
            pool[target] = frame[name].reindex(samples.index).to_numpy()
    for label in ['fixed', 'selected', 'selected_without_encoder', 'selected_nonrecurrent', 'selected_direct_only']:
        frame = pd.read_parquet(folder / f'{label}.parquet').set_index('sample_row')
        pool[label] = frame.prediction.reindex(samples.index).to_numpy()
    return pool


def history_rules(prepared, all_samples):
    """Independent validation tuning for strong inexpensive history references."""
    gap = np.load(prepared / 'gap_events.npy', mmap_mode='r')
    starts = np.load(prepared / 'event_start.npy', mmap_mode='r')
    select = all_samples.loc[(all_samples.phase == 2) & all_samples.observed]
    test = all_samples.loc[(all_samples.phase == 4) & all_samples.observed]
    origin = np.r_[select.origin.to_numpy(), test.origin.to_numpy()]
    pool = {}; trace = []
    for kind in ['mean', 'median']:
        choices = []
        for length in [1, 3, 5, 10, 20]:
            result = np.zeros(len(origin))
            for start in range(0, len(origin), 50000):
                o = origin[start:start + 50000]
                ix = o[:, None] - np.arange(length - 1, -1, -1)
                valid = ix > starts[o, None]
                ix = np.maximum(ix, starts[o, None])
                values = np.where(valid, gap[ix], np.nan)
                result[start:start + len(o)] = (np.nanmean(values, axis=1) if kind == 'mean'
                                                else np.nanmedian(values, axis=1))
            score = float(np.abs(result[:len(select)] - select.duration.to_numpy()).mean())
            choices.append((score, length, result[len(select):]))
            trace.append({'rule': kind, 'length': length, 'selection_MAE': score})
        score, length, pred = min(choices, key=lambda t: t[:2])
        pool['tuned_window_' + kind] = pred
    training = all_samples.loc[(all_samples.phase == 0) & all_samples.observed, 'duration']
    pool['training_median'] = np.full(len(test), float(training.median()))
    return pool, trace


def forecast_graph_coverage(prepared, samples):
    """Origin-weighted coverage, separate from all-source-basket coverage.

    One Instacart order can support several pair forecasts. These rows count
    forecast cases, while prepared/graph_coverage.csv counts original baskets.
    """
    graph_rows = np.load(prepared / 'event_graph_row.npy', mmap_mode='r')
    records = []
    for maximum in [100, 300, 600, 1000]:
        graph = np.load(prepared / f'graph_P{maximum}.npy', mmap_mode='r')
        for phase in range(5):
            for complete in [False, True]:
                d = samples.loc[(samples.phase == phase) & (samples.observed if complete else True)]
                zero = {dimension: 0 for dimension in [8, 16, 32]}
                origins = d.origin.to_numpy()
                for start in range(0, len(origins), 50000):
                    vectors = graph[graph_rows[origins[start:start + 50000]]]
                    for dimension in zero:
                        zero[dimension] += int((np.abs(vectors[:, :dimension]).max(axis=1) < 1e-12).sum())
                for dimension, count in zero.items():
                    records.append({'phase': phase, 'observed_successor_only': complete,
                        'max_products': maximum, 'dimensions': dimension, 'forecast_origins': len(d),
                        'zero_graph_vector_fraction': count / len(d) if len(d) else np.nan,
                        'denominator': 'forecast origins; repeated original baskets count for each item forecast'})
    return records


def aggregate(prepared, runs, destination):
    assert (runs / 'completed_main.json').exists(), 'Main five-seed study is still running; paper gate remains closed.'
    completion = json.loads((runs / 'completed_main.json').read_text())
    assert completion['seeds'] == SEEDS and completion['datasets'] == DATASETS
    destination.mkdir(parents=True, exist_ok=True)
    rows, seed_rows, inference, costs, tails, groups, weights, selections, timings = [], [], [], [], [], [], [], [], []
    for dataset in DATASETS:
        all_samples = pd.read_parquet(prepared / dataset / 'samples.parquet')
        pd.DataFrame(forecast_graph_coverage(prepared / dataset, all_samples)).to_csv(
            destination / f'{dataset}_forecast_graph_coverage.csv', index=False)
        rules, trace = history_rules(prepared / dataset, all_samples)
        pd.DataFrame(trace).to_csv(destination / f'{dataset}_history_rule_validation.csv', index=False)
        samples = all_samples.loc[(all_samples.phase == 4) & all_samples.observed].copy()
        seed_pools = []
        for seed in SEEDS:
            folder = runs / dataset / f'seed_{seed}'
            pool = read_pool(folder, samples)
            pool.update(rules)
            assert all(np.isfinite(v).all() and (v >= 0).all() for v in pool.values())
            seed_pools.append(pool)
            for name, p in pool.items():
                seed_rows.append({'dataset': dataset, 'seed': seed, 'model': name, **metrics(samples.duration.to_numpy(), p)})
            for label in ['fixed', 'selected']:
                rec = json.loads((folder / f'{label}.json').read_text())
                for member, value in rec['weights'].items():
                    weights.append({'dataset': dataset, 'seed': seed, 'arm': label, 'member': member, 'weight': value})
            selection = json.loads((folder / f'selection_s{seed}.json').read_text())
            chosen = selection['selected']
            selections.append({'dataset': dataset, 'seed': seed, 'L': chosen['L'], 'K': chosen['K'],
                'selected_input_channels': len(chosen['columns']), 'selection_seconds': selection['selection_seconds'],
                'evaluated_configurations': len(selection['trials'])})
            for directory, stage in [('fits', 'final_members'), ('baseline_fits', 'external_baselines'), ('selection_fits', 'input_search')]:
                for fit in (folder / directory).glob('*.json'):
                    rec = json.loads(fit.read_text())
                    is_boosting = fit.with_suffix('.joblib').exists()
                    artifact = fit.with_suffix('.joblib' if is_boosting else '.pt')
                    timings.append({'dataset': dataset, 'seed': seed, 'fit': directory + '/' + fit.stem,
                        'stage': stage, 'kind': 'boosting' if is_boosting else rec['config']['kind'],
                        'model_bytes': artifact.stat().st_size,
                        **{key: rec.get(key) for key in ['training_seconds', 'inference_seconds', 'inference_origins', 'parameters', 'peak_cuda_bytes', 'training_origins']}})
        predictions = {name: np.mean([pool[name] for pool in seed_pools], axis=0) for name in seed_pools[0]}
        # Fixed members are reported as a separate full set, never used to pick
        # the strongest test competitor after seeing results.
        frame = samples[['origin', 'series', 'customer', 'product', 'duration']].copy()
        for name, p in predictions.items():
            frame[name] = p
            rows.append({'dataset': dataset, 'model': name, 'origins': len(samples),
                         'original_customers': int(samples.customer.nunique()), **metrics(samples.duration.to_numpy(), p)})
            for early, late in [(1, 1), (2, 1), (5, 1), (1, 2), (1, 5)]:
                e = p - samples.duration.to_numpy()
                cost = early * np.maximum(-e, 0) + late * np.maximum(e, 0)
                costs.append({'dataset': dataset, 'model': name, 'early_weight': early, 'late_weight': late,
                              'hypothetical_cost_per_origin': float(cost.mean()),
                              'meaning': 'illustrative timing penalty, not measured financial profit'})
        frame.to_parquet(destination / f'{dataset}_test_predictions.parquet', index=False)
        comparisons = {name: p for name, p in predictions.items() if not name.startswith('fixed_')}
        inference += [{'dataset': dataset, **row} for row in paired_inference(samples.duration.to_numpy(), comparisons, samples.customer.to_numpy())]
        y, p = samples.duration.to_numpy(), predictions['selected']
        e = np.abs(y - p); threshold = np.quantile(y, .9); selected = y >= threshold
        tails.append({'dataset': dataset, 'threshold': threshold, 'long_gap_n': int(selected.sum()),
                      'long_gap_MAE': float(e[selected].mean()), 'absolute_error_share': float(e[selected].sum() / e.sum())})
        for i, ix in enumerate(np.array_split(np.argsort(p, kind='stable'), 10), 1):
            groups.append({'dataset': dataset, 'group': i, 'n': len(ix), 'mean_prediction': p[ix].mean(),
                           'mean_observed': y[ix].mean(), 'median_observed': np.median(y[ix]),
                           'observed_below_prediction_pct': 100 * (y[ix] <= p[ix]).mean()})
    # One predeclared family across datasets and both paired loss tests.
    family = [row[key] for row in inference for key in ['MAE_p_raw', 'MSE_p_raw']]
    adjusted = iter(holm(family))
    for row in inference:
        row['MAE_p_Holm'] = next(adjusted); row['MSE_p_Holm'] = next(adjusted)
    outputs = {'metrics': rows, 'seed_metrics': seed_rows, 'paired_inference': inference,
               'hypothetical_costs': costs, 'tails': tails, 'grouped_predictions': groups,
               'combiner_weights': weights, 'input_selections': selections, 'measured_fit_costs': timings}
    for name, records in outputs.items():
        pd.DataFrame(records).to_csv(destination / f'{name}.csv', index=False)
    manifest = {'status': 'main results complete; ablation and multi-horizon gates checked separately',
                'seeds': SEEDS, 'datasets': DATASETS, 'bootstrap_resamples': 2000,
                'permutation_resamples': 9999, 'Holm_family_size': len(family),
                'no_test_model_selection': True,
                'input_hashes': {str(p.relative_to(runs)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in runs.rglob('*_members.parquet')}}
    (destination / 'main_results_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Aggregated actual five-seed results and paired customer-level uncertainty. Manuscript unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.prepared, args.runs, args.output)

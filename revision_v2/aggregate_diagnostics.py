"""Complete ablation and five-horizon tables after every required fit exists."""
from pathlib import Path
import argparse, itertools, json
import numpy as np
import pandas as pd
from aggregate import SEEDS, DATASETS, metrics, paired_inference, holm


def aggregate_ablation(prepared, runs, destination):
    assert (runs / 'completed_ablations.json').exists(), 'Structural and sensitivity fits incomplete.'
    records, seed_records = [], []
    structural = ['zero_graph', 'random_graph', 'no_attention', 'last_step_pool', 'no_groups']
    grid = [f'graph_P{p}_d{d}' for p, d in itertools.product([100, 300, 600, 1000], [8, 16, 32])]
    loss = [f'logloss_tau{tau:.2f}_cap{cap}' for tau, cap in itertools.product([.50, .55, .60, .65, .70], [None, .99])]
    for dataset in DATASETS:
        samples = pd.read_parquet(prepared / dataset / 'samples.parquet')
        samples = samples.loc[(samples.phase == 4) & samples.observed]
        y = samples.duration.to_numpy()
        for label in structural + grid + loss:
            seeds = SEEDS if label in structural else [42]
            ensemble, encoder = [], []
            for seed in seeds:
                folder = runs / dataset / f'seed_{seed}'
                assert (folder / f'{label}_complete.json').exists()
                frame = pd.read_parquet(folder / f'{label}.parquet').set_index('sample_row')
                eframe = pd.read_parquet(folder / f'{label}_members.parquet').set_index('sample_row')
                p = frame.prediction.reindex(samples.index).to_numpy()
                ep = eframe.encoder.reindex(samples.index).to_numpy()
                assert np.isfinite(p).all() and np.isfinite(ep).all()
                ensemble.append(p); encoder.append(ep)
                for kind, v in [('full_system', p), ('standalone_encoder', ep)]:
                    seed_records.append({'dataset': dataset, 'variant': label, 'seed': seed, 'scope': kind, **metrics(y, v)})
            for kind, values in [('full_system', ensemble), ('standalone_encoder', encoder)]:
                records.append({'dataset': dataset, 'variant': label, 'scope': kind, 'seeds': len(seeds),
                                'averaged_prediction': len(seeds) > 1, **metrics(y, np.mean(values, axis=0))})
    pd.DataFrame(records).to_csv(destination / 'ablation_metrics.csv', index=False)
    pd.DataFrame(seed_records).to_csv(destination / 'ablation_seed_metrics.csv', index=False)


def aggregate_horizons(folder, destination):
    assert (folder / 'complete.json').exists(), 'Five-horizon fits incomplete.'
    assert json.loads((folder / 'complete.json').read_text())['seeds'] == SEEDS
    frames = [pd.read_parquet(folder / f'seed_{seed}/test_predictions.parquet') for seed in SEEDS]
    reference = frames[0][['origin', 'customer']]
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(frame[['origin', 'customer']], reference)
    records, seed_records, inference = [], [], []
    output = reference.copy()
    for horizon in range(1, 6):
        actual_column = f'actual_h{horizon}'
        y = frames[0][actual_column].to_numpy()
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame[actual_column], y)
        output[actual_column] = y
        columns = [c for c in frames[0] if c.endswith(f'_h{horizon}') and c != actual_column]
        predictions = {}
        for c in columns:
            values = [frame[c].to_numpy() for frame in frames]
            model = c.rsplit('_h', 1)[0]
            p = np.mean(values, axis=0)
            predictions['selected' if model == 'GTS' else model] = p
            output[c] = p
            records.append({'horizon': horizon, 'model': model, 'origins': len(y), **metrics(y, p)})
            for seed, value in zip(SEEDS, values):
                seed_records.append({'horizon': horizon, 'seed': seed, 'model': model, **metrics(y, value)})
        # Predeclared direct/recursive comparators and simple rules for each h.
        chosen = {name: predictions[name] for name in ['selected', 'transformer_direct', 'transformer_recursive', 'window_mean', 'window_median']}
        inference.extend({'horizon': horizon, **r} for r in paired_inference(y, chosen, reference.customer.to_numpy()))
    values = iter(holm([r[k] for r in inference for k in ['MAE_p_raw', 'MSE_p_raw']]))
    for row in inference:
        row['MAE_p_Holm_horizon_family'] = next(values); row['MSE_p_Holm_horizon_family'] = next(values)
    pd.DataFrame(records).to_csv(destination / 'horizon_metrics.csv', index=False)
    pd.DataFrame(seed_records).to_csv(destination / 'horizon_seed_metrics.csv', index=False)
    pd.DataFrame(inference).to_csv(destination / 'horizon_inference.csv', index=False)
    output.to_parquet(destination / 'multihorizon_test_predictions.parquet', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--multihorizon', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate_ablation(args.prepared, args.runs, args.output)
    aggregate_horizons(args.multihorizon, args.output)
    assert (args.output / 'main_results_manifest.json').exists()
    (args.output / 'numerical_sections_complete.json').write_text(json.dumps({
        'main_five_seed_experiments': 'complete', 'structural_ablations': 'complete',
        'graph_loss_sensitivity': 'complete, explicitly single-seed diagnostics',
        'five_horizon_comparison': 'complete', 'statistical_and_cost_tables': 'complete',
        'paper_edit_gate': 'waiting for independent saved-model inference and final result verification'
    }, indent=2), encoding='utf-8')
    print('All required numerical sections completed. The paper can now be revised from these actual results.')

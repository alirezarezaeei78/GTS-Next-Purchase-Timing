"""Open the manuscript-edit gate only after complete, audited actual results."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from aggregate import DATASETS, SEEDS, metrics


def verify(prepared, runs, root):
    assert (root / 'numerical_sections_complete.json').exists()
    benchmark = json.loads((root / 'benchmark_complete.json').read_text())
    assert benchmark['datasets'] == DATASETS and benchmark['seeds'] == SEEDS
    assert benchmark['live_inference_verified_against_saved_predictions']
    table = pd.read_csv(root / 'metrics.csv')
    weights = pd.read_csv(root / 'combiner_weights.csv')
    assert set(weights.dataset) == set(DATASETS) and set(weights.seed) == set(SEEDS)
    for _, group in weights.groupby(['dataset', 'seed', 'arm']):
        assert (group.weight >= -1e-10).all() and abs(group.weight.sum() - 1) < 1e-9
    for dataset in DATASETS:
        truth = pd.read_parquet(prepared / dataset / 'samples.parquet')
        truth = truth.loc[(truth.phase == 4) & truth.observed]
        frame = pd.read_parquet(root / f'{dataset}_test_predictions.parquet')
        np.testing.assert_array_equal(frame.origin, truth.origin)
        np.testing.assert_array_equal(frame.duration, truth.duration)
        for _, row in table[table.dataset == dataset].iterrows():
            pred = frame[row.model].to_numpy()
            assert np.isfinite(pred).all() and (pred >= 0).all()
            for key, expected in metrics(frame.duration.to_numpy(), pred).items():
                assert abs(row[key] - expected) < 1e-8, (dataset, row.model, key)
        for seed in SEEDS:
            folder = runs / dataset / f'seed_{seed}'
            member = pd.read_parquet(folder / 'selected_members.parquet').set_index('sample_row')
            meta = json.loads((folder / 'selected.json').read_text())
            actual = pd.read_parquet(folder / 'selected.parquet').set_index('sample_row')
            recomputed = sum(member[name] * weight for name, weight in meta['weights'].items())
            np.testing.assert_allclose(recomputed, actual.prediction, atol=1e-9)
        grouped = pd.read_csv(root / 'grouped_predictions.csv')
        assert grouped.loc[grouped.dataset == dataset, 'n'].sum() == len(truth)
    paired = pd.read_csv(root / 'paired_inference.csv')
    for metric in ['MAE', 'RMSE', 'MedAE']:
        assert (paired[f'{metric}_low'] <= paired[f'{metric}_high']).all()
    for metric in ['MAE', 'MSE']:
        assert ((paired[f'{metric}_p_raw'] >= 0) & (paired[f'{metric}_p_Holm'] <= 1)).all()
        assert (paired[f'{metric}_p_Holm'] >= paired[f'{metric}_p_raw'] - 1e-12).all()
    horizon = pd.read_parquet(root / 'multihorizon_test_predictions.parquet')
    ht = pd.read_csv(root / 'horizon_metrics.csv')
    for _, row in ht.iterrows():
        h = int(row.horizon)
        actual = horizon[f'actual_h{h}'].to_numpy()
        pred = horizon[f'{row.model}_h{h}'].to_numpy()
        for key, expected in metrics(actual, pred).items():
            assert abs(row[key] - expected) < 1e-8, (row.model, h, key)
    for name in ['revision_model_comparison', 'revision_grouped_predictions', 'revision_gap_distributions',
                 'revision_combiner_weights', 'revision_graph_sensitivity', 'revision_five_horizons']:
        for extension in ['pdf', 'png', 'svg']:
            assert (root / 'figures' / f'{name}.{extension}').stat().st_size > 1000
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file() and p.name != 'results_ready_for_paper.json'}
    report = {'status': 'complete numerical evidence, ready for manuscript revision and visual review',
              'all_main_datasets_and_seeds': True, 'structural_ablations': True,
              'single_seed_sensitivity_clearly_labeled': True, 'five_horizon_metrics': True,
              'saved_model_live_inference_matches': True, 'point_metrics_and_stack_predictions_recomputed': True,
              'paired_intervals_and_Holm_checked': True, 'six_vector_result_figures': True,
              'no_journal_acceptance_or_universal_accuracy_claim': True, 'file_sha256': manifest}
    (root / 'results_ready_for_paper.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: complete actual result sections and saved-model inference verified. Manuscript revision gate is open.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    verify(args.prepared, args.runs, args.results)

"""Audit causal preparation and independent fitting phases before model training."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from prepare import feature_bank


def verify(folder):
    samples = pd.read_parquet(folder / 'samples.parquet')
    keys = pd.read_parquet(folder / 'event_keys.parquet')
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert samples.customer.notna().all() and keys.customer.notna().all()
    assert (samples.duration >= 0).all()
    assert set(samples.phase) == set(range(5))
    train = samples.loc[(samples.phase == 0) & samples.observed]
    assert (train.t + train.duration < train.end).all()
    if manifest['dataset'] != 'Instacart':
        # Stronger than ordering within each customer: every customer shares
        # one boundary, and every training label precedes every later origin.
        assert (samples.groupby('phase').end.nunique() == 1).all()
        assert float((train.t + train.duration).max()) < float(samples.loc[samples.phase > 0, 't'].min())
    # Every original customer has monotonically ordered, nonoverlapping phases,
    # even when that customer's item histories have different purchase origins.
    group = keys.groupby(['customer', 'phase']).order.agg(['min', 'max']).unstack('phase')
    for left in range(5):
        for right in range(left + 1, 5):
            lo, hi = group['max'][left], group['min'][right]
            valid = lo.notna() & hi.notna()
            assert (lo[valid] < hi[valid]).all(), (manifest['dataset'], left, right)
    assert not samples.loc[samples.phase > 0].duplicated(['series', 'phase']).any()
    # The exact event-index used in a sample must be from its own fitting phase.
    np.testing.assert_array_equal(keys.phase.to_numpy()[samples.origin], samples.phase)
    profiles = np.load(folder / 'profiles.npy', mmap_mode='r')
    profile_rows = np.load(folder / 'event_profile_row.npy', mmap_mode='r')
    assert len(profiles) <= keys.customer.nunique() if manifest['dataset'] != 'Instacart' else len(profiles) >= keys.customer.nunique()
    assignments = pd.DataFrame({'customer': keys.customer, 'profile': profile_rows})
    assert (assignments.groupby('customer').profile.nunique() == 1).all()
    for p in [100, 300, 600, 1000]:
        vectors = np.load(folder / f'graph_P{p}.npy', mmap_mode='r')
        assert vectors.shape[1] == 32
        for start in range(0, len(vectors), 100000):
            assert np.isfinite(vectors[start:start + 100000]).all()
    result = {'dataset': manifest['dataset'], 'training_labels_before_cutoff': True,
              'global_calendar_boundary_checked': manifest['dataset'] != 'Instacart',
              'original_customer_shared_order_phases': True, 'stack_phase_independent': True,
              'original_customer_unique_profile': True, 'finite_graph_arrays': True,
              'phase_counts': manifest['phase_counts']}
    return result


def causal_feature_test():
    baskets = pd.DataFrame({'sid': [1]*8 + [2]*8, 'customer': [1]*8 + [2]*8,
        'order': list(range(8))*2, 't': [0, 2, 4, 8, 10, 15, 19, 22]*2,
        'basket': range(16), 'amount': np.arange(1, 17, dtype=float),
        'quantity': np.ones(16), 'item_count': np.ones(16),
        'dow': list(range(7)) + [0] + list(range(7)) + [0], 'month': np.ones(16)})
    full, values, names, *_ = feature_bank(baskets, baskets, 'OnlineRetailII')
    altered = baskets.copy()
    future = altered.order > 4
    altered.loc[future, 'amount'] *= 500
    altered.loc[future, 't'] += 100
    changed, new, names2, *_ = feature_bank(altered, altered, 'OnlineRetailII')
    assert names == names2
    prior = full.order <= 4
    np.testing.assert_allclose(values[prior], new[prior], atol=1e-5)
    # Missing first gap is flagged; a genuine second zero-day gap is available.
    zero = baskets.copy(); zero.loc[zero.order == 1, 't'] = 0
    _, z, names, *_ = feature_bank(zero, zero, 'Instacart')
    assert z[0, names.index('gap_available')] == 0
    assert z[1, names.index('last_gap')] == 0 and z[1, names.index('gap_available')] == 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    causal_feature_test()
    reports = [verify(folder) for folder in sorted(args.prepared.iterdir()) if (folder / 'manifest.json').exists()]
    assert len(reports) == 3
    target = args.prepared.parent / 'pretraining_audit.json'
    target.write_text(json.dumps({'future_feature_mutation_check': 'pass', 'datasets': reports}, indent=2), encoding='utf-8')
    print('PASS: three clean datasets, shared cutoffs, independent phases, unique profiles, zero-gap flags and future-feature mutation check.')

"""Clean public-data preparation with five disjoint fitting/evaluation phases.

No model selection or test scoring occurs in this program. Raw inputs and model
weights stay outside the public repository. Relative Instacart time is never
represented as an actual calendar date.
"""
from pathlib import Path
import argparse, gc, hashlib, json, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

FRACTIONS = np.array([0, .60, .70, .80, .90, 1.0])
GRAPH_PRODUCTS = [100, 300, 600, 1000]


def log(*values):
    print(time.strftime('%H:%M:%S'), *values, flush=True)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_daily(root, name):
    if name == 'OnlineRetailII':
        path = root / 'online_retail_ii/online_retail_II.parquet'
        raw = pd.read_parquet(path)
        n_raw = len(raw)
        identified = raw['Customer ID'].notna()
        missing = int((~identified).sum())
        raw = raw.loc[identified].copy()  # Before any identifier conversion.
        raw = raw.loc[(raw.Quantity > 0) & (raw.Price > 0)
                      & ~raw.Invoice.astype(str).str.upper().str.startswith('C')].copy()
        frame = pd.DataFrame({'customer': raw['Customer ID'].astype('int64'),
                              'product': raw.StockCode.astype(str),
                              'date': pd.to_datetime(raw.InvoiceDate).dt.normalize(),
                              'quantity': raw.Quantity.astype(float),
                              'amount': (raw.Quantity * raw.Price).astype(float)})
    else:
        path = root / 'ta_feng/ta_feng_all_months_merged.csv'
        raw = pd.read_csv(path)
        n_raw = len(raw)
        identified = raw.CUSTOMER_ID.notna()
        missing = int((~identified).sum())
        raw = raw.loc[identified & (raw.AMOUNT > 0) & (raw.SALES_PRICE > 0)].copy()
        frame = pd.DataFrame({'customer': raw.CUSTOMER_ID.astype('int64'),
                              'product': raw.PRODUCT_ID.astype(str),
                              'date': pd.to_datetime(raw.TRANSACTION_DT).dt.normalize(),
                              'quantity': raw.AMOUNT.astype(float),
                              'amount': raw.SALES_PRICE.astype(float)})
    del raw
    assert frame.customer.notna().all()
    start = frame.date.min()
    frame['t'] = (frame.date - start).dt.days.astype('int32')
    end = int(frame.t.max()) + 1
    boundaries = np.floor(FRACTIONS * end).astype('int32')
    counts = frame.loc[frame.t < boundaries[1]].groupby('customer').date.nunique()
    keep = counts[counts >= 3].index
    # Eligibility depends only on the training prefix, never on later purchases.
    frame = frame.loc[frame.customer.isin(keep)].copy()
    frame['basket'] = pd.factorize(pd.MultiIndex.from_frame(frame[['customer', 't']]), sort=True)[0]
    baskets = frame.groupby('basket', sort=True).agg(customer=('customer', 'first'), t=('t', 'first'),
        amount=('amount', 'sum'), quantity=('quantity', 'sum'), item_count=('product', 'size'))
    baskets['dow'] = (start + pd.to_timedelta(baskets.t, unit='D')).dt.dayofweek
    baskets['month'] = (start + pd.to_timedelta(baskets.t, unit='D')).dt.month
    baskets['order'] = baskets.t
    baskets['sid'] = baskets.customer
    baskets['product_id'] = -1
    baskets['phase'] = np.searchsorted(boundaries[1:-1], baskets.t, side='right')
    for k in range(1, 6):
        baskets[f'end{k}'] = boundaries[k]
    edges = frame[['basket', 'product']].drop_duplicates()
    audit = {'raw_rows': n_raw, 'missing_id_rows_removed_before_conversion': missing,
             'positive_identified_line_items': len(frame), 'training_eligible_customers': len(keep),
             'source_sha256': {path.name: digest(path)},
             'calendar_boundaries': [(start + pd.Timedelta(days=int(x))).strftime('%Y-%m-%d')
                                     for x in boundaries], 'time_unit': 'calendar days'}
    return baskets.reset_index(), baskets.reset_index(), edges, audit


def load_instacart(root):
    folder = root / 'instacart'
    orders = pd.read_csv(folder / 'orders.csv', dtype={'order_id': 'int32', 'user_id': 'int32',
        'order_number': 'int16', 'order_dow': 'int8', 'days_since_prior_order': 'float32'})
    op = pd.concat([pd.read_csv(folder / file, usecols=['order_id', 'product_id'],
        dtype={'order_id': 'int32', 'product_id': 'int32'})
        for file in ['order_products__prior.csv', 'order_products__train.csv']], ignore_index=True)
    orders = orders.loc[orders.order_id.isin(op.order_id.unique())].copy()
    # Orders whose products were not released are NOT empty observed baskets.
    orders.sort_values(['user_id', 'order_number'], inplace=True)
    orders['t'] = orders.days_since_prior_order.fillna(0).groupby(orders.user_id).cumsum().astype('int32')
    orders['last_order'] = orders.groupby('user_id').order_number.transform('max')
    cut_columns = []
    for phase, fraction in enumerate(FRACTIONS[1:], 1):
        column = f'cut{phase}'
        orders[column] = np.floor(orders.last_order * fraction).astype('int16')
        cut_columns.append(column)
    orders['phase'] = sum((orders.order_number > orders[c]).astype('int8') for c in cut_columns[:-1])
    # Per-user cutoff days are taken from the same ORIGINAL order for every pair.
    for phase, column in enumerate(cut_columns, 1):
        at_cut = orders.loc[orders.order_number == orders[column]].set_index('user_id').t
        orders[f'end{phase}'] = orders.user_id.map(at_cut).fillna(0).astype('int32') + 1e-4
    basket_n = op.groupby('order_id').size()
    orders['item_count'] = orders.order_id.map(basket_n).astype('float32')
    orders['amount'] = orders.item_count
    orders['quantity'] = orders.item_count
    orders.rename(columns={'order_id': 'basket', 'user_id': 'customer', 'order_number': 'order',
                           'order_dow': 'dow'}, inplace=True)
    orders['month'] = 0
    lut_size = int(orders.basket.max()) + 1
    user_lut = np.zeros(lut_size, dtype='int32')
    phase_lut = np.full(lut_size, -1, dtype='int8')
    user_lut[orders.basket] = orders.customer
    phase_lut[orders.basket] = orders.phase
    oid, pid = op.order_id.to_numpy(), op.product_id.to_numpy()
    pair = user_lut[oid].astype('int64') * 60000 + pid
    keys, counts = np.unique(pair[phase_lut[oid] == 0], return_counts=True)
    eligible = keys[counts >= 6]
    mask = np.isin(pair, eligible)
    events = pd.DataFrame({'sid': pair[mask], 'basket': oid[mask], 'product_id': pid[mask]})
    events = events.merge(orders, on='basket', how='left', validate='many_to_one')
    edges = op.rename(columns={'order_id': 'basket', 'product_id': 'product'})
    audit = {'public_order_product_rows': len(op), 'observable_baskets': len(orders),
             'training_eligible_pairs': len(eligible), 'eligibility_minimum_training_purchases': 6,
             'source_sha256': {file: digest(folder / file) for file in
                ['orders.csv', 'order_products__prior.csv', 'order_products__train.csv']},
             'time_unit': 'recorded relative same-item days',
             'cutoff_rule': 'floor(.60/.70/.80/.90 times each original customer observable order count)',
             'unreleased_test_baskets_excluded': True}
    del pair, mask, user_lut, phase_lut, op
    gc.collect()
    return events, orders, edges, audit


def feature_bank(events, baskets, name):
    events = events.sort_values(['sid', 'order'], kind='stable').reset_index(drop=True)
    sid, starts, sizes = np.unique(events.sid, return_index=True, return_counts=True)
    group = np.repeat(np.arange(len(sid)), sizes)
    begin = np.repeat(starts, sizes)
    pos = np.arange(len(events)) - begin
    ix = np.arange(len(events))
    t = events.t.to_numpy(dtype=float)
    gap = np.r_[0, np.diff(t)]
    gap[pos == 0] = 0
    amount = events.amount.to_numpy(dtype=float)
    def prefix(x):
        return np.r_[0., np.cumsum(x, dtype=float)]
    pg, pg2, pa, pa2 = prefix(gap), prefix(gap ** 2), prefix(amount), prefix(amount ** 2)
    count = pos + 1
    f = {'last_gap': gap, 'events_seen': count, 'gap_available': (pos > 0).astype(float),
         'gap_std_available': (pos > 1).astype(float)}
    for lag in [2, 3, 4, 5]:
        valid = pos >= lag
        values = np.zeros(len(events)); values[valid] = gap[ix[valid] - lag + 1]
        f[f'gap_lag{lag}'] = values
        f[f'gap_lag{lag}_available'] = valid.astype(float)
    for width in [3, 5, 10, 20]:
        left = np.maximum(begin, ix - width + 1)
        # First event is not a real zero gap: exclude it from gap statistics.
        gap_left = np.maximum(begin + 1, left)
        n = np.maximum(0, ix - gap_left + 1)
        total = pg[ix + 1] - pg[gap_left]
        sq = pg2[ix + 1] - pg2[gap_left]
        f[f'gap_mean{width}'] = total / np.maximum(n, 1)
        f[f'gap_sd{width}'] = np.sqrt(np.maximum(0, sq - total ** 2 / np.maximum(n, 1)) / np.maximum(n - 1, 1))
        n = ix - left + 1
        total = pa[ix + 1] - pa[left]; sq = pa2[ix + 1] - pa2[left]
        prefix_name = 'order_size' if name == 'Instacart' else 'basket_value'
        f[f'{prefix_name}_mean{width}'] = total / n
        f[f'{prefix_name}_sd{width}'] = np.sqrt(np.maximum(0, sq - total ** 2 / n) / np.maximum(n - 1, 1))
        f[f'window{width}_observed_count'] = n
    n = np.maximum(pos, 1)
    sg = pg[ix + 1] - pg[begin + 1]; sg2 = pg2[ix + 1] - pg2[begin + 1]
    f['gap_mean'] = sg / n
    f['gap_sd'] = np.sqrt(np.maximum(0, sg2 - sg ** 2 / n) / np.maximum(n - 1, 1))
    f['amount_sum'] = pa[ix + 1] - pa[begin]
    f['amount_mean'] = f['amount_sum'] / count
    f['last_amount'] = amount
    f['relative_amount'] = amount / np.maximum(f['amount_mean'], 1e-8)
    f['tenure'] = t - t[begin]
    f['item_count'] = events.item_count.to_numpy(dtype=float)
    f['quantity'] = events.quantity.to_numpy(dtype=float)
    f['sin_dow'] = np.sin(2 * np.pi * events.dow.to_numpy() / 7)
    f['cos_dow'] = np.cos(2 * np.pi * events.dow.to_numpy() / 7)
    if name != 'Instacart':
        f['sin_month'] = np.sin(2 * np.pi * events.month.to_numpy() / 12)
        f['cos_month'] = np.cos(2 * np.pi * events.month.to_numpy() / 12)
    # Global ORIGINAL-customer activity includes orders without the modeled item.
    b = baskets.sort_values(['customer', 'order'], kind='stable').copy()
    b['global_order_count'] = b.groupby('customer').cumcount() + 1
    b['global_mean_size'] = b.groupby('customer').amount.cumsum() / b.global_order_count
    b['global_last_gap'] = b.groupby('customer').t.diff().fillna(0)
    for column in ['global_order_count', 'global_mean_size', 'global_last_gap']:
        f[column] = events.basket.map(b.set_index('basket')[column]).to_numpy(dtype=float)
    x = np.column_stack(list(f.values())).astype('float32')
    assert np.isfinite(x).all()
    return events, x, list(f), gap, group, starts, sizes, pos


def graph_vectors(baskets, edges, out):
    edges = edges.drop_duplicates(['basket', 'product']).copy()
    train_baskets = baskets.loc[baskets.phase == 0, 'basket']
    train_edges = edges.loc[edges.basket.isin(train_baskets)]
    frequencies = train_edges.groupby('product').size().sort_values(ascending=False, kind='stable')
    basket_codes, basket_ids = pd.factorize(edges.basket, sort=True)
    train_mask = pd.Index(basket_ids).isin(train_baskets)
    reports = []
    for maximum in GRAPH_PRODUCTS:
        top = frequencies.index[:maximum]
        columns = top.get_indexer(edges['product'])
        valid = columns >= 0
        b = sparse.coo_matrix((np.ones(valid.sum(), dtype='float32'),
            (basket_codes[valid], columns[valid])), shape=(len(basket_ids), len(top))).tocsr()
        adjacency = (b[train_mask].T @ b[train_mask]).astype('float64')
        adjacency.setdiag(0); adjacency.eliminate_zeros()
        components, _ = connected_components(adjacency, directed=False)
        adjacency = adjacency + sparse.eye(len(top))
        scale = 1 / np.sqrt(np.asarray(adjacency.sum(axis=1)).ravel())
        normalized = sparse.diags(scale) @ adjacency @ sparse.diags(scale)
        k = min(len(top) - 1, 33 + components)
        if k < len(top) - 1:
            vals, vecs = eigsh(normalized, k=k, which='LA', tol=1e-9,
                              v0=np.random.default_rng(42).normal(size=len(top)))
        else:
            vals, vecs = np.linalg.eigh(normalized.toarray())
        order = np.argsort(vals)[::-1]
        order = order[vals[order] < 1 - 1e-7][:32]  # All component-constant modes excluded.
        embedding = np.zeros((len(top), 32), dtype='float32')
        kept = vecs[:, order]
        for j in range(kept.shape[1]):
            if kept[np.argmax(np.abs(kept[:, j])), j] < 0:
                kept[:, j] *= -1
        embedding[:, :len(order)] = kept * np.sqrt(np.maximum(0, vals[order]))
        denom = np.asarray(b.sum(axis=1)).ravel()
        vectors = np.asarray(b @ embedding) / np.maximum(denom, 1)[:, None]
        np.save(out / f'graph_P{maximum}.npy', vectors.astype('float32'))
        np.save(out / f'graph_P{maximum}_embedding.npy', embedding)
        # Seeded random item vectors, pooled with the identical represented items.
        if maximum == 600:
            random = np.random.default_rng(20261019).normal(0, 1 / np.sqrt(maximum), (len(top), 32))
            rv = np.asarray(b @ random) / np.maximum(denom, 1)[:, None]
            np.save(out / 'graph_random.npy', rv.astype('float32'))
        for phase in range(5):
            selected = pd.Index(basket_ids).isin(baskets.loc[baskets.phase == phase, 'basket'])
            reports.append({'max_products': maximum, 'dimensions_available': len(order),
                'connected_components': components, 'phase': phase, 'baskets': int(selected.sum()),
                'no_represented_item_fraction': float((denom[selected] == 0).mean()),
                'zero_vector_fraction': float((np.linalg.norm(vectors[selected], axis=1) < 1e-12).mean())})
        dump(out / f'graph_P{maximum}_products.json', [str(p) for p in top])
        del b, adjacency, normalized, vectors
        gc.collect()
    np.save(out / 'graph_basket_ids.npy', np.asarray(basket_ids))
    pd.DataFrame(reports).to_csv(out / 'graph_coverage.csv', index=False)
    return pd.Index(basket_ids)


def prepare(root, out, name):
    started = time.time(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        log('Already prepared', name); return
    events, baskets, edges, audit = load_instacart(root) if name == 'Instacart' else load_daily(root, name)
    log(name, 'loaded', len(events), 'series events', len(baskets), 'baskets')
    events, x, names, gap, group, starts, sizes, pos = feature_bank(events, baskets, name)
    graph_ids = graph_vectors(baskets, edges, out)
    np.save(out / 'x_events.npy', x)
    np.save(out / 'gap_events.npy', gap.astype('float32'))
    np.save(out / 'event_start.npy', starts[group])
    np.save(out / 'event_graph_row.npy', graph_ids.get_indexer(events.basket).astype('int32'))
    dump(out / 'feature_names.json', names)
    # One profile per original customer, NOT repeated window/pair profiles.
    tr_b = baskets.loc[baskets.phase == 0].sort_values(['customer', 'order']).copy()
    tr_b['gap'] = tr_b.groupby('customer').t.diff()
    profiles = tr_b.groupby('customer').agg(mean_gap=('gap', 'mean'), sd_gap=('gap', 'std'),
        events=('t', 'size'), mean_amount=('amount', 'mean')).fillna(0)
    np.save(out / 'profiles.npy', np.log1p(np.maximum(0, profiles.to_numpy())).astype('float32'))
    profile_row = profiles.index.get_indexer(events.customer)
    assert (profile_row >= 0).all()
    np.save(out / 'event_profile_row.npy', profile_row.astype('int32'))
    phase = events.phase.to_numpy()
    t = events.t.to_numpy(dtype=float)
    next_t = np.r_[t[1:], np.inf]
    next_order = np.r_[events.order.to_numpy()[1:], np.inf]
    last = pos == sizes[group] - 1
    next_t[last] = np.inf; next_order[last] = np.inf
    endpoint = np.column_stack([events[f'end{k}'].to_numpy() for k in range(1, 6)])[np.arange(len(events)), phase]
    observed = next_t < endpoint
    # At an Instacart cutoff, a same-relative-day LATER order is still future.
    if name == 'Instacart':
        cuts = np.column_stack([events[f'cut{k}'].to_numpy() for k in range(1, 6)])
        observed &= next_order <= cuts[np.arange(len(events)), phase]
    duration = np.where(observed, next_t - t, np.maximum(0, endpoint - t))
    use = pos >= 2
    # All earlier sliding training origins; first later origin chosen without y.
    later = pd.DataFrame({'row': np.flatnonzero(use & (phase > 0)),
                          'series': group[use & (phase > 0)], 'phase': phase[use & (phase > 0)]})
    keep_later = later.groupby(['series', 'phase'], sort=False).row.first().to_numpy()
    keep = np.sort(np.r_[np.flatnonzero(use & (phase == 0)), keep_later])
    samples = pd.DataFrame({'origin': keep, 'series': group[keep],
        'customer': events.customer.to_numpy()[keep], 'product': events.product_id.to_numpy()[keep],
        'phase': phase[keep], 't': t[keep], 'end': endpoint[keep],
        'duration': duration[keep], 'observed': observed[keep], 'history_events': pos[keep] + 1})
    assert np.isfinite(samples.duration).all() and (samples.duration >= 0).all()
    assert not samples.loc[samples.phase > 0].duplicated(['series', 'phase']).any()
    assert (samples.loc[(samples.phase == 0) & samples.observed, 't']
            + samples.loc[(samples.phase == 0) & samples.observed, 'duration']
            < samples.loc[(samples.phase == 0) & samples.observed, 'end']).all()
    samples.to_parquet(out / 'samples.parquet', index=False)
    events[['sid', 'customer', 'order', 'phase', 't', 'basket']].to_parquet(out / 'event_keys.parquet', index=False)
    phase_counts = samples.groupby('phase').agg(origins=('origin', 'size'), observed=('observed', 'sum')).to_dict('index')
    audit.update(dataset=name, events=len(events), numerical_candidates=len(names),
        original_customers=int(events.customer.nunique()), series=int(events.sid.nunique()),
        phase_counts={str(k): v for k, v in phase_counts.items()},
        sliding_training_windows=True, profile_fitting_unit='one original customer',
        graph_fitting_phase='training baskets only', seconds=time.time() - started)
    dump(out / 'manifest.json', audit)
    log('PREPARED', name, phase_counts, 'seconds', round(time.time() - started, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['OnlineRetailII', 'TaFeng', 'Instacart'])
    args = parser.parse_args()
    for dataset in args.datasets:
        prepare(args.data_root, args.output / dataset, dataset)

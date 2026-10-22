"""Clean five-purchase-day forecasts with matched direct/recursive Transformers.

Only positive purchase days from the revised Online Retail II preparation are
used. No returns, sub-day timestamp remainder, or future observed context is
fed into a forecast. Every horizon is scored separately.
"""
from pathlib import Path
import argparse, copy, gc, json, math, sys, time
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared', type=Path, required=True)
parser.add_argument('--runs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cuda-runtime', type=Path)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 1400, 2026, 31337])
args = parser.parse_args() if __name__ == '__main__' else None
if args is not None and args.cuda_runtime:
    sys.path.insert(0, str(args.cuda_runtime))
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from engine import Data, Network, DEVICE, SEEDS, log, dump, simplex_mae


class DirectNetwork(Network):
    def __init__(self, numeric, cfg, scale, center, spread):
        super().__init__(numeric, cfg, scale, 0., 1.)
        self.output = nn.Linear(self.hidden.out_features, 5)
        self.register_buffer('centers', torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer('spreads', torch.as_tensor(spread, dtype=torch.float32))

    def points(self, raw, anchor):
        z = self.centers + self.spreads * raw
        if self.cfg['mode'] == 'residual':
            z = z + torch.log1p(anchor[:, None])
        return torch.expm1(z.clamp(0, 20))


class Transformer(nn.Module):
    def __init__(self, numeric, length, center, spread, recursive):
        super().__init__()
        self.recursive = recursive
        self.project = nn.Linear(numeric, 64)
        self.positions = nn.Parameter(torch.randn(1, length, 64) * .02)
        layer = nn.TransformerEncoderLayer(64, 8, 128, dropout=.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output = nn.Linear(64, 1 if recursive else 5)
        self.register_buffer('centers', torch.as_tensor(center[:1] if recursive else center, dtype=torch.float32))
        self.register_buffer('spreads', torch.as_tensor(spread[:1] if recursive else spread, dtype=torch.float32))

    def forward(self, x, g, valid):
        h = self.project(x) + self.positions
        # Left-padded query rows can otherwise be fully masked by a causal mask.
        # Context is entirely observed; bidirectional encoding of that context is safe.
        h = self.encoder(h, src_key_padding_mask=~valid)
        return self.output(h[:, -1])

    def points(self, raw, anchor):
        return torch.expm1((self.centers + self.spreads * raw).clamp(0, 20))


def forecast_covariate_update(data, cfg, rows, raw, gap_history, amount_history, prediction, elapsed):
    """Future basket attributes are held at the last known value, never observed.

    Gap statistics, event counts, relative amounts, and known calendar encodings
    are updated from recursively PREDICTED gaps. This convention is reported
    alongside the recursive benchmark rather than hidden as test context.
    """
    f = {name: raw[:, i].copy() for i, name in enumerate(data.names)}
    old_count = f['events_seen'] - 1
    old_mean = f['gap_mean'].copy()
    new_mean = (old_mean * old_count + prediction) / (old_count + 1)
    variance_sum = f['gap_sd'] ** 2 * np.maximum(old_count - 1, 0)
    variance_sum += (prediction - old_mean) * (prediction - new_mean)
    f['gap_mean'] = new_mean
    f['gap_sd'] = np.sqrt(np.maximum(0, variance_sum) / np.maximum(old_count, 1))
    f['events_seen'] += 1
    f['gap_available'][:] = 1; f['gap_std_available'][:] = 1
    gap_history = np.column_stack([gap_history[:, 1:], prediction])
    amount_history = np.column_stack([amount_history[:, 1:], f['last_amount']])
    f['last_gap'] = prediction
    for lag in [2, 3, 4, 5]:
        values = gap_history[:, -lag]
        f[f'gap_lag{lag}'] = np.nan_to_num(values)
        f[f'gap_lag{lag}_available'] = np.isfinite(values).astype(float)
    for width in [3, 5, 10, 20]:
        gaps = gap_history[:, -width:]
        amounts = amount_history[:, -width:]
        f[f'gap_mean{width}'] = np.nanmean(gaps, axis=1)
        # Stable population-to-sample conversion avoids warnings at one value.
        count = np.isfinite(gaps).sum(1)
        centered = np.where(np.isfinite(gaps), gaps - np.nanmean(gaps, axis=1)[:, None], 0)
        f[f'gap_sd{width}'] = np.sqrt((centered ** 2).sum(1) / np.maximum(count - 1, 1))
        f[f'basket_value_mean{width}'] = np.nanmean(amounts, axis=1)
        count = np.isfinite(amounts).sum(1)
        centered = np.where(np.isfinite(amounts), amounts - np.nanmean(amounts, axis=1)[:, None], 0)
        f[f'basket_value_sd{width}'] = np.sqrt((centered ** 2).sum(1) / np.maximum(count - 1, 1))
        f[f'window{width}_observed_count'] = count
    f['amount_sum'] += f['last_amount']
    f['amount_mean'] = f['amount_sum'] / f['events_seen']
    f['relative_amount'] = f['last_amount'] / np.maximum(f['amount_mean'], 1e-8)
    f['tenure'] += prediction
    f['global_order_count'] = f['events_seen']
    f['global_mean_size'] = f['amount_mean']
    f['global_last_gap'] = prediction
    start = np.datetime64(data.manifest['calendar_boundaries'][0])
    days = data.samples.t.to_numpy()[rows] + elapsed
    dates = start + np.floor(days).astype('timedelta64[D]')
    dow = (dates.astype('datetime64[D]').astype('int64') + 3) % 7
    month = dates.astype('datetime64[M]').astype('int64') % 12 + 1
    f['sin_dow'] = np.sin(2 * np.pi * dow / 7)
    f['cos_dow'] = np.cos(2 * np.pi * dow / 7)
    f['sin_month'] = np.sin(2 * np.pi * month / 12)
    f['cos_month'] = np.cos(2 * np.pi * month / 12)
    updated = np.column_stack([f[name] for name in data.names]).astype('float32')
    columns = cfg['columns']
    x = (updated[:, columns] - data.mean[columns]) / data.sd[columns]
    x = np.column_stack([x, np.ones(len(x))])
    if cfg['K']:
        labels = data.cluster(cfg['K'], cfg['seed'])[data.profile_row[data.origins[rows]]]
        x = np.column_stack([x, np.eye(cfg['K'], dtype='float32')[labels]])
    return updated, gap_history, amount_history, x.astype('float32')


def predict(data, cfg, model, rows):
    output = np.empty((len(rows), 5))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 2048):
            batch = rows[start:start + 2048]
            x, g, valid, anchor = data.batch(batch, cfg)
            if getattr(model, 'recursive', False):
                indices, mask = data.indices(batch, 20)
                gaps = np.where(mask & (indices > data.starts[indices]), data.gaps[indices], np.nan)
                amounts = np.where(mask, data.x[indices, data.names.index('last_amount')], np.nan)
                raw = np.array(data.x[data.origins[batch]])
                elapsed = np.zeros(len(batch)); prediction = np.empty((len(batch), 5))
                for horizon in range(5):
                    pred = model.points(model(torch.as_tensor(x, device=DEVICE),
                        torch.as_tensor(g, device=DEVICE), torch.as_tensor(valid, device=DEVICE)),
                        torch.as_tensor(anchor, device=DEVICE)).cpu().numpy()[:, 0]
                    prediction[:, horizon] = pred
                    if horizon < 4:
                        elapsed += pred
                        raw, gaps, amounts, next_x = forecast_covariate_update(data, cfg, batch, raw, gaps, amounts, pred, elapsed)
                        x = np.concatenate([x[:, 1:], next_x[:, None]], axis=1)
                        valid = np.column_stack([valid[:, 1:], np.ones(len(batch), dtype=bool)])
            else:
                prediction = model.points(model(torch.as_tensor(x, device=DEVICE), torch.as_tensor(g, device=DEVICE),
                    torch.as_tensor(valid, device=DEVICE)), torch.as_tensor(anchor, device=DEVICE)).cpu().numpy()
            output[start:start + len(batch)] = prediction
    assert np.isfinite(output).all() and (output >= 0).all()
    return output


def fit(data, y, eligible, cfg, kind, destination):
    suffix = f"{kind}_{cfg['mode']}"
    path = destination / f'{suffix}.npz'
    if path.exists():
        return np.load(path)['prediction']
    train = np.flatnonzero(eligible & (data.phase == 0))
    early = np.flatnonzero(eligible & (data.phase == 1))
    rows = np.flatnonzero(eligible & (data.phase > 0))
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    anchor = data.naive(cfg['L'], 'median')
    z = np.log1p(y[train])
    if cfg['mode'] == 'residual':
        z -= np.log1p(anchor[train, None])
    center, spread = z.mean(0), np.maximum(.1, z.std(0))
    numeric = len(cfg['columns']) + 1 + cfg['K']
    if kind.startswith('transformer'):
        model = Transformer(numeric, cfg['L'], center, spread, kind.endswith('recursive'))
    else:
        model = DirectNetwork(numeric, {**cfg, 'kind': kind}, data.scale_y, center, spread)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=2)
    best = float('inf'); state = None; stale = 0; history = []
    rng = np.random.default_rng(cfg['seed']); started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); order = rng.permutation(train)
        for start in range(0, len(order), 4096):
            batch = order[start:start + 4096]
            x, g, valid, a = data.batch(batch, cfg)
            opt.zero_grad(set_to_none=True)
            raw = model(torch.as_tensor(x, device=DEVICE), torch.as_tensor(g, device=DEVICE), torch.as_tensor(valid, device=DEVICE))
            p = model.points(raw, torch.as_tensor(a, device=DEVICE))
            truth = y[batch, :1] if getattr(model, 'recursive', False) else y[batch]
            loss = (p - torch.as_tensor(truth, dtype=torch.float32, device=DEVICE)).abs().mean() / data.scale_y
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        p = predict(data, cfg, model, early)
        score = float(np.abs(p - y[early]).mean())
        scheduler.step(score); history.append({'epoch': epoch + 1, 'early_MAE': score})
        if score < best - 1e-6:
            best = score; state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(state)
    full = np.full_like(y, np.nan); full[rows] = predict(data, cfg, model, rows)
    np.savez_compressed(path, prediction=full)
    dump(destination / f'{suffix}.json', {'config': cfg, 'kind': kind, 'history': history,
        'training_seconds': time.perf_counter() - started, 'parameters': sum(p.numel() for p in model.parameters()),
        'train_origins': len(train), 'early_origins': len(early), 'predict_origins': len(rows),
        'recursive_covariates': 'held last known basket attributes; gap-dependent features updated from predictions only'})
    torch.save(state, destination / f'{suffix}.pt')
    del model, opt, state; gc.collect(); torch.cuda.empty_cache()
    return full


def run():
    args.output.mkdir(parents=True, exist_ok=True)
    data = Data(args.prepared / 'OnlineRetailII')
    keys = pd.read_parquet(data.folder / 'event_keys.parquet')
    origin = data.origins
    last = np.minimum(origin + 5, len(keys) - 1)
    eligible = (origin + 5 < len(keys)) & (data.starts[last] == data.starts[origin])
    eligible &= keys.t.to_numpy()[last] < data.samples.end.to_numpy()
    y = np.column_stack([data.gaps[np.minimum(origin + h, len(keys) - 1)] for h in range(1, 6)])
    dump(args.output / 'cohort.json',
        {'positive_customer_day_only': True, 'origins_by_phase': {str(p): int((eligible & (data.phase == p)).sum()) for p in range(5)},
         'target': 'five consecutive complete positive purchase-day gaps, not timestamp remainders',
         'selection': 'same origin eligibility and outcomes for all direct/recursive comparators'})
    assert all((eligible & (data.phase == p)).sum() > 0 for p in range(5)), 'Insufficient complete horizons in a frozen phase.'
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        destination = args.output / f'seed_{seed}'; destination.mkdir(exist_ok=True)
        if (destination / 'complete.json').exists():
            continue
        selected = json.loads((args.runs / 'OnlineRetailII' / f'seed_{seed}' / f'selection_s{seed}.json').read_text())
        cfg = selected['selected']
        select = np.flatnonzero(eligible & (data.phase == 2))
        stack = np.flatnonzero(eligible & (data.phase == 3))
        test = np.flatnonzero(eligible & (data.phase == 4))
        pool = {}
        for kind in ['encoder', 'lstm', 'gru']:
            choices = {}
            for mode in ['direct', 'residual']:
                choices[mode] = fit(data, y, eligible, {**cfg, 'mode': mode, 'pool': cfg['pool'] if kind == 'encoder' else 'last'}, kind, destination)
            choices['average'] = (choices['direct'] + choices['residual']) / 2
            choice = min(choices, key=lambda k: np.abs(choices[k][select] - y[select]).mean())
            pool[kind] = choices[choice]
        train = np.flatnonzero(eligible & (data.phase == 0))
        x, _, _, _ = data.batch(np.r_[train, select, stack, test], cfg)
        x = x.reshape(len(x), -1)
        indices = np.r_[train, select, stack, test]
        bp = np.full_like(y, np.nan)
        for h in range(5):
            model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200,
                learning_rate=.05, max_leaf_nodes=31, random_state=seed, early_stopping=False)
            with threadpool_limits(limits=4):
                model.fit(x[:len(train)], y[train, h]); bp[indices, h] = np.maximum(0, model.predict(x))
        pool['boosting'] = bp
        pool['window_mean'] = np.repeat(data.naive(cfg['L'])[:, None], 5, axis=1)
        final = np.full_like(y, np.nan); all_weights = []
        for h in range(5):
            matrix = np.column_stack([pool[k][:, h] for k in pool])
            w = simplex_mae(matrix[stack], y[stack, h]); final[:, h] = matrix @ w
            all_weights.append(dict(zip(pool, w.tolist())))
        pool['GTS'] = final
        pool['window_median'] = np.repeat(data.naive(cfg['L'], 'median')[:, None], 5, axis=1)
        for kind in ['transformer_direct', 'transformer_recursive']:
            pool[kind] = fit(data, y, eligible, {**cfg, 'mode': 'direct'}, kind, destination)
        frame = data.samples.loc[test, ['origin', 'customer']].copy()
        for h in range(5):
            frame[f'actual_h{h + 1}'] = y[test, h]
            for name, p in pool.items():
                frame[f'{name}_h{h + 1}'] = p[test, h]
        frame.to_parquet(destination / 'test_predictions.parquet', index=False)
        dump(destination / 'complete.json', {'seed': seed, 'test_origins': len(test), 'combiner_weights_by_horizon': all_weights,
            'runtime': 'one PyTorch CUDA implementation for all neural fits', 'direct_recursive_origins_match': True})
        log('Completed clean five-horizon seed', seed)
    dump(args.output / 'complete.json', {'seeds': args.seeds, 'clean_positive_purchase_days': True, 'horizons': 5})


if __name__ == '__main__':
    run()

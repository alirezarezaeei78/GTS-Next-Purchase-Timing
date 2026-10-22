"""One PyTorch implementation for the clean revision experiments.

All decisions use phases 0/1/2. Ensemble weights use phase 3 only. Phase 4
targets are never exposed to selection, fitting, or early stopping.
"""
from pathlib import Path
import copy, gc, hashlib, json, math, os, shutil, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import linprog
from scipy import sparse
from threadpoolctl import threadpool_limits
import joblib

VERSION = 'clean-v2.0.0'
torch.set_num_threads(4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = False
SEEDS = [42, 7, 1400, 2026, 31337]


def log(*values):
    print(time.strftime('%H:%M:%S'), *values, flush=True)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def mi(a, b, weight):
    size = 24
    table = np.bincount(a.astype('int64') * size + b, weights=weight,
                        minlength=size * size).reshape(size, size)
    table /= table.sum()
    independent = table.sum(1)[:, None] * table.sum(0)[None, :]
    good = table > 0
    return float((table[good] * np.log(table[good] / independent[good])).sum())


class Data:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        self.name = self.manifest['dataset']
        self.x = np.load(self.folder / 'x_events.npy', mmap_mode='r')
        self.names = json.loads((self.folder / 'feature_names.json').read_text())
        self.gaps = np.load(self.folder / 'gap_events.npy', mmap_mode='r')
        self.starts = np.load(self.folder / 'event_start.npy', mmap_mode='r')
        self.graph_row = np.load(self.folder / 'event_graph_row.npy', mmap_mode='r')
        self.profile_row = np.load(self.folder / 'event_profile_row.npy', mmap_mode='r')
        self.profiles = np.load(self.folder / 'profiles.npy')
        self.samples = pd.read_parquet(self.folder / 'samples.parquet')
        self.origins = self.samples.origin.to_numpy()
        self.y = self.samples.duration.to_numpy(dtype='float64')
        self.phase = self.samples.phase.to_numpy()
        self.observed = self.samples.observed.to_numpy()
        self.customers = self.samples.customer.to_numpy()
        self.train = np.flatnonzero((self.phase == 0) & self.observed)
        self.early = np.flatnonzero((self.phase == 1) & self.observed)
        self.select = np.flatnonzero((self.phase == 2) & self.observed)
        self.stack = np.flatnonzero((self.phase == 3) & self.observed)
        self.test = np.flatnonzero((self.phase == 4) & self.observed)
        self.predict_rows = np.flatnonzero(self.phase > 0)
        self.scale_y = max(1., float(np.median(self.y[self.train])))
        train_events = pd.read_parquet(self.folder / 'event_keys.parquet', columns=['phase']).phase.to_numpy() == 0
        # Fit once per event; overlapping windows do not multiply scaler weights.
        scaler = StandardScaler()
        for start in range(0, len(self.x), 100000):
            block = np.asarray(self.x[start:start + 100000])
            valid = train_events[start:start + len(block)]
            if valid.any():
                scaler.partial_fit(block[valid])
        self.mean = scaler.mean_.astype('float32')
        self.sd = np.maximum(scaler.scale_, 1e-6).astype('float32')
        self.graphs, self.clusters, self.naives = {}, {}, {}

    def graph(self, maximum=600):
        if maximum not in self.graphs:
            file = 'graph_random.npy' if maximum == 'random' else f'graph_P{maximum}.npy'
            self.graphs[maximum] = np.load(self.folder / file, mmap_mode='r')
        return self.graphs[maximum]

    def cluster(self, k, seed):
        if k == 0:
            return None
        key = (k, seed)
        if key not in self.clusters:
            profile_scaler = StandardScaler().fit(self.profiles)
            with threadpool_limits(limits=4):
                km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(profile_scaler.transform(self.profiles))
            self.clusters[key] = km.labels_.astype('int16')
        return self.clusters[key]

    def indices(self, sample_rows, length):
        origin = self.origins[sample_rows]
        ix = origin[:, None] - np.arange(length - 1, -1, -1)
        valid = ix >= self.starts[origin, None]
        return np.maximum(ix, self.starts[origin, None]), valid

    def batch(self, sample_rows, cfg):
        ix, valid = self.indices(sample_rows, cfg['L'])
        columns = cfg['columns']
        x = (np.asarray(self.x[ix][:, :, columns]) - self.mean[columns]) / self.sd[columns]
        # Explicit padding indicator; genuine zero-day purchases remain distinct.
        x *= valid[:, :, None]
        x = np.concatenate([x, valid[:, :, None]], axis=2)
        k = cfg.get('K', 0)
        if k:
            lab = self.cluster(k, cfg['seed'])[self.profile_row[ix]]
            x = np.concatenate([x, np.eye(k, dtype='float32')[lab] * valid[:, :, None]], axis=2)
        d = cfg.get('d', 16)
        if cfg.get('graph', 'spectral') == 'zero':
            g = np.zeros((*ix.shape, d), dtype='float32')
        else:
            maximum = 'random' if cfg.get('graph') == 'random' else cfg.get('P', 600)
            g = np.array(self.graph(maximum)[self.graph_row[ix], :d], dtype='float32')
            g *= valid[:, :, None]
        # Anchors and naive rules use only genuine observed gaps.
        available = valid & (ix > self.starts[ix])
        gap = np.where(available, self.gaps[ix], np.nan)
        anchor = np.nanmedian(gap, axis=1)
        return x.astype('float32'), g, valid, np.nan_to_num(anchor).astype('float32')

    def naive(self, length, kind='mean'):
        if (length, kind) in self.naives:
            return self.naives[(length, kind)]
        result = np.zeros(len(self.y))
        for start in range(0, len(result), 50000):
            rows = np.arange(start, min(start + 50000, len(result)))
            ix, valid = self.indices(rows, length)
            valid &= ix > self.starts[ix]
            gap = np.where(valid, self.gaps[ix], np.nan)
            result[rows] = np.nanmean(gap, 1) if kind == 'mean' else np.nanmedian(gap, 1)
        result = np.nan_to_num(result)
        self.naives[(length, kind)] = result
        return result

    def rank(self, destination):
        file = Path(destination) / 'feature_ranking.json'
        if file.exists():
            return json.loads(file.read_text())
        x = np.asarray(self.x[self.origins[self.train]], dtype='float32')
        z = np.log1p(self.y[self.train])
        _, inverse, count = np.unique(self.customers[self.train], return_inverse=True, return_counts=True)
        weight = 1 / count[inverse]
        def discretize(v):
            return np.searchsorted(np.unique(np.quantile(v, np.linspace(0, 1, 25)[1:-1])), v, side='right').astype('int8')
        bins = np.column_stack([discretize(x[:, c]) for c in range(x.shape[1])])
        target = discretize(z)
        keep, removed, hashes = [], {}, {}
        for c, name in enumerate(self.names):
            if np.ptp(x[:, c]) < 1e-7:
                removed[name] = 'constant in completed training origins'
                continue
            key = hashlib.sha256(x[:, c].tobytes()).hexdigest()
            if key in hashes:
                removed[name] = 'identical to ' + self.names[hashes[key]]
            else:
                hashes[key] = c; keep.append(c)
        relevance = {c: mi(bins[:, c], target, weight) for c in keep}
        selected, redundancy = [], {}
        while keep:
            scores = {}
            for c in keep:
                penalty = []
                for other in selected:
                    key = tuple(sorted([c, other]))
                    if key not in redundancy:
                        redundancy[key] = mi(bins[:, c], bins[:, other], weight)
                    penalty.append(redundancy[key])
                scores[c] = relevance[c] - (np.mean(penalty) if penalty else 0)
            best = max(keep, key=lambda c: (scores[c], -c))
            selected.append(best); keep.remove(best)
        result = {'indices': selected, 'names': [self.names[c] for c in selected],
                  'removed': removed, 'training_origins': len(self.train),
                  'target': 'uncapped log1p gap; MI is invariant to monotone transformations in the population',
                  'relevance': {self.names[c]: v for c, v in relevance.items()},
                  'weighting': 'equal total weight per original training customer',
                  'estimator': '24-quantile-bin mutual information minus mean selected-feature mutual information'}
        dump(file, result)
        return result


class Network(nn.Module):
    def __init__(self, numeric, cfg, scale_y, center, spread):
        super().__init__()
        self.cfg, self.scale_y, self.center, self.spread = cfg, scale_y, center, spread
        kind = cfg.get('kind', 'encoder')
        self.kind = kind
        self.use_graph = kind == 'encoder'
        self.attention = self.use_graph and cfg.get('attention', True)
        output = 2 if kind == 'deepar' else 1
        self.project = nn.Linear(numeric + (cfg.get('d', 16) if self.use_graph else 0), 32)
        if kind.startswith('mlp') or kind == 'weibull':
            self.hidden = nn.Linear(32, 8 if kind.startswith('mlp') else 32)
        else:
            self.rnn = (nn.GRU if kind in ['gru', 'deepar'] else nn.LSTM)(32, 64, batch_first=True)
            self.hidden = nn.Linear(64, 64)
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(self.hidden.out_features, output)
        nn.init.zeros_(self.output.bias)
        if self.attention:
            self.q = nn.Linear(64, 128); self.k = nn.Linear(64, 128); self.v = nn.Linear(64, 128)
            self.attention_output = nn.Linear(128, 64); self.norm = nn.LayerNorm(64)
        if kind == 'weibull':
            self.shape_raw = nn.Parameter(torch.tensor(0.))

    def forward(self, x, g, valid):
        if self.use_graph:
            x = torch.cat([x, g], dim=-1)
        h = self.dropout(torch.relu(self.project(x)))
        if hasattr(self, 'rnn'):
            h, _ = self.rnn(h)
            if self.attention:
                n, length, _ = h.shape
                q = self.q(h).view(n, length, 4, 32).transpose(1, 2)
                k = self.k(h).view(n, length, 4, 32).transpose(1, 2)
                v = self.v(h).view(n, length, 4, 32).transpose(1, 2)
                scores = q @ k.transpose(-2, -1) / math.sqrt(32)
                scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
                a = self.dropout(scores.softmax(dim=-1))
                h = self.norm(h + self.attention_output((a @ v).transpose(1, 2).reshape(n, length, 128)))
            if self.cfg.get('pool', 'mean' if self.use_graph else 'last') == 'mean':
                h = (h * valid[:, :, None]).sum(1) / valid.sum(1).clamp(min=1)[:, None]
            else:
                h = h[:, -1]
        else:
            h = h[:, -1]
        return self.output(torch.relu(self.hidden(self.dropout(h))))

    def distribution(self, raw):
        if self.kind == 'deepar':
            mu = self.center + self.spread * raw[:, 0]
            sigma = nn.functional.softplus(raw[:, 1]) + .05
            return mu, sigma
        scale = nn.functional.softplus(raw[:, 0]) * self.scale_y + 1e-3
        shape = nn.functional.softplus(self.shape_raw) + .1
        return scale, shape

    def point(self, raw, anchor):
        if self.kind == 'deepar':
            mu, sigma = self.distribution(raw)
            probability = .5 + .5 * torch.special.ndtr(-mu / sigma)
            z = mu + sigma * torch.special.ndtri(probability.clamp(1e-6, 1 - 1e-6))
            return torch.expm1(z.clamp(0, 20))
        if self.kind == 'weibull':
            scale, shape = self.distribution(raw)
            return torch.floor(scale * math.log(2) ** (1 / shape)).clamp(min=0)
        z = self.center + self.spread * raw[:, 0]
        if self.cfg.get('mode') == 'residual':
            z = z + torch.log1p(anchor)
        return torch.expm1(z.clamp(0, 20))

    def objective(self, raw, y, anchor, observed, cap=None):
        if self.kind == 'deepar':
            mu, sigma = self.distribution(raw)
            z = torch.log1p(y)
            normalizer = torch.special.log_ndtr(mu / sigma)
            density_nll = .5 * ((z - mu) / sigma) ** 2 + torch.log(sigma) + .5 * math.log(2 * math.pi) + z + normalizer
            survival_nll = -torch.special.log_ndtr((mu - z) / sigma) + normalizer
            return torch.where(observed, density_nll, survival_nll).mean()
        if self.kind == 'weibull':
            scale, shape = self.distribution(raw)
            # Avoid the undefined 0**shape gradient for true zero-day outcomes.
            h0 = torch.where(y > 0,
                torch.exp(shape * (torch.log(y.clamp(min=1e-8)) - torch.log(scale))),
                torch.zeros_like(y))
            h1 = torch.exp(shape * (torch.log1p(y) - torch.log(scale)))
            # Exact mass for whole-day outcomes, including real zero-day gaps.
            event_nll = h0 - torch.log((-torch.expm1(-(h1 - h0))).clamp(min=1e-12))
            return torch.where(observed, event_nll, h0).mean()
        if self.cfg.get('loss', 'day_mae') == 'log_pinball':
            truth = torch.log1p(y.clamp(max=cap) if cap is not None else y)
            pred = self.center + self.spread * raw[:, 0]
            if self.cfg.get('mode') == 'residual':
                pred = pred + torch.log1p(anchor)
            error = truth - pred
            tau = self.cfg.get('tau', .5)
            return torch.maximum(tau * error, (tau - 1) * error).mean()
        error = (self.point(raw, anchor) - y) / self.scale_y
        return (error ** 2).mean() if self.kind == 'mlp_mse' else error.abs().mean()


def predict(data, cfg, model, rows, batch_size=4096):
    result = np.zeros(len(rows))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            x, g, valid, anchor = data.batch(batch, cfg)
            output = model(torch.as_tensor(x, device=DEVICE), torch.as_tensor(g, device=DEVICE),
                           torch.as_tensor(valid, device=DEVICE))
            result[start:start + len(batch)] = model.point(output, torch.as_tensor(anchor, device=DEVICE)).cpu().numpy()
    assert np.isfinite(result).all() and (result >= 0).all()
    return result


def fit(data, cfg, folder, epochs=20, patience=4):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    config = {**cfg, 'epochs': epochs, 'patience': patience, 'version': VERSION,
              'dataset_manifest': hashlib.sha256((data.folder / 'manifest.json').read_bytes()).hexdigest()}
    key = identity(config); prefix = folder / key
    if prefix.with_suffix('.json').exists():
        info = json.loads(prefix.with_suffix('.json').read_text())
        with np.load(prefix.with_suffix('.npz')) as saved:
            return saved['prediction'].copy(), info
    torch.manual_seed(cfg['seed'])
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed']); torch.cuda.reset_peak_memory_stats()
    train = np.flatnonzero(data.phase == 0) if cfg['kind'] in ['deepar', 'weibull'] else data.train
    z = np.log1p(data.y[data.train])
    if cfg.get('mode') == 'residual':
        z = z - np.log1p(data.naive(cfg['L'], 'median')[data.train])
    center, spread = float(z.mean()), max(.1, float(z.std()))
    cap = float(np.quantile(data.y[data.train], cfg['cap'])) if cfg.get('cap') else None
    numeric = len(cfg['columns']) + 1 + cfg.get('K', 0)
    model = Network(numeric, cfg, data.scale_y, center, spread).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2)
    rng = np.random.default_rng(cfg['seed']); best = float('inf'); state = None; stale = 0
    history = []; started = time.perf_counter()
    for epoch in range(epochs):
        model.train(); order = rng.permutation(train); total = 0.
        batch_size = cfg.get('batch_size', 4096)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            x, g, valid, anchor = data.batch(batch, cfg)
            optimizer.zero_grad(set_to_none=True)
            output = model(torch.as_tensor(x, device=DEVICE), torch.as_tensor(g, device=DEVICE), torch.as_tensor(valid, device=DEVICE))
            loss = model.objective(output, torch.as_tensor(data.y[batch], dtype=torch.float32, device=DEVICE),
                torch.as_tensor(anchor, device=DEVICE), torch.as_tensor(data.observed[batch], device=DEVICE), cap)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite objective: {config}')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()) * len(batch)
        early = predict(data, cfg, model, data.early)
        score = float(np.abs(early - data.y[data.early]).mean())
        scheduler.step(score)
        history.append({'epoch': epoch + 1, 'training_loss': total / len(train), 'early_MAE': score,
                        'seconds': time.perf_counter() - started})
        if score < best - 1e-6:
            best = score; state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        log(data.name, cfg['seed'], cfg['kind'], key, 'epoch', epoch + 1, 'early MAE', round(score, 5), 'seconds', round(time.perf_counter() - started, 1))
        if stale >= patience:
            break
    model.load_state_dict(state)
    train_seconds = time.perf_counter() - started
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    before = time.perf_counter()
    prediction = predict(data, cfg, model, data.predict_rows)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - before
    full = np.full(len(data.y), np.nan); full[data.predict_rows] = prediction
    info = {'config': config, 'training_seconds': train_seconds, 'inference_seconds': inference_seconds,
            'inference_origins': len(data.predict_rows), 'parameters': sum(p.numel() for p in model.parameters()),
            'peak_cuda_bytes': int(torch.cuda.max_memory_allocated()) if DEVICE.type == 'cuda' else 0,
            'training_origins': len(train), 'history': history, 'best_early_MAE': best,
            'selection_MAE': float(np.abs(full[data.select] - data.y[data.select]).mean()),
            'torch': torch.__version__, 'device': str(DEVICE)}
    torch.save({'state_dict': state, 'config': config, 'center': center, 'spread': spread}, prefix.with_suffix('.pt'))
    np.savez_compressed(prefix.with_suffix('.npz'), prediction=full)
    dump(prefix.with_suffix('.json'), info)
    del model, optimizer, state
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return full, info


def boost(data, cfg, folder):
    key = identity({**cfg, 'kind': 'boost', 'version': VERSION})
    prefix = Path(folder) / key
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if prefix.with_suffix('.json').exists():
        return np.load(prefix.with_suffix('.npz'))['prediction'], json.loads(prefix.with_suffix('.json').read_text())
    # On-disk matrix bounds memory for the full sliding Instacart cohort.
    width = cfg['L'] * (len(cfg['columns']) + 1 + cfg.get('K', 0))
    fit_rows = np.r_[data.train, data.predict_rows]
    cache_root = Path(os.environ.get('GTS_CACHE_ROOT', str(prefix.parent)))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f'{data.name}_{key}.input.npy'
    required = len(fit_rows) * width * np.dtype('float64').itemsize
    if shutil.disk_usage(cache_root).free < required + 2 * 1024 ** 3:
        raise RuntimeError('Insufficient temporary-disk space. Set GTS_CACHE_ROOT to a drive with '
                           f'at least {required / 1024 ** 3 + 2:.1f} GiB free before resuming.')
    # sklearn's histogram binning validates float64 inputs. Store that dtype on
    # disk up front to avoid allocating a second full dense matrix in RAM.
    # Column-major storage makes binning each feature a sequential disk read.
    cache = np.lib.format.open_memmap(cache_path, mode='w+', dtype='float64',
                                     shape=(len(fit_rows), width), fortran_order=True)
    for start in range(0, len(fit_rows), 10000):
        x, _, _, _ = data.batch(fit_rows[start:start + 10000], cfg)
        cache[start:start + len(x)] = x.reshape(len(x), -1)
    started = time.perf_counter()
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200, max_leaf_nodes=31,
        min_samples_leaf=25, learning_rate=.05, l2_regularization=.1,
        early_stopping=False, random_state=cfg['seed'])
    with threadpool_limits(limits=4):
        model.fit(cache[:len(data.train)], data.y[data.train])
        training_seconds = time.perf_counter() - started
        before = time.perf_counter()
        values = np.maximum(0, model.predict(cache[len(data.train):]))
    full = np.full(len(data.y), np.nan); full[data.predict_rows] = values
    info = {'config': cfg, 'training_seconds': training_seconds, 'inference_seconds': time.perf_counter() - before,
            'inference_origins': len(data.predict_rows), 'training_origins': len(data.train),
            'selection_MAE': float(np.abs(full[data.select] - data.y[data.select]).mean())}
    joblib.dump(model, prefix.with_suffix('.joblib'))
    np.savez_compressed(prefix.with_suffix('.npz'), prediction=full); dump(prefix.with_suffix('.json'), info)
    del cache
    cache_path.unlink()
    return full, info


def simplex_mae(predictions, truth):
    """Global MAE optimum on the simplex, using sparse HiGHS linear programming."""
    p = np.asarray(predictions, dtype=float); y = np.asarray(truth, dtype=float)
    assert p.ndim == 2 and len(p) == len(y) and np.isfinite(p).all()
    n, members = p.shape
    eye = sparse.eye(n, format='csr')
    a = sparse.vstack([sparse.hstack([sparse.csr_matrix(p), -eye]),
                       sparse.hstack([-sparse.csr_matrix(p), -eye])], format='csr')
    equality = sparse.csr_matrix((np.ones(members), (np.zeros(members), np.arange(members))), shape=(1, members + n))
    result = linprog(np.r_[np.zeros(members), np.ones(n) / n], A_ub=a, b_ub=np.r_[y, -y],
        A_eq=equality, b_eq=[1.], bounds=[(0, None)] * (members + n), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.maximum(0, result.x[:members]); weights /= weights.sum()
    assert abs(weights.sum() - 1) < 1e-8
    return weights

"""Historical GTS implementation; default inputs remain fixed in this archive."""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as splinalg
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

try:
    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        Add, Concatenate, Dense, Dropout, GlobalAveragePooling1D, Input,
        LayerNormalization, LSTM, MultiHeadAttention,
    )
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    HAVE_TF = True
except Exception:  # pragma: no cover
    HAVE_TF = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if HAVE_TF:
    tf.random.set_seed(RANDOM_SEED)


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class Config:
    """All experiment knobs in one place so runs are reproducible.

    seq_len            basket events per input window (ablate 3/5/10 for paper)
    split_mode         'customer_holdout' (leave-last-two-out per customer) or
                       'temporal' (global 70/15/15 calendar split — stricter)
    fixed_horizons     business-defined τ in days for the horizon head
    horizon_quantiles  TRAIN-target quantiles added as adaptive horizons
    graph_emb_dim      spectral embedding dimension d (ablate 8/16/32)
    max_graph_products graph node cap = most frequent train products (memory)
    epochs/batch_size  training budget; EarlyStopping(patience) guards overfit
    max_rows           row cap when loading huge files (None = all)
    """
    seq_len: int = 5
    split_mode: str = "customer_holdout"
    fixed_horizons: Sequence[float] = (7, 30, 60, 90)
    horizon_quantiles: Sequence[float] = (0.5, 0.75, 0.9)
    graph_emb_dim: int = 16
    max_graph_products: int = 600
    top_k_dynamic: int = 5
    epochs: int = 40
    batch_size: int = 128
    patience: int = 6
    max_rows: Optional[int] = None
    min_events_per_customer: int = 5
    make_plots: bool = True
    # --- dynamic-threshold system (all thresholds derived from TRAIN only) --
    # target_cap_quantile: winsorize TRAINING targets at this train-quantile so
    # extreme outlier gaps cannot dominate the loss. Test targets are NEVER
    # capped — evaluation stays honest. None disables capping.
    target_cap_quantile: Optional[float] = 0.99
    # eval_threshold_quantiles: the τ ladder for threshold-stratified test
    # metrics. Being quantiles of each dataset's own TRAIN gap distribution,
    # the ladder adapts automatically to the rhythm of any database
    # (days-scale fast food vs months-scale retail).
    eval_threshold_quantiles: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99)
    # --- second event target: next-purchase amount (when + how much) --------
    predict_amount: bool = True
    # --- feature-set switch for the literature comparison -------------------
    # "ours" = gap-dynamics families + train-only top-k selection (default);
    # "lit_rfm" = time-varying RFM / multi-window activity features, the
    # dominant recipe in the dynamic-feature-engineering literature.
    feature_set: str = "ours"
    # --- v2 improvements -----------------------------------------------------
    # residual_prior: the regression head predicts the CORRECTION to a strong
    # per-customer prior (the window mean gap) in log space:
    #     r = log1p(y) - log1p(prior),   y_hat = expm1(log1p(prior) + r_hat).
    # If the network learns r=0 it exactly reproduces the strongest naive
    # baseline; any learned signal improves on it. This grounds the model on
    # personal periodicity instead of fighting it.
    residual_prior: bool = True
    # Auxiliary-loss weights: the horizon/amount heads regularize the shared
    # representation but must not tax the primary timing head (the classic
    # multi-task trade-off observed in our ablations).
    horizon_loss_weight: float = 0.3
    amount_loss_weight: float = 0.3
    # --- customer clustering (one GLOBAL configuration) ----------------------
    # Customers are segmented by their TRAIN-split behaviour profile (gap
    # level/dispersion, frequency, monetary) with K-Means. The number of
    # clusters is a single GLOBAL constant — identical for every dataset —
    # selected once by validation MAE across the whole corpus
    # (see select_global_k.py), i.e. by what helps next-purchase-event
    # prediction, not by an unsupervised index. The cluster becomes a one-hot
    # model input, and test metrics are additionally reported PER CLUSTER.
    use_clusters: bool = True
    cluster_k: int = 4          # global; set from the select_global_k.py sweep
    # --- auxiliary event heads ------------------------------------------------
    # The paper's event is (when, how much): gap regression + amount head.
    # The within-horizon classification head remains implemented but is OFF:
    # its thresholds are dataset-specific and therefore not comparable across
    # datasets, which is out of the paper's scope.
    use_horizon_head: bool = False
    within_tolerance_quantile: float = 0.25   # kept for CSV compatibility
    # --- metric-aligned estimation (v3) --------------------------------------
    # The system is scored by MAE in days, and MAE is minimised by the
    # conditional MEDIAN. The defaults below reproduce the published pipeline
    # exactly (Huber on the log target, arithmetic-mean anchor), which estimates
    # a conditional MEAN-like statistic instead:
    #
    #   Huber(delta=1) on a *standardised* target is quadratic across the bulk
    #   of the data, so it is a mean estimator in log space; back-transforming
    #   exp(E[log y]) yields a geometric mean, which equals the median only when
    #   log y is symmetric. Measured log-skew is +0.66 (HyperMe) to -0.34
    #   (Dorsa), so the incumbent is systematically off the statistic it is
    #   scored on.
    #
    # timing_loss="pinball" trains the timing head, the plain RNN members and
    # the gradient-boosting member with the pinball loss at `pinball_tau`.
    # At tau=0.5 this is the L1 loss, whose minimiser is the conditional median;
    # because the median is equivariant under monotone transforms, expm1
    # back-transformation recovers the conditional median of y exactly, with no
    # distributional assumption and no smearing correction.
    #
    # anchor_stat="median" makes the personal-rhythm anchor (the base of the
    # residual head AND a stack member) the window median rather than the mean.
    timing_loss: str = "huber"        # "huber" (published) | "pinball"
    pinball_tau: float = 0.5
    anchor_stat: str = "mean"         # "mean" (published) | "median"


# ===========================================================================
# Split assignment (guard G2) and censoring accounting (guard G5)
# ===========================================================================
def add_gap_target(events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute Δt at basket grain and count right-censored final events.

    Input: one row per (customer, purchase day) with `event_date`.
    The last event of each customer has an unknown next gap — it is *counted*
    (for the paper's censoring statement and future survival head) and then
    excluded from the regression sample, never mislabelled.
    """
    events = events.sort_values(["customer_id", "event_date"]).copy()
    events["next_purchase_date"] = events.groupby("customer_id")["event_date"].shift(-1)
    events["days_to_next_purchase"] = (
        events["next_purchase_date"] - events["event_date"]
    ).dt.days
    # Second event attribute: the VALUE of the next basket. Together with the
    # gap this makes the target a purchase EVENT (when it happens + how large
    # it is), not just a duration.
    events["next_purchase_amount"] = events.groupby("customer_id")["sale_price_total"].shift(-1)

    censored = events["next_purchase_date"].isna()
    stats = {
        "total_purchase_events": int(len(events)),
        "right_censored_events": int(censored.sum()),
        "right_censored_rate": float(censored.mean()) if len(events) else float("nan"),
        "modelable_events": int(((~censored) & (events["days_to_next_purchase"] > 0)).sum()),
    }
    return events[events["days_to_next_purchase"] > 0].copy(), stats


def assign_event_split(events: pd.DataFrame, split_mode: str) -> pd.Series:
    """One split decision for everything downstream (guard G2)."""
    events = events.sort_values(["customer_id", "event_date"])
    split = pd.Series("train", index=events.index, dtype=object)
    if split_mode == "temporal":
        order = events["event_date"].rank(method="first")
        q = order / order.max()
        split[q > 0.70] = "val"
        split[q > 0.85] = "test"
        return split
    if split_mode != "customer_holdout":
        raise ValueError(f"Unknown split_mode: {split_mode}")
    for _, idx in events.groupby("customer_id", sort=False).groups.items():
        pos = list(idx)
        split.loc[pos[-1]] = "test"
        if len(pos) >= 3:
            split.loc[pos[-2]] = "val"
    return split


# ===========================================================================
# Dynamic causal features + train-only correlation selection
# ===========================================================================
def engineer_features(events: pd.DataFrame, cfg: Config
                      ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Build the dynamic feature families (all pre-outcome, guard G1/G2).

    Families: gap lags Gap{2..5}Last, rolling AvgGap/GapStd and
    AvgSpend/SpendStd over windows {3,5,10,20}, expanding AvgGap/GapStd,
    monetary (CumSpend, AvgSpend, RelativeLastAmount), tenure, basket size,
    and cyclic calendar encodings. From the *candidate* rolling/lag features,
    the top-k by |corr(feature, target)| measured ON TRAIN ONLY are kept —
    dataset-adaptive selection without target leakage.
    """
    df = events.sort_values(["customer_id", "event_date"]).copy()
    g = df.groupby("customer_id", sort=False)

    gap = g["event_date"].diff().dt.days.fillna(0).clip(lower=0)
    df["Days_Since_Last_Purchase"] = gap
    df["Transaction_Count"] = g.cumcount() + 1

    gap_g = gap.groupby(df["customer_id"], sort=False)
    candidates: List[str] = []
    for lag in (2, 3, 4, 5):
        col = f"Gap{lag}Last"
        df[col] = gap_g.shift(lag - 1).fillna(0)
        candidates.append(col)
    spend_g = df.groupby("customer_id", sort=False)["sale_price_total"]
    for w in (3, 5, 10, 20):
        df[f"AvgGap_w{w}"] = gap_g.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
        df[f"GapStd_w{w}"] = gap_g.transform(lambda x, w=w: x.rolling(w, min_periods=1).std()).fillna(0)
        df[f"AvgSpend_w{w}"] = spend_g.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
        df[f"SpendStd_w{w}"] = spend_g.transform(lambda x, w=w: x.rolling(w, min_periods=1).std()).fillna(0)
        candidates += [f"AvgGap_w{w}", f"GapStd_w{w}", f"AvgSpend_w{w}", f"SpendStd_w{w}"]

    df["AvgGap"] = gap_g.transform(lambda x: x.expanding().mean()).fillna(0)
    df["GapStd"] = gap_g.transform(lambda x: x.expanding().std()).fillna(0)
    df["CumSpend"] = spend_g.cumsum()
    df["AvgSpend"] = spend_g.transform(lambda x: x.expanding().mean())
    df["LastAmount"] = df["sale_price_total"]
    df["RelativeLastAmount"] = (
        df["LastAmount"] / df["AvgSpend"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0)
    df["CustomerTenure"] = (df["event_date"] - g["event_date"].transform("min")).dt.days

    # Calendar-window activity candidates (the useful part of the RFM-window
    # recipe, folded into OUR candidate pool): causal counts/spend over the
    # past w calendar days. The train-only selector decides, per dataset,
    # whether gap-dynamics or activity-window features carry the signal —
    # this is what makes the feature system adaptive rather than a fixed list.
    for w in (7, 30):
        cnt = np.zeros(len(df))
        spd = np.zeros(len(df))
        pos = 0
        for _, grp in df.groupby("customer_id", sort=False):
            dates = grp["event_date"].to_numpy(dtype="datetime64[D]")
            spend = grp["sale_price_total"].to_numpy(dtype=float)
            n = len(grp)
            left = np.searchsorted(dates, dates - np.timedelta64(w, "D"),
                                   side="left")
            prefix = np.concatenate([[0.0], np.cumsum(spend)])
            cnt[pos:pos + n] = np.arange(n) - left
            spd[pos:pos + n] = prefix[:-1] - prefix[left]
            pos += n
        df[f"Count_{w}d"] = cnt
        df[f"Spend_{w}d"] = spd
        candidates += [f"Count_{w}d", f"Spend_{w}d"]

    df["DayOfWeek"] = df["event_date"].dt.dayofweek.astype(int)
    df["Month"] = df["event_date"].dt.month.astype(int)
    df["sin_dow"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["Month"] / 12)

    # --- train-only correlation ranking (guard G2) -------------------------
    train_mask = df["split"] == "train"
    sub = df.loc[train_mask, candidates + ["days_to_next_purchase"]]
    scores: Dict[str, float] = {}
    for col in candidates:
        valid = sub[[col, "days_to_next_purchase"]].dropna()
        if len(valid) >= 10:
            c = valid[col].corr(valid["days_to_next_purchase"])
            if pd.notna(c):
                scores[col] = abs(float(c))
    selected = sorted(scores, key=scores.get, reverse=True)[: cfg.top_k_dynamic]

    base = ["Days_Since_Last_Purchase", "Transaction_Count", "AvgGap", "GapStd",
            "CumSpend", "AvgSpend", "LastAmount", "RelativeLastAmount",
            "CustomerTenure", "line_items", "unique_products_in_event",
            "quantity", "sin_dow", "cos_dow", "sin_month", "cos_month"]

    # Cluster one-hots (multi-cluster prediction): the behavioural segment of
    # the customer, decided from train events only, becomes a model input so
    # each cluster gets its own learned adjustment of the shared dynamics.
    cluster_cols: List[str] = []
    if "customer_cluster" in df.columns:
        k = int(df["customer_cluster"].max()) + 1
        for c in range(k):
            col = f"Cluster_{c}"
            df[col] = (df["customer_cluster"] == c).astype(float)
            cluster_cols.append(col)

    feature_cols = [c for c in dict.fromkeys(base + cluster_cols + selected)
                    if c in df.columns]
    return df, feature_cols, {"selected_dynamic": selected,
                              "n_features": len(feature_cols),
                              "n_clusters": len(cluster_cols) or 1}


def engineer_features_lit_rfm(events: pd.DataFrame, cfg: Config
                              ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Time-varying RFM / multi-window feature set — the literature recipe.

    This reproduces, faithfully and leakage-safely, the dominant dynamic
    feature-engineering pattern in recent e-commerce prediction work
    (see 'Dynamic Feature Engineering for E-Commerce Prediction' review:
    multi-scale rolling windows of counts and spend at 1/7/30 days, recency,
    monetary trajectories, cumulative RFM — as in Mena et al. 2024's
    time-varying RFM, Kim et al. 2024's RFM+graph, and the review's
    "1-day, 7-day, 30-day counts and spend" build guidance).

    It exists as the head-to-head competitor for OUR feature system. The
    conceptual difference we test: RFM trajectories describe *activity
    intensity* (how often / how much lately) — designed for conversion, churn,
    and intent classification — whereas next-purchase-EVENT timing is driven
    by *inter-purchase gap dynamics* (gap lags, rolling gap statistics),
    which RFM windows only encode indirectly through counts. Both sets are
    point-in-time correct at basket grain, so the comparison isolates feature
    *content*, not hygiene.
    """
    df = events.sort_values(["customer_id", "event_date"]).copy()
    g = df.groupby("customer_id", sort=False)

    # Recency: the one timing feature RFM keeps.
    gap = g["event_date"].diff().dt.days.fillna(0).clip(lower=0)
    df["Recency"] = gap
    df["Frequency_cum"] = g.cumcount() + 1
    df["Monetary_cum"] = g["sale_price_total"].cumsum()
    df["CustomerTenure"] = (df["event_date"] - g["event_date"].transform("min")).dt.days

    # Multi-window counts and spend over CALENDAR windows (1/7/30/90 days),
    # computed causally: each row sees only strictly-prior events.
    for w in (1, 7, 30, 90):
        cnt = np.zeros(len(df))
        spd = np.zeros(len(df))
        pos = 0
        for _, grp in df.groupby("customer_id", sort=False):
            dates = grp["event_date"].to_numpy(dtype="datetime64[D]")
            spend = grp["sale_price_total"].to_numpy(dtype=float)
            n = len(grp)
            left = np.searchsorted(dates, dates - np.timedelta64(w, "D"), side="left")
            cum_n = np.arange(n)                      # events strictly before k
            cum_s = np.concatenate([[0.0], np.cumsum(spend)[:-1]])
            cnt[pos:pos + n] = cum_n - left
            spd[pos:pos + n] = cum_s - np.concatenate([[0.0], np.cumsum(spend)])[left]
            pos += n
        df[f"Count_{w}d"] = cnt
        df[f"Spend_{w}d"] = spd

    df["AvgBasket_cum"] = (df["Monetary_cum"]
                           / df["Frequency_cum"].replace(0, np.nan)).fillna(0)
    df["Rate_cum"] = df["Frequency_cum"] / (df["CustomerTenure"] + 1.0)

    feature_cols = (["Recency", "Frequency_cum", "Monetary_cum",
                     "CustomerTenure", "AvgBasket_cum", "Rate_cum"]
                    + [f"Count_{w}d" for w in (1, 7, 30, 90)]
                    + [f"Spend_{w}d" for w in (1, 7, 30, 90)]
                    + ["line_items", "quantity"])
    feature_cols = [c for c in feature_cols if c in df.columns]
    # The comparison suite reads gap baselines through this alias.
    df["Days_Since_Last_Purchase"] = df["Recency"]
    return df, feature_cols, {"selected_dynamic": [],
                              "n_features": len(feature_cols),
                              "feature_set": "lit_rfm"}


def make_features(events: pd.DataFrame, cfg: Config):
    """Dispatch between our feature system and the literature RFM set."""
    if cfg.feature_set == "lit_rfm":
        return engineer_features_lit_rfm(events, cfg)
    return engineer_features(events, cfg)


# ===========================================================================
# Train-only co-purchase graph + spectral embedding (guard G2/G5)
# ===========================================================================
def build_graph_embeddings(events: pd.DataFrame, cfg: Config
                           ) -> Tuple[Dict[str, int], np.ndarray]:
    """Spectral embedding of the TRAIN co-purchase graph.

    A_ij = number of train baskets containing both product i and j (top-N most
    frequent train products only, for tractability). We embed the symmetrically
    normalized adjacency D^{-1/2}(A+I)D^{-1/2}; each product gets
    e_i = [sqrt(λ_1)v_1(i), ..., sqrt(λ_d)v_d(i)] and each basket is the mean
    of its member products' embeddings. Non-train products -> zero vector.
    """
    train_events = events[events["split"] == "train"]
    freq: Dict[str, int] = {}
    for prods in train_events["product_ids"]:
        for p in prods:
            freq[p] = freq.get(p, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[: cfg.max_graph_products]
    mapping = {p: i for i, p in enumerate(top)}
    n, dim = len(mapping), cfg.graph_emb_dim
    if n < 3:
        return mapping, np.zeros((max(n, 1), dim), dtype=np.float32)

    adj = np.zeros((n, n), dtype=float)
    for prods in train_events["product_ids"]:
        idxs = [mapping[p] for p in prods if p in mapping]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                adj[idxs[a], idxs[b]] += 1.0
                adj[idxs[b], idxs[a]] += 1.0
    np.fill_diagonal(adj, adj.diagonal() + 1.0)

    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    a_norm = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    k = min(dim, n - 1)
    if HAVE_SCIPY:
        vals, vecs = splinalg.eigsh(sp.csr_matrix(a_norm), k=k + 1, which="LA")
    else:
        vals, vecs = np.linalg.eigh(a_norm)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order][1: k + 1], vecs[:, order][:, 1: k + 1]
    emb = vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    return mapping, emb.astype(np.float32)


def basket_graph_vector(prods, mapping, emb) -> np.ndarray:
    idxs = [mapping[p] for p in prods if p in mapping]
    return emb[idxs].mean(axis=0) if idxs else np.zeros(emb.shape[1], np.float32)


# ===========================================================================
# Sequences
# ===========================================================================
def make_sequences(df, feature_cols, cfg, mapping, emb):
    """Length-L windows per customer; each window predicts the gap (and next
    basket amount) after its LAST event, and inherits that event's split label
    (no window straddles the split boundary in a way that leaks a test target
    into training). Returns (x_num, x_graph, y_gap, y_amount, meta)."""
    x_num, x_graph, y, y_amt, rows = [], [], [], [], []
    L = cfg.seq_len
    has_amt = "next_purchase_amount" in df.columns
    for cid, grp in df.groupby("customer_id", sort=False):
        grp = grp.sort_values("event_date")
        if len(grp) < L:
            continue
        vals = grp[feature_cols].to_numpy(float)
        targets = grp["days_to_next_purchase"].to_numpy(float)
        amounts = (grp["next_purchase_amount"].to_numpy(float)
                   if has_amt else np.zeros(len(grp)))
        splits = grp["split"].to_numpy()
        gvecs = np.stack([basket_graph_vector(p, mapping, emb)
                          for p in grp["product_ids"]])
        dates = grp["event_date"].to_numpy()
        for end in range(L - 1, len(grp)):
            s = end - L + 1
            x_num.append(vals[s: end + 1])
            x_graph.append(gvecs[s: end + 1])
            y.append(float(targets[end]))
            y_amt.append(float(amounts[end]) if np.isfinite(amounts[end]) else 0.0)
            rows.append({"customer_id": cid, "split": str(splits[end]),
                         "target_event_date": str(pd.Timestamp(dates[end]).date())})
    if not x_num:
        raise ValueError("No sequences could be built; check seq_len / filters.")
    return (np.asarray(x_num, float), np.asarray(x_graph, float),
            np.asarray(y, float), np.asarray(y_amt, float), pd.DataFrame(rows))


# ===========================================================================
# Model
# ===========================================================================
def timing_loss_fn(cfg: "Config"):
    """The loss for every head that is ultimately scored by MAE in days.

    "huber"   reproduces the published pipeline (a mean estimator in log space).
    "pinball" targets the conditional quantile at cfg.pinball_tau; at tau=0.5
              this is the L1 loss and its minimiser is the conditional median,
              which is the MAE-optimal predictor. See Config for why this
              matters here.
    """
    if cfg.timing_loss == "huber":
        return Huber(delta=1.0)
    if cfg.timing_loss == "pinball":
        tau = float(cfg.pinball_tau)

        def pinball(y_true, y_pred):
            e = y_true - y_pred
            return tf.reduce_mean(tf.maximum(tau * e, (tau - 1.0) * e))
        pinball.__name__ = f"pinball_{tau:.2f}"
        return pinball
    raise ValueError(f"Unknown timing_loss: {cfg.timing_loss}")


def window_anchor(win: np.ndarray, cfg: "Config") -> np.ndarray:
    """The personal-rhythm anchor over a window of past gaps.

    The published pipeline uses the arithmetic mean. Under MAE the median is the
    optimal point summary, and the measured mean/median ratio of the gap
    distribution runs 1.5x-3.3x across this corpus, so the choice is not
    cosmetic. Selected on validation, never on test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stat = np.nanmedian(win, axis=1) if cfg.anchor_stat == "median" \
            else np.nanmean(win, axis=1)
    return np.nan_to_num(np.clip(stat, 0, None))


def build_plain_rnn(seq_len: int, n_numeric: int, cell: str = "lstm",
                    cfg: "Config" = None):
    """A plain single-head recurrent regressor (ensemble member / baseline)."""
    from tensorflow.keras.layers import GRU as _GRU
    num_in = Input(shape=(seq_len, n_numeric), name="numeric_input")
    rnn = (LSTM if cell == "lstm" else _GRU)(
        64, dropout=0.2, kernel_regularizer=regularizers.l2(1e-5))(num_in)
    out = Dense(1, name="reg")(Dense(64, activation="relu")(rnn))
    model = Model(num_in, out)
    model.compile(optimizer=Adam(1e-3),
                  loss=timing_loss_fn(cfg if cfg is not None else Config()))
    return model


def stack_on_validation(test_cols: List[np.ndarray], val_cols: List[np.ndarray],
                        y_val: np.ndarray,
                        groups_val: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Validation-selected stack; returns (test_prediction, convex weights).

    Delegates the combiner choice to `stacking.select_meta_learner` so that this
    entry point and `run_comparison.py` cannot diverge. They did diverge once:
    `run_comparison` was upgraded to a validation-selected combiner while this
    path kept a convex-only blend, and the two produced different numbers for
    the same model on the same data — which then reached two different tables of
    the manuscript.

    They diverged a SECOND time, more quietly. `run_comparison` passes
    `groups=customer_id` so the inner selection split keeps all of one
    customer's rows on the same side; this path passed no groups, so a
    customer's rows could appear on both sides of the split that CHOOSES the
    meta-learner. That inflates the apparent score of the high-capacity
    combiners and made `run_events` optimistic relative to the published
    `run_comparison` numbers (Bergerland 4.85 here vs 5.20 published, with an
    unconstrained meta-learner winning selection). `groups_val` closes it.

    The convex weights are still computed and returned regardless of which
    combiner wins, because callers use them as a diagnostic of how much
    exploitable structure a dataset holds.
    """
    try:
        import stacking
        Tm = np.column_stack([np.clip(c, 0, None) for c in test_cols])
        Vm = np.column_stack([np.clip(c, 0, None) for c in val_cols])
        pred, _kind, _scores = stacking.select_meta_learner(
            Vm, y_val, Tm, groups=groups_val)
        # Record the winner's INNER-validation MAE. The stack weights are fitted
        # on validation, so validation MAE of the final blend is optimistic and
        # cannot be used to choose between pipeline variants; the inner split
        # inside select_meta_learner is the one honest held-out number available
        # here, and arm selection (e.g. the estimand study) must use it.
        stack_on_validation.last_diagnostics = {
            "meta_learner": _kind,
            "inner_val_mae": (float(_scores[_kind]) if _kind in _scores
                              else None),
            "candidate_inner_val_mae": {k: float(v) for k, v in _scores.items()},
        }
        return pred, stacking.fit_convex_weights(Vm, y_val)
    except Exception:
        pass          # fall through to the self-contained implementation below

    Tm = np.column_stack([np.clip(c, 0, None) for c in test_cols])
    Vm = np.column_stack([np.clip(c, 0, None) for c in val_cols])
    try:
        from scipy.optimize import nnls
        w, _ = nnls(Vm, y_val)
    except Exception:
        w = np.ones(Vm.shape[1])
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(Vm.shape[1])
    w = w / w.sum()

    def _vmae(wt):
        return float(np.mean(np.abs(Vm @ wt - y_val)))
    cur = _vmae(w)
    for _ in range(300):
        improved = False
        for i in range(len(w)):
            for delta in (0.10, -0.10, 0.04, -0.04):
                w2 = w.copy()
                w2[i] = max(0.0, w2[i] + delta)
                if w2.sum() <= 0:
                    continue
                w2 = w2 / w2.sum()
                m2 = _vmae(w2)
                if m2 < cur - 1e-9:
                    w, cur, improved = w2, m2, True
        if not improved:
            break
    return Tm @ w, w


def build_model(seq_len: int, n_numeric: int, graph_dim: int, n_horizons: int,
                predict_amount: bool = False,
                horizon_loss_weight: float = 0.3,
                amount_loss_weight: float = 0.3,
                cfg: "Config" = None):
    """Dense projection -> LSTM -> MultiHeadAttention (+residual+LayerNorm)
    -> mean pooling -> shared Dense -> heads:
       reg     log-gap Huber regression (WHEN the next purchase happens)
       horizon per-τ sigmoid classification (WITHIN τ days?)
       amount  log-spend Huber regression (HOW LARGE the next basket is),
               optional — together the heads describe the next purchase EVENT."""
    num_in = Input(shape=(seq_len, n_numeric), name="numeric_input")
    graph_in = Input(shape=(seq_len, graph_dim), name="graph_input")
    x = Concatenate()([num_in, graph_in])
    x = Dropout(0.2)(Dense(32, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-5))(x))
    lstm = LSTM(64, return_sequences=True, dropout=0.2,
                kernel_regularizer=regularizers.l2(1e-5))(x)
    attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(lstm, lstm)
    h = LayerNormalization(epsilon=1e-6)(Add()([lstm, attn]))
    pooled = Dropout(0.2)(GlobalAveragePooling1D()(h))
    shared = Dense(64, activation="relu",
                   kernel_regularizer=regularizers.l2(1e-5))(pooled)
    cfg = cfg if cfg is not None else Config()
    outputs = [Dense(1, name="reg")(shared)]
    losses: Dict[str, Any] = {"reg": timing_loss_fn(cfg)}
    loss_weights: Dict[str, float] = {"reg": 1.0}
    if n_horizons > 0:
        outputs.append(Dense(n_horizons, activation="sigmoid", name="horizon")(shared))
        losses["horizon"] = "binary_crossentropy"
        loss_weights["horizon"] = horizon_loss_weight
    if predict_amount:
        outputs.append(Dense(1, name="amount")(shared))
        losses["amount"] = Huber(delta=1.0)
        loss_weights["amount"] = amount_loss_weight
    model = Model([num_in, graph_in], outputs)
    model.compile(optimizer=Adam(1e-3), loss=losses, loss_weights=loss_weights)
    return model


# ===========================================================================
# Metrics
# ===========================================================================
def resolve_horizons(y_train: np.ndarray, cfg: Config) -> List[float]:
    thr = [float(h) for h in cfg.fixed_horizons if h and h > 0]
    thr += [float(np.quantile(y_train, q)) for q in cfg.horizon_quantiles if 0 < q < 1]
    return sorted({round(t, 4) for t in thr if t > 0})


def regression_metrics(y_true, y_pred, tol_days: Optional[float] = None
                       ) -> Dict[str, float]:
    """Real-day regression metrics.

    `tol_days` is the DYNAMIC hit-rate tolerance (a train-gap quantile chosen
    per dataset, see Config.within_tolerance_quantile) — a fixed tolerance
    like 7 days would be trivial on slow datasets and impossible on fast ones.
    """
    p = np.clip(np.asarray(y_pred, float), 0, None)
    t = np.asarray(y_true, float)
    if len(t) == 0:
        return {"n": 0}
    out = {
        "n": int(len(t)),
        "mae_days": float(mean_absolute_error(t, p)),
        "rmse_days": float(math.sqrt(mean_squared_error(t, p))),
        "median_ae_days": float(np.median(np.abs(p - t))),
        "rmsle": float(math.sqrt(mean_squared_error(np.log1p(t), np.log1p(p)))),
        "within_7_days": float(np.mean(np.abs(p - t) <= 7)),
        "bias_days": float(np.mean(p - t)),
    }
    if tol_days is not None:
        out["tolerance_days"] = float(tol_days)
        out["within_tolerance"] = float(np.mean(np.abs(p - t) <= tol_days))
    return out


def assign_customer_clusters(events: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, int]:
    """Multi-cluster segmentation of customers from TRAIN behaviour only.

    Profile per customer (train-split events only, guard G2): log mean gap,
    log gap dispersion, log event count, log mean basket value — the four
    axes along which purchase processes differ most. K-Means++ (scipy
    implementation — sklearn's KMeans trips a threadpoolctl/OpenMP conflict
    on Windows when TensorFlow is loaded) with the GLOBAL cluster count
    cfg.cluster_k, identical for every dataset (selected once on validation
    MAE across the corpus; see select_global_k.py). Customers without train
    events (possible under the temporal split) fall back to the largest
    cluster. Returns (cluster id per events-row, k).
    """
    from scipy.cluster.vq import kmeans2

    train_ev = events[events["split"] == "train"]
    g = train_ev.sort_values(["customer_id", "event_date"]).groupby("customer_id")
    gaps = g["event_date"].diff().dt.days
    prof = pd.DataFrame({
        "mean_gap": np.log1p(gaps.groupby(train_ev["customer_id"]).mean()),
        "std_gap": np.log1p(gaps.groupby(train_ev["customer_id"]).std()),
        "n_events": np.log1p(g.size()),
        "mean_spend": np.log1p(g["sale_price_total"].mean().clip(lower=0)),
    }).fillna(0.0)

    k = int(cfg.cluster_k)
    if len(prof) < 5 * k or not cfg.use_clusters or k < 2:
        return pd.Series(0, index=events.index), 1

    x = prof.to_numpy(float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    _, labels_all = kmeans2(x, k, minit="++", seed=RANDOM_SEED)
    if len(np.unique(labels_all)) < 2:
        return pd.Series(0, index=events.index), 1

    labels = pd.Series(labels_all, index=prof.index)
    default = int(labels.mode().iloc[0])
    mapped = events["customer_id"].map(labels).fillna(default).astype(int)
    return mapped, k


def stratified_threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_train: np.ndarray,
                                 quantiles: Sequence[float]
                                 ) -> List[Dict[str, float]]:
    """Dynamic-threshold evaluation: metrics inside train-quantile strata.

    For each τ_q = Q_q(train gaps) we report metrics on the test subset with
    actual gap ≤ τ_q, plus its coverage (share of test inside the stratum),
    plus one open 'beyond-τmax' stratum for the outlier tail. Because the τ
    ladder comes from each dataset's own TRAIN distribution, the same code
    yields day-scale thresholds on fast-moving data and month-scale thresholds
    on slow data — the system adapts to the nature of the database with no
    manual tuning, and no test information is used to place the thresholds.

    This replaces the (biased) old idea of *training* on threshold-filtered
    subsets: here every model trains on everything and thresholds are used
    only to *report* where errors live — including how much the outlier tail
    weakens each model.
    """
    rows: List[Dict[str, float]] = []
    p = np.clip(np.asarray(y_pred, float), 0, None)
    t = np.asarray(y_true, float)
    taus = sorted({round(float(np.quantile(y_train, q)), 4)
                   for q in quantiles if 0 < q <= 1})
    for q, tau in zip(sorted(q for q in quantiles if 0 < q <= 1), taus):
        sel = t <= tau
        row = {"train_quantile": float(q), "tau_days": float(tau),
               "coverage": float(sel.mean())}
        row.update({k: v for k, v in regression_metrics(t[sel], p[sel]).items()
                    if k in ("n", "mae_days", "rmse_days", "median_ae_days",
                             "within_7_days")})
        rows.append(row)
    beyond = t > taus[-1] if taus else np.zeros(len(t), bool)
    if beyond.any():
        row = {"train_quantile": None, "tau_days": float(taus[-1]),
               "coverage": float(beyond.mean()), "stratum": "beyond_tau_max"}
        row.update({k: v for k, v in
                    regression_metrics(t[beyond], p[beyond]).items()
                    if k in ("n", "mae_days", "rmse_days", "median_ae_days")})
        rows.append(row)
    return rows


def classification_metrics(z, p) -> Dict[str, float]:
    z = np.asarray(z).astype(int)
    p = np.clip(np.asarray(p, float), 0, 1)
    both = len(z) and z.min() != z.max()
    return {
        "positive_rate": float(z.mean()) if len(z) else float("nan"),
        "f1": float(f1_score(z, (p >= .5).astype(int), zero_division=0)) if len(z) else float("nan"),
        "roc_auc": float(roc_auc_score(z, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(z, p)) if both else float("nan"),
        "brier": float(brier_score_loss(z, p)) if len(z) else float("nan"),
    }


# ===========================================================================
# Orchestration
# ===========================================================================
def run_events(events: pd.DataFrame, cfg: Config, output_dir: Path,
               dataset_name: str) -> Dict[str, Any]:
    """Run the full algorithm on prepared basket events and save everything.

    `events` must have: customer_id, event_date (datetime), sale_price_total,
    quantity, line_items, unique_products_in_event, product_ids (tuple).
    Adapters in npe_data.py produce exactly this schema for every dataset —
    that is what lets ONE algorithm run unchanged on all of them.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events, censor_stats = add_gap_target(events)
    counts = events.groupby("customer_id").size()
    keep = counts[counts >= max(cfg.seq_len, cfg.min_events_per_customer)].index
    events = events[events["customer_id"].isin(keep)].copy()
    if events.empty:
        raise ValueError(f"{dataset_name}: no customers with enough events.")

    events["split"] = assign_event_split(events, cfg.split_mode)
    events["customer_cluster"], n_clusters = assign_customer_clusters(events, cfg)
    featured, feature_cols, feat_info = make_features(events, cfg)
    mapping, emb = build_graph_embeddings(featured, cfg)
    x_num, x_graph, y, y_amt, meta = make_sequences(featured, feature_cols, cfg,
                                                    mapping, emb)

    tr = (meta["split"] == "train").to_numpy()
    va = (meta["split"] == "val").to_numpy()
    te = (meta["split"] == "test").to_numpy()
    if tr.sum() < 50 or te.sum() < 20:
        raise ValueError(f"{dataset_name}: split too small "
                         f"(train={tr.sum()}, test={te.sum()}).")

    scaler_x = StandardScaler().fit(x_num[tr].reshape(-1, x_num.shape[-1]))
    x_num_s = scaler_x.transform(x_num.reshape(-1, x_num.shape[-1])).reshape(x_num.shape)

    # Dynamic-threshold outlier guard: winsorize the *training* target at a
    # train-quantile cap so extreme gaps cannot dominate the loss. The cap is
    # dataset-adaptive (train-derived) and test targets are never touched.
    y_fit = y.copy()
    cap_days = None
    if cfg.target_cap_quantile:
        cap_days = float(np.quantile(y[tr], cfg.target_cap_quantile))
        y_fit = np.minimum(y, cap_days)

    # Residual-over-prior parameterization (v2): the head predicts the
    # log-space correction to the customer's window-mean gap, so the model is
    # anchored on personal periodicity and can only add signal on top of it.
    gap_idx0 = feature_cols.index("Days_Since_Last_Purchase")
    prior_days = window_anchor(x_num[:, :, gap_idx0], cfg)
    if cfg.residual_prior:
        target_log = np.log1p(y_fit) - np.log1p(prior_days)
    else:
        target_log = np.log1p(y_fit)
    scaler_y = StandardScaler().fit(target_log[tr].reshape(-1, 1))
    y_s = scaler_y.transform(target_log.reshape(-1, 1)).flatten()

    def invert_reg(pred_scaled: np.ndarray, idx: np.ndarray) -> np.ndarray:
        logv = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        if cfg.residual_prior:
            logv = logv + np.log1p(prior_days[idx])
        return np.expm1(logv).clip(min=0)

    # Amount target (second event attribute), same treatment: log1p + train
    # scaling + train-quantile winsorization.
    amt_fit = np.clip(y_amt, 0, None)
    if cfg.target_cap_quantile:
        amt_cap = float(np.quantile(amt_fit[tr], cfg.target_cap_quantile))
        amt_fit = np.minimum(amt_fit, max(amt_cap, 1e-9))
    amt_log = np.log1p(amt_fit)
    scaler_amt = StandardScaler().fit(amt_log[tr].reshape(-1, 1))
    amt_s = scaler_amt.transform(amt_log.reshape(-1, 1)).flatten()

    # Horizon head is opt-in (thresholds are dataset-specific and therefore
    # excluded from the cross-dataset study; the event = when + how much).
    horizons = resolve_horizons(y[tr], cfg) if cfg.use_horizon_head else []
    z = (np.stack([(y <= tau).astype(float) for tau in horizons], axis=1)
         if horizons else np.zeros((len(y), 0)))

    # Dynamic hit-rate tolerance: a train-gap quantile, so 'accurate enough'
    # means the same thing on daily-rhythm and monthly-rhythm datasets.
    tol_days = float(max(1.0, np.quantile(y[tr], cfg.within_tolerance_quantile)))

    # Naive baselines every reviewer asks about (real days).
    gap_idx = feature_cols.index("Days_Since_Last_Purchase")
    baselines = {
        "global_train_median": np.full(len(y), float(np.median(y[tr]))),
        "last_gap": x_num[:, -1, gap_idx].clip(min=0),
        "customer_mean_gap": np.nan_to_num(
            np.clip(np.nanmean(x_num[:, :, gap_idx], axis=1), 0, None)),
    }

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "split_mode": cfg.split_mode,
        "seq_len": cfg.seq_len,
        "feature_set": cfg.feature_set,
        "censoring_profile": censor_stats,
        "selected_dynamic_features": feat_info["selected_dynamic"],
        "n_numeric_features": len(feature_cols),
        "graph_products": len(mapping),
        "horizons_days": horizons,
        "target_cap_train_quantile": cfg.target_cap_quantile,
        "target_cap_days": cap_days,
        "residual_prior": cfg.residual_prior,
        "loss_weights": {"reg": 1.0, "horizon": cfg.horizon_loss_weight,
                         "amount": cfg.amount_loss_weight},
        "split_counts": {"train": int(tr.sum()), "val": int(va.sum()),
                         "test": int(te.sum())},
        "n_customer_clusters": int(n_clusters),
        "tolerance_days": tol_days,
        "baseline_test": {k: regression_metrics(y[te], v[te], tol_days=tol_days)
                          for k, v in baselines.items()},
    }

    history = None
    reg_days = hz_pred = amt_pred_units = None
    if HAVE_TF:
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        model = build_model(cfg.seq_len, x_num.shape[-1], emb.shape[-1],
                            len(horizons), predict_amount=cfg.predict_amount,
                            horizon_loss_weight=cfg.horizon_loss_weight,
                            amount_loss_weight=cfg.amount_loss_weight,
                            cfg=cfg)
        monitor = "val_loss" if va.sum() else "loss"
        fit_kwargs: Dict[str, Any] = dict(
            epochs=cfg.epochs, batch_size=cfg.batch_size, shuffle=True, verbose=0,
            callbacks=[
                EarlyStopping(monitor=monitor, patience=cfg.patience,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3,
                                  min_lr=1e-5),
            ])
        targets = {"reg": y_s[tr]}
        val_targets = {"reg": y_s[va]}
        if horizons:
            targets["horizon"] = z[tr]
            val_targets["horizon"] = z[va]
        if cfg.predict_amount:
            targets["amount"] = amt_s[tr]
            val_targets["amount"] = amt_s[va]
        if va.sum():
            fit_kwargs["validation_data"] = ([x_num_s[va], x_graph[va]], val_targets)
        history = model.fit([x_num_s[tr], x_graph[tr]], targets, **fit_kwargs)

        # Head output ordering: reg, [horizon], [amount] — resolve by config
        # so indexing is correct whether or not the horizon head is enabled.
        hz_idx = 1 if horizons else None
        amt_idx = (2 if horizons else 1) if cfg.predict_amount else None

        def _predict_reg_days(inp, idx):
            out = model.predict(inp, verbose=0)
            out = out if isinstance(out, list) else [out]
            return invert_reg(out[0].flatten(), idx), out

        deep_test, preds = _predict_reg_days([x_num_s[te], x_graph[te]],
                                             np.flatnonzero(te))
        hz_pred = preds[hz_idx] if hz_idx is not None else None

        # ---- self-configuring STACKED ensemble (the full system) -----------
        # Validation-weighted convex combination of complementary learners,
        # all on the same auto-engineered features (no per-dataset feature
        # engineering): deep graph-temporal encoder, gradient boosting, plain
        # LSTM and GRU, and the per-customer mean-gap prior. Weights fit on
        # VALIDATION only -> provably no worse than the best member there.
        reg_days = deep_test
        stack_weights = None
        if va.sum():
            from sklearn.ensemble import HistGradientBoostingRegressor as _GBM
            from tensorflow.keras.callbacks import EarlyStopping as _ES
            flat = x_num.reshape(len(x_num), -1)
            gbm_loss = (dict(loss="quantile", quantile=float(cfg.pinball_tau))
                        if cfg.timing_loss == "pinball"
                        else dict(loss="squared_error"))
            gbm = _GBM(**gbm_loss, learning_rate=0.05, max_iter=300,
                       max_leaf_nodes=31, min_samples_leaf=25,
                       l2_regularization=0.1, random_state=RANDOM_SEED)
            gbm.fit(flat[tr], np.log1p(y_fit[tr]))
            gbm_val = np.expm1(gbm.predict(flat[va])).clip(min=0)
            gbm_test = np.expm1(gbm.predict(flat[te])).clip(min=0)

            deep_val, _ = _predict_reg_days([x_num_s[va], x_graph[va]],
                                            np.flatnonzero(va))
            gap_prior_val = window_anchor(x_num[va][:, :, gap_idx0], cfg)
            gap_prior_test = window_anchor(x_num[te][:, :, gap_idx0], cfg)

            test_cols = [deep_test, gbm_test, gap_prior_test]
            val_cols = [deep_val, gbm_val, gap_prior_val]
            # Plain LSTM/GRU members on the direct log target.
            y_dir = np.log1p(y_fit)
            sc_dir = StandardScaler().fit(y_dir[tr].reshape(-1, 1))
            y_dir_s = sc_dir.transform(y_dir.reshape(-1, 1)).flatten()
            for cell in ("lstm", "gru"):
                mdl = build_plain_rnn(cfg.seq_len, x_num.shape[-1], cell, cfg)
                mdl.fit(x_num_s[tr], y_dir_s[tr], epochs=cfg.epochs,
                        batch_size=cfg.batch_size, shuffle=True, verbose=0,
                        validation_data=(x_num_s[va], y_dir_s[va]),
                        callbacks=[_ES(monitor="val_loss", patience=cfg.patience,
                                       restore_best_weights=True)])
                def _to_days(inp):
                    p = sc_dir.inverse_transform(
                        mdl.predict(inp, verbose=0).reshape(-1, 1)).flatten()
                    return np.expm1(p).clip(min=0)
                test_cols.append(_to_days(x_num_s[te]))
                val_cols.append(_to_days(x_num_s[va]))

            reg_days, stack_weights = stack_on_validation(
                test_cols, val_cols, y[va],
                groups_val=meta["customer_id"].to_numpy()[va])
        summary["stack_diagnostics"] = getattr(
            stack_on_validation, "last_diagnostics", None)
        summary["ensemble_stack_weights"] = (
            [float(w) for w in stack_weights] if stack_weights is not None
            else None)
        summary["model_test_regression"] = regression_metrics(
            y[te], reg_days, tol_days=tol_days)

        # Multi-cluster reporting: prediction quality for EACH behavioural
        # segment separately (its own MAE/tolerance hit-rate), so the paper
        # and the operator can see where the model is reliable per cluster.
        cluster_map = (featured.drop_duplicates("customer_id")
                       .set_index("customer_id")["customer_cluster"])
        seq_cluster = meta["customer_id"].map(cluster_map).fillna(0).astype(int)
        per_cluster = {}
        for c in sorted(seq_cluster.unique()):
            sel = te & (seq_cluster == c).to_numpy()
            if sel.sum() >= 10:
                # positions of sel within the test subset
                pos = seq_cluster[te].reset_index(drop=True) == c
                per_cluster[f"cluster_{c}"] = regression_metrics(
                    y[te][pos.to_numpy()], reg_days[pos.to_numpy()],
                    tol_days=tol_days)
        summary["per_cluster_test"] = per_cluster

        if horizons and hz_pred is not None:
            summary["model_test_horizons"] = {
                f"{tau:g}d": classification_metrics(z[te][:, m], hz_pred[:, m])
                for m, tau in enumerate(horizons)}
        best_base = min(v["mae_days"] for v in summary["baseline_test"].values())
        summary["test_mae_lift_vs_best_baseline"] = float(
            (best_base - summary["model_test_regression"]["mae_days"]) / best_base)

        # Second event attribute: next-basket amount, reported in the
        # dataset's own monetary units against a naive mean-basket baseline.
        if amt_idx is not None and len(preds) > amt_idx:
            amt_pred_units = np.expm1(scaler_amt.inverse_transform(
                preds[amt_idx].flatten().reshape(-1, 1)).flatten()).clip(min=0)
            amt_true = np.clip(y_amt[te], 0, None)
            amt_base = np.nan_to_num(
                np.clip(np.nanmean(x_num[te][:, :,
                        feature_cols.index("LastAmount")]
                        if "LastAmount" in feature_cols else
                        x_num[te][:, :, 0], axis=1), 0, None))
            summary["model_test_amount"] = {
                "mae_currency": float(mean_absolute_error(amt_true, amt_pred_units)),
                "rmse_currency": float(math.sqrt(mean_squared_error(
                    amt_true, amt_pred_units))),
                "baseline_customer_mean_basket_mae": float(
                    mean_absolute_error(amt_true, amt_base)),
            }

        # Dynamic-threshold stratified evaluation (train-quantile τ ladder).
        summary["threshold_stratified_test"] = stratified_threshold_metrics(
            y[te], reg_days, y[tr], cfg.eval_threshold_quantiles)
        model.save(output_dir / "model.keras")
    else:
        summary["model_test_regression"] = {
            "note": "TensorFlow unavailable — baselines only."}

    # Persist predictions for reproducibility and offline plotting.
    pred_df = meta.loc[te].copy()
    pred_df["actual_days"] = y[te]
    if reg_days is not None:
        pred_df["predicted_days"] = reg_days
        if horizons and hz_pred is not None:
            for m, tau in enumerate(horizons):
                pred_df[f"actual_within_{tau:g}d"] = z[te][:, m].astype(int)
                pred_df[f"prob_within_{tau:g}d"] = hz_pred[:, m]
    if amt_pred_units is not None:
        pred_df["actual_next_amount"] = y_amt[te]
        pred_df["predicted_next_amount"] = amt_pred_units
    for k, v in baselines.items():
        pred_df[f"baseline_{k}"] = v[te]
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Feature evaluation: permutation importance of each numeric feature on
    # the test regression error (model-agnostic, works on the trained heads).
    perm_importance = None
    if HAVE_TF and reg_days is not None and te.sum() <= 20_000:
        rng = np.random.default_rng(RANDOM_SEED)
        base_mae = summary["model_test_regression"]["mae_days"]
        imp = {}
        for j, col in enumerate(feature_cols):
            xp = x_num_s[te].copy()
            xp[:, :, j] = rng.permutation(xp[:, :, j])
            p = model.predict([xp, x_graph[te]], verbose=0)
            pd_days = invert_reg(p[0].flatten(), np.flatnonzero(te))
            imp[col] = float(np.mean(np.abs(pd_days - y[te])) - base_mae)
        perm_importance = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
        pd.DataFrame({"feature": list(perm_importance),
                      "delta_mae_days": list(perm_importance.values())}
                     ).to_csv(output_dir / "feature_importance.csv", index=False)
        summary["top5_features_by_permutation"] = list(perm_importance)[:5]

    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    if cfg.make_plots:
        try:
            import npe_plots
            npe_plots.make_all_plots(
                output_dir=output_dir, dataset_name=dataset_name,
                y_all=y, split_test_mask=te, y_test=y[te],
                reg_pred=reg_days, horizons=horizons,
                z_test=z[te], hz_pred=hz_pred,
                baselines_test={k: v[te] for k, v in baselines.items()},
                history=history,
                perm_importance=perm_importance,
                threshold_rows=summary.get("threshold_stratified_test"))
            summary["plots"] = "saved to plots/"
        except Exception as exc:  # plots must never kill a benchmark run
            summary["plots_error"] = str(exc)
            (output_dir / "summary.json").write_text(
                json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    return summary


def json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, np.integer):
        return int(v)
    return v

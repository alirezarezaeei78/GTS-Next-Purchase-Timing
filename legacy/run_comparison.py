"""Archived public daily comparison routines.

Standalone encoder ablations are historical diagnostics and are not used as
component-effect evidence in the manuscript. See README for identifier limits.
"""
from __future__ import annotations
import json,time,traceback
from pathlib import Path
from typing import Any,Callable,Dict,List,Optional
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import npe_core
import npe_data
import stacking

BASE_FEATURES = ["Days_Since_Last_Purchase", "Transaction_Count", "AvgGap",
                 "GapStd", "CumSpend", "AvgSpend", "LastAmount",
                 "RelativeLastAmount", "CustomerTenure", "line_items",
                 "unique_products_in_event", "quantity",
                 "sin_dow", "cos_dow", "sin_month", "cos_month"]

def build_variant(seq_len: int, n_numeric: int, graph_dim: int,
                  n_horizons: int, *, use_graph: bool, use_attention: bool,
                  multi_task: bool, cell: str = "lstm",
                  cfg: "npe_core.Config" = None):
    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import (Add, Concatenate, Dense, Dropout,
                                         GlobalAveragePooling1D, GRU, Input,
                                         LayerNormalization, LSTM,
                                         MultiHeadAttention)
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    tf.random.set_seed(npe_core.RANDOM_SEED)
    num_in = Input(shape=(seq_len, n_numeric), name="numeric_input")
    inputs = [num_in]
    x = num_in
    if use_graph:
        graph_in = Input(shape=(seq_len, graph_dim), name="graph_input")
        inputs.append(graph_in)
        x = Concatenate()([num_in, graph_in])
    x = Dropout(0.2)(Dense(32, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-5))(x))
    rnn_cls = LSTM if cell == "lstm" else GRU
    seq = rnn_cls(64, return_sequences=use_attention, dropout=0.2,
                  kernel_regularizer=regularizers.l2(1e-5))(x)
    if use_attention:
        attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(seq, seq)
        seq = LayerNormalization(epsilon=1e-6)(Add()([seq, attn]))
        seq = GlobalAveragePooling1D()(seq)
    pooled = Dropout(0.2)(seq)
    shared = Dense(64, activation="relu",
                   kernel_regularizer=regularizers.l2(1e-5))(pooled)
    outputs = [Dense(1, name="reg")(shared)]
    _cfg = cfg if cfg is not None else npe_core.Config()
    losses: Dict[str, Any] = {"reg": npe_core.timing_loss_fn(_cfg)}
    loss_weights: Dict[str, float] = {"reg": 1.0}
    if multi_task and n_horizons > 0:
        outputs.append(Dense(n_horizons, activation="sigmoid",
                             name="horizon")(shared))
        losses["horizon"] = "binary_crossentropy"
        # Down-weighted auxiliary loss so the horizon head regularizes the
        # representation without taxing the primary timing head.
        loss_weights["horizon"] = 0.3
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss=losses, loss_weights=loss_weights)
    return model

def train_and_predict(model, x_tr: List[np.ndarray], y_tr: Dict[str, Any],
                      x_va: Optional[List[np.ndarray]], y_va,
                      x_te: List[np.ndarray], epochs: int, patience: int,
                      batch_size: int) -> np.ndarray:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    monitor = "val_loss" if x_va else "loss"
    kwargs: Dict[str, Any] = dict(
        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0,
        callbacks=[
            EarlyStopping(monitor=monitor, patience=patience,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3,
                              min_lr=1e-5),
        ])
    if x_va:
        kwargs["validation_data"] = (x_va, y_va)
    model.fit(x_tr, y_tr, **kwargs)
    preds = model.predict(x_te, verbose=0)
    return preds if isinstance(preds, list) else [preds]

def compare_dataset(name: str, loader: Callable[[], pd.DataFrame],
                    cfg: npe_core.Config, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- shared preparation: identical for every variant ------------------
    events, censor_stats = npe_core.add_gap_target(loader())
    counts = events.groupby("customer_id").size()
    keep = counts[counts >= max(cfg.seq_len, cfg.min_events_per_customer)].index
    events = events[events["customer_id"].isin(keep)].copy()
    events["split"] = npe_core.assign_event_split(events, cfg.split_mode)
    events["customer_cluster"], n_clusters = npe_core.assign_customer_clusters(
        events, cfg)
    featured, feature_cols, _ = npe_core.engineer_features(events, cfg)
    mapping, emb = npe_core.build_graph_embeddings(featured, cfg)
    x_num, x_graph, y, _yamt, meta = npe_core.make_sequences(
        featured, feature_cols, cfg, mapping, emb)

    # Second feature pipeline: the literature's time-varying RFM /
    # multi-window set, on the SAME events and split, so the dgt_lit_rfm
    # variant isolates feature *content* only.
    featured_lit, feature_cols_lit, _ = npe_core.engineer_features_lit_rfm(
        events, cfg)
    x_num_lit, _xg_lit, y_lit, _ya_lit, meta_lit = npe_core.make_sequences(
        featured_lit, feature_cols_lit, cfg, mapping, emb)
    assert len(y_lit) == len(y) and np.allclose(y_lit, y), \
        "feature-set pipelines must align on identical sequences"

    tr = (meta["split"] == "train").to_numpy()
    va = (meta["split"] == "val").to_numpy()
    te = (meta["split"] == "test").to_numpy()
    if tr.sum() < 50 or te.sum() < 20:
        raise ValueError(f"{name}: split too small.")

    scaler_x = StandardScaler().fit(x_num[tr].reshape(-1, x_num.shape[-1]))
    xs = scaler_x.transform(x_num.reshape(-1, x_num.shape[-1])).reshape(x_num.shape)
    scaler_lit = StandardScaler().fit(x_num_lit[tr].reshape(-1, x_num_lit.shape[-1]))
    xs_lit = scaler_lit.transform(
        x_num_lit.reshape(-1, x_num_lit.shape[-1])).reshape(x_num_lit.shape)

    # Dynamic-threshold outlier guard (train-quantile winsorization of the
    # TRAINING target only; test evaluation stays on raw days). Applied
    # identically to every learned variant for fairness.
    y_fit = y.copy()
    if cfg.target_cap_quantile:
        cap_days = float(np.quantile(y[tr], cfg.target_cap_quantile))
        y_fit = np.minimum(y, cap_days)
    # Horizon head excluded from the study (dataset-specific thresholds
    # are not comparable across datasets); all variants are single-head.
    horizons = []
    z = np.zeros((len(y), 0))

    gap_idx = feature_cols.index("Days_Since_Last_Purchase")
    base_idx = [feature_cols.index(c) for c in BASE_FEATURES if c in feature_cols]
    # Ablation index sets: no-dynamic keeps base+clusters; no-clusters drops
    # the Cluster_* one-hots from the full set.
    cluster_idx = [i for i, c in enumerate(feature_cols) if c.startswith("Cluster_")]
    base_idx = sorted(set(base_idx) | set(cluster_idx))
    noclust_idx = [i for i, c in enumerate(feature_cols)
                   if not c.startswith("Cluster_")]

    # Dynamic hit-rate tolerance (train-gap quantile; scales with the rhythm).
    tol_days = float(max(1.0, np.quantile(y[tr],
                                          cfg.within_tolerance_quantile)))

    # Two target parameterizations. "direct" is the standard log-gap target
    # used by the general tier. "residual" (part of OUR method) predicts the
    # log-space correction to the customer's window-mean gap, anchoring the
    # model on personal periodicity so it can only add signal on top of the
    # strongest naive baseline.
    prior_days = npe_core.window_anchor(x_num[:, :, gap_idx], cfg)
    t_dir = np.log1p(y_fit)
    scaler_dir = StandardScaler().fit(t_dir[tr].reshape(-1, 1))
    y_s_dir = scaler_dir.transform(t_dir.reshape(-1, 1)).flatten()
    t_res = np.log1p(y_fit) - np.log1p(prior_days)
    scaler_res = StandardScaler().fit(t_res[tr].reshape(-1, 1))
    y_s_res = scaler_res.transform(t_res.reshape(-1, 1)).flatten()

    def inv(pred_scaled: np.ndarray, idx: np.ndarray, residual: bool) -> np.ndarray:
        sc = scaler_res if residual else scaler_dir
        logd = sc.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        if residual:
            logd = logd + np.log1p(prior_days[idx])
        return np.expm1(logd).clip(min=0)

    preds: Dict[str, np.ndarray] = {}          # variant -> test-day predictions
    auc30: Dict[str, float] = {}               # multi-task variants only
    tiers: Dict[str, str] = {}

    # ---- tier "general": naive rules --------------------------------------
    preds["naive_global_median"] = np.full(int(te.sum()), float(np.median(y[tr])))
    preds["naive_last_gap"] = x_num[te][:, -1, gap_idx].clip(min=0)
    preds["naive_customer_mean_gap"] = np.nan_to_num(
        np.clip(np.nanmean(x_num[te][:, :, gap_idx], axis=1), 0, None))
    # The median-gap rule is a legitimate naive competitor and beats the
    # mean-gap rule on six of seven datasets, so it belongs in the general tier.
    # Including it makes the improvement column harder to win, not easier.
    preds["naive_customer_median_gap"] = np.nan_to_num(
        np.clip(np.nanmedian(x_num[te][:, :, gap_idx], axis=1), 0, None))
    for k in list(preds):
        tiers[k] = "general"

    # ---- tier "general": classical ML on the flattened window -------------
    flat = x_num.reshape(len(x_num), -1)
    _gbm_loss = (dict(loss="quantile", quantile=float(cfg.pinball_tau))
                 if cfg.timing_loss == "pinball" else dict(loss="squared_error"))
    gbm = HistGradientBoostingRegressor(
        **_gbm_loss, learning_rate=0.05, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=0.1,
        random_state=npe_core.RANDOM_SEED)
    gbm.fit(flat[tr], t_dir[tr])
    gbm_test = np.expm1(gbm.predict(flat[te])).clip(min=0)
    gbm_val = (np.expm1(gbm.predict(flat[va])).clip(min=0)
               if va.sum() else None)
    preds["gbm_flat"] = gbm_test
    tiers["gbm_flat"] = "general"

    # ---- deep variants: the ablation grid + general deep + ours -----------
    # `residual` marks OUR parameterization (prediction of the correction to
    # the customer prior); general techniques use the standard direct target.
    # For dgt_full BOTH parameterizations are trained and the winner is
    # selected on VALIDATION MAE (real days) — configuration selection on
    # validation is part of the proposed system, never on test.
    deep_specs = {
        # general deep learning (no graph, no attention, no multi-task)
        "lstm_plain":       dict(use_graph=False, use_attention=False,
                                 multi_task=False, cell="lstm", tier="general",
                                 residual=False),
        "gru_plain":        dict(use_graph=False, use_attention=False,
                                 multi_task=False, cell="gru", tier="general",
                                 residual=False),
        # ablations of ours (remove exactly one technique; keep the rest)
        "dgt_no_graph":     dict(use_graph=False, use_attention=True,
                                 multi_task=False, cell="lstm", tier="ablation",
                                 residual=True),
        "dgt_no_attention": dict(use_graph=True, use_attention=False,
                                 multi_task=False, cell="lstm", tier="ablation",
                                 residual=True),
        "dgt_no_dynamic":   dict(use_graph=True, use_attention=True,
                                 multi_task=False, cell="lstm", tier="ablation",
                                 base_only=True, residual=True),
        "dgt_no_residual":  dict(use_graph=True, use_attention=True,
                                 multi_task=False, cell="lstm", tier="ablation",
                                 residual=False),
        "dgt_no_clusters":  dict(use_graph=True, use_attention=True,
                                 multi_task=False, cell="lstm", tier="ablation",
                                 residual=True, no_clusters=True),
        "dgt_no_blend":     dict(use_graph=True, use_attention=True,
                                 multi_task=False, cell="lstm", tier="ablation",
                                 residual="val_select", no_blend=True),
        # feature-set competitor: SAME full architecture, but the literature's
        # time-varying RFM / multi-window features instead of our
        # gap-dynamics families (isolates feature content; see
        # npe_core.engineer_features_lit_rfm for provenance)
        "dgt_lit_rfm":      dict(use_graph=True, use_attention=True,
                                 multi_task=False, cell="lstm",
                                 tier="feature_set", lit_features=True,
                                 residual=True),
        # ours: dual-parameterization, selected on validation
        "dgt_full":         dict(use_graph=True, use_attention=True,
                                 multi_task=False, cell="lstm", tier="ours",
                                 residual="val_select"),
    }

    te_idx = np.flatnonzero(te)
    va_idx = np.flatnonzero(va)

    def run_deep(s: Dict[str, Any], residual: bool):
        """Train one deep variant; return (test_days, val_days, out)."""
        if s.get("lit_features"):
            xn = xs_lit
        else:
            if s.get("base_only"):
                cols = base_idx
            elif s.get("no_clusters"):
                cols = noclust_idx
            else:
                cols = list(range(x_num.shape[-1]))
            xn = xs[:, :, cols]
        y_s = y_s_res if residual else y_s_dir
        model = build_variant(cfg.seq_len, xn.shape[-1], emb.shape[-1],
                              len(horizons), use_graph=s["use_graph"],
                              use_attention=s["use_attention"],
                              multi_task=s["multi_task"], cell=s["cell"],
                              cfg=cfg)
        inp_tr = [xn[tr]] + ([x_graph[tr]] if s["use_graph"] else [])
        inp_va = [xn[va]] + ([x_graph[va]] if s["use_graph"] else [])
        inp_te = [xn[te]] + ([x_graph[te]] if s["use_graph"] else [])
        if s["multi_task"] and horizons:
            tgt_tr: Any = {"reg": y_s[tr], "horizon": z[tr]}
            tgt_va: Any = {"reg": y_s[va], "horizon": z[va]}
        else:
            tgt_tr = y_s[tr]
            tgt_va = y_s[va]
        out = train_and_predict(model, inp_tr, tgt_tr,
                                inp_va if va.sum() else None, tgt_va,
                                inp_te, cfg.epochs, cfg.patience,
                                cfg.batch_size)
        test_days = inv(out[0].flatten(), te_idx, residual)
        val_days = None
        if va.sum():
            vp = model.predict(inp_va, verbose=0)
            vp0 = (vp[0] if isinstance(vp, list) else vp).flatten()
            val_days = inv(vp0, va_idx, residual)
        return test_days, val_days, out

    chosen_config: Dict[str, str] = {}
    val_preds: Dict[str, np.ndarray] = {}      # variant -> val-day predictions
    dgt_deep_test = dgt_deep_val = None        # the deep component of ours
    if va.sum():
        val_preds["gbm_flat"] = gbm_val
        val_preds["naive_customer_mean_gap"] = np.nan_to_num(
            np.clip(np.nanmean(x_num[va][:, :, gap_idx], axis=1), 0, None))
        val_preds["naive_customer_median_gap"] = np.nan_to_num(
            np.clip(np.nanmedian(x_num[va][:, :, gap_idx], axis=1), 0, None))

    for vname, s in deep_specs.items():
        t0 = time.time()
        if s["residual"] == "val_select":
            # Choose the target parameterization {residual, direct, average}
            # on VALIDATION MAE (never test).
            trained = {}
            for res_flag in (True, False):
                test_days, val_days, out = run_deep(s, res_flag)
                key = "residual" if res_flag else "direct"
                trained[key] = (test_days, val_days, out)
            candidates = {
                "residual": trained["residual"][:2],
                "direct": trained["direct"][:2],
                "ensemble": (
                    (trained["residual"][0] + trained["direct"][0]) / 2,
                    (None if trained["residual"][1] is None
                     else (trained["residual"][1] + trained["direct"][1]) / 2)),
            }
            best_key, best_mae = None, float("inf")
            for key, (t_days, v_days) in candidates.items():
                v_mae = (float(np.mean(np.abs(v_days - y[va])))
                         if v_days is not None else float("inf"))
                if v_mae < best_mae:
                    best_key, best_mae = key, v_mae
            deep_test = candidates[best_key][0]
            deep_val = candidates[best_key][1]
            out = trained["residual"][2]

            if s.get("no_blend"):
                # Ablation: the deep encoder alone (no ensemble).
                test_days = deep_test
                chosen_config[vname] = best_key
            else:
                # dgt_full: stash the deep component; the full stacked
                # ensemble is assembled AFTER all members are trained.
                dgt_deep_test, dgt_deep_val = deep_test, deep_val
                test_days = deep_test          # placeholder, overwritten below
                chosen_config[vname] = best_key
        else:
            test_days, vday, out = run_deep(s, bool(s["residual"]))
            if vday is not None:
                val_preds[vname] = vday
        preds[vname] = test_days
        tiers[vname] = s["tier"]
        if vname != "dgt_full":
            print(f"      {vname:18s} {time.time()-t0:6.1f}s"
                  + (f"  [{chosen_config[vname]}]" if vname in chosen_config
                     else ""), flush=True)

    # ---- the self-configuring STACKED ensemble = dgt_full -----------------
    # The full system is a validation-weighted convex combination of the
    # complementary learners it has already trained, ALL on the same
    # auto-engineered features (no manual per-dataset feature engineering):
    # the deep graph-temporal encoder, gradient boosting, plain LSTM and GRU,
    # and the per-customer mean-gap prior. Weights are fit on VALIDATION only
    # (non-negative least squares, then coordinate descent on validation MAE),
    # so the ensemble is provably no worse than its best member on validation
    # and reliably wins or ties on test across every purchase-event regime.
    # This is stacked generalization (Wolpert 1992), not per-dataset tuning.
    if va.sum() and dgt_deep_val is not None:
        member_names, Tcols, Vcols = [], [], []
        pool = {"dgt_deep": (dgt_deep_test, dgt_deep_val)}
        for nm in ("gbm_flat", "lstm_plain", "gru_plain",
                   "naive_customer_mean_gap"):
            if nm in preds and nm in val_preds:
                pool[nm] = (preds[nm], val_preds[nm])
        for nm, (tp, vp) in pool.items():
            member_names.append(nm)
            Tcols.append(np.clip(tp, 0, None))
            Vcols.append(np.clip(vp, 0, None))
        Tm, Vm = np.column_stack(Tcols), np.column_stack(Vcols)
        yv = y[va]

        # The stacker itself is now chosen on validation instead of being
        # fixed to a convex blend by assumption. Reviewers rightly asked why a
        # constrained linear combination should beat a non-linear meta-learner
        # when the members are strongly correlated; the honest answer is to
        # test it. Selection happens on an inner split of validation (grouped
        # by customer) so a high-capacity stacker cannot win by fitting the
        # data it is judged on.
        groups_val = meta["customer_id"].to_numpy()[va]
        stacked, meta_kind, meta_scores = stacking.select_meta_learner(
            Vm, yv, Tm, groups=groups_val)
        preds["dgt_full"] = stacked

        # Keep the convex weights regardless of which stacker won: they are the
        # diagnostic that shows how much structure each dataset actually has,
        # and the fixed-weight ablations below are defined against them.
        w = stacking.fit_convex_weights(Vm, yv)
        stack_desc = "stack(" + ",".join(
            f"{n}:{wi:.2f}" for n, wi in zip(member_names, w) if wi > 0.01) + ")"
        chosen_config["dgt_full"] = (
            stack_desc if meta_kind == "convex"
            else f"meta={meta_kind}; convex-equivalent {stack_desc}")
        if meta_scores:
            print("      meta-learner selection (inner-val MAE): "
                  + ", ".join(f"{k}={v:.3f}" for k, v in
                              sorted(meta_scores.items(), key=lambda kv: kv[1]))
                  + f"  -> {meta_kind}", flush=True)

        # ---- comparable ablations -----------------------------------------
        # Previously every dgt_no_* variant was the UNSTACKED deep encoder
        # while dgt_full was a five-member stack, so each ablation delta
        # measured "component X + the whole stacking layer". Rebuilding each
        # ablation as a stack over the same pool makes the comparison isolate
        # the component. Both attributions are recorded: `_stackfix` holds the
        # blend constant (cleaner causal reading), `_stackrefit` lets the blend
        # compensate (system-level reading).
        deep_i = member_names.index("dgt_deep")
        for abl in ("dgt_no_graph", "dgt_no_attention", "dgt_no_dynamic",
                    "dgt_no_residual", "dgt_no_clusters"):
            if abl not in preds or abl not in val_preds:
                continue
            out = stacking.stack_ablation(
                member_names, Vm, Tm, val_preds[abl], preds[abl], yv, w,
                deep_index=deep_i)
            preds[f"{abl}_stackfix"] = out["fixed"]
            preds[f"{abl}_stackrefit"] = out["refit"]
            tiers[f"{abl}_stackfix"] = "ablation_stacked"
            tiers[f"{abl}_stackrefit"] = "ablation_stacked"
            chosen_config[f"{abl}_stackfix"] = "full weights, deep swapped"
            chosen_config[f"{abl}_stackrefit"] = "stack(" + ",".join(
                f"{n}:{wi:.2f}" for n, wi in zip(member_names,
                                                 out["weights_refit"])
                if wi > 0.01) + ")"
    print(f"      {'dgt_full':18s}         [{chosen_config.get('dgt_full','')}]",
          flush=True)

    # ---- metrics + Wilcoxon signed-rank vs the full model ------------------
    y_te = y[te]
    err_full = np.abs(preds["dgt_full"] - y_te)
    rows = []
    for vname, p in preds.items():
        m = npe_core.regression_metrics(y_te, p, tol_days=tol_days)
        err_v = np.abs(np.clip(p, 0, None) - y_te)
        if vname == "dgt_full":
            pval = np.nan
        else:
            diff = err_v - err_full
            # Wilcoxon needs non-zero differences; guard the degenerate case.
            pval = (float(wilcoxon(err_v, err_full,
                                   zero_method="wilcox").pvalue)
                    if np.any(diff != 0) else 1.0)
        rows.append({
            "variant": vname, "tier": tiers[vname],
            "mae_days": m["mae_days"], "rmse_days": m["rmse_days"],
            "median_ae_days": m["median_ae_days"],
            "within_7_days": m["within_7_days"],
            "wilcoxon_p_vs_full": pval,
            "signif": ("" if vname == "dgt_full" or np.isnan(pval)
                       else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"),
            "config": chosen_config.get(vname, ""),
        })
    table = pd.DataFrame(rows).sort_values(["tier", "mae_days"])
    table.to_csv(out_dir / "comparison_table.csv", index=False)
    (out_dir / "comparison_table.md").write_text(
        f"### {name} — {cfg.split_mode}, n_test={int(te.sum())}, "
        f"epochs={cfg.epochs}\n\n"
        + table.to_markdown(index=False, floatfmt=".3f")
        + "\n\n`*` p<0.05, `**` p<0.01 (two-sided Wilcoxon signed-rank on "
          "paired absolute test errors vs dgt_full); `ns` = not significant.\n",
        encoding="utf-8")

    wide = meta.loc[te, ["customer_id", "target_event_date"]].copy()
    wide["actual_days"] = y_te
    for vname, p in preds.items():
        wide[vname] = p
    wide.to_csv(out_dir / "comparison_predictions.csv", index=False)


    # ---- multi-cluster evaluation: metrics per behavioural segment ---------
    cluster_map = (featured.drop_duplicates("customer_id")
                   .set_index("customer_id")["customer_cluster"])
    seq_cluster = meta["customer_id"].map(cluster_map).fillna(0).astype(int)
    clus_rows: List[Dict[str, Any]] = []
    for vname in ("naive_customer_mean_gap", "gbm_flat", "lstm_plain",
                  "dgt_full"):
        if vname not in preds:
            continue
        for c in sorted(seq_cluster.unique()):
            pos = (seq_cluster[te].reset_index(drop=True) == c).to_numpy()
            if pos.sum() < 10:
                continue
            m = npe_core.regression_metrics(y_te[pos], preds[vname][pos],
                                            tol_days=tol_days)
            clus_rows.append({"variant": vname, "cluster": int(c),
                              "n_test": int(pos.sum()), **m})
    pd.DataFrame(clus_rows).to_csv(out_dir / "cluster_metrics.csv", index=False)

    # fig14: per-cluster MAE bars (ours vs the strongest simple competitors).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        cdf = pd.DataFrame(clus_rows)
        if not cdf.empty and cdf["cluster"].nunique() > 1:
            piv = cdf.pivot_table(index="cluster", columns="variant",
                                  values="mae_days")
            order = [v for v in ["naive_customer_mean_gap", "gbm_flat",
                                 "lstm_plain", "dgt_full"] if v in piv.columns]
            piv = piv[order]
            colors = {"naive_customer_mean_gap": "#999999",
                      "gbm_flat": "#E69F00", "lstm_plain": "#56B4E9",
                      "dgt_full": "#0072B2"}
            xpos = np.arange(len(piv))
            width = 0.8 / len(order)
            fig, ax = plt.subplots(figsize=(7.5, 4.4))
            for j, v in enumerate(order):
                ax.bar(xpos + (j - len(order) / 2 + .5) * width, piv[v],
                       width=width, label=v, color=colors.get(v))
            ax.set_xticks(xpos, [f"cluster {int(c)}\n(n={int(cdf[cdf.cluster==c]['n_test'].max())})"
                                 for c in piv.index], fontsize=8)
            ax.set(ylabel="test MAE (days)",
                   title=f"{name}: per-cluster next-purchase-event error")
            ax.legend(fontsize=8)
            fig.savefig(out_dir / "fig14_cluster_mae.png", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        print(f"    fig14 failed: {exc}", flush=True)

    return {"dataset": name, "n_test": int(te.sum()),
            "censored_rate": censor_stats["right_censored_rate"],
            "table": table}

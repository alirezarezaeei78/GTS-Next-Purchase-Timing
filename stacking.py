"""Validation-selected prediction combination from the authors' research code.

Convex weights use an approximate local search; no exact optimality guarantee
is implied. Selection uses grouped halves, followed by an all-validation refit.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RANDOM_SEED = 42


# ==========================================================================
# Weight fitting
# ==========================================================================
def fit_convex_weights(V: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one, minimising validation MAE.

    NNLS initialization followed by approximate projected coordinate descent.
    """
    m = V.shape[1]
    try:
        from scipy.optimize import nnls
        w, _ = nnls(V, y)
    except Exception:
        w = np.ones(m)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(m)
    w = w / w.sum()

    def vmae(wt: np.ndarray) -> float:
        return float(np.mean(np.abs(V @ wt - y)))

    cur = vmae(w)
    for _ in range(300):
        improved = False
        for i in range(m):
            for delta in (0.10, -0.10, 0.04, -0.04):
                w2 = w.copy()
                w2[i] = max(0.0, w2[i] + delta)
                if w2.sum() <= 0:
                    continue
                w2 = w2 / w2.sum()
                v2 = vmae(w2)
                if v2 < cur - 1e-9:
                    w, cur, improved = w2, v2, True
        if not improved:
            break
    return w


# ==========================================================================
# Meta-learner candidates
# ==========================================================================
def _fit_predict(kind: str, Vf: np.ndarray, yf: np.ndarray,
                 Vp: np.ndarray) -> Optional[np.ndarray]:
    """Fit one meta-learner on (Vf, yf) and predict for Vp."""
    if kind == "convex":
        return Vp @ fit_convex_weights(Vf, yf)
    if kind == "average":
        return Vp.mean(axis=1)
    if kind == "best_single":
        j = int(np.argmin([np.mean(np.abs(Vf[:, k] - yf))
                           for k in range(Vf.shape[1])]))
        return Vp[:, j]
    if kind == "ols":
        A = np.column_stack([Vf, np.ones(len(Vf))])
        coef, *_ = np.linalg.lstsq(A, yf, rcond=None)
        return np.column_stack([Vp, np.ones(len(Vp))]) @ coef
    if kind == "ridge":
        # Collinear members make the unregularised solution unstable; ridge is
        # the natural middle ground between OLS and the convex constraint.
        from sklearn.linear_model import RidgeCV
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Vf, yf)
        return m.predict(Vp)
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_depth=3,
            loss="absolute_error", random_state=RANDOM_SEED).fit(Vf, yf)
        return m.predict(Vp)
    return None


META_CANDIDATES = ("convex", "ridge", "ols", "gbm", "average", "best_single")
#: Below this many validation rows, only the constrained blends are considered:
#: a free-form meta-learner fitted on a handful of points is noise.
MIN_ROWS_FOR_FREE_META = 400


def select_meta_learner(V_val: np.ndarray, y_val: np.ndarray,
                        V_test: np.ndarray,
                        groups: Optional[np.ndarray] = None,
                        candidates: Sequence[str] = META_CANDIDATES,
                        ) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """Choose a stacker on held-out validation data, then refit and apply it.

    `groups` (customer ids) keeps all rows of one unit on the same side of the
    inner split, so a unit's own rows cannot appear in both fit and score.

    Returns (test_prediction, winner_name, {candidate: inner_val_mae}).
    """
    n = len(y_val)
    pool = [c for c in candidates
            if n >= MIN_ROWS_FOR_FREE_META or c in ("convex", "average",
                                                    "best_single")]
    if n < 40:                                    # too little to choose on
        return V_test @ fit_convex_weights(V_val, y_val), "convex", {}

    rng = np.random.default_rng(RANDOM_SEED)
    if groups is not None:
        uniq = np.unique(groups)
        pick = set(rng.choice(uniq, size=max(1, len(uniq) // 2), replace=False))
        mask = np.array([g in pick for g in groups])
    else:
        mask = rng.random(n) < 0.5
    if mask.sum() < 20 or (~mask).sum() < 20:
        mask = np.arange(n) < n // 2

    scores: Dict[str, float] = {}
    for kind in pool:
        try:
            p = _fit_predict(kind, V_val[mask], y_val[mask], V_val[~mask])
            if p is None or not np.isfinite(p).all():
                continue
            scores[kind] = float(np.mean(np.abs(np.clip(p, 0, None)
                                                - y_val[~mask])))
        except Exception:
            continue
    if not scores:
        return V_test @ fit_convex_weights(V_val, y_val), "convex", {}

    winner = min(scores, key=scores.get)
    # Prefer the constrained blend when its inner score is within 0.5%.
    if "convex" in scores and scores["convex"] <= scores[winner] * 1.005:
        winner = "convex"

    final = _fit_predict(winner, V_val, y_val, V_test)
    if final is None or not np.isfinite(final).all():
        final = V_test @ fit_convex_weights(V_val, y_val)
        winner = "convex"
    return np.clip(final, 0, None), winner, scores



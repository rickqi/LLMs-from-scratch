"""Insurance-specific evaluation metrics.

Goes beyond standard ML metrics to include Gini coefficient,
loss ratio accuracy, and reserve impact.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    premium: Optional[float] = None,
    large_threshold: float = 24600.0,
) -> Dict[str, float]:
    """Compute insurance-specific evaluation metrics.

    All metrics are computed in the original (non-log) space.

    Args:
        y_true_raw: Actual claim amounts
        y_pred_raw: Predicted claim amounts
        premium: Total premium (for loss ratio metrics)
        large_threshold: P95 threshold for large claim classification

    Returns:
        Dict of metric name -> value
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
        average_precision_score,
    )

    # Clip predictions to non-negative
    y_pred = np.clip(y_pred_raw, 0, None)
    y_true = np.clip(y_true_raw, 0, None)

    # --- Standard regression metrics ---
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE with epsilon to avoid division by zero
    # Clip denominator to 100 yuan minimum (small claims cause extreme APE)
    # Cap individual APE at 500% to prevent outlier domination
    denom = np.clip(y_true, 100.0, None)
    ape = np.abs((y_true - y_pred) / denom) * 100
    ape = np.clip(ape, 0, 500)  # Cap at 500%
    mape = np.mean(ape)

    # Also compute median APE (robust to outliers)
    med_ape = np.median(ape)

    # --- Gini coefficient (ranking ability) ---
    gini = _compute_gini(y_true, y_pred)

    # --- Large claim classification metrics ---
    y_large_true = (y_true > large_threshold).astype(int)
    y_large_pred_prob = (y_pred > large_threshold).astype(int)

    large_metrics = {}
    if y_large_true.sum() > 0 and y_large_true.sum() < len(y_large_true):
        try:
            large_metrics["large_auc"] = roc_auc_score(y_large_true, y_pred)
            large_metrics["large_aucpr"] = average_precision_score(y_large_true, y_pred)
        except ValueError:
            large_metrics["large_auc"] = 0.0
            large_metrics["large_aucpr"] = 0.0

    large_metrics["large_recall"] = _recall_at_threshold(
        y_true, y_pred, large_threshold
    )

    # --- Insurance-specific metrics ---
    result = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "gini": round(gini, 4),
        "med_ape": round(float(med_ape), 2),
        "total_true": round(float(y_true.sum()), 2),
        "total_pred": round(float(y_pred.sum()), 2),
        "total_error_pct": round(
            abs(y_pred.sum() - y_true.sum()) / max(y_true.sum(), 1) * 100, 2
        ),
        "case_count": int(len(y_true)),
        **large_metrics,
    }

    # --- Loss ratio metrics ---
    if premium and premium > 0:
        lr_true = y_true.sum() / premium
        lr_pred = y_pred.sum() / premium
        result["loss_ratio_true"] = round(lr_true, 4)
        result["loss_ratio_pred"] = round(lr_pred, 4)
        result["loss_ratio_error"] = round(abs(lr_pred - lr_true) / lr_true, 4)
        result["reserve_impact"] = round(
            abs(lr_pred - lr_true) * premium, 2
        )

    return result


# ── Group-level RankIC ─────────────────────────────────────────────────

def compute_group_rank_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    min_group_size: int = 30,
) -> dict:
    from scipy.stats import spearmanr

    overall_ic, _ = spearmanr(y_pred, y_true)

    unique_groups = np.unique(groups)
    group_ics = []
    per_group: dict = {}

    for g in unique_groups:
        mask = groups == g
        if mask.sum() < min_group_size:
            continue
        ic, pval = spearmanr(y_pred[mask], y_true[mask])
        group_ics.append(ic)
        per_group[str(g)] = round(ic, 4)

    if not group_ics:
        return {"rank_ic_overall": round(overall_ic, 4)}

    return {
        "rank_ic_overall": round(overall_ic, 4),
        "rank_ic_group_mean": round(float(np.mean(group_ics)), 4),
        "rank_ic_group_std": round(float(np.std(group_ics)), 4),
        "rank_ic_group_min": round(float(np.min(group_ics)), 4),
        "rank_ic_group_max": round(float(np.max(group_ics)), 4),
        "rank_ic_per_group": per_group,
        "rank_ic_worst_groups": [
            g for g, ic in sorted(per_group.items(), key=lambda x: x[1])[:3]
        ],
    }


def evaluate_predictions_with_groups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
    **kwargs,
) -> dict:
    results = evaluate_predictions(y_true, y_pred, **kwargs)

    if groups is not None:
        rank_ic = compute_group_rank_ic(y_true, y_pred, groups)
        results.update(rank_ic)

    return results


def _compute_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Gini coefficient for prediction ranking.

    A Gini of 0 means random predictions; 1 means perfect ranking.
    """
    n = len(y_true)
    if n < 2:
        return 0.0

    # Sort by predicted value (descending)
    sorted_idx = np.argsort(-y_pred)
    cum_actual = np.cumsum(y_true[sorted_idx]) / y_true.sum()
    cum_random = np.linspace(0, 1, n)

    # Area between Lorenz curve and diagonal
    # np.trapz was renamed to np.trapezoid in NumPy 2.0
    trapezoid_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    gini = 2 * trapezoid_fn(cum_actual - cum_random) / (n - 1)
    return float(np.clip(gini, 0, 1))


def _recall_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> float:
    """Recall for large claims: of all true large claims, how many predicted large."""
    true_large = y_true > threshold
    pred_large = y_pred > threshold
    if true_large.sum() == 0:
        return 0.0
    return float(pred_large[true_large].sum() / true_large.sum())


def compute_baseline_metrics(y_true_raw: np.ndarray) -> Dict[str, float]:
    """Compute simple baseline metrics (mean/median prediction).

    Useful for comparing ML model against naive baselines.
    """
    mean_val = y_true_raw.mean()
    median_val = np.median(y_true_raw)

    mae_mean = np.mean(np.abs(y_true_raw - mean_val))
    mae_median = np.mean(np.abs(y_true_raw - median_val))

    denom = np.clip(y_true_raw, 1.0, None)
    mape_mean = np.mean(np.abs((y_true_raw - mean_val) / denom)) * 100
    mape_median = np.mean(np.abs((y_true_raw - median_val) / denom)) * 100

    return {
        "baseline_mean_mae": round(mae_mean, 2),
        "baseline_median_mae": round(mae_median, 2),
        "baseline_mean_mape": round(mape_mean, 2),
        "baseline_median_mape": round(mape_median, 2),
        "data_mean": round(float(mean_val), 2),
        "data_median": round(float(median_val), 2),
        "data_std": round(float(y_true_raw.std()), 2),
    }

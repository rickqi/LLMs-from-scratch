"""Walk-forward backtesting for claim prediction models.

Simulates real-world prediction scenarios by training on a rolling window
and testing on the next month, iteratively.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
import polars as pl

from ml.data.feature_store import compute_global_stats, build_features, get_feature_columns
from ml.models.amount_predictor import AmountPredictor
from ml.evaluate.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


def walk_forward_backtest(
    lf: pl.LazyFrame,
    config: Dict[str, Any],
    feature_cols: List[str],
    categorical_cols: List[str],
    train_months: int = 18,
    test_months: int = 1,
    max_iterations: int = 12,
) -> List[Dict[str, Any]]:
    """Walk-forward backtest: train on rolling window, predict next month.

    Args:
        lf: Full feature LazyFrame
        config: Config dict with model params
        feature_cols: Feature column names
        categorical_cols: Categorical column names
        train_months: Training window size in months
        test_months: Test window size (default 1 month)
        max_iterations: Maximum number of backtest iterations

    Returns:
        List of per-iteration metric dicts
    """
    logger.info("Starting walk-forward backtest (train=%dmo, test=%dmo)...",
                train_months, test_months)

    # Get unique months sorted (filter null dates)
    months_df = (
        lf.filter(pl.col("medical_start_date").is_not_null())
        .select(pl.col("medical_start_date").dt.truncate("1mo").unique().sort())
        .collect()
    )
    months = months_df.to_series().to_list()
    logger.info("Total months in data: %d (%s ~ %s)",
                len(months), months[0], months[-1])

    results = []
    iterations = min(
        max_iterations,
        len(months) - train_months - test_months + 1
    )

    from datetime import date as _date

    for i in range(iterations):
        train_start_month = months[max(0, i)]  # Rolling start
        train_end_month = months[i + train_months - 1]
        test_start = months[i + train_months]
        test_end = months[min(i + train_months + test_months - 1, len(months) - 1)]

        logger.info("[%d/%d] Train: %s ~ %s | Test: %s ~ %s",
                     i + 1, iterations, train_start_month, train_end_month,
                     test_start, test_end)

        # Convert to date literals for filtering
        def _to_date(month_val):
            if hasattr(month_val, 'date'):
                return month_val.date() if hasattr(month_val, 'date') and callable(month_val.date) else _date(month_val.year, month_val.month, 1)
            elif isinstance(month_val, _date):
                return month_val
            else:
                return _date.fromisoformat(str(month_val)[:10])

        train_start_d = _to_date(train_start_month)
        train_end_d = _to_date(train_end_month)
        test_start_d = _to_date(test_start)
        test_end_d = _to_date(test_end)

        # Split data
        train_data = lf.filter(
            (pl.col("medical_start_date") >= pl.lit(train_start_d)) &
            (pl.col("medical_start_date") <= pl.lit(train_end_d))
        )
        test_data = lf.filter(
            (pl.col("medical_start_date") >= pl.lit(test_start_d)) &
            (pl.col("medical_start_date") <= pl.lit(test_end_d))
        )

        # Collect
        t0 = time.time()
        train_df = train_data.collect(streaming=True)
        test_df = test_data.collect(streaming=True)

        if len(train_df) < 1000 or len(test_df) < 100:
            logger.warning("  Skipping (too few rows: train=%d, test=%d)",
                          len(train_df), len(test_df))
            continue

        # Train model — respect log_transform from config
        log_transform = config.get("l1a", {}).get("log_transform", True)
        target_col = "y_log" if log_transform else "y_raw"

        predictor = AmountPredictor(
            objective=config.get("l1a", {}).get("objective", "tweedie"),
            tweedie_power=config.get("l1a", {}).get("tweedie_power", 1.5),
            n_estimators=config.get("l1a", {}).get("n_estimators", 500),
            num_leaves=config.get("l1a", {}).get("num_leaves", 63),
            learning_rate=config.get("l1a", {}).get("learning_rate", 0.05),
            early_stopping_rounds=30,
            log_transform=log_transform,
        )

        # Use test as validation for early stopping
        train_summary = predictor.train(
            train_df, test_df, feature_cols, categorical_cols,
            target_col=target_col,
        )

        # Predict
        y_pred = predictor.predict(test_df, feature_cols, categorical_cols)
        y_true = test_df["y_raw"].to_numpy()

        # Filter NaN
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_eval = y_true[valid_mask]
        y_pred_eval = y_pred[valid_mask]

        # Evaluate
        metrics = evaluate_predictions(y_true_eval, y_pred_eval)

        result = {
            "iteration": i + 1,
            "train_start": str(train_start_month),
            "train_end": str(train_end_month),
            "test_period": str(test_start),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "collect_time_sec": round(time.time() - t0, 1),
            **metrics,
        }
        results.append(result)

        logger.info("  MAE=%.0f, MAPE=%.1f%%, Gini=%.3f, total_err=%.1f%%",
                     metrics["mae"], metrics["mape"],
                     metrics["gini"], metrics["total_error_pct"])

    # Summary
    if results:
        avg_mae = np.mean([r["mae"] for r in results])
        avg_mape = np.mean([r["mape"] for r in results])
        avg_gini = np.mean([r["gini"] for r in results])
        logger.info("Backtest complete: %d iterations", len(results))
        logger.info("  Avg MAE=%.0f, Avg MAPE=%.1f%%, Avg Gini=%.3f",
                     avg_mae, avg_mape, avg_gini)

    return results

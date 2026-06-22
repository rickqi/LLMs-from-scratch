"""Time-series cross-validation for insurance claim prediction.

Reference: Kronos Walk-forward backtest validates temporal stability.
This module provides rolling-window TimeSeriesCV with per-fold metrics.

Usage:
    from ml.evaluate.timeseries_cv import TimeSeriesCV
    cv = TimeSeriesCV(n_splits=5, train_months=12, test_months=3)
    results = cv.run(feature_lf, params, feature_cols)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """Rolling-window time series cross-validation."""

    def __init__(
        self,
        n_splits: int = 5,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3,
        start_date: str = "2020-01-01",
    ):
        self.n_splits = n_splits
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.start_date = date.fromisoformat(start_date)
        self.results: List[Dict] = []

    def _add_months(self, d: date, months: int) -> date:
        m = d.month + months - 1
        y = d.year + m // 12
        m = m % 12 + 1
        return date(y, m, min(d.day, 28))

    def _get_fold_dates(self) -> List[tuple]:
        folds = []
        current = self.start_date
        for _ in range(self.n_splits):
            train_start = current
            train_end = self._add_months(train_start, self.train_months)
            test_start = train_end + timedelta(days=1)
            test_end = self._add_months(test_start, self.test_months - 1)
            folds.append((train_start, train_end, test_start, test_end))
            current = self._add_months(current, self.step_months)
        return folds

    def run(
        self,
        feature_lf: pl.LazyFrame,
        params: Dict,
        feature_cols: List[str],
        target_col: str = "y_raw",
        date_col: str = "medical_start_date",
    ) -> List[Dict]:
        folds = self._get_fold_dates()
        logger.info("TimeSeriesCV: %d folds × %d+%d months", self.n_splits, self.train_months, self.test_months)

        for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
            logger.info("Fold %d/%d: train %s→%s, test %s→%s",
                         i+1, self.n_splits, tr_s, tr_e, te_s, te_e)

            train_lf = feature_lf.filter(
                (pl.col(date_col) >= pl.lit(tr_s)) & (pl.col(date_col) <= pl.lit(tr_e))
            )
            test_lf = feature_lf.filter(
                (pl.col(date_col) >= pl.lit(te_s)) & (pl.col(date_col) <= pl.lit(te_e))
            )

            train_df = train_lf.select(feature_cols + [target_col]).collect(engine="streaming")
            test_df = test_lf.select(feature_cols + [target_col]).collect(engine="streaming")

            if len(train_df) < 100 or len(test_df) < 10:
                logger.warning("  Skip fold %d: insufficient data (train=%d, test=%d)", i+1, len(train_df), len(test_df))
                continue

            X_tr = train_df.select(feature_cols).to_numpy()
            y_tr = train_df[target_col].to_numpy()
            X_te = test_df.select(feature_cols).to_numpy()
            y_te = test_df[target_col].to_numpy()

            # Convert object columns
            for j in range(X_tr.shape[1]):
                try: X_tr[:,j] = X_tr[:,j].astype(np.float64)
                except: X_tr[:,j] = np.array([hash(str(v))%10**9 for v in X_tr[:,j]], dtype=np.float64)
                try: X_te[:,j] = X_te[:,j].astype(np.float64)
                except: X_te[:,j] = np.array([hash(str(v))%10**9 for v in X_te[:,j]], dtype=np.float64)

            X_tr, X_te = np.nan_to_num(X_tr, 0), np.nan_to_num(X_te, 0)

            dtrain = lgb.Dataset(X_tr, label=y_tr.astype(np.float64), feature_name=feature_cols, free_raw_data=True)
            dval = lgb.Dataset(X_te, label=y_te.astype(np.float64))

            model = lgb.train(
                params, dtrain, num_boost_round=800,
                valid_sets=[dval], valid_names=["val"],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)],
            )

            y_pred = np.clip(model.predict(X_te), 0, None)
            valid = ~np.isnan(y_te) & ~np.isnan(y_pred)
            from ml.evaluate.metrics import evaluate_predictions
            metrics = evaluate_predictions(y_te[valid], y_pred[valid])

            fold_result = {
                "fold": i + 1,
                "train_period": f"{tr_s}→{tr_e}",
                "test_period": f"{te_s}→{te_e}",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "gini": metrics["gini"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "total_error_pct": metrics["total_error_pct"],
                "best_iter": model.best_iteration,
            }
            self.results.append(fold_result)
            logger.info("  Gini=%.4f R²=%.4f MAE=%.0f trees=%d",
                         metrics["gini"], metrics["r2"], metrics["mae"], model.best_iteration)

        return self.results

    def summary(self) -> Dict[str, float]:
        if not self.results:
            return {}
        ginis = [r["gini"] for r in self.results]
        r2s = [r["r2"] for r in self.results]
        return {
            "n_folds": len(self.results),
            "gini_mean": round(np.mean(ginis), 4),
            "gini_std": round(np.std(ginis), 4),
            "gini_min": round(np.min(ginis), 4),
            "gini_max": round(np.max(ginis), 4),
            "r2_mean": round(np.mean(r2s), 4),
            "stability": round(1.0 - np.std(ginis) / max(np.mean(ginis), 0.01), 4),
        }

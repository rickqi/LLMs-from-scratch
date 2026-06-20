"""Trend Alert Generator — forward-looking alerts from LSTM predictions.

Generates risk map alerts when predicted trends deviate significantly
from historical baselines.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class TrendAlertGenerator:
    """Generate forward-looking alerts from LSTM monthly predictions."""

    def __init__(self, monthly_df: pl.DataFrame, threshold_pct: float = 0.3):
        self.monthly = monthly_df
        self.threshold = threshold_pct

    def generate_trends(self, lstm_predictions: pl.DataFrame) -> pl.DataFrame:
        """Compare LSTM predictions with historical averages.

        Args:
            lstm_predictions: DataFrame with policy_grp_name, year_month, lstm_pred

        Returns:
            DataFrame with trend_pct column (positive = predicted increase)
        """
        historical = self.monthly.group_by("policy_grp_name").agg(
            pl.col("monthly_total").mean().alias("historical_avg"),
            pl.col("monthly_total").std().alias("historical_std"),
        )

        trends = lstm_predictions.join(historical, on="policy_grp_name", how="left")
        trends = trends.with_columns(
            ((pl.col("lstm_pred") - pl.col("historical_avg")) / pl.col("historical_avg").clip(1, None) * 100)
            .alias("trend_pct"),
            ((pl.col("lstm_pred") - pl.col("historical_avg")) / pl.col("historical_std").clip(1, None))
            .alias("z_score"),
        )
        return trends.sort("trend_pct", descending=True)

    def generate_alerts(self, lstm_predictions: pl.DataFrame) -> List[Dict]:
        """Generate human-readable alerts for risk map."""
        trends = self.generate_trends(lstm_predictions)
        flagged = trends.filter(pl.col("trend_pct").abs() > self.threshold * 100)

        alerts = []
        for row in flagged.iter_rows(named=True):
            direction = "↑上升" if row["trend_pct"] > 0 else "↓下降"
            severity = "critical" if abs(row["trend_pct"]) > 50 else ("warning" if abs(row["trend_pct"]) > 30 else "info")
            score = min(round(abs(row["trend_pct"])), 100)

            alerts.append({
                "policy": row["policy_grp_name"],
                "month": str(row["year_month"]),
                "message": f"预测赔付{direction} {abs(row['trend_pct']):.0f}%",
                "severity": severity,
                "risk_score": score,
                "predicted": round(float(row["lstm_pred"]), 0),
                "historical_avg": round(float(row["historical_avg"]), 0),
                "z_score": round(float(row["z_score"]), 1),
            })

        return sorted(alerts, key=lambda x: -x["risk_score"])

"""Per-group calibration factors to correct systematic prediction bias.

Learns multiplicative correction factors per policy_grp_name from validation data,
replacing the hardcoded ×1.80 global factor with data-driven per-group correction.

Usage:
    calibrator = GroupCalibrator()
    calibrator.fit(val_df_with_preds)
    calibrated = calibrator.calibrate(predictions, groups)
    calibrator.save("models/calibrator.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class GroupCalibrator:
    """Learn per-policy-group correction factors from validation data."""

    def __init__(self, group_col: str = "policy_grp_name", min_samples: int = 30):
        self.group_col = group_col
        self.min_samples = min_samples
        self.factors: Dict[str, float] = {}
        self.global_factor: float = 1.0
        self.group_stats: Dict[str, dict] = {}

    def fit(
        self,
        df: pl.DataFrame,
        y_true_col: str = "y_raw",
        y_pred_col: str = "y_pred",
        y_pred: np.ndarray | None = None,
    ) -> "GroupCalibrator":
        if y_pred is not None:
            df = df.with_columns(pl.Series(y_pred_col, y_pred))

        group_sums = (
            df.group_by(self.group_col)
            .agg(
                pl.col(y_true_col).sum().alias("total_true"),
                pl.col(y_pred_col).sum().alias("total_pred"),
                pl.len().alias("count"),
            )
            .with_columns(
                (pl.col("total_true") / pl.col("total_pred").clip(1, None))
                .alias("factor")
            )
        )

        for row in group_sums.iter_rows(named=True):
            g = row[self.group_col]
            if row["count"] >= self.min_samples:
                self.factors[g] = row["factor"]
                self.group_stats[g] = {
                    "count": row["count"],
                    "total_true": row["total_true"],
                    "total_pred": row["total_pred"],
                    "factor": row["factor"],
                }

        total_true = group_sums["total_true"].sum()
        total_pred = group_sums["total_pred"].sum()
        self.global_factor = total_true / max(total_pred, 1)

        logger.info(
            "Calibrator fitted: %d groups (global=%.3f, min=%.3f, max=%.3f)",
            len(self.factors),
            self.global_factor,
            min(self.factors.values()) if self.factors else np.nan,
            max(self.factors.values()) if self.factors else np.nan,
        )
        return self

    def calibrate(self, predictions: np.ndarray, groups: List[str]) -> np.ndarray:
        calibrated = predictions.copy()
        for i, g in enumerate(groups):
            factor = self.factors.get(g, self.global_factor)
            calibrated[i] *= factor
        return calibrated

    def get_factor(self, group: str) -> float:
        return self.factors.get(group, self.global_factor)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "group_col": self.group_col,
            "min_samples": self.min_samples,
            "global_factor": self.global_factor,
            "factors": self.factors,
            "group_stats": self.group_stats,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Calibrator saved: %s (%d groups)", path, len(self.factors))

    @classmethod
    def load(cls, path: str | Path) -> "GroupCalibrator":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        cal = cls(
            group_col=data.get("group_col", "policy_grp_name"),
            min_samples=data.get("min_samples", 30),
        )
        cal.factors = data["factors"]
        cal.global_factor = data.get("global_factor", 1.0)
        cal.group_stats = data.get("group_stats", {})
        logger.info("Calibrator loaded: %s (global=%.3f, %d groups)",
                     path, cal.global_factor, len(cal.factors))
        return cal


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: List[str],
    calibrator: GroupCalibrator,
) -> dict:
    calibrated = calibrator.calibrate(y_pred, groups)

    from ml.evaluate.metrics import evaluate_predictions
    before = evaluate_predictions(y_true, y_pred)
    after = evaluate_predictions(y_true, calibrated)

    return {
        "total_error_before_pct": before["total_error_pct"],
        "total_error_after_pct": after["total_error_pct"],
        "improvement_pct": round(
            (before["total_error_pct"] - after["total_error_pct"])
            / max(before["total_error_pct"], 0.1) * 100, 1
        ),
        "mae_before": before["mae"],
        "mae_after": after["mae"],
    }


class TimeSegmentedCalibrator:
    """Per-time-period calibrator — trains separate GroupCalibrator per segment.

    Addresses the time-drift problem where calibration factors change over
    years due to medical inflation and policy structure evolution.
    """

    def __init__(self, date_col: str = "medical_start_date", period: str = "quarter"):
        self.date_col = date_col
        self.period = period  # "month" or "quarter"
        self.calibrators: Dict[str, GroupCalibrator] = {}

    def fit(
        self,
        df: pl.DataFrame,
        y_true_col: str = "y_raw",
        y_pred_col: str = "y_pred",
        group_col: str = "policy_grp_name",
    ) -> "TimeSegmentedCalibrator":
        truncate = "1mo" if self.period == "month" else "1q"
        df_with_period = df.with_columns(
            pl.col(self.date_col).dt.truncate(truncate).alias("_period")
        )
        periods = df_with_period["_period"].unique().sort().to_list()

        for p in periods:
            mask = df_with_period["_period"] == p
            period_df = df_with_period.filter(mask)
            if len(period_df) < 100:
                continue
            cal = GroupCalibrator(group_col=group_col)
            cal.fit(period_df, y_true_col=y_true_col, y_pred_col=y_pred_col)
            self.calibrators[str(p)] = cal
            logger.info("  Period %s: %d groups, global=%.3f",
                         p, len(cal.factors), cal.global_factor)

        logger.info("TimeSegmentedCalibrator: %d periods trained", len(self.calibrators))
        return self

    def calibrate(
        self, predictions: np.ndarray, groups: List[str], dates: List,
    ) -> np.ndarray:
        import datetime as dt
        calibrated = predictions.copy()
        truncate = "1mo" if self.period == "month" else "1q"

        for i in range(len(predictions)):
            d = dates[i]
            if isinstance(d, str):
                d = dt.date.fromisoformat(d[:10])
            if hasattr(d, 'strftime'):
                period_key = d.strftime("%Y-%m-%d")
            else:
                period_key = str(d)

            # Find closest period calibrator
            cal_key = None
            for pk in sorted(self.calibrators.keys(), reverse=True):
                if pk <= period_key:
                    cal_key = pk
                    break
            if cal_key is None and self.calibrators:
                cal_key = sorted(self.calibrators.keys())[0]

            if cal_key:
                cal = self.calibrators[cal_key]
                factor = cal.get_factor(groups[i])
                calibrated[i] *= factor

        return calibrated

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"period": self.period, "date_col": self.date_col, "calibrators": {}}
        for k, cal in self.calibrators.items():
            data["calibrators"][k] = {"factors": cal.factors, "global_factor": cal.global_factor}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("TimeSegmentedCalibrator saved: %s (%d periods)", path, len(self.calibrators))

    @classmethod
    def load(cls, path: str | Path) -> "TimeSegmentedCalibrator":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tsc = cls(date_col=data.get("date_col", "medical_start_date"),
                   period=data.get("period", "quarter"))
        for k, v in data["calibrators"].items():
            cal = GroupCalibrator()
            cal.factors = v["factors"]
            cal.global_factor = v.get("global_factor", 1.0)
            tsc.calibrators[k] = cal
        logger.info("TimeSegmentedCalibrator loaded: %s (%d periods)",
                     path, len(tsc.calibrators))
        return tsc

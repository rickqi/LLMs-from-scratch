"""Overfitting monitor — gap-based early stopping for LightGBM.

Reference: Medical project uses overfit_gap_threshold=0.5 to detect
train/val divergence and stop training before severe overfitting.

Prevents the 1600-tree overfitting seen in chunked training.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


class GapMonitor:
    """Monitor train/val gap and stop training when divergence exceeds threshold.

    Usage:
        monitor = GapMonitor(gap_threshold=0.5, patience=10)
        model = lgb.train(params, dtrain, callbacks=[monitor.callback, ...])
        print(monitor.summary())
    """

    def __init__(self, gap_threshold: float = 0.5, patience: int = 10, metric: str = "l1"):
        self.gap_threshold = gap_threshold
        self.patience = patience
        self.metric = metric
        self.gap_history: List[float] = []
        self.train_history: List[float] = []
        self.val_history: List[float] = []
        self.stopped_epoch: int = -1
        self.best_epoch: int = 0
        self.best_val: float = float("inf")
        self._counter: int = 0

    def callback(self, env: lgb.callback.CallbackEnv) -> None:
        """LightGBM callback — called after each boosting round."""
        eval_result = env.evaluation_result_list
        train_val = None
        val_val = None

        for name, metric_name, value, _ in eval_result:
            if metric_name == self.metric:
                if name == "train":
                    train_val = value
                elif name == "val":
                    val_val = value

        if train_val is None or val_val is None:
            return

        self.train_history.append(train_val)
        self.val_history.append(val_val)

        if val_val < self.best_val:
            self.best_val = val_val
            self.best_epoch = env.iteration
            self._counter = 0

        gap = abs(train_val - val_val) / max(abs(val_val), 1e-8)
        self.gap_history.append(gap)

        if gap > self.gap_threshold:
            self._counter += 1
            if self._counter >= self.patience:
                self.stopped_epoch = env.iteration
                logger.warning("Gap detected! train=%.1f val=%.1f gap=%.2f > %.2f (epoch %d)",
                               train_val, val_val, gap, self.gap_threshold, env.iteration)
                raise lgb.callback.EarlyStopException(env.iteration, f"gap={gap:.2f}")

    def summary(self) -> Dict:
        return {
            "stopped_epoch": self.stopped_epoch,
            "best_epoch": self.best_epoch,
            "best_val": round(self.best_val, 2),
            "final_gap": round(self.gap_history[-1], 4) if self.gap_history else 0,
            "max_gap": round(max(self.gap_history), 4) if self.gap_history else 0,
            "stopped_by_gap": self.stopped_epoch > 0,
        }


def create_overfit_callbacks(gap_threshold: float = 0.5, patience: int = 10) -> List:
    """Create a complete set of overfitting-prevention callbacks.

    Returns:
        List of callbacks: [GapMonitor, early_stopping, log_evaluation]
    """
    monitor = GapMonitor(gap_threshold=gap_threshold, patience=patience)
    return [
        monitor.callback,
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=50),
    ]

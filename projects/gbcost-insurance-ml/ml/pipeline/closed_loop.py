"""Closed-Loop Monitor — detect prediction drift and trigger auto-retraining.

Monitors prediction quality over time, detects when the model's performance
degrades (drift), and triggers automatic retraining.

Usage:
    monitor = ClosedLoopMonitor()
    drift = monitor.check_prediction_drift(y_true, y_pred)
    if monitor.should_retrain(drift):
        run_full_training()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ClosedLoopMonitor:
    """Monitor prediction quality and trigger retraining on drift."""

    def __init__(
        self,
        model_dir: str = "models",
        drift_gini_threshold: float = 0.15,
        drift_error_threshold: float = 0.20,
        check_history: int = 3,
    ):
        self.model_dir = Path(model_dir)
        self.drift_gini_threshold = drift_gini_threshold
        self.drift_error_threshold = drift_error_threshold
        self.check_history = check_history

    def check_prediction_drift(
        self, y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Check if predictions have drifted from expected accuracy.

        Returns:
            Dict with drift_detected, severity, metrics, and recommended action.
        """
        from ml.evaluate.metrics import evaluate_predictions

        metrics = evaluate_predictions(y_true, y_pred)
        current_gini = metrics.get("gini", 0)
        current_error = metrics.get("total_error_pct", 100)

        # Load baseline from registry
        registry = self._load_registry()
        versions = registry.get("versions", [])
        if not versions:
            return {"drift_detected": False, "metrics": metrics, "note": "no_baseline"}

        # Use average of last N versions as baseline
        recent = versions[-self.check_history:]
        baseline_gini = np.mean([v["metrics"].get("gini", 0) for v in recent])
        baseline_error = np.mean([v["metrics"].get("total_error_pct", 100) for v in recent])

        gini_drift = (baseline_gini - current_gini) / max(baseline_gini, 0.01)
        error_drift = (current_error - baseline_error) / max(baseline_error, 0.01)

        drift_detected = gini_drift > self.drift_gini_threshold or error_drift > self.drift_error_threshold

        if drift_detected:
            if gini_drift > 0.3 or error_drift > 0.4:
                severity = "critical"
                action = "retrain_now"
            elif gini_drift > self.drift_gini_threshold:
                severity = "warning"
                action = "schedule_retrain"
            else:
                severity = "info"
                action = "monitor"
        else:
            severity = "normal"
            action = "none"

        result = {
            "drift_detected": drift_detected,
            "severity": severity,
            "action": action,
            "metrics": metrics,
            "baseline": {
                "gini": round(baseline_gini, 4),
                "error_pct": round(baseline_error, 1),
                "versions_checked": len(recent),
            },
            "drift": {
                "gini_drift_pct": round(gini_drift * 100, 1),
                "error_drift_pct": round(error_drift * 100, 1),
            },
            "checked_at": datetime.now().isoformat(),
        }

        if drift_detected:
            logger.warning("DRIFT: Gini %.4f→%.4f (%.1f%%), Error %.1f→%.1f%% (%.1f%%) — %s",
                           baseline_gini, current_gini, gini_drift * 100,
                           baseline_error, current_error, error_drift * 100, action)

        return result

    def should_retrain(self, drift_result: Dict) -> bool:
        return drift_result.get("action") == "retrain_now"

    def _load_registry(self) -> dict:
        path = self.model_dir / "model_registry.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"versions": []}

    def log_drift_history(self, drift_result: Dict, log_dir: str = "reports/ml") -> None:
        """Append drift check result to history log."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "drift_history.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(drift_result, ensure_ascii=False) + "\n")

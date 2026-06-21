"""Probability calibration for LightGBM predictions.

Reference: Insurance Risk Modeling — Isotonic calibration reduces Brier score by 47%.

Adds Isotonic/Platt calibration on top of L1-A predictions to produce
well-calibrated probability estimates for risk assessment.

Usage:
    cal = ProbabilityCalibrator(method="isotonic")
    cal.fit(y_true, y_pred_raw)
    cal.save("models/calibrated_predictor.pkl")
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrates LightGBM predictions using Isotonic or Platt scaling.

    Unlike GroupCalibrator (which corrects total amount bias via multiplicative
    factors), this calibrates individual probability estimates for reliability.
    """

    def __init__(self, method: Literal["isotonic", "platt"] = "isotonic"):
        self.method = method
        self.calibrator: Optional[IsotonicRegression] = None
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> "ProbabilityCalibrator":
        mask = np.isfinite(y_true) & np.isfinite(y_pred_raw)
        yt, yp = y_true[mask], y_pred_raw[mask]

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        else:
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()

        self.calibrator.fit(yp.reshape(-1, 1), yt)
        self._fitted = True

        pre_brier = brier_score_loss(yt, yp)
        yp_cal = self.calibrator.predict(yp.reshape(-1, 1))
        post_brier = brier_score_loss(yt, np.clip(yp_cal, 0, 1))
        self.improvement_pct = (pre_brier - post_brier) / max(pre_brier, 1e-8) * 100

        logger.info("Calibration: %s | Brier %.4f→%.4f (%.1f%% improvement)",
                     self.method, pre_brier, post_brier, self.improvement_pct)
        return self

    def calibrate(self, y_pred_raw: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call .fit() first.")
        return np.clip(self.calibrator.predict(y_pred_raw.reshape(-1, 1)), 0, 1)

    def get_calibration_curve(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> Dict:
        """Generate calibration curve data for plotting."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred_raw)
        prob_true, prob_pred = calibration_curve(y_true[mask], y_pred_raw[mask], n_bins=10)
        return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"method": self.method, "calibrator": self.calibrator,
                "improvement_pct": getattr(self, "improvement_pct", 0)}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("ProbabilityCalibrator saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        with open(path, "rb") as f:
            data = pickle.load(f)
        cal = cls(method=data["method"])
        cal.calibrator = data["calibrator"]
        cal._fitted = True
        cal.improvement_pct = data.get("improvement_pct", 0)
        return cal


def evaluate_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute calibration metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    brier = brier_score_loss(yt, yp)

    # ECE (Expected Calibration Error)
    prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10, strategy="uniform")
    bin_counts = np.histogram(yp, bins=10)[0]
    bin_counts = bin_counts[:len(prob_true)]  # Align lengths
    ece = np.sum(np.abs(prob_pred - prob_true) * bin_counts / bin_counts.sum())

    return {
        "brier_score": round(brier, 4),
        "ece": round(float(ece), 4),
        "n_samples": int(mask.sum()),
    }

"""Ensemble predictor — weighted fusion of LightGBM + XGBoost.

Combines predictions from multiple Tweedie regressors for improved accuracy.
Reference: Insurance-Claims-Prediction-ML uses LightGBM + XGBoost ensemble.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Weighted ensemble of multiple regressors."""

    def __init__(self, models: List = None, weights: Optional[List[float]] = None):
        self.models = models or []
        self.weights = weights or []
        self.feature_names: List[str] = []

    def add_model(self, model, weight: float = 1.0):
        self.models.append(model)
        self.weights.append(weight)
        if hasattr(model, 'feature_names'):
            if not self.feature_names:
                self.feature_names = model.feature_names

    def predict(self, df: pl.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        if not self.models:
            raise RuntimeError("No models in ensemble.")
        fc = feature_cols or self.feature_names
        preds = []
        for model in self.models:
            p = model.predict(df, fc)
            preds.append(p)
        weights = np.array(self.weights) / sum(self.weights)
        return np.average(np.array(preds), axis=0, weights=weights)

    def predict_proba(self, df: pl.DataFrame, feature_cols=None) -> np.ndarray:
        """For L1-B ensemble — average of probabilities."""
        preds = [m.predict_proba(df, feature_cols) for m in self.models if hasattr(m, 'predict_proba')]
        return np.mean(preds, axis=0) if preds else np.zeros(len(df))

    def save(self, model_dir: str | Path, name: str = "ensemble") -> Path:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        meta = {"n_models": len(self.models), "weights": self.weights, "feature_names": self.feature_names}
        (model_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return model_dir

    @classmethod
    def load(cls, model_dir: str | Path, name: str = "ensemble") -> "EnsemblePredictor":
        from ml.models.amount_predictor import AmountPredictor
        from ml.models.xgboost_predictor import XGBoostPredictor

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / f"{name}_meta.json").read_text(encoding="utf-8"))

        ensemble = cls(weights=meta["weights"])
        ensemble.feature_names = meta["feature_names"]

        # Load each model based on available files
        for i in range(meta["n_models"]):
            lgb_path = model_dir / f"l1a_amount.txt"
            xgb_path = model_dir / f"xgb_amount.json"
            if lgb_path.exists():
                ensemble.add_model(AmountPredictor.load(model_dir, "l1a_amount"))
                break
            if xgb_path.exists():
                ensemble.add_model(XGBoostPredictor.load(model_dir, "xgb_amount"))
                break
        return ensemble


def optimize_ensemble_weights(
    models: List, val_df: pl.DataFrame, feature_cols: List[str], y_true: np.ndarray,
) -> List[float]:
    """Find optimal ensemble weights by minimizing MAE on validation set."""
    preds = [m.predict(val_df, feature_cols) for m in models]
    n = len(models)

    best_mae = float("inf")
    best_weights = [1.0 / n] * n

    # Grid search over weight combinations
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1 - w1, 11):
            w3 = 1 - w1 - w2
            ws = [w1, w2, w3][:n]
            if sum(ws) == 0:
                continue
            ws_norm = np.array(ws) / sum(ws)
            ensemble_pred = np.average(preds, axis=0, weights=ws_norm)
            mae = np.mean(np.abs(y_true - ensemble_pred))
            if mae < best_mae:
                best_mae = mae
                best_weights = ws_norm.tolist()

    logger.info("Ensemble weights: %s (MAE=%.0f)", [f"{w:.2f}" for w in best_weights], best_mae)
    return best_weights

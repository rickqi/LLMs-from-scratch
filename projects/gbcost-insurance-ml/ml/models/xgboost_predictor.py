"""XGBoost Tweedie regression — alternative to LightGBM for claim amount prediction.

Reference: Insurance Risk Modeling project uses Tweedie XGBoost + Isotonic LightGBM.
Provides a drop-in alternative for model comparison and ensemble.

XGBoost advantages over LightGBM for Tweedie:
  - Better handling of extreme values (more regularized)
  - Native GPU support (DMatrix)
  - Different tree growth strategy (level-wise vs leaf-wise)

Usage:
    xgb = XGBoostPredictor(tweedie_power=1.35)
    xgb.train(train_df, val_df, feature_cols, categorical_cols)
    y_pred = xgb.predict(test_df)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost-based claim amount predictor with Tweedie objective."""

    def __init__(
        self,
        tweedie_power: float = 1.35,
        n_estimators: int = 800,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 50,
        early_stopping_rounds: int = 50,
        seed: int = 42,
    ) -> None:
        self.params: Dict[str, Any] = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": tweedie_power,
            "eval_metric": ["mae", "rmse"],
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "verbosity": 0,
            "seed": seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Any = None
        self.best_iteration: int = 0
        self.feature_names: List[str] = []
        self.train_time_sec: float = 0.0
        self.train_eval: Dict[str, float] = {}

    def _to_numpy(self, df: pl.DataFrame, feature_cols: List[str], target_col: str = "y_raw"):
        X = np.zeros((len(df), len(feature_cols)), dtype=np.float64)
        for i, col in enumerate(feature_cols):
            try:
                X[:, i] = df[col].to_numpy().astype(np.float64)
            except Exception:
                try:
                    X[:, i] = df[col].cast(pl.Utf8).hash(42).mod(10 ** 9).cast(pl.Float64).to_numpy()
                except Exception:
                    X[:, i] = 0.0
        y = df[target_col].to_numpy().astype(np.float64) if target_col in df.columns else None
        return X, y

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str] = None,
        target_col: str = "y_raw",
    ) -> Dict[str, Any]:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. pip install xgboost")

        self.feature_names = feature_cols
        X_train, y_train = self._to_numpy(train_df, feature_cols, target_col)
        X_val, y_val = self._to_numpy(val_df, feature_cols, target_col)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        logger.info("Training XGBoost (tweedie_power=%.2f, n_estimators=%d)...",
                     self.params["tweedie_variance_power"], self.n_estimators)

        t0 = time.time()
        self.model = xgb.train(
            self.params, dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=50,
        )
        self.train_time_sec = time.time() - t0
        self.best_iteration = self.model.best_iteration

        self.train_eval = {
            "best_iteration": self.best_iteration,
            "train_time_sec": round(self.train_time_sec, 1),
        }
        if hasattr(self.model, "best_score"):
            self.train_eval["val_rmse"] = round(float(self.model.best_score), 2)

        logger.info("XGBoost complete: best_iter=%d, time=%.1fs",
                     self.best_iteration, self.train_time_sec)
        return self.train_eval

    def predict(self, df: pl.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        import xgboost as xgb

        if self.model is None:
            raise RuntimeError("Model not trained.")
        fc = feature_cols or self.feature_names
        X, _ = self._to_numpy(df, fc)
        dtest = xgb.DMatrix(X)
        return np.clip(self.model.predict(dtest, ntree_limit=self.best_iteration), 0, None)

    def save(self, model_dir: str | Path, name: str = "xgb_amount") -> Path:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{name}.json"
        self.model.save_model(str(model_path))

        meta = {
            "params": self.params, "n_estimators": self.n_estimators,
            "best_iteration": self.best_iteration, "feature_names": self.feature_names,
            "train_eval": self.train_eval, "train_time_sec": self.train_time_sec,
        }
        (model_dir / f"{name}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("XGBoost saved: %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path, name: str = "xgb_amount") -> "XGBoostPredictor":
        import xgboost as xgb

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / f"{name}_meta.json").read_text(encoding="utf-8"))

        p = cls(
            tweedie_power=meta["params"]["tweedie_variance_power"],
            n_estimators=meta["n_estimators"],
            max_depth=meta["params"]["max_depth"],
            learning_rate=meta["params"]["learning_rate"],
            subsample=meta["params"]["subsample"],
            colsample_bytree=meta["params"]["colsample_bytree"],
            reg_alpha=meta["params"]["reg_alpha"],
            reg_lambda=meta["params"]["reg_lambda"],
            min_child_weight=meta["params"]["min_child_weight"],
        )
        p.model = xgb.Booster()
        p.model.load_model(str(model_dir / f"{name}.json"))
        p.best_iteration = meta["best_iteration"]
        p.feature_names = meta["feature_names"]
        p.train_eval = meta.get("train_eval", {})
        return p

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if self.model is None:
            return {}
        scores = self.model.get_score(importance_type=importance_type)
        # Map f0, f1, ... to feature names
        result = {}
        for k, v in scores.items():
            idx = int(k.replace("f", "")) if k.startswith("f") else None
            name = self.feature_names[idx] if idx is not None and idx < len(self.feature_names) else k
            result[name] = float(v)
        return dict(sorted(result.items(), key=lambda x: -x[1]))

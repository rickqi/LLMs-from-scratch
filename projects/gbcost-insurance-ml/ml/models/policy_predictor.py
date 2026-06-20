"""L2: Policy-level loss ratio prediction.

Aggregates L1 case-level predictions to policy level, then trains a regression
model to predict policy loss ratio or total claim amount.

Complements L1-A (case-level amount) with L2 (policy-level) for:
  - Underwriting decisions
  - Pricing and reserve setting
  - Portfolio risk assessment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class PolicyPredictor:
    """LightGBM regression for policy-level loss ratio prediction."""

    def __init__(
        self,
        objective: str = "regression",
        n_estimators: int = 500,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_samples: int = 20,
        early_stopping_rounds: int = 30,
        seed: int = 42,
    ) -> None:
        self.params: Dict[str, Any] = {
            "objective": objective,
            "metric": ["mae", "rmse"],
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "subsample_freq": 1,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_samples": min_child_samples,
            "verbose": -1,
            "seed": seed,
            "n_jobs": -1,
        }
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[lgb.Booster] = None
        self.best_iteration: int = 0
        self.feature_names: List[str] = []
        self.train_eval: Dict[str, float] = {}

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str = "loss_ratio",
    ) -> Dict[str, Any]:
        X_train = train_df.select(feature_cols).to_pandas()
        y_train = train_df[target_col].to_numpy()
        X_val = val_df.select(feature_cols).to_pandas()
        y_val = val_df[target_col].to_numpy()

        for col in X_train.columns:
            dtype = str(X_train[col].dtype)
            if dtype not in ("int8", "int16", "int32", "int64",
                             "float32", "float64", "bool", "category"):
                X_train[col] = X_train[col].astype("category")
            dtype_v = str(X_val[col].dtype)
            if dtype_v not in ("int8", "int16", "int32", "int64",
                               "float32", "float64", "bool", "category"):
                X_val[col] = X_val[col].astype("category")

        self.feature_names = feature_cols

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols,
                             categorical_feature=[], free_raw_data=True)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        logger.info("Training L2 PolicyPredictor (%d policies, %d features)...",
                     len(train_df), len(feature_cols))

        self.model = lgb.train(
            self.params, dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(20),
            ],
        )
        self.best_iteration = self.model.best_iteration

        self.train_eval = {"best_iteration": self.best_iteration}
        if self.model.best_score:
            for name, score in self.model.best_score.items():
                for metric, val in score.items():
                    self.train_eval[f"{name}_{metric}"] = round(val, 4)

        logger.info("L2 training complete: best_iter=%d, val_mae=%.4f",
                     self.best_iteration,
                     self.train_eval.get("val_mae", float("nan")))

        return self.train_eval

    def predict(self, df: pl.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        fc = feature_cols or self.feature_names
        X = df.select(fc).to_pandas()
        for col in X.columns:
            dtype = str(X[col].dtype)
            if dtype not in ("int8","int16","int32","int64","float32","float64","bool"):
                X[col] = X[col].astype(float)
        return np.clip(
            self.model.predict(X, num_iteration=self.best_iteration,
                               predict_disable_shape_check=True),
            0, None)

    def save(self, model_dir: str | Path, name: str = "l2_policy") -> Path:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{name}.txt"
        self.model.save_model(str(model_path))
        meta = {"params": self.params, "n_estimators": self.n_estimators,
                "best_iteration": self.best_iteration, "feature_names": self.feature_names,
                "train_eval": self.train_eval}
        (model_dir / f"{name}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("L2 saved: %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path, name: str = "l2_policy") -> "PolicyPredictor":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / f"{name}_meta.json").read_text(encoding="utf-8"))
        p = cls(
            objective=meta["params"]["objective"],
            n_estimators=meta["n_estimators"],
            num_leaves=meta["params"]["num_leaves"],
            learning_rate=meta["params"]["learning_rate"],
            subsample=meta["params"]["subsample"],
            colsample_bytree=meta["params"]["colsample_bytree"],
            reg_alpha=meta["params"]["reg_alpha"],
            reg_lambda=meta["params"]["reg_lambda"],
            min_child_samples=meta["params"]["min_child_samples"],
        )
        p.model = lgb.Booster(model_file=str(model_dir / f"{name}.txt"))
        p.best_iteration = meta["best_iteration"]
        p.feature_names = meta["feature_names"]
        p.train_eval = meta.get("train_eval", {})
        return p

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if self.model is None:
            return {}
        imp = self.model.feature_importance(importance_type=importance_type)
        return dict(sorted(zip(self.feature_names, imp.tolist()), key=lambda x: -x[1]))

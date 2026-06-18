"""L1-A: Claim amount prediction using LightGBM.

Supports Tweedie / Gamma / Regression objectives, quantile prediction,
and Optuna hyperparameter tuning.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class AmountPredictor:
    """LightGBM-based claim amount predictor."""

    def __init__(
        self,
        objective: str = "tweedie",
        tweedie_power: float = 1.5,
        n_estimators: int = 800,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_samples: int = 50,
        early_stopping_rounds: int = 50,
        log_transform: bool = True,
        seed: int = 42,
    ) -> None:
        self.params = {
            "objective": objective,
            "tweedie_variance_power": tweedie_power,
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
            "force_col_wise": True,
        }
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.log_transform = log_transform
        self.model: Optional[lgb.Booster] = None
        self.best_iteration: int = 0
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.category_mappings: Dict[str, List[str]] = {}
        self.train_time_sec: float = 0.0
        self.train_eval: Dict[str, float] = {}

    def _prepare_data(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str],
        target_col: str = "y_log",
    ) -> Tuple["pd.DataFrame", np.ndarray]:
        """Convert Polars DataFrame to LightGBM-compatible pandas DataFrame."""
        X_df = df.select(feature_cols)
        y = df[target_col].to_numpy()

        X_pd = X_df.to_pandas()

        # Cast ALL non-numeric columns to pandas category type
        # LightGBM requires int, float, bool, or category
        for col in X_pd.columns:
            dtype = str(X_pd[col].dtype)
            if dtype not in ("int8", "int16", "int32", "int64",
                             "float32", "float64", "bool", "category"):
                X_pd[col] = X_pd[col].astype("category")

        return X_pd, y

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str],
        target_col: str = "y_log",
    ) -> Dict[str, Any]:
        """Train the LightGBM model.

        Args:
            train_df: Training data (Polars DataFrame)
            val_df: Validation data (Polars DataFrame)
            feature_cols: List of feature column names
            categorical_cols: Subset of feature_cols that are categorical
            target_col: Target column name ('y_log' or 'y_raw')

        Returns:
            Training summary dict
        """
        logger.info("Preparing training data...")
        X_train, y_train = self._prepare_data(
            train_df, feature_cols, categorical_cols, target_col
        )
        X_val, y_val = self._prepare_data(
            val_df, feature_cols, categorical_cols, target_col
        )

        cat_valid = [c for c in categorical_cols if c in feature_cols]
        self.feature_names = feature_cols
        self.categorical_features = cat_valid

        # Persist category mappings for consistent encoding during prediction
        self.category_mappings = {}
        for col in cat_valid:
            series = X_train[col]
            if hasattr(series, 'cat'):
                # Store as strings to ensure JSON serializable
                self.category_mappings[col] = [str(c) for c in series.cat.categories]
        logger.info("Category mappings saved: %d cols (%d total categories)",
                     len(self.category_mappings),
                     sum(len(v) for v in self.category_mappings.values()))

        logger.info("Building LightGBM Datasets...")
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_cols,
            categorical_feature=cat_valid if cat_valid else "auto",
            free_raw_data=True,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            reference=dtrain,
        )

        logger.info("Training LightGBM (objective=%s, n_estimators=%d)...",
                     self.params["objective"], self.n_estimators)

        t0 = time.time()
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(50),
            ],
        )
        self.train_time_sec = time.time() - t0
        self.best_iteration = self.model.best_iteration

        # Collect eval metrics
        self.train_eval = {
            "best_iteration": self.best_iteration,
            "train_time_sec": round(self.train_time_sec, 1),
        }
        if self.model.best_score:
            for name, score in self.model.best_score.items():
                for metric, val in score.items():
                    self.train_eval[f"{name}_{metric}"] = round(val, 4)

        logger.info("Training complete: best_iter=%d, time=%.1fs",
                     self.best_iteration, self.train_time_sec)

        return self.train_eval

    def predict(
        self,
        df: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        inverse_log: bool = True,
    ) -> np.ndarray:
        """Predict claim amounts.

        Args:
            df: Data to predict on
            feature_cols: Override feature columns (default: self.feature_names)
            categorical_cols: Override categorical columns
            inverse_log: If True, apply expm1 to convert from log space

        Returns:
            Array of predicted amounts
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        fc = feature_cols or self.feature_names
        cc = categorical_cols or self.categorical_features

        X_pred = df.select(fc).to_pandas()

        # Apply persisted category mappings for consistent encoding.
        # Without this, Polars Categorical creates independent code tables
        # per DataFrame, causing prediction-quality collapse (R2 0.91→0.02).
        import pandas as _pd

        if self.category_mappings:
            for col, known_cats in self.category_mappings.items():
                if col in X_pred.columns:
                    # Convert column to string, then apply known categories.
                    # Values not in known_cats get code=-1 (LightGBM handles unknown).
                    X_pred[col] = _pd.Categorical(
                        X_pred[col].astype(str),
                        categories=known_cats,
                    )

        # For non-category columns, cast to appropriate dtype
        for col in X_pred.columns:
            dtype = str(X_pred[col].dtype)
            if dtype not in ("int8", "int16", "int32", "int64",
                             "float32", "float64", "bool", "category"):
                if dtype == "bool":
                    X_pred[col] = X_pred[col].astype(np.int8)
                else:
                    # Remaining non-category columns -> category
                    X_pred[col] = X_pred[col].astype("category")

        # Convert to numpy float64 matrix (category → integer codes)
        X_np = np.empty((len(X_pred), len(X_pred.columns)), dtype=np.float64)
        for i, col in enumerate(X_pred.columns):
            series = X_pred[col]
            if str(series.dtype) == "category":
                X_np[:, i] = _pd.Categorical(series).codes.astype(np.float64)
            else:
                X_np[:, i] = series.to_numpy(dtype=np.float64, na_value=0.0)

        preds_log = self.model.predict(X_np, num_iteration=self.best_iteration)

        if inverse_log and self.log_transform:
            return np.expm1(preds_log)
        return preds_log

    def save(self, model_dir: str | Path, name: str = "l1a_amount") -> Path:
        """Save model and metadata."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No model to save.")

        model_path = model_dir / f"{name}.txt"
        self.model.save_model(str(model_path))

        meta = {
            "params": self.params,
            "n_estimators": self.n_estimators,
            "best_iteration": self.best_iteration,
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "category_mappings": self.category_mappings,
            "log_transform": self.log_transform,
            "train_eval": self.train_eval,
            "train_time_sec": self.train_time_sec,
        }
        meta_path = model_dir / f"{name}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Model saved: %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path, name: str = "l1a_amount") -> "AmountPredictor":
        """Load model and metadata."""
        model_dir = Path(model_dir)
        model_path = model_dir / f"{name}.txt"
        meta_path = model_dir / f"{name}_meta.json"

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        predictor = cls(
            objective=meta["params"]["objective"],
            tweedie_power=meta["params"].get("tweedie_variance_power", 1.5),
            n_estimators=meta["n_estimators"],
            num_leaves=meta["params"]["num_leaves"],
            learning_rate=meta["params"]["learning_rate"],
            subsample=meta["params"]["subsample"],
            colsample_bytree=meta["params"]["colsample_bytree"],
            reg_alpha=meta["params"]["reg_alpha"],
            reg_lambda=meta["params"]["reg_lambda"],
            min_child_samples=meta["params"]["min_child_samples"],
            log_transform=meta["log_transform"],
        )
        predictor.model = lgb.Booster(model_file=str(model_path))
        predictor.best_iteration = meta["best_iteration"]
        predictor.feature_names = meta["feature_names"]
        predictor.categorical_features = meta["categorical_features"]
        predictor.category_mappings = meta.get("category_mappings", {})
        predictor.train_eval = meta.get("train_eval", {})

        logger.info("Model loaded: %s (best_iter=%d)", model_path, predictor.best_iteration)
        return predictor

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """Get feature importance sorted by value."""
        if self.model is None:
            return {}
        imp = self.model.feature_importance(importance_type=importance_type)
        result = dict(zip(self.feature_names, imp.tolist()))
        return dict(sorted(result.items(), key=lambda x: -x[1]))


def train_quantile_models(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    quantiles: List[float] = None,
    target_col: str = "y_log",
    n_estimators: int = 500,
    model_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Train separate quantile regression models for uncertainty estimation.

    Args:
        quantiles: List of quantile levels (default: [0.05, 0.50, 0.95])

    Returns:
        Dict with quantile predictions and model paths
    """
    if quantiles is None:
        quantiles = [0.05, 0.50, 0.95]

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df[target_col].to_numpy()

    # Cast ALL non-numeric columns to category (same as AmountPredictor._prepare_data)
    for col in X_train.columns:
        dtype = str(X_train[col].dtype)
        if dtype not in ("int8", "int16", "int32", "int64",
                         "float32", "float64", "bool", "category"):
            X_train[col] = X_train[col].astype("category")

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)

    results = {}
    for q in quantiles:
        name = f"p{int(q * 100):02d}"
        logger.info("Training quantile model: %s (q=%.2f)", name, q)

        params = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=[lgb.log_evaluation(100)],
        )

        if model_dir:
            path = Path(model_dir) / f"l1a_{name}.txt"
            model.save_model(str(path))
            results[name] = {"model_path": str(path), "quantile": q}
        else:
            results[name] = {"quantile": q, "model": model}

        logger.info("  %s trained (%d iterations)", name, model.current_iteration())

    return results

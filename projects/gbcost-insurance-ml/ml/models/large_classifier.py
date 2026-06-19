"""L1-B: Large claim binary classifier using LightGBM.

Complementary to L1-A (Tweedie regression). Identifies claims likely to exceed
the P95 threshold, enabling targeted investigation and boosting large-claim recall.

Usage:
    classifier = LargeClaimClassifier()
    classifier.train(X_train, y_train, X_val, y_val, feature_names, cat_features)
    probas = classifier.predict_proba(X_test)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class LargeClaimClassifier:
    """Binary classifier: predicts probability of claim exceeding a threshold."""

    def __init__(
        self,
        threshold_value: float = 24600.0,
        n_estimators: int = 500,
        num_leaves: int = 31,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_samples: int = 50,
        scale_pos_weight: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        self.threshold = threshold_value
        self.n_estimators = n_estimators
        self.params: Dict[str, Any] = {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
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
        if scale_pos_weight is not None:
            self.params["scale_pos_weight"] = scale_pos_weight

        self.model: Optional[lgb.Booster] = None
        self.best_iteration: int = 0
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.train_eval: Dict[str, float] = {}

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str],
        y_train: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        exclude_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if exclude_cols:
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        if y_train is None:
            y_train = (train_df["y_raw"].to_numpy() > self.threshold).astype(np.float64)
        if y_val is None:
            y_val = (val_df["y_raw"].to_numpy() > self.threshold).astype(np.float64)

        from ml.models.amount_predictor import AmountPredictor
        dummy = AmountPredictor(log_transform=False)
        target_col = "y_raw"
        X_train_pd, _ = dummy._prepare_data(train_df, feature_cols, categorical_cols, target_col)
        X_val_pd, _ = dummy._prepare_data(val_df, feature_cols, categorical_cols, target_col)

        self.feature_names = feature_cols
        cat_valid = [c for c in categorical_cols if c in feature_cols]
        self.categorical_features = cat_valid

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if "scale_pos_weight" not in self.params or self.params["scale_pos_weight"] is None:
            self.params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            logger.info("Auto weight: scale_pos_weight=%.1f (neg=%d, pos=%d)",
                         self.params["scale_pos_weight"], n_neg, n_pos)

        dtrain = lgb.Dataset(
            X_train_pd, label=y_train,
            feature_name=feature_cols,
            categorical_feature=cat_valid if cat_valid else "auto",
            free_raw_data=True,
        )
        dval = lgb.Dataset(X_val_pd, label=y_val, reference=dtrain)

        logger.info("Training L1-B classifier (threshold=%.0f, n_estimators=%d)...",
                     self.threshold, self.n_estimators)

        self.model = lgb.train(
            self.params, dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(30, verbose=True),
                lgb.log_evaluation(50),
            ],
        )
        self.best_iteration = self.model.best_iteration

        self.train_eval = {"best_iteration": self.best_iteration}
        if self.model.best_score:
            for name, score in self.model.best_score.items():
                for metric, val in score.items():
                    self.train_eval[f"{name}_{metric}"] = round(val, 4)

        logger.info("L1-B complete: best_iter=%d, val_auc=%.4f, val_ap=%.4f",
                     self.best_iteration,
                     self.train_eval.get("val_auc", 0),
                     self.train_eval.get("val_average_precision", 0))

        return self.train_eval

    def predict_proba(self, df: pl.DataFrame, feature_cols=None, categorical_cols=None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained.")

        fc = feature_cols or self.feature_names
        X_pd = df.select(fc).to_pandas()

        for col in X_pd.columns:
            dtype = str(X_pd[col].dtype)
            if dtype not in ("int8", "int16", "int32", "int64",
                             "float32", "float64", "bool", "category"):
                X_pd[col] = X_pd[col].astype("category")

        import pandas as _pd
        X_np = np.empty((len(X_pd), len(X_pd.columns)), dtype=np.float64)
        for i, col in enumerate(X_pd.columns):
            series = X_pd[col]
            if str(series.dtype) == "category":
                X_np[:, i] = _pd.Categorical(series).codes.astype(np.float64)
            else:
                X_np[:, i] = series.to_numpy(dtype=np.float64, na_value=0.0)

        return self.model.predict(X_np, num_iteration=self.best_iteration)

    def save(self, model_dir: str | Path, name: str = "l1b_classifier") -> Path:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No model to save.")

        model_path = model_dir / f"{name}.txt"
        self.model.save_model(str(model_path))

        meta = {
            "threshold": self.threshold,
            "params": self.params,
            "n_estimators": self.n_estimators,
            "best_iteration": self.best_iteration,
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "train_eval": self.train_eval,
        }
        (model_dir / f"{name}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("L1-B saved: %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path, name: str = "l1b_classifier") -> "LargeClaimClassifier":
        model_dir = Path(model_dir)
        model_path = model_dir / f"{name}.txt"
        meta_path = model_dir / f"{name}_meta.json"

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        classifier = cls(
            threshold_value=meta["threshold"],
            n_estimators=meta["n_estimators"],
            num_leaves=meta["params"]["num_leaves"],
            learning_rate=meta["params"]["learning_rate"],
            subsample=meta["params"]["subsample"],
            colsample_bytree=meta["params"]["colsample_bytree"],
            reg_alpha=meta["params"]["reg_alpha"],
            reg_lambda=meta["params"]["reg_lambda"],
            min_child_samples=meta["params"]["min_child_samples"],
        )
        classifier.model = lgb.Booster(model_file=str(model_path))
        classifier.best_iteration = meta["best_iteration"]
        classifier.feature_names = meta["feature_names"]
        classifier.categorical_features = meta["categorical_features"]
        classifier.train_eval = meta.get("train_eval", {})

        logger.info("L1-B loaded: %s (best_iter=%d)", model_path, classifier.best_iteration)
        return classifier

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if self.model is None:
            return {}
        imp = self.model.feature_importance(importance_type=importance_type)
        result = dict(zip(self.feature_names, imp.tolist()))
        return dict(sorted(result.items(), key=lambda x: -x[1]))


def ensemble_predict(
    amount_predictor,
    large_classifier: LargeClaimClassifier,
    df: pl.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    boost_factor: float = 0.5,
) -> np.ndarray:
    base_pred = amount_predictor.predict(df, feature_cols, categorical_cols)
    large_proba = large_classifier.predict_proba(df, feature_cols, categorical_cols)

    adjusted = base_pred.copy()
    boost_mask = large_proba > 0.5
    adjusted[boost_mask] *= (1.0 + large_proba[boost_mask] * boost_factor)

    return adjusted

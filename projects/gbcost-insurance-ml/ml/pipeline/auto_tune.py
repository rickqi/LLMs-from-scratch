"""Auto hyperparameter tuning with Optuna + gap detection.

Drop-in wrapper: runs Optuna study and selects best model using
train/val gap monitoring to prevent overfitting during tuning.

Usage:
    python -m ml.pipeline.auto_tune --quick   # 20 trials, fast preview
    python -m ml.pipeline.auto_tune           # 50 trials, full search
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger("ml.auto_tune")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auto hyperparameter tuning")
    parser.add_argument("--config", default="ml/config_ml.yaml")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Fast mode: 20 trials")
    parser.add_argument("--output", default="models/optuna")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    if args.quick:
        args.trials = 20
        logger.info("Quick mode: %d trials", args.trials)

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data from parquet cache
    from ml.data.feature_store import build_features, get_feature_columns
    with open("data/ml_cache/global_stats.pkl", "rb") as f:
        raw = pickle.load(f)
    global_stats = {k: (pl.DataFrame(v) if isinstance(v, dict) else v) for k, v in raw.items()}

    cache_lf = pl.scan_parquet("data/ml_cache/train_cache.parquet")
    flf = build_features(cache_lf, global_stats, config["features"]["categorical_cols"], log_transform=False)
    all_f = get_feature_columns(flf, config["features"]["exclude_cols"])

    # Sample for tuning speed
    df = flf.select(all_f + ["y_raw", "case_no"]).filter(
        pl.col("case_no").hash(seed=42) % 100 < 30
    ).collect(engine="streaming")

    split_n = int(len(df) * 0.8)
    train_df = df.slice(0, split_n)
    val_df = df.slice(split_n, len(df) - split_n)
    logger.info("Tune data: train=%s val=%s features=%d", f"{len(train_df):,}", f"{len(val_df):,}", len(all_f))

    # Build numpy matrices
    X_tr = _to_numpy(train_df, all_f)
    y_tr = train_df["y_raw"].to_numpy().astype(np.float64)
    X_vl = _to_numpy(val_df, all_f)
    y_vl = val_df["y_raw"].to_numpy().astype(np.float64)
    X_tr, X_vl = np.nan_to_num(X_tr, 0), np.nan_to_num(X_vl, 0)

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=all_f, free_raw_data=True)
    dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

    # Optuna study
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_dir = Path(args.output)
    db_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="auto_tune_v1",
        direction="minimize",
        storage=f"sqlite:///{db_dir / 'optuna_study.db'}",
        load_if_exists=True,
    )

    def objective(trial):
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": trial.suggest_float("tweedie_power", 1.1, 1.7, step=0.05),
            "metric": ["l1", "rmse"],
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=32),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, step=20),
            "verbose": -1, "seed": 42, "n_jobs": -1, "force_col_wise": True,
            "feature_pre_filter": False,
        }

        from ml.models.gap_monitor import GapMonitor
        monitor = GapMonitor(gap_threshold=0.5, patience=10)

        try:
            model = lgb.train(params, dtrain, num_boost_round=800,
                valid_sets=[dtrain, dval], valid_names=["train", "val"],
                callbacks=[monitor.callback, lgb.log_evaluation(period=0)])
        except lgb.callback.EarlyStopException:
            pass
        except Exception:
            return float("inf")

        val_l1 = model.best_score.get("val", {}).get("l1", float("inf"))
        gap = monitor.summary()
        if gap.get("stopped_by_gap"):
            val_l1 *= 1.5  # Penalize overfitting trials

        trial.set_user_attr("best_iter", model.best_iteration)
        trial.set_user_attr("max_gap", gap.get("max_gap", 0))
        return val_l1

    logger.info("Starting auto-tune: %d trials...", args.trials)
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_params
    best_val = study.best_value
    logger.info("Auto-tune complete: %d trials in %.0fs", args.trials, elapsed)
    logger.info("Best params: tweedie=%.2f leaves=%d lr=%.4f sub=%.2f col=%.2f alpha=%.4f lambda=%.2f",
                 best["tweedie_power"], best["num_leaves"], best["learning_rate"],
                 best["subsample"], best["colsample_bytree"], best["reg_alpha"], best["reg_lambda"])
    logger.info("Best val_l1: %.1f", best_val)

    result = {"best_params": best, "best_val_l1": best_val, "n_trials": args.trials, "time_sec": round(elapsed, 1)}
    (db_dir / "auto_tune_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved: %s", db_dir / "auto_tune_result.json")

    return 0


def _to_numpy(df: pl.DataFrame, feature_cols: list) -> np.ndarray:
    X = np.zeros((len(df), len(feature_cols)), dtype=np.float64)
    for i, col in enumerate(feature_cols):
        try:
            X[:, i] = df[col].cast(pl.Float64, strict=False).fill_null(0).to_numpy()
        except Exception:
            try:
                X[:, i] = df[col].dt.epoch("s").cast(pl.Float64).fill_null(0).to_numpy()
            except Exception:
                try:
                    X[:, i] = df[col].cast(pl.Utf8).hash(42).mod(10 ** 9).cast(pl.Float64).fill_null(0).to_numpy()
                except Exception:
                    pass
    return X


if __name__ == "__main__":
    sys.exit(main())

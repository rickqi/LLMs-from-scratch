"""Hyperparameter tuning with Optuna for L1-A claim amount prediction.

Usage:
    python -m ml.pipeline.tune --config ml/config_ml.yaml
    python -m ml.pipeline.tune --config ml/config_ml.yaml --trials 30 --no-prune
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger("ml.tune")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for LightGBM")
    parser.add_argument("--config", default="ml/config_ml.yaml", help="Config file path")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--no-prune", action="store_true", help="Disable early pruning")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    config = load_config(args.config)

    # --- Load data (reuse training pipeline) ---
    from ml.data.loader import load_doris_csv
    from ml.data.feature_store import compute_global_stats, build_features, get_feature_columns
    from ml.data.split import split_by_time
    from ml.models.amount_predictor import AmountPredictor
    from ml.evaluate.metrics import evaluate_predictions

    csv_path = config["data"]["csv_path"]
    logger.info("Loading data from %s", csv_path)
    lf = load_doris_csv(csv_path)

    # Global stats
    split_cfg = config["split"]
    global_stats = compute_global_stats(lf, train_end=split_cfg["train_end"])

    # Features
    l1a_cfg = config["l1a"]
    feature_lf = build_features(
        lf, global_stats,
        categorical_cols=config["features"]["categorical_cols"],
        log_transform=l1a_cfg.get("log_transform", False),
    )
    feature_cols = get_feature_columns(feature_lf, config["features"]["exclude_cols"])
    categorical_cols = [c for c in config["features"]["categorical_cols"] if c in feature_cols]
    target_col = "y_raw"  # Tweedie on raw target

    # Split
    splits = split_by_time(feature_lf, train_end=split_cfg["train_end"],
                           val_end=split_cfg["val_end"], test_end=split_cfg["test_end"])
    select_cols = feature_cols + ["y_raw", "is_large"]
    train_df = splits["train"].select(select_cols).collect(streaming=True)
    val_df = splits["val"].select(select_cols).collect(streaming=True)
    logger.info("Train: %s rows | Val: %s rows", f"{len(train_df):,}", f"{len(val_df):,}")

    # --- Build train/val Datasets once (reused across trials) ---
    # Use a dummy predictor just for _prepare_data
    dummy = AmountPredictor(log_transform=False)
    X_train_pd, y_train = dummy._prepare_data(train_df, feature_cols, categorical_cols, target_col)
    X_val_pd, y_val = dummy._prepare_data(val_df, feature_cols, categorical_cols, target_col)

    import lightgbm as lgb
    dtrain = lgb.Dataset(X_train_pd, label=y_train, feature_name=feature_cols,
                         categorical_feature=categorical_cols if categorical_cols else "auto",
                         free_raw_data=True)
    dval = lgb.Dataset(X_val_pd, label=y_val, reference=dtrain)

    # --- Optuna objective ---
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_val_l1 = float("inf")
    best_params: dict = {}
    trial_results: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_val_l1, best_params

        params_search = {
            "objective": "tweedie",
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.05, 1.5, step=0.05),
            "metric": ["mae", "rmse"],
            "num_leaves": trial.suggest_int("num_leaves", 31, 127, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, step=20),
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
            "force_col_wise": True,
            "feature_pre_filter": False,  # Required for Optuna's dynamic min_data_in_leaf
        }

        # Also tune n_estimators via early_stopping (capped at 1000)
        n_estimators_tune = trial.suggest_int("n_estimators", 400, 1000, step=100)

        t0 = time.time()
        model = lgb.train(
            params_search,
            dtrain,
            num_boost_round=n_estimators_tune,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
            ],
        )

        val_l1 = model.best_score["val"]["l1"]
        elapsed = time.time() - t0

        # Prune bad trials early (optional, controlled by --no-prune)
        if not args.no_prune:
            trial.report(val_l1, model.best_iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial_results.append({
            "trial": trial.number,
            "val_l1": round(val_l1, 1),
            "best_iter": model.best_iteration,
            "time_sec": round(elapsed, 1),
            "params": {k: v for k, v in params_search.items()
                       if k not in ("metric", "verbose", "seed", "n_jobs", "force_col_wise")},
        })

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            best_params = dict(params_search)
            best_params["best_iteration"] = model.best_iteration

        trial_number = trial.number + 1  # 1-based display
        logger.info("Trial %d/%d: val_l1=%.0f, best_iter=%d, time=%.1fs",
                     trial_number, args.trials, val_l1, model.best_iteration, elapsed)

        return val_l1

    # --- Run Optuna ---
    logger.info("Starting Optuna study: %d trials", args.trials)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner() if not args.no_prune else optuna.pruners.NopPruner(),
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    total_time = time.time() - t0

    # --- Results ---
    logger.info("=" * 60)
    logger.info("Optuna tuning complete: %d trials in %.1f min", args.trials, total_time / 60)
    logger.info("Best val_l1: %.1f (improvement: %.1f%%)",
                 study.best_value,
                 (1 - study.best_value / dummy.train_eval.get("val_l1", study.best_value)) * 100)

    logger.info("Best parameters:")
    for k, v in sorted(best_params.items()):
        if k not in ("metric", "verbose", "seed", "n_jobs", "force_col_wise", "subsample_freq"):
            logger.info("  %s: %s", k, v)

    # Save results
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    tune_result = {
        "study_name": "l1a_tweedie_tune",
        "n_trials": args.trials,
        "total_time_sec": round(total_time, 1),
        "best_val_l1": round(study.best_value, 1),
        "best_iteration": best_params.pop("best_iteration", 0),
        "best_params": {k: v for k, v in sorted(best_params.items())},
        "trial_history": trial_results,
    }

    tune_path = model_dir / "optuna_tune_result.json"
    tune_path.write_text(json.dumps(tune_result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Tuning results saved: %s", tune_path)

    # Print config update snippet
    logger.info("=" * 60)
    logger.info("Update ml/config_ml.yaml l1a section with:")
    bp = best_params
    logger.info("  n_estimators: %d", bp.get("n_estimators", 800))
    logger.info("  tweedie_power: %.1f", bp.get("tweedie_variance_power", 1.2))
    logger.info("  num_leaves: %d", bp.get("num_leaves", 63))
    logger.info("  learning_rate: %.4f", bp.get("learning_rate", 0.05))
    logger.info("  subsample: %.2f", bp.get("subsample", 0.8))
    logger.info("  colsample_bytree: %.2f", bp.get("colsample_bytree", 0.8))
    logger.info("  reg_alpha: %.4f", bp.get("reg_alpha", 0.1))
    logger.info("  reg_lambda: %.2f", bp.get("reg_lambda", 1.0))
    logger.info("  min_child_samples: %d", bp.get("min_child_samples", 50))
    logger.info("  early_stopping_rounds: 50")
    logger.info("  log_transform: false")

    return 0


if __name__ == "__main__":
    sys.exit(main())

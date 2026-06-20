"""Training pipeline CLI entry point.

Usage:
    python -m ml.pipeline.train --config ml/config_ml.yaml
    python -m ml.pipeline.train --config ml/config_ml.yaml --mode tune
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import polars as pl

logger = logging.getLogger("ml.train")


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


# ── Checkpoint / Resume ────────────────────────────────────────────────
_CHECKPOINT_DIR = Path("models/checkpoint")
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "training_checkpoint.pkl"

def save_checkpoint(state: dict) -> None:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state["timestamp"] = datetime.now().isoformat()
    with open(_CHECKPOINT_PATH, "wb") as f:
        pickle.dump(state, f)

def load_checkpoint() -> dict | None:
    if _CHECKPOINT_PATH.exists():
        with open(_CHECKPOINT_PATH, "rb") as f:
            return pickle.load(f)
    return None

from datetime import datetime  # noqa: E402 (placed here to avoid forward ref in checkpoint)


# ── Model Version Registry ─────────────────────────────────────────────
VERSION_DIR = Path("models/versions")
VERSION_REGISTRY = Path("models/model_registry.json")

def create_model_version(model_dir: Path, version: str, metrics: dict) -> Path:
    import shutil

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = VERSION_DIR / f"v{version}_{timestamp}"
    version_path.mkdir(parents=True, exist_ok=True)

    for f in model_dir.glob("*"):
        if f.is_file():
            shutil.copy2(f, version_path / f.name)

    latest_link = Path("models/latest")
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(
        str(version_path.relative_to("models")),
        target_is_directory=True,
    )

    registry = load_registry()
    registry["versions"].append({
        "version": version,
        "timestamp": timestamp,
        "dir": str(version_path),
        "metrics": metrics,
    })
    registry["latest"] = version
    save_registry(registry)

    logger.info("Model version created: v%s → %s", version, version_path)
    return version_path

def load_registry() -> dict:
    if VERSION_REGISTRY.exists():
        return json.loads(VERSION_REGISTRY.read_text(encoding="utf-8"))
    return {"versions": [], "latest": None}

def save_registry(registry: dict) -> None:
    VERSION_REGISTRY.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

import shutil  # noqa: E402

EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_INDEX = EXPERIMENT_DIR / "experiment_index.json"


def _archive_experiment(
    exp_name: str, version_tag: str, report_data: dict, top_features: list,
) -> None:
    exp_path = EXPERIMENT_DIR / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)

    config_snapshot = report_data.copy()
    config_snapshot.pop("top_features", None)
    (exp_path / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (exp_path / "metrics.json").write_text(
        json.dumps(report_data.get("val_metrics", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (exp_path / "model_params.json").write_text(
        json.dumps(report_data.get("model_params", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    import csv
    with open(exp_path / "feature_importance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows([[feat["feature"], feat["importance"]] for feat in top_features])

    index = []
    if EXPERIMENT_INDEX.exists():
        index = json.loads(EXPERIMENT_INDEX.read_text(encoding="utf-8"))
    index.append({
        "name": exp_name,
        "version": version_tag,
        "timestamp": datetime.now().isoformat(),
        "r2": report_data["val_metrics"]["r2"],
        "gini": report_data["val_metrics"]["gini"],
        "mape": report_data["val_metrics"]["mape"],
        "feature_count": report_data["feature_count"],
    })
    EXPERIMENT_INDEX.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Experiment archived: %s", exp_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train L1-A claim amount prediction model")
    parser.add_argument("--config", default="ml/config_ml.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["full", "l1a_only", "tune"], default="full",
                        help="Training mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--no-backtest", action="store_true", help="Skip walk-forward backtest")
    parser.add_argument("--version", default="auto", help="Model version tag (default: auto-detect)")
    parser.add_argument("--exp-name", default=None, help="Experiment name for tracking")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use Parquet cache (skip CSV re-scan)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild Parquet cache from CSV")
    parser.add_argument("--skip-quantile", action="store_true",
                        help="Skip quantile model training (P05/P50/P95)")
    parser.add_argument("--skip-l2", action="store_true",
                        help="Skip L2 policy-level training")
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of training data to use (0-1.0, default 1.0)")
    parser.add_argument("--chunked", action="store_true",
                        help="Chunked training from Parquet (for low-RAM environments)")
    parser.add_argument("--chunk-size", type=int, default=500000,
                        help="Rows per chunk in chunked mode (default 500K)")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    config = load_config(args.config)
    logger.info("Config loaded from %s", args.config)

    # ── Checkpoint / Resume ──
    ckpt = load_checkpoint() if args.resume else None
    completed_step = ckpt["step"] if ckpt else 0
    if ckpt:
        logger.info("Resuming from checkpoint: step %d/%d", ckpt["step"], ckpt["total"])

    # --- Step 1: Load data (via Parquet cache or CSV) ---
    from ml.data.loader import load_doris_csv
    from ml.data.feature_store import (
        compute_global_stats, build_features, get_feature_columns,
        prepare_train_cache, load_train_lf,
    )
    from ml.data.split import split_by_time
    from ml.models.amount_predictor import AmountPredictor, train_quantile_models
    from ml.evaluate.metrics import evaluate_predictions, compute_baseline_metrics

    csv_path = config["data"]["csv_path"]
    cache_dir = Path(config["data"]["cache_dir"])
    split_cfg = config["split"]

    logger.info("=" * 60)
    logger.info("Step 1: Loading data")

    if args.use_cache and not args.rebuild_cache:
        cache_path = prepare_train_cache(
            csv_path, cache_dir, train_end=split_cfg["train_end"],
            force=args.rebuild_cache,
        )
        cache_lf = load_train_lf(cache_path)
        logger.info("  Parquet cache: %s (%.0f MB)", cache_path.name,
                     cache_path.stat().st_size / (1024 * 1024))
    elif args.rebuild_cache:
        logger.info("  Rebuilding cache (--rebuild-cache)...")
        cache_path = prepare_train_cache(
            csv_path, cache_dir, train_end=split_cfg["train_end"], force=True,
        )
        cache_lf = load_train_lf(cache_path)
    else:
        logger.info("  Loading from CSV (no cache)...")
        cache_lf = None

    # Full CSV LazyFrame for feature building (needs val/test periods too)
    full_lf = load_doris_csv(csv_path) if cache_lf is None else load_doris_csv(csv_path)

    # --- Step 2: Compute global stats (from Parquet cache if available) ---
    logger.info("=" * 60)
    logger.info("Step 2: Computing global statistics (train-only data)")
    stats_lf = cache_lf if cache_lf is not None else full_lf
    global_stats = compute_global_stats(stats_lf, train_end=split_cfg["train_end"])

    # Save global stats for prediction reuse
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / "global_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump({k: v.to_dict(as_series=False) if hasattr(v, 'to_dict') else v
                      for k, v in global_stats.items()}, f)
    logger.info("Global stats saved: %s", stats_path)

    if completed_step <= 2:
        save_checkpoint({"step": 2, "total": 9, "stats_path": str(stats_path)})

    # --- Step 3: Build features ---
    logger.info("=" * 60)
    logger.info("Step 3: Building feature LazyFrame")
    feature_lf = build_features(
        full_lf,
        global_stats,
        categorical_cols=config["features"]["categorical_cols"],
        log_transform=config["l1a"].get("log_transform", True),
    )

    # Determine feature columns
    feature_cols = get_feature_columns(feature_lf, config["features"]["exclude_cols"])
    categorical_cols = [c for c in config["features"]["categorical_cols"]
                        if c in feature_cols]

    # Save feature schema
    schema = {"feature_cols": feature_cols, "categorical_cols": categorical_cols}
    schema_path = model_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Feature schema saved: %d features (%d categorical)",
                len(feature_cols), len(categorical_cols))

    # --- Step 4: Split data ---
    logger.info("=" * 60)
    logger.info("Step 4: Splitting data (train/val/test)")
    splits = split_by_time(
        feature_lf,
        train_end=split_cfg["train_end"],
        val_end=split_cfg["val_end"],
        test_end=split_cfg["test_end"],
    )

    # Determine target column based on log_transform setting
    l1a_cfg = config["l1a"]
    use_log = l1a_cfg.get("log_transform", True)
    target_col = "y_log" if use_log else "y_raw"
    select_cols = feature_cols + ["y_raw", "is_large"]
    if use_log:
        select_cols.append("y_log")

    logger.info("Collecting train/val DataFrames...")
    t0 = time.time()

    if args.chunked:
        # Chunked mode: skip train collection, sample ~1M val rows for early stopping
        train_df = pl.DataFrame()
        val_df = splits["val"].filter(
            pl.col("case_no").hash(seed=42) % 100 < 50
        ).select(select_cols).collect(engine="streaming")
        logger.info("Chunked mode: val %s rows (sampled ~50%%) | train → Parquet sink",
                     f"{len(val_df):,}")
    elif args.sample < 1.0 and args.sample > 0:
        sample_pct = int(args.sample * 100)
        train_lf = splits["train"].filter(
            pl.col("case_no").hash(seed=42) % 100 < sample_pct
        ).select(select_cols)
        train_df = train_lf.collect(engine="streaming")
        val_df = splits["val"].select(select_cols).collect(engine="streaming")
    else:
        train_lf = splits["train"].select(select_cols)
        train_df = train_lf.collect(engine="streaming")
        val_df = splits["val"].select(select_cols).collect(engine="streaming")
        logger.info("Train: %s rows | Val: %s rows | Time: %.1fs",
                     f"{len(train_df):,}", f"{len(val_df):,}", time.time() - t0)

    if completed_step <= 4:
        save_checkpoint({"step": 4, "total": 9, "stats_path": str(stats_path),
                          "schema_path": str(schema_path)})

    # --- Step 5: Train L1-A model ---
    logger.info("=" * 60)

    if args.chunked:
        # Chunked mode: sink features to Parquet, then train in batches
        logger.info("Step 5: Training L1-A (CHUNKED from Parquet, %d rows/chunk)", args.chunk_size)

        train_parquet = cache_dir / "train_features.parquet"
        logger.info("  Sinking training features to Parquet...")
        t_sink = time.time()
        splits["train"].select(select_cols).sink_parquet(
            str(train_parquet), compression="zstd", compression_level=1,
        )
        logger.info("  Parquet ready: %.0f MB (%.1fs)",
                     train_parquet.stat().st_size / (1024 * 1024), time.time() - t_sink)

        predictor = AmountPredictor(
            objective=l1a_cfg["objective"],
            tweedie_power=l1a_cfg.get("tweedie_power", 1.5),
            n_estimators=l1a_cfg["n_estimators"],
            num_leaves=l1a_cfg["num_leaves"],
            learning_rate=l1a_cfg["learning_rate"],
            subsample=l1a_cfg["subsample"],
            colsample_bytree=l1a_cfg["colsample_bytree"],
            reg_alpha=l1a_cfg["reg_alpha"],
            reg_lambda=l1a_cfg["reg_lambda"],
            min_child_samples=l1a_cfg["min_child_samples"],
            early_stopping_rounds=l1a_cfg["early_stopping_rounds"],
            log_transform=use_log,
        )

        train_summary = predictor.train_chunked(
            train_parquet, val_df, feature_cols, categorical_cols,
            target_col=target_col, chunk_size=args.chunk_size,
        )
    else:
        logger.info("Step 5: Training L1-A LightGBM model (target=%s)", target_col)
        predictor = AmountPredictor(
            objective=l1a_cfg["objective"],
            tweedie_power=l1a_cfg.get("tweedie_power", 1.5),
            n_estimators=l1a_cfg["n_estimators"],
            num_leaves=l1a_cfg["num_leaves"],
            learning_rate=l1a_cfg["learning_rate"],
            subsample=l1a_cfg["subsample"],
            colsample_bytree=l1a_cfg["colsample_bytree"],
            reg_alpha=l1a_cfg["reg_alpha"],
            reg_lambda=l1a_cfg["reg_lambda"],
            min_child_samples=l1a_cfg["min_child_samples"],
            early_stopping_rounds=l1a_cfg["early_stopping_rounds"],
            log_transform=use_log,
        )

        train_summary = predictor.train(
            train_df, val_df, feature_cols, categorical_cols, target_col=target_col
        )
    logger.info("L1-A training summary: %s", json.dumps(train_summary, indent=2))

    if completed_step <= 5:
        save_checkpoint({"step": 5, "total": 9, "stats_path": str(stats_path),
                          "schema_path": str(schema_path)})

    # Save model
    predictor.save(model_dir, name="l1a_amount")

    # Save standalone category mappings for predict.py quantile encoding
    cat_map_path = model_dir / "category_mappings.json"
    cat_map_path.write_text(
        json.dumps(predictor.category_mappings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Category mappings saved: %s (%d columns)",
                cat_map_path, len(predictor.category_mappings))

    # --- Step 6: Evaluate on val set ---
    logger.info("=" * 60)
    logger.info("Step 6: Evaluating on validation set")
    y_val_pred = predictor.predict(val_df, feature_cols, categorical_cols)
    y_val_true = val_df["y_raw"].to_numpy()

    val_metrics = evaluate_predictions(y_val_true, y_val_pred)
    baseline_metrics = compute_baseline_metrics(y_val_true)

    logger.info("Validation metrics:")
    for k, v in val_metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("Baseline (mean prediction): MAE=%.0f, MAPE=%.1f%%",
                baseline_metrics["baseline_mean_mae"],
                baseline_metrics["baseline_mean_mape"])
    logger.info("Improvement over baseline: MAE %.1f%% / MAPE %.1f%%",
                (1 - val_metrics["mae"] / baseline_metrics["baseline_mean_mae"]) * 100,
                (1 - val_metrics["mape"] / baseline_metrics["baseline_mean_mape"]) * 100)

    # --- Step 7: Quantile models ---
    logger.info("=" * 60)
    skip_q = args.skip_quantile or args.mode == "l1a_only"
    if skip_q:
        logger.info("Step 7: Quantile models SKIPPED (--skip-quantile or --mode l1a_only)")
        quantile_results = {}
    else:
        logger.info("Step 7: Training quantile models (P05/P50/P95)")
        quantile_results = train_quantile_models(
            train_df, val_df, feature_cols, categorical_cols,
            quantiles=config["quantiles"]["levels"],
            n_estimators=config["quantiles"]["n_estimators"],
            target_col=target_col,
            model_dir=str(model_dir),
        )
        logger.info("Quantile models saved: %s", list(quantile_results.keys()))

    # --- Step 8: Feature importance ---
    logger.info("=" * 60)
    logger.info("Step 8: Feature importance")
    importance = predictor.get_feature_importance(importance_type="gain")
    top_features = [{"feature": k, "importance": v} for k, v in importance.items()][:20]
    for feat in top_features:
        logger.info("  %-30s %s", feat["feature"], feat["importance"])

    # --- Step 9: Backtest (optional) ---
    backtest_results = []
    if not args.no_backtest:
        logger.info("=" * 60)
        logger.info("Step 9: Walk-forward backtest")
        from ml.evaluate.backtest import walk_forward_backtest
        backtest_results = walk_forward_backtest(
            feature_lf,
            {"l1a": l1a_cfg},
            feature_cols,
            categorical_cols,
            train_months=18,
            test_months=3,       # P1: quarterly windows for stability
            max_iterations=4,    # P1: fewer windows, each with 3x data
        )

    # --- Step 10: L2 Policy-level model (P3) ---
    l2_results: dict = {}
    if not args.skip_l2:
        logger.info("=" * 60)
        logger.info("Step 10: Training L2 Policy-level predictor")

        from ml.data.policy_aggregator import aggregate_policy_features, get_policy_feature_cols
        from ml.models.policy_predictor import PolicyPredictor

        policy_lf = aggregate_policy_features(full_lf, train_end=split_cfg["train_end"])
        policy_df = policy_lf.collect(engine="streaming")
        logger.info("  Policies: %d, features: %d", len(policy_df),
                     len(policy_lf.collect_schema().names()))

        p_feature_cols = get_policy_feature_cols(policy_lf)
        p_target = "loss_ratio"

        policy_df = policy_df.filter(
            pl.col(p_target).is_not_null() & (pl.col(p_target) > 0)
            & (pl.col("policy_avg_premium") > 0)
        )
        logger.info("  Valid policies: %d", len(policy_df))

        split_n = int(len(policy_df) * 0.8)
        p_train = policy_df.slice(0, split_n)
        p_val = policy_df.slice(split_n, len(policy_df) - split_n)

        l2 = PolicyPredictor(n_estimators=500)
        l2_results = l2.train(p_train, p_val, p_feature_cols, target_col=p_target)

        y_pred = l2.predict(p_val)
        y_true = p_val[p_target].to_numpy()
        from ml.evaluate.metrics import evaluate_predictions
        l2_metrics = evaluate_predictions(y_true, y_pred)
        logger.info("  L2 validation: R²=%.4f, MAE=%.4f, Gini=%.4f",
                     l2_metrics["r2"], l2_metrics["mae"], l2_metrics["gini"])

        l2.save(model_dir, name="l2_policy")
    else:
        logger.info("Step 10: L2 SKIPPED (--skip-l2)")

    # --- Step 11: ML Risk Map Scores (P0) ---
    if not args.skip_l2:
        logger.info("=" * 60)
        logger.info("Step 11: Generating ML risk map scores")
        try:
            from ml.data.risk_scorer import MLRiskScorer
            scorer = MLRiskScorer(predictor, None, l2, None, None)
            risk_map = scorer.generate_risk_map(
                train_df,  # Use training data for risk scoring
                p_train,
                feature_cols,
                categorical_cols,
                p_feature_cols,
            )
            risk_path = model_dir / "ml_risk_map.json"
            risk_path.write_text(
                json.dumps(risk_map, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            risk_count = len(risk_map.get("disease", []))
            logger.info("  Risk map saved: %s (%d disease groups, %d members)",
                         risk_path, risk_count, len(risk_map.get("member", [])))
        except Exception as e:
            logger.warning("  Risk map generation skipped: %s", e)

    # --- Save training report data ---
    report_data = {
        "config_path": args.config,
        "mode": args.mode,
        "csv_path": csv_path,
        "feature_count": len(feature_cols),
        "categorical_feature_count": len(categorical_cols),
        "train_rows": train_summary.get("total_rows", len(train_df)),
        "val_rows": len(val_df),
        "split_config": split_cfg,
        "model_params": predictor.params,
        "train_summary": train_summary,
        "val_metrics": val_metrics,
        "baseline_metrics": baseline_metrics,
        "top_features": top_features,
        "quantile_models": {k: v.get("quantile", 0) for k, v in quantile_results.items()},
        "backtest_results": backtest_results,
    }

    report_dir = Path(config["output"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "training_report_data.json"
    report_path.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )
    logger.info("=" * 60)
    logger.info("Training report data saved: %s", report_path)
    logger.info("DONE. Validation MAPE: %.1f%%", val_metrics["mape"])

    # ── Model Versioning ──
    version_tag = args.version
    if version_tag == "auto":
        registry = load_registry()
        existing = len(registry["versions"])
        version_tag = f"{existing + 1}.0"

    create_model_version(
        model_dir, version_tag,
        metrics={
            "r2": val_metrics["r2"],
            "gini": val_metrics["gini"],
            "mae": val_metrics["mae"],
            "mape": val_metrics["mape"],
            "total_error_pct": val_metrics.get("total_error_pct", 0),
            "train_time_sec": train_summary.get("train_time_sec", 0),
            "best_iteration": train_summary.get("best_iteration", 0),
            "feature_count": len(feature_cols),
            "train_rows": train_summary.get("total_rows", len(train_df) if len(train_df) > 0 else 0),
        },
    )

    # ── Experiment Archiving (P2-9) ──
    if args.exp_name:
        _archive_experiment(args.exp_name, version_tag, report_data, top_features)

    # Cleanup checkpoint on successful full run
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleaned (run complete)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

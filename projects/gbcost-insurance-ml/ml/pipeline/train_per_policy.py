"""Per-policy training — loads 1797 policy CSV files one at a time.

Reference: company-regulations-training backup pattern
  COS: ins-kq6zz7wo-1313469539 / LLMs-from-scratch/projects/gbcost-insurance-ml/
  Backup: python -m ml.pipeline.cos_backup models predictions reports

Accumulates policies in batches, trains LightGBM on each batch from scratch,
avoiding both OOM (incremental loading) and overfitting (no keep_training_booster).

Usage:
    python -m ml.pipeline.train_per_policy --policy-dir data/doris/policies --batch-size 10
    python -m ml.pipeline.train_per_policy --policy-dir data/doris/policies --batch-size 10 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger("ml.per_policy")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def load_policy_csv(path: Path) -> pl.DataFrame:
    """Load a single wide_*.csv policy file."""
    return pl.read_csv(
        str(path), try_parse_dates=True, infer_schema_length=10000,
    )


def build_policy_features(
    df: pl.DataFrame, global_stats: dict, categorical_cols: List[str],
) -> pl.DataFrame:
    """Build features for a single policy (or merged batch)."""
    from ml.data.feature_store import build_features
    lf = df.lazy()
    return build_features(lf, global_stats, categorical_cols=categorical_cols, log_transform=False)


def train_on_batch(
    train_df: pl.DataFrame, val_df: pl.DataFrame,
    feature_cols: List[str], categorical_cols: List[str],
    params: dict, n_estimators: int, early_stopping: int,
) -> lgb.Booster:
    """Train LightGBM using numpy arrays (avoids pandas categorical serialization bug)."""
    # Cast categorical columns to integer codes before numpy conversion
    train_df_fixed, val_df_fixed = train_df, val_df
    for col in categorical_cols:
        if col in train_df.columns:
            train_df_fixed = train_df_fixed.with_columns(
                pl.col(col).rank("dense").cast(pl.Int64).alias(col))
        if col in val_df.columns:
            val_df_fixed = val_df_fixed.with_columns(
                pl.col(col).rank("dense").cast(pl.Int64).alias(col))

    existing = [c for c in feature_cols if c in train_df_fixed.columns]
    train_np = train_df_fixed.select(existing).to_numpy()
    val_np = val_df_fixed.select(existing).to_numpy()

    # Convert any remaining object columns to float via hash
    for i, col in enumerate(existing):
        try:
            train_np[:, i] = train_np[:, i].astype(np.float64)
        except (ValueError, TypeError):
            train_np[:, i] = np.array([hash(str(v)) % 10**9 for v in train_np[:, i]], dtype=np.float64)
        try:
            val_np[:, i] = val_np[:, i].astype(np.float64)
        except (ValueError, TypeError):
            val_np[:, i] = np.array([hash(str(v)) % 10**9 for v in val_np[:, i]], dtype=np.float64)

    dtrain = lgb.Dataset(train_np, label=train_df["y_raw"].to_numpy().astype(np.float64),
                         feature_name=existing, free_raw_data=True)
    dval = lgb.Dataset(val_np, label=val_df["y_raw"].to_numpy().astype(np.float64), reference=dtrain)

    try:
        model = lgb.train(dict(params), dtrain, num_boost_round=n_estimators,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(period=0)])
    except (ValueError, TypeError):
        model = lgb.train(dict(params), dtrain, num_boost_round=n_estimators,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(period=0)])
    return model


def compute_global_stats_from_sample(policy_files: List[Path], sample_n: int = 50) -> dict:
    """Compute global stats from a random sample of policies."""
    sample = random.sample(policy_files, min(sample_n, len(policy_files)))
    dfs = []
    for f in sample:
        try:
            df = load_policy_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning("Skip %s: %s", f.name, e)
    merged = pl.concat(dfs, how="diagonal_relaxed")
    from ml.data.feature_store import compute_global_stats
    return compute_global_stats(merged.lazy())


CKPT_PATH = Path("models/checkpoint/per_policy_ckpt.json")


def save_ckpt(state: dict) -> None:
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_ckpt() -> dict | None:
    if CKPT_PATH.exists():
        return json.loads(CKPT_PATH.read_text(encoding="utf-8"))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-policy incremental training")
    parser.add_argument("--policy-dir", default="data/doris/policies")
    parser.add_argument("--batch-size", type=int, default=10, help="Policies per training batch")
    parser.add_argument("--val-policies", type=int, default=50, help="Policies held out for validation")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    random.seed(args.seed)

    policy_dir = Path(args.policy_dir)
    all_files = sorted(policy_dir.glob("wide_*.csv"))
    if not all_files:
        logger.error("No wide_*.csv files found in %s", policy_dir)
        return 1

    random.shuffle(all_files)
    val_files = all_files[:args.val_policies]
    train_files = all_files[args.val_policies:]
    logger.info("Policies: %d train + %d val (total %d)", len(train_files), len(val_files), len(all_files))

    # Load config
    import yaml
    with open("ml/config_ml.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    categorical_cols = config["features"]["categorical_cols"]
    l1a_cfg = config["l1a"]
    params = {
        "objective": "tweedie", "tweedie_variance_power": l1a_cfg.get("tweedie_power", 1.35),
        "metric": ["l1", "rmse"], "num_leaves": l1a_cfg["num_leaves"],
        "learning_rate": l1a_cfg["learning_rate"], "subsample": l1a_cfg["subsample"],
        "subsample_freq": 1, "colsample_bytree": l1a_cfg["colsample_bytree"],
        "reg_alpha": l1a_cfg["reg_alpha"], "reg_lambda": l1a_cfg["reg_lambda"],
        "min_child_samples": l1a_cfg["min_child_samples"],
        "verbose": -1, "seed": args.seed, "n_jobs": -1, "force_col_wise": True,
    }

    # Compute global stats from sample
    logger.info("Computing global stats from %d sample policies...", min(50, len(all_files)))
    global_stats = compute_global_stats_from_sample(all_files, sample_n=50)
    cache_dir = Path(config["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "global_stats.pkl", "wb") as f:
        pickle.dump({k: v.to_dict(as_series=False) if hasattr(v, 'to_dict') else v
                      for k, v in global_stats.items()}, f)

    # Load validation data
    logger.info("Loading %d validation policies...", len(val_files))
    val_dfs = []
    for f in val_files:
        try:
            val_dfs.append(load_policy_csv(f))
        except Exception as e:
            logger.warning("Skip val %s: %s", f.name, e)
    val_raw = pl.concat(val_dfs, how="diagonal_relaxed")
    val_feat = build_policy_features(val_raw, global_stats, categorical_cols)
    val_df = val_feat.collect(engine="streaming")
    feature_cols = [c for c in val_feat.collect_schema().names()
                    if c not in ("y_raw", "y_log", "is_large", "clm_acount_amnt_cny",
                                 "medical_start_date", "case_no")]
    logger.info("Validation: %s rows, %d features", f"{len(val_df):,}", len(feature_cols))

    # Resume
    ckpt = load_ckpt() if args.resume else None
    start_batch = ckpt["batch"] + 1 if ckpt else 0
    best_model = None
    best_val_l1 = float("inf")

    if ckpt:
        logger.info("Resuming from batch %d/%d", start_batch, len(train_files) // args.batch_size)

    # Train batch by batch
    buffer: List[pl.DataFrame] = []
    batch_idx = start_batch

    for i, policy_file in enumerate(train_files):
        if i < start_batch * args.batch_size:
            continue

        try:
            df = load_policy_csv(policy_file)
            buffer.append(df)
        except Exception as e:
            logger.warning("Skip %s: %s", policy_file.name, e)
            continue

        if len(buffer) >= args.batch_size or i == len(train_files) - 1:
            t0 = time.time()
            merged = pl.concat(buffer, how="diagonal_relaxed")
            feat_lf = build_policy_features(merged, global_stats, categorical_cols)
            train_df = feat_lf.collect(engine="streaming")
            buffer = []

            model = train_on_batch(train_df, val_df, feature_cols, categorical_cols,
                                   params, args.n_estimators, args.early_stopping)

            val_l1 = model.best_score["val"]["l1"]
            elapsed = time.time() - t0

            logger.info("Batch %3d: %s rows, val_l1=%.1f, best_iter=%d, time=%.0fs",
                         batch_idx, f"{len(train_df):,}", val_l1, model.best_iteration, elapsed)

            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                model.save_model(f"{args.output_dir}/l1a_amount.txt", num_iteration=model.best_iteration)
                best_model = model

            save_ckpt({"batch": batch_idx, "total_batches": len(train_files) // args.batch_size,
                        "best_val_l1": best_val_l1, "files_processed": i + 1})
            batch_idx += 1

    # Final save
    if best_model:
        from ml.models.amount_predictor import AmountPredictor
        p = AmountPredictor(log_transform=False)
        p.model = best_model
        p.best_iteration = best_model.best_iteration
        p.feature_names = feature_cols
        p.categorical_features = [c for c in categorical_cols if c in feature_cols]
        p.save(args.output_dir, "l1a_amount")

        # Evaluate
        y_pred = p.predict(val_df, feature_cols, categorical_cols)
        y_true = val_df["y_raw"].to_numpy()
        from ml.evaluate.metrics import evaluate_predictions
        metrics = evaluate_predictions(y_true, y_pred)
        logger.info("=" * 50)
        logger.info("Final: batches=%d, val_l1=%.1f, Gini=%.4f, R²=%.4f, MAE=%.0f, err=%.1f%%",
                     batch_idx, best_val_l1, metrics["gini"], metrics["r2"],
                     metrics["mae"], metrics["total_error_pct"])

    CKPT_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

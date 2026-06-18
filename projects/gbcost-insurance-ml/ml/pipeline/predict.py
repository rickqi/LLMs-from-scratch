"""Prediction pipeline CLI entry point.

Usage:
    python -m ml.pipeline.predict --csv data/doris/c001_ghb_poicy_clm_duty_d.csv
    python -m ml.pipeline.predict --csv ... --policy GP2202301028502
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger("ml.predict")


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


def _run_chunked_predict(
    feature_lf, feature_cols, extra_cols, categorical_cols,
    predictor, model_dir, output_dir, args, config,
) -> None:
    import tempfile
    from pathlib import Path
    import lightgbm as lgb
    import pandas as _pd

    chunk_size = args.chunk_size
    all_cols = feature_cols + extra_cols
    logger.info("Chunked prediction: sinking features to temp Parquet...")
    t_sink = time.time()

    tmpdir = Path(tempfile.mkdtemp(prefix="ml_predict_"))
    tmp_parquet = tmpdir / "features.parquet"
    feature_lf.select(all_cols).sink_parquet(str(tmp_parquet), compression="zstd", compression_level=1)
    logger.info("  Parquet ready: %.0f MB (%.1fs)", tmp_parquet.stat().st_size / (1024*1024), time.time()-t_sink)

    total_rows = pl.scan_parquet(str(tmp_parquet)).select(pl.len()).collect().item()
    logger.info("  Total: %s rows, chunking by %s", f"{total_rows:,}", f"{chunk_size:,}")

    cat_mappings_path = model_dir / "category_mappings.json"
    category_mappings = {}
    if cat_mappings_path.exists():
        category_mappings = json.loads(cat_mappings_path.read_text(encoding="utf-8"))

    results = []
    for offset in range(0, total_rows, chunk_size):
        chunk_df = pl.scan_parquet(str(tmp_parquet)).slice(offset, chunk_size).collect(engine="streaming")
        chunk_n = offset // chunk_size + 1

        y_pred = predictor.predict(chunk_df, feature_cols, categorical_cols)

        buckets = np.select(
            [y_pred < 500, y_pred < 2000, y_pred < 10000, y_pred < 50000],
            ["0-500", "500-2K", "2K-10K", "10K-50K"], default="50K+",
        )

        meta_cols = [pl.col(c).cast(pl.Utf8) if c in {"duty_code","hosp_grade","policy_grp_name","claim_type",
                        "fee_item_type","insured_gender","rnew_flag","sale_chnl","main_insured_rela",
                        "is_public","is_expensive"} else pl.col(c)
                     for c in extra_cols if c in chunk_df.columns]

        chunk_result = chunk_df.select(meta_cols).with_columns([
            pl.Series("predicted_amount", y_pred),
            pl.Series("amount_bucket", buckets),
        ])
        results.append(chunk_result)
        logger.info("  Chunk %d: %s rows predicted", chunk_n, f"{len(chunk_df):,}")

    result_df = pl.concat(results, how="vertical")
    output_path = output_dir / f"{args.policy}_case_predictions.parquet"
    result_df.write_parquet(output_path)
    logger.info("Predictions saved: %s (%d rows)", output_path, len(result_df))

    # Summary
    y_true_all = result_df["y_raw"].to_numpy() if "y_raw" in result_df.columns else None
    y_pred_all = result_df["predicted_amount"].to_numpy()
    from ml.evaluate.metrics import evaluate_predictions
    if y_true_all is not None:
        valid = ~(np.isnan(y_true_all) | np.isnan(y_pred_all))
        m = evaluate_predictions(y_true_all[valid], y_pred_all[valid])
    else:
        m = {}
    summary = {"policy_id": args.policy, "case_count": len(result_df),
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **m}
    (output_dir / f"{args.policy}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Summary saved. MAE: %.0f | Gini: %.3f | Error: %.1f%%",
                 m.get("mae", 0), m.get("gini", 0), m.get("total_error_pct", 0))

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict claim amounts using trained model")
    parser.add_argument("--csv", required=True, help="Doris CSV path")
    parser.add_argument("--policy", default="ALL", help="Policy group to predict (default: all)")
    parser.add_argument("--config", default="ml/config_ml.yaml")
    parser.add_argument("--model-dir", default=None, help="Override model directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--ensemble", action="store_true", help="Use L1-A + L1-B ensemble prediction")
    parser.add_argument("--calibrate", action="store_true", help="Apply group calibration factors")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Chunked prediction: rows per batch (0=all at once)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    config = load_config(args.config)
    model_dir = Path(args.model_dir or config["output"]["model_dir"])
    output_dir = Path(args.output_dir or config["output"]["prediction_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from ml.models.amount_predictor import AmountPredictor
    logger.info("Loading model from %s", model_dir)
    predictor = AmountPredictor.load(model_dir, name="l1a_amount")

    # Load feature schema
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    feature_cols = schema["feature_cols"]
    categorical_cols = schema["categorical_cols"]

    # Load global stats
    cache_dir = Path(config["data"]["cache_dir"])
    stats_path = cache_dir / "global_stats.pkl"
    with open(stats_path, "rb") as f:
        raw_stats = pickle.load(f)

    # Reconstruct stats DataFrames
    global_stats = {}
    for k, v in raw_stats.items():
        if isinstance(v, dict):
            global_stats[k] = pl.DataFrame(v)
        else:
            global_stats[k] = v

    # Load and build features
    from ml.data.loader import load_doris_csv
    from ml.data.feature_store import build_features

    logger.info("Loading CSV and building features...")
    lf = load_doris_csv(args.csv)

    if args.policy != "ALL":
        lf = lf.filter(pl.col("policy_grp_name") == args.policy)
        logger.info("Filtered to policy: %s", args.policy)

    feature_lf = build_features(
        lf, global_stats,
        categorical_cols=categorical_cols,
        log_transform=predictor.log_transform,
    )

    # Collect for prediction — avoid duplicate column names
    logger.info("Collecting data...")
    t0 = time.time()
    # Metadata columns that are NOT already in feature_cols
    _desired_meta = ["y_raw", "medical_start_date", "policy_grp_name", "group_code",
                     "duty_code", "hosp_grade", "case_no"]
    extra_cols = [c for c in _desired_meta if c not in feature_cols]

    if args.chunk_size > 0:
        # Chunked prediction: sink to Parquet → predict in batches
        _run_chunked_predict(
            feature_lf, feature_cols, extra_cols, categorical_cols,
            predictor, model_dir, output_dir, args, config,
        )
        return 0

    pred_df = feature_lf.select(feature_cols + extra_cols).collect(engine="streaming")
    logger.info("Collected %s rows in %.1fs", f"{len(pred_df):,}", time.time() - t0)

    # Predict
    logger.info("Predicting...")
    y_pred = predictor.predict(pred_df, feature_cols, categorical_cols)

    # Quantile predictions — reuse the same numpy matrix from main prediction
    # (AmountPredictor.predict already built X_np internally, but we need to
    # rebuild it here for quantile models since it wasn't returned)
    import pandas as _pd
    
    # Load category mappings for consistent encoding
    cat_mappings_path = model_dir / "category_mappings.json"
    category_mappings = {}
    if cat_mappings_path.exists():
        category_mappings = json.loads(cat_mappings_path.read_text(encoding="utf-8"))
        logger.info("Loaded category mappings: %d cols", len(category_mappings))
    
    X_pred_pd = pred_df.select(feature_cols).to_pandas()
    
    # Apply category mappings BEFORE encoding
    if category_mappings:
        for col, known_cats in category_mappings.items():
            if col in X_pred_pd.columns:
                X_pred_pd[col] = _pd.Categorical(
                    X_pred_pd[col].astype(str),
                    categories=known_cats,
                )
    
    for col in X_pred_pd.columns:
        dtype = str(X_pred_pd[col].dtype)
        if dtype not in ("int8", "int16", "int32", "int64", "float32", "float64"):
            if dtype == "bool":
                X_pred_pd[col] = X_pred_pd[col].astype(np.int8)
            elif dtype != "category":
                X_pred_pd[col] = X_pred_pd[col].astype("category")
    X_np = np.empty((len(X_pred_pd), len(X_pred_pd.columns)), dtype=np.float64)
    for i, col in enumerate(X_pred_pd.columns):
        series = X_pred_pd[col]
        if str(series.dtype) == "category":
            X_np[:, i] = _pd.Categorical(series).codes.astype(np.float64)
        else:
            X_np[:, i] = series.to_numpy(dtype=np.float64, na_value=0.0)

    quantile_preds = {}
    for q_name in ["p05", "p50", "p95"]:
        q_path = model_dir / f"l1a_{q_name}.txt"
        if q_path.exists():
            import lightgbm as lgb
            q_model = lgb.Booster(model_file=str(q_path))
            try:
                q_pred = q_model.predict(X_np)
            except Exception:
                logger.warning("Quantile model %s has feature mismatch — skipping", q_name)
                continue
            if predictor.log_transform:
                q_pred = np.expm1(q_pred)
            quantile_preds[q_name] = np.clip(q_pred, 0, None)

    # ── Ensemble mode (P2-12): L1-A + L1-B + Calibration ──
    ensemble_probas: np.ndarray | None = None
    if args.ensemble:
        from ml.models.large_classifier import LargeClaimClassifier, ensemble_predict

        l1b_path = model_dir / "l1b_classifier.txt"
        if l1b_path.exists():
            l1b = LargeClaimClassifier.load(model_dir, name="l1b_classifier")
            ensemble_probas = l1b.predict_proba(pred_df, feature_cols, categorical_cols)
            logger.info("L1-B loaded: large_claim_prob >0.5: %d cases",
                        int((ensemble_probas > 0.5).sum()))
        else:
            logger.warning("L1-B model not found at %s — skipping ensemble", l1b_path)

    if args.calibrate:
        from ml.models.calibration import GroupCalibrator
        cal_path = model_dir / "calibrator.json"
        if cal_path.exists():
            calibrator = GroupCalibrator.load(cal_path)
            groups = pred_df["policy_grp_name"].cast(pl.Utf8).to_list()
            y_pred = calibrator.calibrate(y_pred, groups)
            logger.info("Applied group calibration (global=%.3f)", calibrator.global_factor)
        else:
            logger.warning("Calibrator not found at %s", cal_path)

    # Amount bucket
    buckets = np.select(
        [y_pred < 500, y_pred < 2000, y_pred < 10000, y_pred < 50000],
        ["0-500", "500-2K", "2K-10K", "10K-50K"],
        default="50K+",
    )

    # Apply bucket correction factors if available
    corr_path = model_dir / "bucket_correction.json"
    if corr_path.exists():
        corr_data = json.loads(corr_path.read_text(encoding="utf-8"))
        bucket_map = {b: s["factor"] for b, s in corr_data.get("bucket_factors", {}).items()}
        y_pred_corrected = y_pred.copy()
        for b, factor in bucket_map.items():
            mask = buckets == b
            y_pred_corrected[mask] = y_pred[mask] * factor
        logger.info("Applied bucket correction factors (global=%.4f)", corr_data.get("global_factor", 1.0))
    else:
        y_pred_corrected = y_pred

    # Build output DataFrame — keep metadata columns + predictions
    output_cols: dict = {
        "predicted_amount": y_pred_corrected,
        "amount_bucket": buckets,
    }
    if corr_path.exists():
        output_cols["predicted_raw"] = y_pred  # pre-correction for audit
    if "p05" in quantile_preds:
        output_cols["predicted_p05"] = quantile_preds["p05"]
    if "p95" in quantile_preds:
        output_cols["predicted_p95"] = quantile_preds["p95"]
    if "p50" in quantile_preds:
        output_cols["predicted_p50"] = quantile_preds["p50"]
    if ensemble_probas is not None:
        output_cols["large_claim_proba"] = ensemble_probas

    # Preserve metadata columns that exist (cast categorical→string for Parquet)
    meta_keep = []
    _cat_cols = {"duty_code", "hosp_grade", "policy_grp_name", "claim_type",
                 "fee_item_type", "insured_gender", "rnew_flag", "sale_chnl",
                 "main_insured_rela", "is_public", "is_expensive"}
    for c in ["medical_start_date", "policy_grp_name", "group_code",
              "duty_code", "hosp_grade", "case_no"]:
        if c in pred_df.columns:
            if c in _cat_cols:
                meta_keep.append(pl.col(c).cast(pl.Utf8))
            else:
                meta_keep.append(pl.col(c))

    result_df = pred_df.select(meta_keep + [
        pl.col("y_raw").alias("actual_amount"),
    ]).with_columns([
        pl.Series(name, values) for name, values in output_cols.items()
    ])

    # Save Parquet
    output_path = output_dir / f"{args.policy}_case_predictions.parquet"
    result_df.write_parquet(output_path)
    logger.info("Predictions saved: %s (%d rows)", output_path, len(result_df))

    # Summary JSON (use corrected predictions if available)
    from ml.evaluate.metrics import evaluate_predictions
    y_true = pred_df["y_raw"].to_numpy()
    y_pred_eval = y_pred_corrected  # use corrected for evaluation
    # Filter out NaN rows (missing actual amounts)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred_eval))
    if valid_mask.sum() < len(y_true):
        logger.info("Filtered %d NaN rows from evaluation",
                    len(y_true) - valid_mask.sum())
    y_true_eval = y_true[valid_mask]
    y_pred_eval_arr = y_pred_eval[valid_mask]
    summary_metrics = evaluate_predictions(y_true_eval, y_pred_eval_arr)

    # Per-bucket stats
    bucket_stats = {}
    for b in ["0-500", "500-2K", "2K-10K", "10K-50K", "50K+"]:
        mask = result_df["amount_bucket"] == b
        cnt = int(mask.sum())
        if cnt > 0:
            bucket_stats[b] = {
                "count": cnt,
                "actual_total": float(result_df.filter(mask)["actual_amount"].sum()),
                "predicted_total": float(result_df.filter(mask)["predicted_amount"].sum()),
            }

    summary = {
        "policy_id": args.policy,
        "case_count": len(pred_df),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_version": "l1a_v1",
        "ensemble": bool(args.ensemble and ensemble_probas is not None),
        "calibrated": bool(args.calibrate),
        **summary_metrics,
        "bucket_stats": bucket_stats,
    }

    summary_path = output_dir / f"{args.policy}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Summary saved: %s", summary_path)
    logger.info("MAE: %.0f | MAPE: %.1f%% | Gini: %.3f | 总量误差: %.1f%%",
                summary_metrics["mae"], summary_metrics["mape"],
                summary_metrics["gini"], summary_metrics.get("total_error_pct", 0))

    return 0


if __name__ == "__main__":
    sys.exit(main())

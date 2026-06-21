"""SHAP explainability for LightGBM insurance claim models.

Reference: Insurance-Claims-Prediction-ML uses SHAP for regulatory compliance.

Generates feature importance, SHAP values, and summary plots for model
interpretability — required for insurance regulatory compliance (OSFI E-23).

Usage:
    python -m ml.evaluate.shap_explainer --model models/l1a_amount.txt --output reports/shap/
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger("ml.shap")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def compute_shap_values(
    model_path: str | Path,
    data_path: str | Path = "data/ml_cache/train_cache.parquet",
    sample_n: int = 5000,
    output_dir: str | Path = "reports/shap",
) -> Dict:
    """Compute SHAP values and generate interpretability report.

    Args:
        model_path: Path to LightGBM model (.txt)
        data_path: Path to training data parquet
        sample_n: Number of samples for SHAP computation
        output_dir: Output directory for SHAP report

    Returns:
        Dict with top features, SHAP summary, and report path
    """
    import lightgbm as lgb

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = lgb.Booster(model_file=str(model_path))
    feature_names = model.feature_name()
    logger.info("Model: %d trees, %d features", model.num_trees(), len(feature_names))

    # Load data sample
    df = pl.read_parquet(data_path).sample(n=min(sample_n, pl.read_parquet(data_path).height), seed=42)
    logger.info("SHAP data: %d rows", len(df))

    # Build feature matrix (numeric only)
    existing = [c for c in feature_names if c in df.columns]
    X = np.zeros((len(df), len(existing)), dtype=np.float64)
    for i, col in enumerate(existing):
        try:
            X[:, i] = df[col].to_numpy().astype(np.float64)
        except Exception:
            try:
                X[:, i] = df[col].cast(pl.Utf8).hash(42).mod(10 ** 9).cast(pl.Float64).to_numpy()
            except Exception:
                X[:, i] = 0.0

    X = np.nan_to_num(X, 0)

    # Compute SHAP via TreeSHAP
    logger.info("Computing SHAP values (TreeSHAP)...")
    try:
        import shap
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X[:sample_n])
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For multi-output models

        # Feature importance from SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(-shap_importance)[:20]

        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": existing[idx],
                "shap_importance": float(shap_importance[idx]),
                "mean_shap": float(shap_values[:, idx].mean()),
                "std_shap": float(shap_values[:, idx].std()),
                "max_shap": float(shap_values[:, idx].max()),
            })

        # Generate SHAP summary plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, X[:sample_n], feature_names=existing,
                              max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("SHAP summary saved: %s", output_dir / "shap_summary.png")
        except Exception as e:
            logger.warning("SHAP plot generation failed: %s", e)

        # Generate bar plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_names = [f["feature"] for f in top_features[:15]]
            top_vals = [f["shap_importance"] for f in top_features[:15]]
            ax.barh(range(len(top_names)), top_vals[::-1])
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names[::-1])
            ax.set_xlabel("mean(|SHAP|)")
            ax.set_title("SHAP Feature Importance (Top 15)")
            plt.tight_layout()
            plt.savefig(output_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning("SHAP bar plot failed: %s", e)

        # Save SHAP report
        report = {
            "model": str(model_path),
            "n_samples": len(X),
            "n_features": len(existing),
            "top_features": top_features,
            "calibration_note": "SHAP values show directional impact. Use for regulatory compliance documentation.",
        }
        (output_dir / "shap_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("SHAP report saved: %s", output_dir / "shap_report.json")

        # Print top 5
        logger.info("Top 5 SHAP features:")
        for f in top_features[:5]:
            logger.info("  %-30s SHAP=%.4f (mean=%.4f)", f["feature"], f["shap_importance"], f["mean_shap"])

        return report

    except ImportError:
        logger.warning("shap not installed. Install: pip install shap")
        return {"error": "shap_not_installed", "top_features": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SHAP explainability analysis")
    parser.add_argument("--model", default="models/l1a_amount.txt")
    parser.add_argument("--data", default="data/ml_cache/train_cache.parquet")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output", default="reports/shap")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    compute_shap_values(args.model, args.data, args.samples, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

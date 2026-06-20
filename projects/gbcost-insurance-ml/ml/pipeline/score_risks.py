"""Generate ML risk scores for risk map G10/G11 consumption.

Usage:
    python -m ml.pipeline.score_risks --csv data/doris/c001_ghb_poicy_clm_duty_d.csv
    python -m ml.pipeline.score_risks --output data/ml_cache/ml_risk_map.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import polars as pl

logger = logging.getLogger("ml.risk_score")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ML risk scores for risk map")
    parser.add_argument("--csv", default="data/doris/c001_ghb_poicy_clm_duty_d.csv")
    parser.add_argument("--output", default="data/ml_cache/ml_risk_map.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    from ml.models.amount_predictor import AmountPredictor
    from ml.models.large_classifier import LargeClaimClassifier
    from ml.models.policy_predictor import PolicyPredictor
    from ml.data.risk_scorer import MLRiskScorer
    from ml.data.loader import load_doris_csv
    from ml.data.feature_store import build_features
    from ml.data.policy_aggregator import aggregate_policy_features, get_policy_feature_cols

    logger.info("Loading models...")
    l1a = AmountPredictor.load("models", "l1a_amount")
    l1b = LargeClaimClassifier.load("models", "l1b_classifier")
    l2 = PolicyPredictor.load("models", "l2_policy")

    logger.info("Loading features...")
    schema = json.loads(open("models/feature_schema.json").read())
    feature_cols = schema["feature_cols"]
    cat_cols = schema["categorical_cols"]
    l1b_feature_cols = [c for c in feature_cols if c in l1b.feature_names]
    logger.info("  L1-A: %d, L1-B: %d features", len(feature_cols), len(l1b_feature_cols))

    class _Dummy:
        def predict(self, x): return [0]
    scorer = MLRiskScorer(l1a, l1b, l2, _Dummy(), _Dummy())

    with open("data/ml_cache/global_stats.pkl", "rb") as f:
        raw = pickle.load(f)
    global_stats = {k: (pl.DataFrame(v) if isinstance(v, dict) else v) for k, v in raw.items()}

    lf = load_doris_csv(args.csv)
    feature_lf = build_features(lf, global_stats, categorical_cols=cat_cols, log_transform=False)

    extra = ["y_raw", "policy_grp_name", "group_code", "hosp_grade", "insured_no", "medical_start_date"]
    sel = feature_cols + [c for c in extra if c not in feature_cols]
    case_df = feature_lf.select(sel).filter(
        pl.col("medical_start_date").hash(seed=42) % 100 < 30
    ).collect(engine="streaming")
    logger.info("Cases: %s", f"{len(case_df):,}")

    policy_lf = aggregate_policy_features(lf)
    p_feature_cols = get_policy_feature_cols(policy_lf)
    policy_df = policy_lf.filter(
        pl.col("loss_ratio").is_not_null() & (pl.col("loss_ratio") > 0)
    ).collect(engine="streaming")
    logger.info("Policies: %d", len(policy_df))

    logger.info("Scoring risk map dimensions...")
    risk_map = scorer.generate_risk_map(
        case_df, policy_df, feature_cols, cat_cols,
        l1b_feature_cols, cat_cols, p_feature_cols,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(risk_map, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    disease_high = sum(1 for d in risk_map["disease"] if d.get("risk_level") == "high")
    member_top = risk_map["member"][0]["risk_score"] if risk_map["member"] else 0
    unit_critical = sum(1 for u in risk_map["policy_unit"] if u.get("risk_level") == "critical")

    logger.info("Risk map saved: %s", output)
    logger.info("  Disease high-risk groups: %d", disease_high)
    logger.info("  Member top risk score: %.0f", member_top)
    logger.info("  Policy unit critical: %d", unit_critical)

    return 0


if __name__ == "__main__":
    sys.exit(main())

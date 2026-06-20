"""ML Risk Scorer — converts model predictions to risk map dimension scores.

Each method maps to a G10 risk map dimension (disease/hospital/member/policy-unit).
Output is JSON-compatible for direct TUI consumption.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class MLRiskScorer:
    """Compute ML-driven risk scores for all 4 risk map dimensions."""

    def __init__(self, l1a_predictor, l1b_classifier, l2_predictor, lstm_model, calibrator):
        self.l1a = l1a_predictor
        self.l1b = l1b_classifier
        self.l2 = l2_predictor
        self.lstm = lstm_model
        self.calibrator = calibrator

    def _add_predictions(self, df, l1a_cols, l1a_cats, l1b_cols, l1b_cats):
        df = df.with_columns([
            pl.Series("_l1a_pred", self.l1a.predict(df, l1a_cols, l1a_cats)),
            pl.Series("_l1b_proba", self.l1b.predict_proba(df, l1b_cols, l1b_cats)),
        ])
        return df

    def score_disease_dimension(self, df, l1a_cols, l1a_cats, l1b_cols, l1b_cats):
        df = self._add_predictions(df, l1a_cols, l1a_cats, l1b_cols, l1b_cats)
        scores = df.group_by("group_code").agg(
            pl.col("_l1a_pred").mean().alias("avg_predicted"),
            pl.col("_l1a_pred").sum().alias("total_predicted"),
            pl.col("_l1b_proba").mean().alias("large_prob"),
            pl.len().alias("case_count"),
        ).filter(pl.col("case_count") >= 5).sort("avg_predicted", descending=True)
        if len(scores) > 0:
            scores = scores.with_columns(
                pl.col("avg_predicted")
                .qcut([0.33, 0.67], labels=["low", "medium", "high"], include_breaks=False)
                .alias("risk_level")
            )
        return scores

    def score_hospital_dimension(self, df, l1a_cols, l1a_cats, l1b_cols, l1b_cats):
        df = self._add_predictions(df, l1a_cols, l1a_cats, l1b_cols, l1b_cats)
        scores = df.group_by("hosp_grade").agg(
            pl.col("_l1a_pred").mean().alias("avg_predicted"),
            pl.col("_l1a_pred").std().alias("std_predicted"),
            pl.col("_l1b_proba").mean().alias("large_prob"),
            pl.col("_l1b_proba").quantile(0.95).alias("p95_large"),
            pl.len().alias("case_count"),
        ).filter(pl.col("case_count") >= 5)
        grand_mean = scores["large_prob"].mean() or 1.0
        scores = scores.with_columns((pl.col("large_prob") / grand_mean).alias("risk_ratio"))
        return scores.sort("risk_ratio", descending=True)

    def score_member_dimension(self, df, l1a_cols, l1a_cats, l1b_cols, l1b_cats):
        df = self._add_predictions(df, l1a_cols, l1a_cats, l1b_cols, l1b_cats)
        scores = df.group_by("insured_no").agg(
            pl.col("_l1a_pred").sum().alias("total_predicted"),
            pl.col("_l1b_proba").mean().alias("large_prob"),
            pl.col("_l1b_proba").max().alias("max_large_prob"),
            pl.len().alias("claim_count"),
        ).filter(pl.col("claim_count") >= 2)
        if len(scores) > 0:
            rank_total = pl.col("total_predicted").rank("ordinal") / pl.len() * 50
            rank_prob = pl.col("large_prob").rank("ordinal") / pl.len() * 50
            scores = scores.with_columns((rank_total + rank_prob).alias("risk_score"))
        return scores.sort("risk_score", descending=True).head(100)

    def score_policy_unit_dimension(self, policy_df, feature_cols):
        preds = self.l2.predict(policy_df, feature_cols)
        df = policy_df.select("policy_grp_name").with_columns(
            pl.Series("predicted_loss_ratio", np.clip(preds, 0, None)),
        )
        if len(df) > 0:
            df = df.with_columns(
                pl.col("predicted_loss_ratio")
                .qcut([0.25, 0.50, 0.75], labels=["low", "medium", "high", "critical"], include_breaks=False)
                .alias("risk_level")
            )
        return df.sort("predicted_loss_ratio", descending=True)

    def generate_risk_map(self, case_df, policy_df, l1a_cols, l1a_cats, l1b_cols, l1b_cats, p_feature_cols):
        l1b_c = [c for c in l1a_cats if c in l1b_cols]
        return {
            "disease": self.score_disease_dimension(case_df, l1a_cols, l1a_cats, l1b_cols, l1b_c).to_dicts(),
            "hospital": self.score_hospital_dimension(case_df, l1a_cols, l1a_cats, l1b_cols, l1b_c).to_dicts(),
            "member": self.score_member_dimension(case_df, l1a_cols, l1a_cats, l1b_cols, l1b_c).to_dicts(),
            "policy_unit": self.score_policy_unit_dimension(policy_df, p_feature_cols).to_dicts(),
        }

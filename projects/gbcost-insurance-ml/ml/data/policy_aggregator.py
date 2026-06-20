"""Policy-level feature aggregation from case-level data.

Transforms case-level claims into policy-level features for L2 prediction.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)


def aggregate_policy_features(
    lf: pl.LazyFrame,
    train_end: str = "2024-12-31",
    target_col: str = "clm_acount_amnt_cny",
    premium_col: str = "annual_prem",
) -> pl.LazyFrame:
    """Build policy-level feature set from case-level data.

    Groups by policy_grp_name and computes:
      - Claim statistics: count, sum, mean, median, std, skew, max, p95
      - Member statistics: unique members, claims per member
      - Temporal: months active, claims per month
      - Premium & loss ratio
      - Hospital & disease diversity

    Args:
        lf: Case-level LazyFrame (from load_doris_csv)
        train_end: Training period cutoff (for loss ratio calculation)
        target_col: Claim amount column
        premium_col: Annual premium column

    Returns:
        LazyFrame with one row per policy
    """
    train_cutoff = date.fromisoformat(train_end)

    # Filter train period for target leakage prevention
    train_lf = lf.filter(pl.col("medical_start_date") <= pl.lit(train_cutoff))

    # Per-policy aggregations
    policy_lf = (
        train_lf
        .group_by("policy_grp_name")
        .agg(
            # Claim statistics
            pl.col(target_col).sum().alias("policy_total_claims"),
            pl.col(target_col).mean().alias("policy_avg_claim"),
            pl.col(target_col).median().alias("policy_median_claim"),
            pl.col(target_col).std().alias("policy_std_claim"),
            pl.col(target_col).max().alias("policy_max_claim"),
            pl.col(target_col).quantile(0.95).alias("policy_p95_claim"),
            pl.col(target_col).count().alias("policy_claim_count"),

            # Member statistics
            pl.col("insured_no").n_unique().alias("policy_member_count"),
            (pl.col(target_col).count() / pl.col("insured_no").n_unique()).alias("claims_per_member"),

            # Temporal
            (pl.col("medical_start_date").max() - pl.col("medical_start_date").min())
            .dt.total_days().alias("policy_active_days"),
            pl.col("medical_start_date").dt.truncate("1mo").n_unique().alias("policy_active_months"),
            (pl.col(target_col).count() / pl.col("medical_start_date").dt.truncate("1mo").n_unique())
            .alias("claims_per_month"),

            # Hospital & disease diversity
            pl.col("hosp_grade").n_unique().alias("policy_hosp_diversity"),
            pl.col("group_code").n_unique().alias("policy_disease_diversity"),
            pl.col("fee_item_type").n_unique().alias("policy_fee_diversity"),

            # Large claim proportion
            ((pl.col(target_col) > 24600).sum() / pl.col(target_col).count())
            .alias("policy_large_claim_ratio"),

            # Day count stats
            pl.col("day_count").mean().alias("policy_avg_days"),
            pl.col("day_count").max().alias("policy_max_days"),

            # Metadata (take max/first)
            pl.col("sale_chnl").first().alias("policy_sale_chnl"),
            pl.col("rnew_flag").first().alias("policy_renewal"),
        )
    )

    # Premium and loss ratio
    premium_df = (
        train_lf
        .group_by("policy_grp_name")
        .agg(
            pl.col(premium_col).mean().alias("policy_avg_premium"),
            pl.col("pass_months").max().alias("policy_pass_months"),
        )
    )

    policy_lf = policy_lf.join(premium_df, on="policy_grp_name", how="left")

    # Derived features — computed in stages to avoid Polars circular reference
    policy_lf = policy_lf.with_columns([
        (pl.col("policy_total_claims") / pl.col("policy_avg_premium").clip(1, None))
        .alias("loss_ratio"),
        (pl.col("policy_total_claims") / pl.col("policy_member_count").clip(1, None))
        .alias("cost_per_member"),
        (pl.col("policy_active_days").cast(pl.Float64) / 365.25)
        .alias("policy_active_years"),
    ])

    policy_lf = policy_lf.with_columns(
        (pl.col("claims_per_month") / pl.col("policy_member_count").clip(1, None))
        .alias("claims_per_member_per_month"),
    )

    logger.info("Policy features built: %d columns", len(policy_lf.collect_schema().names()))
    return policy_lf


def get_policy_feature_cols(lf: pl.LazyFrame, target_col: str = "loss_ratio") -> List[str]:
    """Get policy-level feature column names, excluding IDs, targets, and strings."""
    exclude = {
        "policy_grp_name", target_col, "policy_total_claims",
        "policy_avg_premium", "policy_pass_months",
        "policy_sale_chnl", "policy_renewal",  # string columns
    }
    return sorted(c for c in lf.collect_schema().names() if c not in exclude)

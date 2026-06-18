"""Two-phase feature engineering for claim amount prediction.

Phase 1 (compute_global_stats): Scan full dataset once to compute aggregation
statistics — group_code target encoding, monthly aggregations. Results are
small DataFrames stored in memory.

Phase 2 (build_features): Transform LazyFrame with all feature expressions,
including joins to global stats. No data is collected until train/predict
calls .collect(streaming=True).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)

# P95 threshold for large claim (from v2 data profiling)
_LARGE_CLAIM_THRESHOLD = 24600.0

# Bayesian smoothing factor for target encoding
_TE_SMOOTHING = 10.0

# Numeric features that get log1p transformation
_LOG_COLS = ["annual_prem", "pass_prem", "year_actual_claim_paid"]


def compute_global_stats(
    lf: pl.LazyFrame,
    train_end: str = "2024-12-31",
    target_col: str = "clm_acount_amnt_cny",
) -> Dict[str, pl.DataFrame]:
    """Pass 1: Scan full dataset to compute aggregation statistics.

    Only uses training-period data (before train_end) to prevent leakage.

    Args:
        lf: Source LazyFrame
        train_end: Last training date (inclusive)
        target_col: Target column for encoding

    Returns:
        Dict of small DataFrames:
            - 'group_code_te': ICD group target encoding map
            - 'native_place_te': Native place target encoding map
            - 'monthly': Monthly aggregation by policy group
            - 'global_mean': Overall target mean (for smoothing)
    """
    logger.info("Computing global stats (train period: <= %s)...", train_end)
    from datetime import date as _date
    train_cutoff_date = _date.fromisoformat(train_end)

    train_lf = lf.filter(
        (pl.col("medical_start_date") <= pl.lit(train_cutoff_date))
        & pl.col(target_col).is_not_null()
        & (pl.col(target_col) > 0)
    )

    stats: Dict[str, pl.DataFrame] = {}

    # --- Global mean (for Bayesian smoothing) ---
    global_mean = train_lf.select(
        pl.col(target_col).mean().alias("global_mean"),
        pl.col(target_col).std().alias("global_std"),
        pl.len().alias("global_count"),
    ).collect()
    stats["global"] = global_mean
    logger.info("  Global mean: %s", global_mean["global_mean"][0])

    # --- Target Encoding for high-cardinality columns ---
    for col in ["group_code", "native_place_name"]:
        te_df = (
            train_lf
            .group_by(col)
            .agg(
                pl.col(target_col).mean().alias(f"{col}_mean"),
                pl.col(target_col).std().alias(f"{col}_std"),
                pl.col(target_col).count().alias(f"{col}_count"),
            )
            .with_columns(
                # Bayesian smoothing: shrink low-count groups toward global mean
                (
                    (pl.col(f"{col}_mean") * pl.col(f"{col}_count") +
                     global_mean["global_mean"][0] * _TE_SMOOTHING) /
                    (pl.col(f"{col}_count") + _TE_SMOOTHING)
                ).alias(f"{col}_te")
            )
            .collect()
        )
        stats[col] = te_df
        logger.info("  %s TE: %d groups", col, len(te_df))

    # --- Monthly aggregations by policy group (for lag features) ---
    monthly_df = (
        train_lf
        .with_columns(
            pl.col("medical_start_date").dt.truncate("1mo").alias("year_month")
        )
        .group_by(["policy_grp_name", "year_month"])
        .agg(
            pl.col(target_col).sum().alias("monthly_total"),
            pl.col(target_col).mean().alias("monthly_avg"),
            pl.col(target_col).median().alias("monthly_median"),
            pl.col("case_no").count().alias("monthly_count"),
            pl.col("clm_acount_amnt_cny_duty").sum().alias("monthly_duty_total"),
        )
        .sort(["policy_grp_name", "year_month"])
        .collect()
    )
    stats["monthly"] = monthly_df
    logger.info("  Monthly stats: %d rows (%d policy groups)",
                len(monthly_df), monthly_df["policy_grp_name"].n_unique())

    logger.info("Global stats computed: %d tables", len(stats))
    return stats


def build_features(
    lf: pl.LazyFrame,
    global_stats: Dict[str, pl.DataFrame],
    categorical_cols: Optional[List[str]] = None,
    log_transform: bool = True,
) -> pl.LazyFrame:
    """Pass 2: Transform LazyFrame with all feature expressions.

    Returns a LazyFrame (not collected). Caller decides when to materialize.

    Args:
        lf: Source LazyFrame
        global_stats: Output from compute_global_stats()
        categorical_cols: Columns to cast as Categorical for LightGBM
        log_transform: Whether to apply log1p to target

    Returns:
        LazyFrame with engineered features
    """
    if categorical_cols is None:
        categorical_cols = [
            "duty_code", "policy_grp_name", "insured_gender",
            "rnew_flag", "claim_type", "sale_chnl", "hosp_grade",
            "main_insured_rela", "is_public", "is_expensive",
            "fee_item_type",
        ]

    feature_lf = lf

    # ================================================================
    # 1. Identity features
    # ================================================================
    feature_lf = feature_lf.with_columns([
        # Age at time of medical visit
        ((pl.col("medical_start_date") - pl.col("cust_birthday")).dt.total_days() / 365.25)
            .alias("age"),
    ]).with_columns([
        pl.when(pl.col("age") < 18).then(pl.lit("0-18"))
          .when(pl.col("age") < 30).then(pl.lit("18-30"))
          .when(pl.col("age") < 45).then(pl.lit("30-45"))
          .when(pl.col("age") < 60).then(pl.lit("45-60"))
          .otherwise(pl.lit("60+"))
          .alias("age_bucket"),
    ])

    # ================================================================
    # 2. Policy features
    # ================================================================
    policy_exprs = [
        pl.col("pass_months").cast(pl.Float64).alias("pass_months"),
    ]

    for col in _LOG_COLS:
        if col in feature_lf.collect_schema().names():
            policy_exprs.append(
                pl.col(col).cast(pl.Float64).log1p().alias(f"{col}_log")
            )

    # Policy age in days
    policy_exprs.append(
        (pl.col("medical_start_date") - pl.col("cont_valid_date"))
            .dt.total_days().alias("policy_age_days")
    )

    # Premium per month
    if "annual_prem" in feature_lf.collect_schema().names():
        policy_exprs.append(
            (pl.col("annual_prem").cast(pl.Float64) /
             pl.when(pl.col("pass_months") > 0).then(pl.col("pass_months"))
             .otherwise(12))
            .alias("monthly_prem")
        )

    feature_lf = feature_lf.with_columns(policy_exprs)

    # ================================================================
    # 3. Medical features
    # ================================================================
    medical_exprs = [
        pl.col("day_count").cast(pl.Float64).fill_null(0).alias("day_count_filled"),
        pl.col("est_loss_ratio").cast(pl.Float64).fill_null(0).alias("est_loss_ratio"),
    ]

    # Large case amount fill
    if "large_case_amt" in feature_lf.collect_schema().names():
        medical_exprs.append(
            pl.col("large_case_amt").cast(pl.Float64).fill_null(0).alias("large_case_filled")
        )

    feature_lf = feature_lf.with_columns(medical_exprs)

    # Cast categorical columns
    cat_exprs = []
    schema_names = feature_lf.collect_schema().names()
    for col in categorical_cols:
        if col in schema_names:
            cat_exprs.append(pl.col(col).cast(pl.Categorical))
    if cat_exprs:
        feature_lf = feature_lf.with_columns(cat_exprs)

    # ================================================================
    # 4. Temporal features
    # ================================================================
    feature_lf = feature_lf.with_columns([
        pl.col("medical_start_date").dt.month().alias("month"),
        pl.col("medical_start_date").dt.quarter().alias("quarter"),
        pl.col("medical_start_date").dt.weekday().alias("day_of_week"),
    ]).with_columns([
        (pl.col("month").cast(pl.Float64) * 2.0 * math.pi / 12.0).sin().alias("month_sin"),
        (pl.col("month").cast(pl.Float64) * 2.0 * math.pi / 12.0).cos().alias("month_cos"),
        pl.col("month").is_in([11, 12, 1]).alias("is_yearend"),
    ])

    # ================================================================
    # 5. Target Encoding joins
    # ================================================================
    # Join group_code TE (convert DataFrame → LazyFrame for join)
    if "group_code" in global_stats:
        feature_lf = feature_lf.join(
            global_stats["group_code"].select(["group_code", "group_code_te"]).lazy(),
            on="group_code",
            how="left",
        ).with_columns(
            pl.col("group_code_te").fill_null(
                global_stats["global"]["global_mean"][0]
            )
        )

    # Join native_place TE
    if "native_place_name" in global_stats:
        feature_lf = feature_lf.join(
            global_stats["native_place_name"].select(
                ["native_place_name", "native_place_name_te"]
            ).lazy(),
            on="native_place_name",
            how="left",
        ).with_columns(
            pl.col("native_place_name_te").fill_null(
                global_stats["global"]["global_mean"][0]
            )
        )

    # ================================================================
    # 6. Monthly aggregated features (join + lag)
    # ================================================================
    monthly = global_stats.get("monthly")
    if monthly is not None:
        feature_lf = feature_lf.with_columns(
            pl.col("medical_start_date").dt.truncate("1mo").alias("year_month")
        ).join(
            monthly.lazy().with_columns(
                pl.col("policy_grp_name").cast(pl.Categorical)
            ),
            on=["policy_grp_name", "year_month"],
            how="left",
        )

    # ================================================================
    # 7. Target variables
    # ================================================================
    target_exprs = [
        pl.col("clm_acount_amnt_cny").cast(pl.Float64).alias("y_raw"),
        (pl.col("clm_acount_amnt_cny") > _LARGE_CLAIM_THRESHOLD).alias("is_large"),
    ]
    if log_transform:
        target_exprs.append(
            pl.col("clm_acount_amnt_cny").cast(pl.Float64).log1p().alias("y_log")
        )

    feature_lf = feature_lf.with_columns(target_exprs)

    logger.info("Feature LazyFrame built (not yet collected)")
    return feature_lf


def get_feature_columns(
    lf: pl.LazyFrame,
    exclude_cols: List[str],
    target_cols: Optional[List[str]] = None,
) -> List[str]:
    """Determine which columns are features (excluding IDs, targets, dates).

    Args:
        lf: Feature LazyFrame
        exclude_cols: Explicitly excluded columns
        target_cols: Target-related columns to exclude

    Returns:
        Sorted list of feature column names
    """
    if target_cols is None:
        target_cols = ["y_raw", "y_log", "is_large", "clm_acount_amnt_cny"]

    all_cols = set(lf.collect_schema().names())
    excluded = set(exclude_cols + target_cols + [
        "year_month", "medical_start_date", "medical_end_date",
        "cust_birthday", "cont_valid_date", "cont_end_date",
        "fir_vald_date", "end_case_date", "etl_date",
        "case_no", "old_clm_no", "insured_no", "appnt_no", "rgt_no",
        "insured_name", "appnt_name", "hospital_name",
        "group_name", "duty_name", "fee_name", "fee_desc",
        "fee_item_detail_name", "fee_code", "sub_duty", "sub_duty_code",
        "claim_duty", "grpcont_claim_duty_code", "sale_chnl_code",
        "main_hospital_name", "grp_hospital_name", "medical_group_name",
        "acc_result1_name", "acc_result2_name",
        "sale_depart", "agent_com_name", "agent_name",
        "policy_grp_no", "grp_cont_no",
        "sale_us_eva", "min_uw_raise", "investment_approver", "change_fee",
        "vip_type", "pass_year",
        "native_place_name", "group_code",  # TE versions used instead
        # Raw versions that have log counterparts
        "annual_prem", "pass_prem", "year_actual_claim_paid",
        "clm_acount_amnt_cny_duty",
        "ytd_actual_claim_paid",
        "actual_sales_lr",
    ])

    features = sorted(all_cols - excluded)
    logger.info("Feature columns: %d", len(features))
    return features

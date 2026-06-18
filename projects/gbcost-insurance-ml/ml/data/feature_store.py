"""Two-phase feature engineering for claim amount prediction.

Phase 0 (prepare_train_cache): Single CSV scan → filtered Parquet cache.
    Subsequent runs read from Parquet (columnar, compressed) — 10x faster.

Phase 1 (compute_global_stats): Compute aggregation statistics from LazyFrame
    (CSV or Parquet source).

Phase 2 (build_features): Transform LazyFrame with all feature expressions.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)

# P95 threshold for large claim (from v2 data profiling)
_LARGE_CLAIM_THRESHOLD = 24600.0

# Bayesian smoothing factor for target encoding
_TE_SMOOTHING = 10.0

# Numeric features that get log1p transformation
_LOG_COLS = ["annual_prem", "pass_prem", "year_actual_claim_paid"]

# Columns needed for training (subset of the 71 CSV columns)
_CACHE_COLS = [
    # Identity
    "cust_birthday", "medical_start_date", "medical_end_date",
    # Target
    "clm_acount_amnt_cny", "clm_acount_amnt_cny_duty",
    # Policy
    "policy_grp_name", "duty_code", "duty_name",
    "pass_months", "annual_prem", "pass_prem",
    "cont_valid_date", "cont_end_date",
    "rnew_flag", "sale_chnl", "sale_chnl_code",
    # Member
    "insured_no", "insured_gender", "insured_name",
    "main_insured_rela", "appnt_no", "appnt_name",
    # Medical
    "group_code", "group_name",
    "claim_type", "hosp_grade", "fee_item_type",
    "hospital_name", "main_hospital_name",
    "day_count", "large_case_amt",
    "case_no", "old_clm_no",
    # Finance
    "est_loss_ratio", "actual_sales_lr",
    "year_actual_claim_paid", "ytd_actual_claim_paid",
    # Geography
    "native_place_name",
    # Flags
    "is_public", "is_expensive",
    # Additional IDs
    "rgt_no", "policy_grp_no", "grp_cont_no",
    "fee_code", "fee_name", "fee_desc", "fee_item_detail_name",
    "sub_duty", "sub_duty_code", "claim_duty", "grpcont_claim_duty_code",
    "vip_type", "pass_year",
    # Dates
    "end_case_date", "fir_vald_date", "etl_date",
    # Agent info
    "sale_depart", "agent_com_name", "agent_name",
    "sale_us_eva", "min_uw_raise", "investment_approver", "change_fee",
    "grp_hospital_name", "medical_group_name",
    "acc_result1_name", "acc_result2_name",
]


def prepare_train_cache(
    csv_path: str | Path,
    cache_dir: str | Path = "data/ml_cache",
    train_end: str = "2024-12-31",
    force: bool = False,
) -> Path:
    """Single-pass CSV → filtered Parquet cache.

    Scans the 13GB CSV once, filters to training period (≤ train_end),
    and writes compressed Parquet (~2-3GB). All subsequent loads use
    this cache — no more full CSV scans.

    Args:
        csv_path: Path to Doris CSV
        cache_dir: Directory for cached parquet
        train_end: Training period cutoff date
        force: If True, regenerate even if cache exists

    Returns:
        Path to cached Parquet file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "train_cache.parquet"

    if cache_path.exists() and not force:
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info("Cache exists: %s (%.0f MB) — skipping CSV scan", cache_path.name, size_mb)
        return cache_path

    logger.info("Building train cache from CSV → Parquet (train_end <= %s)...", train_end)

    from ml.data.loader import load_doris_csv
    from datetime import date as _date

    train_cutoff = _date.fromisoformat(train_end)
    t0 = time.time()

    lf = load_doris_csv(csv_path)
    train_lf = lf.filter(
        (pl.col("medical_start_date") <= pl.lit(train_cutoff))
        & pl.col("clm_acount_amnt_cny").is_not_null()
        & (pl.col("clm_acount_amnt_cny") > 0)
    )

    # Only keep columns we need for training (reduces Parquet size)
    schema = train_lf.collect_schema()
    available_cols = [c for c in _CACHE_COLS if c in schema.names()]

    train_lf = train_lf.select(available_cols)

    train_lf.sink_parquet(
        str(cache_path),
        compression="zstd",
        compression_level=3,
        statistics=True,
    )

    elapsed = time.time() - t0
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    logger.info("Train cache ready: %s (%.0f MB, %.1f min)", cache_path.name, size_mb, elapsed / 60)

    # Print policy count for completeness check
    policy_count = (
        pl.scan_parquet(str(cache_path))
        .select(pl.col("policy_grp_name").n_unique())
        .collect()
        .item()
    )
    logger.info("  Policies in cache: %d", policy_count)

    return cache_path


def load_train_lf(cache_path: str | Path) -> pl.LazyFrame:
    """Load cached training data as LazyFrame (no data loaded yet)."""
    return pl.scan_parquet(str(cache_path))


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

    # --- Member-level historical features (P1-6) ---
    if "insured_no" in train_lf.collect_schema().names():
        member_stats = (
            train_lf
            .group_by("insured_no")
            .agg(
                pl.col(target_col).sum().alias("member_total_paid"),
                pl.col(target_col).mean().alias("member_avg_paid"),
                pl.col(target_col).std().alias("member_std_paid"),
                pl.col(target_col).count().alias("member_claim_count"),
                pl.col("case_no").n_unique().alias("member_case_count"),
                pl.col("medical_start_date").max().alias("member_last_claim_date"),
                pl.col("medical_start_date").min().alias("member_first_claim_date"),
                pl.col("hosp_grade").n_unique().alias("member_hosp_count"),
                pl.col("group_code").n_unique().alias("member_disease_count"),
            )
            .with_columns(
                (pl.col("member_claim_count") /
                 ((pl.col("member_last_claim_date") - pl.col("member_first_claim_date"))
                  .dt.total_days() / 365.25).clip(0.5, None)
                ).alias("member_claim_rate"),
                (pl.col("member_total_paid") / pl.col("member_claim_count").clip(1, None))
                .alias("member_avg_claim_size"),
            )
            .collect()
        )
        stats["member"] = member_stats
        logger.info("  Member stats: %d members, avg claims=%.1f",
                    len(member_stats),
                    member_stats["member_claim_count"].mean())

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

        # P1-4: Lag features (shift by 1/2/3 months within each policy group)
        lag_exprs = []
        _lag_cols = ["monthly_total", "monthly_avg", "monthly_median",
                      "monthly_count", "monthly_duty_total"]
        for _lag in [1, 2, 3]:
            for _col in _lag_cols:
                lag_exprs.append(
                    pl.col(_col).shift(_lag).over("policy_grp_name")
                    .alias(f"{_col}_lag{_lag}")
                )

        # Month-over-month change rate for total and average
        for _col in ["monthly_total", "monthly_avg"]:
            lag_exprs.append(
                (pl.col(_col) / pl.col(_col).shift(1).over("policy_grp_name").clip(1, None) - 1.0)
                .alias(f"{_col}_mom")
            )

        if lag_exprs:
            feature_lf = feature_lf.with_columns(lag_exprs)

    # ================================================================
    # 6a. Member-level features (join)
    # ================================================================
    member_stats = global_stats.get("member")
    if member_stats is not None and "insured_no" in feature_lf.collect_schema().names():
        _member_cols = [
            "member_total_paid", "member_avg_paid", "member_std_paid",
            "member_claim_count", "member_case_count", "member_claim_rate",
            "member_avg_claim_size", "member_hosp_count", "member_disease_count",
        ]
        feature_lf = feature_lf.join(
            member_stats.select(["insured_no"] + _member_cols).lazy(),
            on="insured_no",
            how="left",
        )
        for _col in _member_cols:
            feature_lf = feature_lf.with_columns(
                pl.col(_col).fill_null(0).alias(_col)
            )

    # ================================================================
    # 6b. Cross-feature interactions (P2-11)
    # ================================================================
    _interact_pairs = [
        ("age_bucket", "duty_code"),
        ("hosp_grade", "fee_item_type"),
        ("is_yearend", "hosp_grade"),
        ("rnew_flag", "duty_code"),
    ]
    schema = feature_lf.collect_schema()
    for a, b in _interact_pairs:
        if a in schema.names() and b in schema.names():
            interact_name = f"{a}_x_{b}"
            feature_lf = feature_lf.with_columns(
                (pl.col(a).cast(pl.Utf8) + pl.lit("_") + pl.col(b).cast(pl.Utf8))
                .cast(pl.Categorical)
                .alias(interact_name)
            )
            if categorical_cols is not None:
                categorical_cols.append(interact_name)

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

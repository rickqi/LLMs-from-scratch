"""Doris CSV loader using Polars lazy evaluation.

Handles the 5.84GB / 18.9M-row claims CSV without loading into memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

# Columns that should be parsed as dates
_DATE_COLS = [
    "cont_valid_date",
    "medical_start_date",
    "medical_end_date",
    "cust_birthday",
    "end_case_date",
    "cont_end_date",
    "fir_vald_date",
]

# Columns that should be numeric
_NUMERIC_COLS = [
    "clm_acount_amnt_cny",
    "annual_prem",
    "pass_prem",
    "day_count",
    "large_case_amt",
    "clm_acount_amnt_cny_duty",
    "year_actual_claim_paid",
    "ytd_actual_claim_paid",
    "est_loss_ratio",
    "actual_sales_lr",
    "pass_months",
]

# Columns with mixed alphanumeric/numeric values — force to Utf8
_MIXED_TYPE_COLS = [
    "sub_duty_code", "fee_code", "grpcont_claim_duty_code",
    "sale_chnl_code", "policy_grp_no", "grp_cont_no", "rgt_no",
    "case_no", "old_clm_no", "insured_no", "appnt_no",
    "vip_type", "pass_year", "investment_approver",
]

# Schema overrides: force all non-numeric, non-date columns to Utf8
_SCHEMA_OVERRIDES = {
    col: pl.Utf8 for col in _MIXED_TYPE_COLS
}


def load_doris_csv(csv_path: str | Path) -> pl.LazyFrame:
    """Lazily scan the Doris claims CSV.

    Returns a LazyFrame — no data is loaded until .collect() is called.
    Date and numeric columns are cast on the fly.

    Args:
        csv_path: Path to c001_ghb_poicy_clm_duty_d.csv

    Returns:
        Polars LazyFrame with proper type casts
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    size_gb = csv_path.stat().st_size / (1024 ** 3)
    logger.info("Scanning CSV: %s (%.2f GB)", csv_path.name, size_gb)

    lf = pl.scan_csv(
        str(csv_path),
        try_parse_dates=True,
        infer_schema_length=10000,
        schema_overrides=_SCHEMA_OVERRIDES,
    )

    # Ensure date columns are Date type (handle both string and datetime)
    exprs = []
    schema = lf.collect_schema()
    for col in _DATE_COLS:
        if col in schema.names():
            dtype = schema[col]
            if dtype == pl.Date:
                continue  # Already Date
            elif dtype.base_type() == pl.Datetime:
                # Cast Datetime → Date (drop time component)
                exprs.append(pl.col(col).cast(pl.Date).alias(col))
            # If it's Utf8, try_parse_dates already handled it or we leave as-is

    # Ensure numeric columns are Float64
    for col in _NUMERIC_COLS:
        if col in schema.names():
            dtype = schema[col]
            if dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))

    if exprs:
        lf = lf.with_columns(exprs)

    logger.info("LazyFrame ready: %d columns", len(schema.names()))
    return lf


def get_row_count(csv_path: str | Path) -> int:
    """Fast row count without loading data (wc -l approach)."""
    import subprocess

    result = subprocess.run(
        ["wc", "-l", str(csv_path)],
        capture_output=True, text=True, shell=True
    )
    # On Windows fallback
    if result.returncode != 0:
        # Polars approach: count just one column
        count = (
            pl.scan_csv(str(csv_path))
            .select(pl.len())
            .collect()
            .item()
        )
        return count
    return int(result.stdout.strip().split()[0]) - 1  # minus header

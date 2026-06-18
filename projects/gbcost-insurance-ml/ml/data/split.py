"""Time-safe data splitting for insurance claims.

Ensures no future data leaks into training.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict

import polars as pl

logger = logging.getLogger(__name__)


def split_by_time(
    lf: pl.LazyFrame,
    date_col: str = "medical_start_date",
    train_end: str = "2024-12-31",
    val_end: str = "2025-06-30",
    test_end: str = "2026-06-30",
) -> Dict[str, pl.LazyFrame]:
    """Split LazyFrame by time boundaries.

    Args:
        lf: Source LazyFrame
        date_col: Date column to split on
        train_end: Last date (inclusive) for training set
        val_end: Last date (inclusive) for validation set
        test_end: Last date (inclusive) for test set

    Returns:
        Dict with keys 'train', 'val', 'test' — each a filtered LazyFrame
    """
    from datetime import date as _date
    train_d = _date.fromisoformat(train_end)
    val_d = _date.fromisoformat(val_end)
    test_d = _date.fromisoformat(test_end)

    splits = {
        "train": lf.filter(pl.col(date_col) <= pl.lit(train_d)),
        "val": lf.filter(
            (pl.col(date_col) > pl.lit(train_d)) & (pl.col(date_col) <= pl.lit(val_d))
        ),
        "test": lf.filter(
            (pl.col(date_col) > pl.lit(val_d)) & (pl.col(date_col) <= pl.lit(test_d))
        ),
    }

    return splits


def get_split_info(splits: Dict[str, pl.LazyFrame]) -> Dict[str, int]:
    """Get row counts for each split (triggers a collect).

    Returns:
        Dict mapping split name to row count
    """
    info = {}
    for name, lf in splits.items():
        count = lf.select(pl.len()).collect().item()
        info[name] = count
        logger.info("Split '%s': %s rows", name, f"{count:,}")
    return info

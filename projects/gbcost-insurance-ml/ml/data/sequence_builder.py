"""Build LSTM training sequences from monthly policy-group aggregations.

Converts global_stats['monthly'] DataFrame into sliding-window sequences
for PolicyLSTM training and prediction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Feature columns for LSTM input
_SEQ_FEATURE_COLS = [
    "monthly_total",
    "monthly_count",
    "monthly_avg",
]  # These are always present in monthly stats


def build_sequences(
    monthly_df: pl.DataFrame,
    seq_len: int = 12,
    target_col: str = "monthly_total",
    min_sequence_length: int = 18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build sliding-window sequences from monthly policy-group data.

    Args:
        monthly_df: DataFrame from global_stats['monthly'] with columns:
                    policy_grp_name, year_month, monthly_total, monthly_avg,
                    monthly_median, monthly_count, monthly_duty_total
        seq_len: Number of months in each input window
        target_col: Column to predict (default: monthly_total)
        min_sequence_length: Minimum months needed per policy

    Returns:
        sequences: (n_samples, seq_len, n_features) numpy array
        targets: (n_samples,) numpy array
        months: (n_samples,) year_month labels for each prediction point
        policy_names: list of policy_grp_name for each sample
    """
    feature_cols = [c for c in _SEQ_FEATURE_COLS if c in monthly_df.columns]
    if not feature_cols:
        raise ValueError(f"No feature columns found in monthly_df. Available: {monthly_df.columns}")

    logger.info("Building sequences: seq_len=%d, features=%s", seq_len, feature_cols)

    sequences_list = []
    targets_list = []
    months_list = []
    policies_list = []

    for policy in monthly_df["policy_grp_name"].unique().to_list():
        if policy is None:
            continue

        policy_data = monthly_df.filter(pl.col("policy_grp_name") == policy).sort("year_month")

        if len(policy_data) < min_sequence_length:
            continue

        values = policy_data.select(feature_cols).to_numpy().astype(np.float32)
        months = policy_data["year_month"].to_list()

        for i in range(len(values) - seq_len):
            seq = values[i:i + seq_len]
            target = policy_data[target_col][i + seq_len]

            if np.isnan(seq).any() or np.isnan(target) or target <= 0:
                continue

            # Normalize by per-policy mean
            seq_mean = seq.mean(axis=0, keepdims=True)
            seq_mean[seq_mean == 0] = 1.0
            seq_norm = seq / seq_mean
            target_norm = target / seq_mean[0, 0]  # normalize target by total mean

            sequences_list.append(seq_norm)
            targets_list.append(target_norm)
            months_list.append(months[i + seq_len])
            policies_list.append(policy)

    if not sequences_list:
        raise ValueError("No sequences built. Check min_sequence_length or data availability.")

    sequences = np.stack(sequences_list)
    targets = np.array(targets_list, dtype=np.float32)
    months_arr = np.array(months_list)
    policies_arr = policies_list

    logger.info("Built %d sequences from %d policies (shape=%s)",
                 len(sequences), len(set(policies_list)), sequences.shape)

    # Log target statistics
    logger.info("  Target stats: mean=%.0f, median=%.0f, p95=%.0f",
                 np.mean(targets), np.median(targets), np.percentile(targets, 95))

    return sequences, targets, months_arr, policies_arr


def predict_next_month(
    lstm_model,
    monthly_df: pl.DataFrame,
    policy_grp_names: Optional[List[str]] = None,
    seq_len: int = 12,
) -> pl.DataFrame:
    """Generate LSTM predictions for the next month of each policy.

    Args:
        lstm_model: Trained PolicyLSTM instance
        monthly_df: Monthly stats DataFrame
        policy_grp_names: Specific policies to predict (default: all)
        seq_len: Sequence length matching model's training

    Returns:
        DataFrame with columns: policy_grp_name, year_month, lstm_pred
    """
    feature_cols = [c for c in _SEQ_FEATURE_COLS if c in monthly_df.columns]

    results = []
    policies = policy_grp_names or monthly_df["policy_grp_name"].unique().to_list()

    for policy in policies:
        if policy is None:
            continue
        policy_data = monthly_df.filter(
            pl.col("policy_grp_name") == policy
        ).sort("year_month")

        if len(policy_data) < seq_len + 1:
            continue

        values = policy_data.select(feature_cols).to_numpy().astype(np.float32)
        months = policy_data["year_month"].to_list()

        for i in range(len(values) - seq_len):
            seq = values[i:i + seq_len]
            seq_mean = seq.mean(axis=0, keepdims=True)
            seq_mean[seq_mean == 0] = 1.0
            seq_norm = seq / seq_mean

            pred_norm = float(lstm_model.predict(seq_norm.reshape(1, seq_len, -1))[0])
            pred = max(pred_norm * float(seq_mean[0, 0]), 0)  # denormalize
            results.append({
                "policy_grp_name": policy,
                "year_month": months[i + seq_len],
                "lstm_pred": max(pred, 0),
            })

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)

"""Label enrichment from Phase 2 agent_state.json analysis results.

Extracts anomaly labels, FWA flags, DRG markers, and risk scores from
agent_state.json and joins them with case-level ML training data.

This creates a closed-loop: Phase 2 analysis → ML training labels → better models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def extract_case_labels(agent_state: dict) -> pl.DataFrame:
    """Extract per-case labels from a single agent_state.json.

    Args:
        agent_state: Parsed agent_state.json dict

    Returns:
        DataFrame with columns: case_no, anomaly_level, fwa_flag,
        drg_over_budget, medical_irrational, hospital_z_score,
        member_risk_score, health_score
    """
    records: List[Dict[str, Any]] = []

    case_map: Dict[str, dict] = {}

    def _get_or_create(case_no: str) -> dict:
        if case_no not in case_map:
            case_map[case_no] = {"case_no": case_no}
        return case_map[case_no]

    # 1. FWA fraud/waste/abuse matches
    for fwa_key in ["fwa_fraud_matches", "fwa_waste_matches", "fwa_abuse_matches"]:
        fwa_list = agent_state.get(fwa_key, [])
        if isinstance(fwa_list, list):
            for item in fwa_list:
                if isinstance(item, dict):
                    cn = str(item.get("claim_id", item.get("case_no", "")))
                    if cn:
                        rec = _get_or_create(cn)
                        rec["fwa_flag"] = 1
                        rec.setdefault("fwa_type", item.get("rule_name", item.get("category", "")))
                        rec.setdefault("fwa_rule", item.get("rule_id", ""))

    # 2. DRG over-budget
    drg = agent_state.get("drg_result", {})
    if isinstance(drg, dict):
        ob_list = drg.get("over_budget_claims", [])
        if isinstance(ob_list, list):
            for item in ob_list:
                if isinstance(item, dict):
                    cn = str(item.get("claim_id", item.get("case_no", "")))
                    if cn:
                        rec = _get_or_create(cn)
                        rec["drg_over_budget"] = 1
                        actual = float(item.get("actual_amount", 0))
                        base = float(item.get("adjusted_base", item.get("base_amount", 1)))
                        if base > 0:
                            rec["drg_cost_ratio"] = actual / base

    # 3. Medical rationality — extract from by_disease/by_hospital
    med = agent_state.get("medical_rationality", {})
    if isinstance(med, dict):
        for section in ["by_disease", "by_hospital"]:
            section_data = med.get(section, {})
            if isinstance(section_data, dict):
                for _, items in section_data.items():
                    items_list = items if isinstance(items, list) else [items]
                    for item in items_list:
                        if isinstance(item, dict):
                            cn = str(item.get("claim_id", item.get("case_no", "")))
                            if cn:
                                rec = _get_or_create(cn)
                                rec["medical_irrational"] = 1

    # 4. Hospital fee anomaly
    hosp = agent_state.get("hospital_fee_anomaly", {})
    if isinstance(hosp, dict):
        ext = hosp.get("extreme_deviations", [])
        if isinstance(ext, list):
            for item in ext:
                if isinstance(item, dict):
                    cn = str(item.get("claim_id", item.get("case_no", "")))
                    if cn:
                        rec = _get_or_create(cn)
                        z_val = abs(float(item.get("z_score", item.get("deviation", 0))))
                        rec["hospital_z_score"] = max(rec.get("hospital_z_score", 0), z_val)

    # 5. Health score (policy-level, applies to all cases)
    hs = agent_state.get("health_score", {})
    health_val = float(hs.get("score", 0)) if isinstance(hs, dict) else 0.0

    # 6. FWA top risk claims
    fwa_top = agent_state.get("fwa_top_risk_claims", [])
    if isinstance(fwa_top, list):
        for item in fwa_top:
            if isinstance(item, dict):
                cn = str(item.get("claim_id", item.get("case_no", "")))
                if cn:
                    rec = _get_or_create(cn)
                    rec["fwa_high_risk"] = 1
                    rec["fwa_risk_score"] = float(item.get("risk_score", item.get("score", 0)))

    # 7. Hospital anomalies (aggregated by hospital)
    hosp_anom = agent_state.get("hospital_anomalies", [])
    if isinstance(hosp_anom, list):
        for item in hosp_anom:
            if isinstance(item, dict):
                cases = item.get("cases", item.get("claim_ids", []))
                if isinstance(cases, list):
                    for cn in cases:
                        cn_str = str(cn)
                        rec = _get_or_create(cn_str)
                        rec["hospital_anomaly"] = 1

    # Build DataFrame
    if not case_map:
        return pl.DataFrame()

    df = pl.DataFrame(list(case_map.values()))
    # Fill defaults
    defaults = {
        "fwa_flag": 0, "drg_over_budget": 0,
        "medical_irrational": 0, "hospital_z_score": 0.0,
        "fwa_high_risk": 0, "fwa_risk_score": 0.0,
        "hospital_anomaly": 0, "drg_cost_ratio": 1.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))
        else:
            df = df.with_columns(pl.col(col).fill_null(default))

    logger.info("Extracted labels: %d cases (fwa=%d, drg=%d, medical=%d, hosp=%d)",
                 len(df),
                 (df["fwa_flag"] > 0).sum(),
                 (df["drg_over_budget"] > 0).sum(),
                 (df["medical_irrational"] > 0).sum(),
                 (df["hospital_anomaly"] > 0).sum())

    return df


def enrich_features(
    case_lf: pl.LazyFrame,
    labels_df: pl.DataFrame,
) -> pl.LazyFrame:
    """Join extracted labels with case-level feature LazyFrame.

    Args:
        case_lf: Case-level feature LazyFrame (from build_features)
        labels_df: Labels DataFrame (from extract_case_labels)

    Returns:
        LazyFrame with additional label columns
    """
    if len(labels_df) == 0:
        return case_lf

    label_cols = [c for c in labels_df.columns if c != "case_no"]
    labels_lf = labels_df.lazy()

    enriched = case_lf.join(
        labels_lf.select(["case_no"] + label_cols),
        on="case_no",
        how="left",
    )

    for col in label_cols:
        enriched = enriched.with_columns(
            pl.col(col).fill_null(0).alias(col)
        )

    return enriched


def batch_extract_labels(
    agent_state_dir: str | Path,
    pattern: str = "*/agent_state.json",
) -> pl.DataFrame:
    """Batch extract labels from multiple agent_state.json files.

    Args:
        agent_state_dir: Directory containing agent_state.json files
        pattern: Glob pattern for finding files

    Returns:
        Combined labels DataFrame from all agent_state files
    """
    agent_state_dir = Path(agent_state_dir)
    files = sorted(agent_state_dir.glob(pattern))
    logger.info("Scanning %d agent_state files in %s", len(files), agent_state_dir)

    all_labels = []
    for f in files:
        try:
            state = json.loads(f.read_text(encoding="utf-8"))
            labels = extract_case_labels(state)
            if len(labels) > 0:
                all_labels.append(labels)
                logger.debug("  %s: %d cases", f.parent.name, len(labels))
        except Exception as e:
            logger.warning("  Skipping %s: %s", f.name, e)

    if not all_labels:
        logger.warning("No labels extracted from %d files", len(files))
        return pl.DataFrame()

    # Cast all numeric columns to consistent types before concat
    numeric_cols = {"fwa_flag": pl.Int64, "fwa_high_risk": pl.Int64,
                    "drg_over_budget": pl.Int64, "medical_irrational": pl.Int64,
                    "hospital_anomaly": pl.Int64,
                    "fwa_risk_score": pl.Float64, "hospital_z_score": pl.Float64,
                    "drg_cost_ratio": pl.Float64}
    cast_labels = []
    for df in all_labels:
        for col, dtype in numeric_cols.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(dtype))
        cast_labels.append(df)

    combined = pl.concat(cast_labels, how="vertical").unique(subset=["case_no"], keep="last")
    logger.info("Total: %d unique cases labeled across %d files",
                 len(combined), len(all_labels))
    return combined

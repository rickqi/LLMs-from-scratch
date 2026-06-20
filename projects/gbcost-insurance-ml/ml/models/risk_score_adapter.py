"""Risk Score Adapter — convert ML outputs to risk map compatible scores.

Provides continuous risk scoring that replaces discrete rule-engine scores.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class RiskScoreAdapter:
    """Convert ML model outputs to risk map compatible scores (0-100)."""

    @staticmethod
    def case_to_member_risk(l1b_probas: np.ndarray, case_amounts: np.ndarray) -> np.ndarray:
        """Per-case risk score: 70% large-claim probability + 30% amount severity."""
        max_amt_log = np.log1p(24600)
        amount_score = np.clip(np.log1p(np.clip(case_amounts, 0, None)) / max_amt_log, 0, 1)
        return (l1b_probas * 0.7 + amount_score * 0.3) * 100

    @staticmethod
    def case_to_disease_risk(l1b_probas: np.ndarray, group_codes: List[str]) -> Dict[str, dict]:
        """Per-disease risk scores."""
        codes = np.array(group_codes)
        result = {}
        for code in np.unique(codes):
            mask = codes == code
            result[str(code)] = {
                "large_prob": round(float(np.mean(l1b_probas[mask])), 4),
                "max_prob": round(float(np.max(l1b_probas[mask])), 4),
                "case_count": int(mask.sum()),
                "risk_score": round(float(np.mean(l1b_probas[mask]) * 100), 1),
            }
        return result

    @staticmethod
    def case_to_hospital_risk(l1b_probas: np.ndarray, l1a_preds: np.ndarray, hosp_grades: List[str]) -> Dict[str, dict]:
        """Per-hospital risk scores."""
        grades = np.array(hosp_grades)
        result = {}
        for g in np.unique(grades):
            if g is None:
                continue
            mask = grades == g
            result[str(g)] = {
                "avg_predicted": round(float(np.mean(l1a_preds[mask])), 0),
                "large_prob": round(float(np.mean(l1b_probas[mask])), 4),
                "p95_predicted": round(float(np.percentile(l1a_preds[mask], 95)), 0),
                "case_count": int(mask.sum()),
                "risk_score": round(float(np.mean(l1b_probas[mask]) * 100), 1),
            }
        return result

    @staticmethod
    def policy_to_unit_risk(l2_preds: np.ndarray, policy_names: List[str]) -> Dict[str, dict]:
        """Per-policy-unit risk from L2 predictions."""
        result = {}
        for i, name in enumerate(policy_names):
            lr = float(l2_preds[i])
            if lr > 1.0:
                risk = "critical"
            elif lr > 0.75:
                risk = "high"
            elif lr > 0.5:
                risk = "medium"
            else:
                risk = "low"
            result[str(name)] = {"predicted_loss_ratio": round(lr, 4), "risk_level": risk}
        return result

# ML 风险地图 — 训练方案与完整实现计划

> 基于现有 ML 模型 (L1-A/L1-B/L2/LSTM) 增强风险地图 G10/G11 | 2026-06-20

---

## 一、现状分析

### 1.1 风险地图当前状态

```
G10 风险地图 (4维×23项) — 全部"向后看"
┌─────────────────────────────────────────────┐
│ 🔴 疾病(6)  基于 agent_state 已发生数据      │
│ 🔵 医院(6)  事后分析 + knowledge 规则        │
│ 🟢 会员(6)  历史行为聚合                     │
│ 🟡 单位(5)  同比环比分析                     │
├─────────────────────────────────────────────┤
│ G11 ML预测(3) 已有基础，待扩展               │
│   • 预测汇总概览 (R²/Gini/MAE KPI)           │
│   • 保单级预测对比 (预测vs实际分桶)           │
│   • 模型评估详情 (特征重要性/训练/回测)       │
└─────────────────────────────────────────────┘
```

### 1.2 ML 模型矩阵 (可直接利用)

| 模型 | 粒度 | 输出 | 指标 | 风险地图映射 |
|------|------|------|:---:|------|
| **L1-A** Tweedie | 案件级 | 预测金额 (元) | Gini=0.51 | 四维风险评分 |
| **L1-B** Binary | 案件级 | 大额概率 (0-1) | AUC=0.995 | 欺诈/过度风险 |
| **L2** Regression | 保单级 | 赔付率预测 | Gini=0.63 | 投保单位健康度 |
| **LSTM** | 月度级 | 次月总额预测 | Gini=0.75 | 四维趋势预警 |
| **Calibrator** | 保单组 | 校正因子 | 误差→0% | 预测精度保证 |
| **Label Enricher** | 案件级 | FWA/DRG标签 | 477K cases | 欺诈/超支标记 |

---

## 二、P0 — 快速增强（工作量 ~3h）

### 2.1 目标

在不改变风险地图架构的前提下，为 G10 每个维度增加"ML预测风险"子项，让风险统计从"事后统计"变为"事前预测"。

### 2.2 实现方案

#### 2.2.1 数据层：ML 风险评分计算器

```python
# ml/data/risk_scorer.py (新增)

"""ML Risk Scorer — converts ML predictions to risk map scores.

Each output maps directly to a G10 risk map dimension.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional


class MLRiskScorer:
    """Compute ML-driven risk scores for all 4 risk map dimensions."""

    def __init__(
        self,
        l1a_predictor,       # AmountPredictor
        l1b_classifier,      # LargeClaimClassifier
        l2_predictor,        # PolicyPredictor
        lstm_model,          # PolicyLSTM
        calibrator,          # GroupCalibrator
    ):
        self.l1a = l1a_predictor
        self.l1b = l1b_classifier
        self.l2 = l2_predictor
        self.lstm = lstm_model
        self.calibrator = calibrator

    def score_disease_dimension(
        self, df: pl.DataFrame, feature_cols: List[str], cat_cols: List[str],
    ) -> pl.DataFrame:
        """🔴 Disease dimension risk scores.

        Returns per-group_code:
          - avg_predicted_amount: 预测均额
          - large_claim_prob: 大额概率均值
          - risk_level: low/medium/high (分位数)
        """
        df = df.with_columns([
            pl.Series("l1a_pred", self.l1a.predict(df, feature_cols, cat_cols)),
            pl.Series("l1b_proba", self.l1b.predict_proba(df, feature_cols, cat_cols)),
        ])

        scores = df.group_by("group_code").agg(
            pl.col("l1a_pred").mean().alias("avg_predicted_amount"),
            pl.col("l1a_pred").sum().alias("total_predicted"),
            pl.col("l1b_proba").mean().alias("large_claim_prob"),
            pl.len().alias("case_count"),
        ).with_columns(
            pl.col("avg_predicted_amount")
            .qcut([0.33, 0.67], labels=["low", "medium", "high"])
            .alias("risk_level")
        )
        return scores

    def score_hospital_dimension(
        self, df: pl.DataFrame, feature_cols: List[str], cat_cols: List[str],
    ) -> pl.DataFrame:
        """🔵 Hospital dimension risk scores."""
        df = df.with_columns([
            pl.Series("l1a_pred", self.l1a.predict(df, feature_cols, cat_cols)),
            pl.Series("l1b_proba", self.l1b.predict_proba(df, feature_cols, cat_cols)),
        ])

        return df.group_by("hosp_grade").agg(
            pl.col("l1a_pred").mean().alias("avg_predicted"),
            pl.col("l1a_pred").std().alias("std_predicted"),
            pl.col("l1b_proba").mean().alias("large_prob"),
            pl.col("l1b_proba").quantile(0.95).alias("p95_large_prob"),
            pl.len().alias("case_count"),
        ).with_columns(
            (pl.col("large_prob") / pl.col("large_prob").mean())
            .alias("risk_ratio")  # >1 means higher than average
        )

    def score_member_dimension(
        self, df: pl.DataFrame, feature_cols: List[str], cat_cols: List[str],
    ) -> pl.DataFrame:
        """🟢 Member dimension risk scores."""
        df = df.with_columns([
            pl.Series("l1a_pred", self.l1a.predict(df, feature_cols, cat_cols)),
            pl.Series("l1b_proba", self.l1b.predict_proba(df, feature_cols, cat_cols)),
        ])

        scores = df.group_by("insured_no").agg(
            pl.col("l1a_pred").sum().alias("total_predicted"),
            pl.col("l1a_pred").mean().alias("avg_predicted"),
            pl.col("l1b_proba").mean().alias("large_prob"),
            pl.col("l1b_proba").max().alias("max_large_prob"),
            pl.len().alias("claim_count"),
        )

        # Composite risk score: 0-100
        scores = scores.with_columns(
            (pl.col("total_predicted").rank("ordinal") / pl.len() * 50 +
             pl.col("large_prob").rank("ordinal") / pl.len() * 50)
            .alias("risk_score")
        )
        return scores.sort("risk_score", descending=True)

    def score_policy_unit_dimension(
        self, df: pl.DataFrame,
    ) -> pl.DataFrame:
        """🟡 Policy unit dimension — uses L2 predictions."""
        l2_preds = self.l2.predict(df)
        return df.select("policy_grp_name").with_columns(
            pl.Series("predicted_loss_ratio", l2_preds),
        ).with_columns(
            pl.col("predicted_loss_ratio")
            .qcut([0.25, 0.50, 0.75], labels=["low", "medium", "high", "critical"])
            .alias("risk_level")
        )

    def generate_risk_map_json(
        self,
        case_df: pl.DataFrame,
        policy_df: pl.DataFrame,
        feature_cols: List[str],
        cat_cols: List[str],
    ) -> Dict:
        """Generate complete ML risk map as JSON for TUI consumption."""
        return {
            "generated_at": str(pl.datetime),
            "disease": self.score_disease_dimension(case_df, feature_cols, cat_cols).to_dicts(),
            "hospital": self.score_hospital_dimension(case_df, feature_cols, cat_cols).to_dicts(),
            "member": self.score_member_dimension(case_df, feature_cols, cat_cols).to_dicts()[:50],
            "policy_unit": self.score_policy_unit_dimension(policy_df).to_dicts(),
        }
```

#### 2.2.2 管线层：风险评分生成 CLI

```python
# ml/pipeline/score_risks.py (新增)

"""Generate ML risk scores for risk map consumption.

Usage:
    python -m ml.pipeline.score_risks --csv data/doris/c001_ghb_poicy_clm_duty_d.csv
    python -m ml.pipeline.score_risks --output data/ml_cache/ml_risk_map.json
"""

import argparse, json, logging, sys
from pathlib import Path

logger = logging.getLogger("ml.risk_score")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate ML risk scores for risk map")
    parser.add_argument("--csv", default="data/doris/c001_ghb_poicy_clm_duty_d.csv")
    parser.add_argument("--output", default="data/ml_cache/ml_risk_map.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    from ml.models.amount_predictor import AmountPredictor
    from ml.models.large_classifier import LargeClaimClassifier
    from ml.models.policy_predictor import PolicyPredictor
    from ml.models.lstm_model import PolicyLSTM
    from ml.models.calibration import GroupCalibrator
    from ml.data.risk_scorer import MLRiskScorer
    from ml.data.loader import load_doris_csv
    from ml.data.feature_store import build_features

    import pickle, polars as pl

    # Load models
    l1a = AmountPredictor.load("models", "l1a_amount")
    l1b = LargeClaimClassifier.load("models", "l1b_classifier")
    l2 = PolicyPredictor.load("models", "l2_policy")
    lstm = PolicyLSTM.load("models/lstm_policy.pt")
    cal = GroupCalibrator.load("models/calibrator.json")

    scorer = MLRiskScorer(l1a, l1b, l2, lstm, cal)

    # Load data and features
    schema = json.loads(open("models/feature_schema.json").read())
    with open("data/ml_cache/global_stats.pkl", "rb") as f:
        raw = pickle.load(f)
    global_stats = {k: (pl.DataFrame(v) if isinstance(v, dict) else v) for k, v in raw.items()}

    lf = load_doris_csv(args.csv)
    feature_lf = build_features(lf, global_stats, categorical_cols=schema["categorical_cols"])
    case_df = feature_lf.select(schema["feature_cols"] + ["y_raw", "policy_grp_name",
                                 "group_code", "hosp_grade", "insured_no",
                                 "medical_start_date"]).collect(engine="streaming")

    # Policy-level aggregation for L2
    from ml.data.policy_aggregator import aggregate_policy_features
    policy_lf = aggregate_policy_features(lf)
    policy_df = policy_lf.collect(engine="streaming")

    # Generate risk map
    risk_map = scorer.generate_risk_map_json(
        case_df, policy_df, schema["feature_cols"], schema["categorical_cols"]
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(risk_map, indent=2, ensure_ascii=False, default=str),
                                 encoding="utf-8")
    logger.info("ML risk map saved: %s", args.output)

    # Summary
    disease_risks = len([d for d in risk_map["disease"] if d.get("risk_level") == "high"])
    logger.info("Disease high-risk groups: %d", disease_risks)
    logger.info("Member top-10 risk scores: %.0f-%.0f",
                 risk_map["member"][0]["risk_score"] if risk_map["member"] else 0,
                 risk_map["member"][9]["risk_score"] if len(risk_map["member"]) > 9 else 0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

#### 2.2.3 集成到训练管线

在 `train.py` 末尾自动调用：

```python
# train.py — Step 11 (P0): Generate ML risk map
logger.info("=" * 60)
logger.info("Step 11: Generating ML risk map scores")
from ml.pipeline.score_risks import main as score_main
score_main(["--csv", csv_path, "--output", "data/ml_cache/ml_risk_map.json"])
```

### 2.3 P0 产出

| 文件 | 用途 |
|------|------|
| `ml/data/risk_scorer.py` | ML 四维风险评分核心逻辑 |
| `ml/pipeline/score_risks.py` | CLI + 训练集成 |
| `data/ml_cache/ml_risk_map.json` | 风险评分输出（TUI 直接读取） |

---

## 三、P1 — ML 风险评分替代规则引擎（工作量 ~3h）

### 3.1 目标

用 ML 预测替代现有的规则引擎风险评分，提供更精准、连续的风险量化。

### 3.2 实现方案

#### 3.2.1 L1-B 概率 → 会员风险评分

```
现有规则引擎: IF (claims > threshold) AND (fwa_match) THEN risk_high
              → 离散、规则覆盖不全

ML 替代: L1-B predict_proba(case) → 0-1 连续概率
         → 每个案件都有一个风险概率
         → 聚合到会员/疾病/医院维度
```

```python
# ml/models/risk_score_adapter.py (新增)

class RiskScoreAdapter:
    """Convert ML outputs to risk map compatible scores."""

    @staticmethod
    def case_to_member_risk(l1b_probas: np.ndarray, case_amounts: np.ndarray) -> np.ndarray:
        """Per-case risk score: probability × amount severity."""
        amount_score = np.clip(np.log1p(case_amounts) / np.log1p(24600), 0, 1)
        return l1b_probas * 0.7 + amount_score * 0.3  # 70% prob + 30% amount

    @staticmethod
    def case_to_disease_risk(l1b_probas: np.ndarray, group_codes: List[str]) -> Dict[str, float]:
        """Per-disease risk: mean large-claim probability."""
        unique_codes = set(group_codes)
        return {code: float(np.mean(l1b_probas[np.array(group_codes) == code]))
                for code in unique_codes}

    @staticmethod
    def policy_to_unit_risk(l2_preds: np.ndarray, calibrator_factors: Dict[str, float]) -> Dict:
        """Per-unit risk: predicted loss_ratio × group calibration factor."""
        # Higher predicted loss ratio + higher calibration factor = higher risk
        pass
```

#### 3.2.2 风险地图数据结构增强

```json
// 现有 agent_state 输出
{
  "disease_analysis": { "top_diseases": [...], "anomaly_flags": [...] },
  "member_risk_scores": { "rule_based": 72.5 }
}

// P1 增强后
{
  "ml_risk_scores": {
    "disease": {
      "I10": { "large_prob": 0.72, "avg_predicted": 8500, "risk": "high" },
      "J06": { "large_prob": 0.12, "avg_predicted": 1200, "risk": "low" }
    },
    "member": [
      { "insured_no": "G0001", "risk_score": 85.3, "top_risk": "large_claim_prob=0.94" }
    ],
    "hospital": {
      "三级甲等": { "risk_ratio": 1.45, "p95_large": 0.88 }
    },
    "policy_unit": {
      "SPX Cooling": { "predicted_lr": 0.78, "risk_level": "critical" }
    }
  }
}
```

### 3.3 P1 产出

| 文件 | 用途 |
|------|------|
| `ml/models/risk_score_adapter.py` | ML→风险评分转换器 |
| 增强 `ml_risk_map.json` | 连续评分替代离散标签 |

---

## 四、P2 — 趋势预警 + 闭环反馈（工作量 ~4h）

### 4.1 目标

- LSTM 提供未来月份趋势预警
- ML 预测结果回溯验证 → 模型自动重训练触发器
- 风险地图增加"未来趋势"维度

### 4.2 实现方案

#### 4.2.1 LSTM 趋势预警

```python
# ml/pipeline/trend_alerts.py (新增)

class TrendAlertGenerator:
    """Generate forward-looking alerts from LSTM predictions."""

    def __init__(self, lstm_model, monthly_df: pl.DataFrame, threshold_pct: float = 0.3):
        self.lstm = lstm_model
        self.monthly = monthly_df
        self.threshold = threshold_pct

    def generate_disease_trends(self) -> pl.DataFrame:
        """LSTM predictions per disease group for next 3 months."""
        from ml.data.sequence_builder import predict_next_month
        preds = predict_next_month(self.lstm, self.monthly)
        # Compare predicted vs historical average
        preds = preds.join(
            self.monthly.group_by("policy_grp_name").agg(
                pl.col("monthly_total").mean().alias("historical_avg")
            ), on="policy_grp_name"
        )
        preds = preds.with_columns(
            ((pl.col("lstm_pred") / pl.col("historical_avg") - 1) * 100)
            .alias("trend_pct")
        )
        return preds.filter(pl.col("trend_pct").abs() > self.threshold * 100)

    def generate_alerts(self) -> List[Dict]:
        """Generate risk map alerts from LSTM trends."""
        trends = self.generate_disease_trends()
        alerts = []
        for row in trends.iter_rows(named=True):
            direction = "上升" if row["trend_pct"] > 0 else "下降"
            severity = "red" if abs(row["trend_pct"]) > 50 else ("yellow" if abs(row["trend_pct"]) > 30 else "green")
            alerts.append({
                "dimension": "disease",
                "entity": row["policy_grp_name"],
                "message": f"预测赔付{direction} {abs(row['trend_pct']):.0f}%",
                "severity": severity,
                "predicted": row["lstm_pred"],
                "historical_avg": row["historical_avg"],
            })
        return alerts
```

#### 4.2.2 闭环验证触发器

```python
# ml/pipeline/closed_loop.py (新增)

class ClosedLoopMonitor:
    """Monitor prediction vs actual, trigger retraining when drift detected."""

    def __init__(self, model_dir: str = "models", drift_threshold: float = 0.15):
        self.model_dir = Path(model_dir)
        self.drift_threshold = drift_threshold

    def check_prediction_drift(
        self, y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Check if predictions have drifted from expected accuracy."""
        from ml.evaluate.metrics import evaluate_predictions
        metrics = evaluate_predictions(y_true, y_pred)

        # Load baseline metrics from model registry
        registry = json.loads((self.model_dir / "model_registry.json").read_text())
        latest = registry.get("versions", [])[-1] if registry.get("versions") else None

        if latest:
            baseline_gini = latest["metrics"].get("gini", 0)
            current_gini = metrics.get("gini", 0)
            drift = (baseline_gini - current_gini) / max(baseline_gini, 0.01)

            if drift > self.drift_threshold:
                return {
                    "drift_detected": True,
                    "severity": "high" if drift > 0.3 else "medium",
                    "message": f"Gini drift: {baseline_gini:.4f} → {current_gini:.4f} ({drift:.0%})",
                    "action": "retrain_required",
                    "metrics": metrics,
                }

        return {"drift_detected": False, "metrics": metrics}

    def should_retrain(self, drift_result: Dict) -> bool:
        return drift_result.get("drift_detected", False) and drift_result.get("severity") == "high"
```

#### 4.2.3 集成到月度管线

```python
# monthly_pipeline.py 增强

def run_monthly_pipeline_with_monitoring():
    # ... existing steps ...

    # Step 6: Closed-loop monitoring
    monitor = ClosedLoopMonitor()
    drift = monitor.check_prediction_drift(y_true_new, y_pred_new)
    if monitor.should_retrain(drift):
        logger.warning("DRIFT DETECTED: triggering auto-retraining")
        run_full_training()
        send_alert("Model drift detected — retrained")
```

### 4.3 P2 产出

| 文件 | 用途 |
|------|------|
| `ml/pipeline/trend_alerts.py` | LSTM 趋势预警生成器 |
| `ml/pipeline/closed_loop.py` | 预测漂移监控 + 自动重训练触发器 |
| 增强 `monthly_pipeline.py` | 闭环监控集成 |

---

## 五、实施路线图

```
Week 1: P0 快速增强 (3h)
├── ml/data/risk_scorer.py        四维 ML 风险评分
├── ml/pipeline/score_risks.py    CLI + 训练集成
└── train.py §11                  自动生成 ml_risk_map.json

Week 2: P1 评分替代 (3h)
├── ml/models/risk_score_adapter.py  ML→风险评分转换
├── 增强 ml_risk_map.json           连续评分输出
└── G10 各维度增加"ML风险"子项

Week 3: P2 趋势+闭环 (4h)
├── ml/pipeline/trend_alerts.py     LSTM 趋势预警
├── ml/pipeline/closed_loop.py      漂移监控 + 自动重训练
└── monthly_pipeline.py 增强        闭环集成
```

## 六、预期效果矩阵

| 风险地图维度 | 当前 | P0 后 | P1 后 | P2 后 |
|------|:---:|:---:|:---:|:---:|
| 疾病风险 | 事后统计 | +ML预测金额 | +连续概率评分 | +3月趋势预警 |
| 医院风险 | z-score规则 | +预测偏差 | +ML风险比 | +趋势+漂移监控 |
| 会员风险 | 规则引擎(离散) | +Top50风险排序 | +ML连续评分(0-100) | +高频行为预警 |
| 单位风险 | 同比环比 | +L2赔付率预测 | +风险等级(low→critical) | +自动重训练触发 |
| 时间维度 | 无 | 无 | 无 | +未来3月预测 |

# 团险控费 ML — 预训练方案评估与优化路线图

> 基于四项目交叉分析 | 2026-06-18 | 优先级: P0→P1→P2→P3

---

## 一、当前方案全景评估

### 1.1 架构总览

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 数据层       │ →  │ 特征工程      │ →  │ 模型层        │ →  │ 评估层        │
│ Polars惰性   │    │ 两阶段FE     │    │ LightGBM     │    │ Walk-Forward │
│ 5.8GB CSV   │    │ TE+贝叶斯平滑 │    │ Tweedie回归  │    │ Gini/MAE/MAPE│
│ 605万行     │    │ 20+特征      │    │ Optuna调优   │    │ 总量误差     │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 1.2 成熟度评分

| 维度 | 评分 | 说明 |
|------|:---:|------|
| 模型选型 | ⭐⭐⭐⭐⭐ | Tweedie + LightGBM 是结构化保险数据最优解 |
| 特征工程 | ⭐⭐⭐ | 两阶段 TE 正确，缺时序滞后和交互特征 |
| 超参优化 | ⭐⭐⭐⭐ | Optuna 贝叶斯已运行一轮，缺断点续传 |
| 时序安全 | ⭐⭐⭐⭐⭐ | medical_start_date 分割，无前视偏差 |
| 工程基础设施 | ⭐⭐ | 🔴 无断点续传、无版本管理、无云备份 |
| 评估体系 | ⭐⭐⭐⭐ | Gini/MAE/MAPE/Walk-Forward 完整，可引入 RankIC |
| 生产就绪度 | ⭐⭐⭐ | 有 predict.py，缺自动化管线和监控 |

### 1.3 当前模型指标

| 指标 | 值 | 评价 |
|------|:---:|------|
| R² (全量) | 0.647 | 中等，有提升空间 |
| Gini | 0.688 | 排序能力尚可（>0.5 有效信号） |
| MAE | 2,994元 | 较基线改善 43% |
| 总量误差 | 49.6% | 🔴 系统性低估，依赖硬编码 ×1.80 |
| 回测 avg Gini | 0.723 | 季度窗口稳定 |
| large_recall | 0.424 | 🟡 大额识别有提升空间 |
| 训练耗时 | 65s | 极快，支持快速迭代 |

---

## 二、跨项目对比诊断

### 2.1 四项目能力矩阵

| 能力 | Kronos | 医疗 | 法规 | 控费 | 差距 |
|------|:---:|:---:|:---:|:---:|------|
| 断点续传 | ✅ | ✅ | ✅ | ❌ | 🔴 |
| COS云备份 | ✅ | ✅ | ✅ | ❌ | 🔴 |
| 模型版本管理 | ✅ | ✅ | ✅ | ❌ | 🔴 |
| 实验管理系统 | ✅ 15组 | ❌ | ❌ | ❌ | 🟡 |
| 每日自动化管线 | ✅ | ❌ | ❌ | ❌ | 🟡 |
| 特征交互工程 | ✅ 技术指标 | ❌ | ❌ | ❌ | 🟡 |
| 多阶段训练 | ✅ 两阶段 | ✅ 两阶段 | ✅ | ⚠️ 计划中 | 🟢 |
| 评估指标丰富度 | ✅ RankIC | ❌ | ❌ | ⚠️ 可补充 | 🟢 |

### 2.2 Kronos 的核心教训

Kronos 经历了 **12 组 BSQ 失败 → LSTM 突破** 的路径，关键启示：

> **特征工程比架构重要。** Kronos 最佳结果（RankIC +0.579）来自切换目标变量（波动率）和延长历史窗口（2005-2026），而非换模型架构。

对控费的映射：
- 当前 20 个特征可能不够 → 引入时序滞后、会员行为、异常评分特征
- 目标变量可探索变体 → 对数变换 vs 原始、案件级 vs 保单月聚合
- 必须系统记录每次实验 → 参考 Kronos 15 组实验的归档方式

### 2.3 医疗项目的两阶段训练映射

```
医疗 NLP                             控费 ML
┌──────────────────┐              ┌──────────────────┐
│ Stage 1: 续写微调 │    ←→       │ L1-A: 金额回归    │ ✅ 已完成
│ 语言领域适应     │              │ Tweedie Loss     │
├──────────────────┤              ├──────────────────┤
│ Stage 2: 指令微调 │    ←→       │ L1-B: 大额二分类  │ ⬜ 计划中
│ ChatML QA格式    │              │ Binary + AUC     │
├──────────────────┤              ├──────────────────┤
│ Stage 3: 自动评估 │    ←→       │ L2: 保单级聚合    │ ⬜ 计划中
│ BLEU/ROUGE      │              │ Group-level      │
└──────────────────┘              └──────────────────┘
```

### 2.4 Comprehensive Analysis 的技术债务确认

来自 `comprehensive-analysis.md` 的诊断：
- 🔴 **控费：无断点续传，无模型版本管理** — 严重度最高
- P0 控费添加断点续传 (2h)
- P1 统一训练框架 (8h)
- P2 控费添加 COS 备份 (1h)

---

## 三、基于 Doris 分析数据的反馈闭环

### 3.1 agent_state.json → ML 特征增强

Phase 2 分析产出的 `agent_state.json`（114 keys）可以反哺 ML 训练：

| agent_state 字段 | 转化为 ML 特征/标签 | 预估增益 |
|---|---|---|
| `anomaly_results.level` | 异常等级 → L1/L2/L3 多分类标签 | 中 |
| `fwa_results` | 欺诈标记（22欺诈+18浪费+21滥用）→ 二分类标签 | 🔥 高 |
| `drg_analysis.over_budget` | DRG 超支标记 → 二分类特征 | 中 |
| `medical_rationality.flags` | 医疗合理性标记（4类规则）→ 类别特征 | 中 |
| `hospital_fee_anomaly.z_score` | 医院费用偏差 Z-score → 连续特征 | 高 |
| `member_risk_scores` | 会员风险评分 → 连续特征 | 高 |
| `ibnr_results.estimated` | IBNR 估值 → 连续特征 | 低 |
| `health_score.overall` | 健康评分 → 连续特征 | 中 |

### 3.2 闭环数据流设计

```
┌─────────────────────────────────────────────────────────────┐
│  闭环分析模型 — 数据反馈流                                  │
│                                                             │
│  ┌──────────────────┐                                       │
│  │ Phase 2 Agent    │  37节点 LangGraph 分析                │
│  │ agent_state.json │                                       │
│  └────────┬─────────┘                                       │
│           │ 提取异常标签、FWA标记、DRG分组                  │
│           ▼                                                 │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │ 标签增强器       │ ──→ │ ML 训练数据集     │             │
│  │ label_enricher   │     │ (新增 ~8 列)      │             │
│  └──────────────────┘     └────────┬─────────┘             │
│                                    │ 训练                    │
│                                    ▼                         │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │ ML 预测结果       │ ←── │ LightGBM 模型     │             │
│  │ predictions.parquet│    │ (含异常特征)      │             │
│  └────────┬─────────┘     └──────────────────┘             │
│           │ 回溯验证                                         │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ 交叉验证矩阵     │  异常等级 × 预测金额 → 四象限优先级   │
│  │ (阶段三 综合决策) │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、优先级排序的优化路线图

### 🔴 P0 — 阻断级（必须立即补齐，总工作量 ~4h）

#### P0-1: Optuna Study 断点续传

**现状**：Optuna 调优 50 trials 若中断则全部丢失，无法从断点恢复。

**方案**：使用 SQLite 持久化 Optuna study + 训练管线 checkpoint。

```python
# ml/pipeline/tune.py 修改

# === 1. Optuna Study SQLite 持久化 ===
import optuna

DB_PATH = "models/optuna/optuna_study.db"
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

study = optuna.create_study(
    study_name="l1a_tweedie_v2",
    direction="minimize",
    storage=f"sqlite:///{DB_PATH}",
    load_if_exists=True,          # 🔥 中断后可恢复
    pruner=optuna.pruners.MedianPruner(),
)

# study.optimize(objective, n_trials=args.trials)
# 中断后重新运行 → 自动从已完成 trial 之后继续
```

```python
# ml/pipeline/train.py 新增 checkpoint 机制

import pickle
from datetime import datetime

CHECKPOINT_DIR = Path("models/checkpoint")
CHECKPOINT_PATH = CHECKPOINT_DIR / "training_checkpoint.pkl"

def save_checkpoint(state: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state["timestamp"] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(state, f)
    logger.info("Checkpoint saved: step %d/%d", state["step"], state["total"])

def load_checkpoint() -> dict | None:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            return pickle.load(f)
    return None

# 训练流程集成:
#   Step 1-4 每步完成后调用 save_checkpoint({"step": i, "total": 9, ...})
#   启动时 load_checkpoint() → 跳过已完成步骤
```

**验证标准**：
- [ ] Optuna 50 trials 中断后重新运行，已完成 trial 不重复
- [ ] `python -m ml.pipeline.train` 中断后恢复，跳过已完成步骤
- [ ] Checkpoint 文件大小 < 1MB

**工作量**：1h

---

#### P0-2: 模型版本管理系统

**现状**：每次训练覆盖 `models/l1a_amount.txt`，历史模型不可追溯。

**方案**：版本号 + 时间戳命名 + 元数据索引。

```
models/
├── versions/
│   ├── v1.0_20260520/           # 初始版本
│   │   ├── l1a_amount.txt
│   │   ├── l1a_amount_meta.json
│   │   ├── feature_schema.json
│   │   └── training_report.json
│   ├── v1.1_20260601/           # 修复类别编码
│   ├── v1.2_20260610/           # P1 tweedie=1.2 + 季度回测
│   └── v1.3_20260618/           # P2a Optuna调优
├── latest -> versions/v1.3_20260618/   # 符号链接
└── model_registry.json                  # 版本元数据索引
```

```python
# ml/pipeline/train.py 新增版本管理

from datetime import datetime

VERSION_FILE = Path("models/model_registry.json")

def create_model_version(model_dir: Path, version: str, metrics: dict) -> Path:
    """创建带版本的模型目录，更新 registry。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = Path(f"models/versions/v{version}_{timestamp}")
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # 移动模型文件到版本目录
    for f in model_dir.glob("*"):
        if f.is_file():
            shutil.copy2(f, version_dir / f.name)
    
    # 更新 latest 符号链接
    latest_link = Path("models/latest")
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(version_dir.relative_to("models"), target_is_directory=True)
    
    # 更新 registry
    registry = load_registry()
    registry["versions"].append({
        "version": version,
        "timestamp": timestamp,
        "dir": str(version_dir),
        "metrics": metrics,
    })
    registry["latest"] = version
    save_registry(registry)
    
    return version_dir

def load_registry() -> dict:
    if VERSION_FILE.exists():
        return json.loads(VERSION_FILE.read_text(encoding="utf-8"))
    return {"versions": [], "latest": None}

def save_registry(registry: dict) -> None:
    VERSION_FILE.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
```

**验证标准**：
- [ ] 连续训练 3 次，每次生成独立版本目录
- [ ] `models/latest/` 始终指向最新版本
- [ ] `model_registry.json` 包含完整版本历史及指标
- [ ] predict.py 默认加载 `models/latest/`

**工作量**：2h

---

#### P0-3: COS 云备份

**现状**：`models/`、`predictions/`、`reports/ml/` 仅本地存储，无容灾。

**方案**：复用兄弟项目的 `cos_backup.py` 模式。

```python
# ml/pipeline/cos_backup.py (新增)

"""COS cloud backup for ML model artifacts.

Bucket: ins-kq6zz7wo-1313469539 (ap-guangzhou)
Path:   LLMs-from-scratch/projects/gbcost-insurance-ml/
"""

import os
from pathlib import Path
from qcloud_cos import CosConfig, CosS3Client

BUCKET = "ins-kq6zz7wo-1313469539"
REGION = "ap-guangzhou"
BASE_PREFIX = "LLMs-from-scratch/projects/gbcost-insurance-ml"

def backup_models(model_dir: str = "models/versions") -> None:
    """Upload all model versions to COS."""
    client = _get_client()
    for version_dir in Path(model_dir).iterdir():
        if version_dir.is_dir():
            for f in version_dir.glob("*"):
                cos_key = f"{BASE_PREFIX}/{version_dir.name}/{f.name}"
                client.upload_file(Bucket=BUCKET, Key=cos_key, LocalFilePath=str(f))

def backup_predictions(pred_dir: str = "predictions") -> None:
    """Upload prediction outputs to COS."""
    client = _get_client()
    for f in Path(pred_dir).glob("*"):
        if f.is_file():
            cos_key = f"{BASE_PREFIX}/predictions/{f.name}"
            client.upload_file(Bucket=BUCKET, Key=cos_key, LocalFilePath=str(f))

def backup_reports(report_dir: str = "reports/ml") -> None:
    """Upload evaluation reports to COS."""
    client = _get_client()
    for f in Path(report_dir).glob("*"):
        if f.is_file():
            cos_key = f"{BASE_PREFIX}/reports/{f.name}"
            client.upload_file(Bucket=BUCKET, Key=cos_key, LocalFilePath=str(f))

def _get_client() -> CosS3Client:
    config = CosConfig(
        Region=REGION,
        SecretId=os.environ["COS_SECRET_ID"],
        SecretKey=os.environ["COS_SECRET_KEY"],
    )
    return CosS3Client(config)

if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] or ["models", "predictions", "reports"]
    for target in targets:
        {"models": backup_models, "predictions": backup_predictions, "reports": backup_reports}[target]()
    print("Backup complete.")
```

**验证标准**：
- [ ] `python ml/pipeline/cos_backup.py` 上传成功
- [ ] COS 控制台可确认文件完整
- [ ] `train.py` 训练完成后自动触发备份（可选开关）

**工作量**：1h

---

### 🟡 P1 — 模型效果提升（短期可见收益，总工作量 ~13h）

#### P1-4: 时序滞后特征（Lag Features）

**动机**：当前 `build_features()` 计算了 `monthly` 聚合统计但仅做 left join，没有使用滞后值。Kronos 的经验表明时序滞后是预测能力的核心来源。

**实现**：在 `ml/data/feature_store.py` 的 `build_features()` 中扩展。

```python
# ml/data/feature_store.py — build_features() 新增 §6a

# 当前 §6 只做了 join，没有 lag
# 新增：为每个 policy_grp_name 生成 lag-1, lag-2, lag-3 特征

if monthly is not None:
    # ... 现有 join 逻辑 ...
    
    # === 新增: 滞后特征 ===
    lag_exprs = []
    lag_cols = ["monthly_total", "monthly_avg", "monthly_median", "monthly_count", "monthly_duty_total"]
    for lag in [1, 2, 3]:
        for col in lag_cols:
            lag_exprs.append(
                pl.col(col).shift(lag).over("policy_grp_name").alias(f"{col}_lag{lag}")
            )
        # 环比变化率
        if lag == 1:
            for col in ["monthly_total", "monthly_avg"]:
                lag_exprs.append(
                    (pl.col(col) / pl.col(col).shift(1).over("policy_grp_name") - 1)
                    .alias(f"{col}_mom_change")
                )
    
    feature_lf = feature_lf.with_columns(lag_exprs)
```

**验证标准**：
- [ ] 新增 lag 特征列数 = 5 cols × 3 lags + 2 mom_change = 17 个
- [ ] 训练后 `get_feature_importance("gain")` 中 lag 特征出现在 top 15
- [ ] 回测 Gini 提升 ≥ 0.02

**工作量**：3h

---

#### P1-5: L1-B 大额二分类模型

**动机**：当前 `large_recall=0.424`，大额案件识别不足。参考 Kronos Ch6 分类微调模式，构建独立的二分类模型。

**实现**：

```python
# ml/models/large_classifier.py (新增)

"""L1-B: Large claim binary classifier using LightGBM.

Maps to curriculum Ch6: Classification fine-tuning pattern.
- Replace regression head with binary classifier
- Use AUC/AUCPR as primary metrics
- Output probability can be combined with L1-A amount prediction
"""

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class LargeClaimClassifier:
    """Binary classifier for large claim detection (amount > P95 threshold)."""
    
    def __init__(
        self,
        threshold_quantile: float = 0.95,
        threshold_value: float = 24600.0,  # P95 from data profiling
        n_estimators: int = 500,
        num_leaves: int = 31,
        learning_rate: float = 0.03,
        scale_pos_weight: float | None = None,  # Auto-compute from class imbalance
    ):
        self.threshold = threshold_value
        self.n_estimators = n_estimators
        self.params = {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 50,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }
        if scale_pos_weight:
            self.params["scale_pos_weight"] = scale_pos_weight
        self.model: lgb.Booster | None = None
    
    def train(self, X_train, y_train, X_val, y_val, feature_names, categorical_features):
        """Train binary classifier with class weight balancing."""
        # Auto-compute scale_pos_weight if not set
        if "scale_pos_weight" not in self.params:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            self.params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        
        dtrain = lgb.Dataset(X_train, label=y_train, 
                            feature_name=feature_names,
                            categorical_feature=categorical_features)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        self.model = lgb.train(
            self.params, dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
        )
        return self.model
    
    def predict_proba(self, X) -> np.ndarray:
        """Return probability of being a large claim."""
        return self.model.predict(X)

# --- 双模型融合预测 ---
def ensemble_predict(
    amount_predictor,      # L1-A Tweedie regression
    large_classifier,      # L1-B binary classification
    df, feature_cols, categorical_cols,
    large_threshold: float = 24600.0,
) -> np.ndarray:
    """Combine L1-A + L1-B for enhanced prediction.
    
    Strategy: L1-A predicts base amount; L1-B identifies large claims.
    For predicted-large cases, adjust upward by predicted amount quantile.
    """
    base_pred = amount_predictor.predict(df, feature_cols, categorical_cols)
    large_proba = large_classifier.predict_proba(df)
    
    # For cases with high large-claim probability, boost prediction
    boost_mask = large_proba > 0.5
    adjusted_pred = base_pred.copy()
    adjusted_pred[boost_mask] *= (1.0 + large_proba[boost_mask] * 0.5)
    
    return adjusted_pred
```

**验证标准**：
- [ ] AUC ≥ 0.85 且 AUCPR ≥ 0.50
- [ ] large_recall 从 0.424 提升至 ≥ 0.60
- [ ] 双模型融合后 Gini ≥ 0.70
- [ ] 训练耗时 < 30s（不增加显著开销）

**工作量**：4h

---

#### P1-6: 会员历史行为特征

**动机**：当前仅有 `group_code` 和 `native_place_name` 的 Target Encoding，缺单个会员的历史行为画像。

**实现**：在 `compute_global_stats()` 中新增会员级聚合。

```python
# ml/data/feature_store.py — compute_global_stats() 新增

# --- Member-level historical features ---
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
        # Claims per year rate
        (pl.col("member_claim_count") / 
         ((pl.col("member_last_claim_date") - pl.col("member_first_claim_date"))
          .dt.total_days() / 365.25).clip(0.5, None)
        ).alias("member_claim_rate"),
    )
    .collect()
)
stats["member"] = member_stats

# build_features() 中 join:
if "insured_no" in feature_lf.collect_schema().names():
    member_cols = ["member_total_paid", "member_avg_paid", "member_std_paid",
                   "member_claim_count", "member_case_count", "member_claim_rate",
                   "member_hosp_count", "member_disease_count"]
    feature_lf = feature_lf.join(
        stats["member"].select(["insured_no"] + member_cols).lazy(),
        on="insured_no", how="left"
    )
    # Fill null for first-time claimants
    for col in member_cols:
        feature_lf = feature_lf.with_columns(
            pl.col(col).fill_null(0).alias(col)
        )
```

**验证标准**：
- [ ] 新增 8 个会员特征列
- [ ] 特征重要性 top 20 中包含 ≥ 2 个会员特征
- [ ] 不会导致 target leakage（train 时用 train_end 之前数据）

**工作量**：3h

---

#### P1-7: 分保单组总量偏差校正

**动机**：当前总量误差 49.6%，使用全局硬编码 `×1.80`。不同保单组偏差模式不同，应分组校正。

**实现**：

```python
# ml/models/calibration.py (新增)

"""Per-group calibration factors to correct systematic prediction bias."""

import numpy as np
import polars as pl

class GroupCalibrator:
    """Learn per-policy-group correction factors from validation data."""
    
    def __init__(self, group_col: str = "policy_grp_name"):
        self.group_col = group_col
        self.factors: dict[str, float] = {}
        self.global_factor: float = 1.0
    
    def fit(self, df: pl.DataFrame, y_true_col: str = "y_raw", y_pred_col: str = "y_pred"):
        """Compute per-group correction factors.
        
        factor_g = sum(true_g) / sum(pred_g)
        """
        df_with_pred = df.with_columns(pl.Series(y_pred_col, self._preds))
        
        group_sums = (
            df_with_pred
            .group_by(self.group_col)
            .agg(
                pl.col(y_true_col).sum().alias("total_true"),
                pl.col(y_pred_col).sum().alias("total_pred"),
            )
            .with_columns(
                (pl.col("total_true") / pl.col("total_pred").clip(1, None))
                .alias("factor")
            )
        )
        
        for row in group_sums.iter_rows(named=True):
            self.factors[row[self.group_col]] = row["factor"]
        
        # Global fallback
        total_true = group_sums["total_true"].sum()
        total_pred = group_sums["total_pred"].sum()
        self.global_factor = total_true / max(total_pred, 1)
        
        return self
    
    def calibrate(self, predictions: np.ndarray, groups: list[str]) -> np.ndarray:
        """Apply per-group correction to predictions."""
        calibrated = predictions.copy()
        for i, g in enumerate(groups):
            factor = self.factors.get(g, self.global_factor)
            calibrated[i] *= factor
        return calibrated
    
    def save(self, path: str) -> None:
        import json
        data = {"factors": self.factors, "global_factor": self.global_factor}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "GroupCalibrator":
        import json
        with open(path) as f:
            data = json.load(f)
        cal = cls()
        cal.factors = data["factors"]
        cal.global_factor = data["global_factor"]
        return cal
```

**验证标准**：
- [ ] 总量误差从 49.6% 降至 ≤ 30%
- [ ] 分组 RMS 误差 ≤ 全局 RMS 误差
- [ ] calibrator 在 train 时自动保存，predict 时自动加载

**工作量**：2h

---

#### P1-8: 分组 RankIC 评估

**动机**：当前用 Gini 衡量排序能力，但 Kronos 的 RankIC 提供了更精细的分组评估视角。

**实现**：

```python
# ml/evaluate/metrics.py 新增

def compute_group_rank_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Compute RankIC within each group, then average.
    
    RankIC = Spearman correlation between predicted rank and true rank.
    Group-level RankIC removes cross-group mean differences.
    
    Returns:
        dict with 'rank_ic_overall', 'rank_ic_group_mean', 
        'rank_ic_group_std', 'rank_ic_per_group'
    """
    from scipy.stats import spearmanr
    
    overall_ic, _ = spearmanr(y_pred, y_true)
    
    unique_groups = np.unique(groups)
    group_ics = []
    per_group = {}
    
    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 30:  # Skip tiny groups
            continue
        ic, _ = spearmanr(y_pred[mask], y_true[mask])
        group_ics.append(ic)
        per_group[str(g)] = round(ic, 4)
    
    return {
        "rank_ic_overall": round(overall_ic, 4),
        "rank_ic_group_mean": round(np.mean(group_ics), 4),
        "rank_ic_group_std": round(np.std(group_ics), 4),
        "rank_ic_per_group": per_group,
    }

def evaluate_predictions_with_groups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """Extended evaluation including group-level RankIC."""
    from ml.evaluate.metrics import evaluate_predictions
    results = evaluate_predictions(y_true, y_pred, **kwargs)
    
    if groups is not None:
        rank_ic = compute_group_rank_ic(y_true, y_pred, groups)
        results.update(rank_ic)
    
    return results
```

**验证标准**：
- [ ] RankIC overall ≥ 0.30（有效信号阈值）
- [ ] RankIC group_std ≤ 0.15（组间一致性良好）
- [ ] 识别 RankIC < 0 的保单组（模型失效组）

**工作量**：1h

---

### 🟢 P2 — 工程化提升（中长期，总工作量 ~13h）

#### P2-9: 实验管理系统

参照 Kronos 15 组实验的归档方式。

```
experiments/
├── exp_001_baseline_tweedie15/
│   ├── config_snapshot.yaml      # 参数快照
│   ├── metrics.json              # 评估指标
│   ├── feature_importance.csv    # 特征重要性
│   └── notes.md                  # 实验目的+结论
├── exp_002_log_transform_test/
├── exp_003_tweedie_power_scan/
├── ...
└── experiment_index.json         # 总索引
```

```bash
# 新增命令
python -m ml.pipeline.train --config ml/config_ml.yaml --exp-name "exp_004_lag_features"
```

**工作量**：3h

#### P2-10: 月度自动再训练管线

```python
# ml/pipeline/monthly_pipeline.py (新增)

"""Monthly automated ML retraining pipeline.
Triggered after Doris data update.
"""

def run_monthly_pipeline(
    csv_path: str = "data/doris/c001_ghb_poicy_clm_duty_d.csv",
    config_path: str = "ml/config_ml.yaml",
):
    """Complete monthly pipeline: train → predict → evaluate → backup."""
    steps = [
        ("train", lambda: train_new_model(csv_path, config_path)),
        ("predict", lambda: run_full_prediction(csv_path)),
        ("evaluate", lambda: generate_report()),
        ("calibrate", lambda: update_calibration()),
        ("backup", lambda: backup_to_cos()),
    ]
    
    for name, step_fn in steps:
        logger.info(f"[{name}] Starting...")
        try:
            step_fn()
            logger.info(f"[{name}] Complete.")
        except Exception as e:
            logger.error(f"[{name}] FAILED: {e}")
            send_alert(f"ML monthly pipeline failed at step: {name}")
            raise
    
    logger.info("Monthly pipeline complete.")

if __name__ == "__main__":
    run_monthly_pipeline()
```

**工作量**：3h

#### P2-11: 特征交互工程

```python
# ml/data/feature_store.py — build_features() 新增交互特征

# === 交互特征 ===
interaction_exprs = [
    # 年龄 × 险种（不同年龄段对不同险种的使用模式）
    (pl.col("age_bucket").cast(pl.Utf8) + "_" + pl.col("duty_code").cast(pl.Utf8))
        .alias("age_duty_interact"),
    
    # 医院等级 × 费用类型（高等级医院的手术费用特征）
    (pl.col("hosp_grade").cast(pl.Utf8) + "_" + pl.col("fee_item_type").cast(pl.Utf8))
        .alias("hosp_fee_interact"),
    
    # 就诊月份 × 疾病组（季节性 × 疾病）
    (pl.col("month").cast(pl.Utf8) + "_" + pl.col("group_code").cast(pl.Utf8))
        .alias("month_disease_interact"),
    
    # 是否年末 × 医院等级（年末体检高峰）
    (pl.col("is_yearend").cast(pl.Utf8) + "_" + pl.col("hosp_grade").cast(pl.Utf8))
        .alias("yearend_hosp_interact"),
    
    # 续保 × 险种（续保客户的使用行为）
    (pl.col("rnew_flag").cast(pl.Utf8) + "_" + pl.col("duty_code").cast(pl.Utf8))
        .alias("renewal_duty_interact"),
]

# 将这些交互列加入 categorical_cols 用于 LightGBM 原生类别处理
categorical_cols_extended = categorical_cols + [
    "age_duty_interact", "hosp_fee_interact", "month_disease_interact",
    "yearend_hosp_interact", "renewal_duty_interact",
]
```

**工作量**：4h

#### P2-12: 双模型集成预测

```python
# ml/pipeline/predict.py 新增集成模式

def ensemble_predict_pipeline(
    csv_path: str,
    amount_model_path: str = "models/latest/l1a_amount.txt",
    classifier_model_path: str = "models/latest/l1b_classifier.txt",
    calibrator_path: str = "models/latest/calibrator.json",
) -> pl.DataFrame:
    """Full ensemble prediction pipeline."""
    
    # 1. 加载模型
    amount_predictor = AmountPredictor.load(amount_model_path)
    classifier = LargeClaimClassifier.load(classifier_model_path)
    calibrator = GroupCalibrator.load(calibrator_path)
    
    # 2. 特征工程
    lf = load_doris_csv(csv_path)
    feature_lf = build_features(lf, global_stats)
    df = feature_lf.collect(streaming=True)
    
    # 3. 预测
    base_preds = amount_predictor.predict(df)
    large_probas = classifier.predict_proba(df)
    
    # 4. 集成融合
    ensemble_preds = ensemble_predict(amount_predictor, classifier, df)
    
    # 5. 分组校正
    calibrated_preds = calibrator.calibrate(ensemble_preds, df["policy_grp_name"].to_list())
    
    # 6. 输出
    result = df.with_columns([
        pl.Series("pred_base", base_preds),
        pl.Series("pred_large_proba", large_probas),
        pl.Series("pred_ensemble", ensemble_preds),
        pl.Series("pred_calibrated", calibrated_preds),
    ])
    
    return result
```

**工作量**：3h

---

### 🔵 P3 — 前沿探索（可选）

| # | 任务 | 说明 | 工作量 |
|---|------|------|:---:|
| P3-13 | **L2 保单级预测** | 从案件级聚合到保单级，引入保单画像特征（保费规模、行业、地域） | 6h |
| P3-14 | **LSTM 序列建模** | 将会员/保单组的历史建模为时序，Kronos 验证了 LSTM 有效性。可与 LightGBM 做集成 | 8h |
| P3-15 | **自动化评估报告** | 参考医疗项目 BLEU/ROUGE 自动化评分，生成模型质量月度自动评估 | 3h |
| P3-16 | **异常标签反哺** | 从 agent_state.json 批量提取异常标签，构建标注训练集，实现"分析→标注→训练"闭环 | 6h |

---

## 五、实施顺序建议

```
Week 1: P0 全部 (4h)
  ├── P0-1: Optuna 断点续传 (1h)
  ├── P0-2: 模型版本管理 (2h)
  └── P0-3: COS 云备份 (1h)

Week 2-3: P1 核心 (13h)
  ├── P1-4: 时序滞后特征 (3h)  ← 预期增益最高
  ├── P1-5: L1-B 大额二分类 (4h)
  ├── P1-6: 会员行为特征 (3h)
  ├── P1-7: 分组偏差校正 (2h)
  └── P1-8: 分组 RankIC (1h)

Week 4-5: P2 工程化 (13h)
  ├── P2-9: 实验管理系统 (3h)
  ├── P2-10: 月度自动管线 (3h)
  ├── P2-11: 特征交互 (4h)
  └── P2-12: 双模型集成 (3h)
```

---

## 六、预期效果矩阵

完成 P0+P1 后的指标预测：

| 指标 | 当前值 | P1 后预期 | 改善幅度 | 依据 |
|------|:---:|:---:|:---:|------|
| Gini | 0.688 | ≥ 0.72 | +5% | 时序特征 + 会员特征增强排序信号 |
| R² | 0.647 | ≥ 0.70 | +8% | 更多预测变量解释方差 |
| 总量误差 | 49.6% | ≤ 30% | -40% | 分组校正替代全局硬编码 |
| large_recall | 0.424 | ≥ 0.60 | +42% | L1-B 专用二分类模型 |
| 回测 Gini 稳定性 | 0.72±3% | 0.72±2% | 更稳定 | 滞后特征平滑时序波动 |
| 训练中断恢复 | ❌ | ✅ | — | P0-1 断点续传 |
| 模型可追溯 | ❌ | ✅ | — | P0-2 版本管理 |
| 容灾能力 | ❌ | ✅ | — | P0-3 COS 备份 |

---

## 七、依赖与前置条件

| 依赖项 | 状态 | 说明 |
|------|:---:|------|
| Polars ≥ 0.20 | ✅ | 已安装，惰性管线正常运行 |
| LightGBM ≥ 4.0 | ✅ | Tweedie objective 可用 |
| Optuna ≥ 3.0 | ✅ | SQLite storage 需 optuna 3.0+ |
| scikit-learn ≥ 1.0 | ✅ | AUC/AUCPR 计算 |
| scipy ≥ 1.7 | ✅ | spearmanr for RankIC |
| qcloud_cos | ⚠️ | P0-3 需要 `pip install cos-python-sdk-v5` |
| Doris CSV 数据 | ✅ | `data/doris/c001_ghb_poicy_clm_duty_d.csv` |
| agent_state.json | ⚠️ | P3-16 需要批量分析产出，路径由用户指定 |

---

## 八、文件变更清单

### 新增文件

| 文件 | 说明 | 对应任务 |
|------|------|---------|
| `ml/pipeline/cos_backup.py` | COS 云备份脚本 | P0-3 |
| `ml/models/large_classifier.py` | L1-B 大额二分类模型 | P1-5 |
| `ml/models/calibration.py` | 分组偏差校正器 | P1-7 |
| `ml/pipeline/monthly_pipeline.py` | 月度自动再训练管线 | P2-10 |
| `experiments/` 目录 | 实验管理系统 | P2-9 |

### 修改文件

| 文件 | 变更内容 | 对应任务 |
|------|---------|---------|
| `ml/pipeline/train.py` | 断点续传 + 版本管理 + 实验命名 | P0-1, P0-2 |
| `ml/pipeline/tune.py` | Optuna SQLite 持久化 | P0-1 |
| `ml/data/feature_store.py` | 时序滞后特征 + 会员特征 + 交互特征 | P1-4, P1-6, P2-11 |
| `ml/evaluate/metrics.py` | 分组 RankIC + 扩展评估函数 | P1-8 |
| `ml/pipeline/predict.py` | 集成预测模式 + 分组校正 | P1-7, P2-12 |
| `ml/config_ml.yaml` | 新增 l1b section + 实验配置 | P1-5 |

---

> **核心理念**：不换模型架构（LightGBM 已经是结构化保险数据的最优解），通过**补齐基础设施（P0）→ 增强特征工程（P1）→ 工程化运维（P2）**的路径，将方案从"可工作"提升到"生产级"。

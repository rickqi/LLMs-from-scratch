# 团险控费 ML — 详细技术文档

> v2.0 | 2026-06-18 | P0-P2 全部完成

## 项目定位

基于 LightGBM + Optuna 的保险理赔金额预测系统。结构化数据 ML 的最佳实践：Tweedie 损失 + 贝叶斯超参优化 + Walk-forward 回测。

## 技术架构

```
                    ┌─ Parquet 缓存层 (54MB, 225x 压缩) ─┐
                    │   train_cache.parquet              │
                    │   prepare_train_cache()             │
                    └──────────────┬──────────────────────┘
                                   │ <1s (vs 11min)
13GB CSV (71列) ──→ 特征工程 (71列) ──→ LightGBM ──→ Walk-Forward回测
       │                  │              Tweedie            │
       │            ┌──────┴──────┐    + Optuna        MAE/MAPE/RMSE
       │            │ lag-1/2/3   │    + 断点续传        Gini/RankIC
       │            │ member行为  │    + 版本管理
       │            │ 交叉交互    │    + COS备份
       │            └─────────────┘
       │
  polars lazy scan
  (1次扫描缓存, 1次扫描特征)
```

## 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 数据加载 | `ml/data/loader.py` | polars CSV lazy scan + Parquet 缓存 |
| 缓存层 | `ml/data/feature_store.py` | `prepare_train_cache()` 一趟扫描 13GB→54MB |
| 特征工程 | `ml/data/feature_store.py` | TE + lag + member + 交叉交互 (71列) |
| 数据分割 | `ml/data/split.py` | 时序 split (train/val/test) |
| L1-A 金额回归 | `ml/models/amount_predictor.py` | LightGBM Tweedie + 分位数 |
| L1-B 大额分类 | `ml/models/large_classifier.py` | 二分类 + scale_pos_weight |
| 偏差校正 | `ml/models/calibration.py` | 分组校正因子 (替代 ×1.80) |
| 训练流水线 | `ml/pipeline/train.py` | CLI + checkpoint + 版本管理 + 实验归档 |
| 超参优化 | `ml/pipeline/tune.py` | Optuna SQLite 持久化断点续传 |
| 预测推断 | `ml/pipeline/predict.py` | 批量预测 + ensemble + calibrate |
| 月度管线 | `ml/pipeline/monthly_pipeline.py` | train→predict→eval→backup 全自动 |
| 云备份 | `ml/pipeline/cos_backup.py` | COS 容灾 |
| 回测 | `ml/pipeline/run_backtest.py` | Walk-forward 时序回测 |
| 评估 | `ml/evaluate/metrics.py` | Gini/MAE/MAPE + 分组 RankIC |

## 模型配置

```yaml
# config_ml.yaml — P2a Optuna 调优后
l1a:
  objective: tweedie
  tweedie_power: 1.35
  n_estimators: 800
  num_leaves: 127
  learning_rate: 0.118
  subsample: 0.6
  colsample_bytree: 1.0
  reg_alpha: 0.014
  reg_lambda: 0.12
  min_child_samples: 200
  early_stopping_rounds: 50
  log_transform: false

l1b:
  threshold_quantile: 0.95
  n_estimators: 500
  num_leaves: 31
  learning_rate: 0.03
  metric: "average_precision"
```

## 特征列表 (71 列, 15 分类)

| 类别 | 特征 | 新增 |
|------|------|:---:|
| 身份信息 | age, age_bucket, insured_gender, main_insured_rela | |
| 保单信息 | duty_code, policy_grp_name, rnew_flag, sale_chnl, pass_months, policy_age_days, monthly_prem | |
| 就诊信息 | hosp_grade, claim_type, fee_item_type, is_public, is_expensive, day_count, est_loss_ratio | |
| 时序特征 | month, quarter, month_sin, month_cos, is_yearend | |
| 目标编码 | group_code_te, native_place_name_te (Bayesian smoothed) | |
| **月度滞后** | monthly_total_lag{1,2,3}, monthly_avg_lag{1,2,3}, ... | 🆕 P1-4 |
| **环比变化** | monthly_total_mom, monthly_avg_mom | 🆕 P1-4 |
| **会员行为** | member_total_paid, member_avg_paid, member_claim_rate, ... (10列) | 🆕 P1-6 |
| **交叉交互** | age_bucket_x_duty_code, hosp_grade_x_fee_item_type, ... (4组) | 🆕 P2-11 |
| 金融特征 | annual_prem_log, pass_prem_log, year_actual_claim_paid_log | |
| 大额标记 | large_case_filled | |

## 训练命令

```bash
# ── 数据缓存（首次运行，后续自动跳过）──
python -m ml.pipeline.train --config ml/config_ml.yaml --rebuild-cache
#   13GB CSV → 54MB Parquet (225x 压缩, ~3 min)

# ── 完整训练（使用缓存）──
python -m ml.pipeline.train --config ml/config_ml.yaml --mode full
#   全局统计: <1s (缓存) | 特征构建: ~5min (1次CSV扫描)
#   训练: ~5min | 总计: ~10min

# ── 快速训练（跳过回测）──
python -m ml.pipeline.train --config ml/config_ml.yaml --mode full --no-backtest

# ── 实验追踪 ──
python -m ml.pipeline.train --config ml/config_ml.yaml --exp-name "lag_features_v1"

# ── 超参优化（断点续传）──
python -m ml.pipeline.tune --config ml/config_ml.yaml --trials 50

# ── 预测 ──
python -m ml.pipeline.predict --csv data/doris/c001_ghb_poicy_clm_duty_d.csv --policy ALL
python -m ml.pipeline.predict --csv ... --policy GP2202301028502 --ensemble --calibrate

# ── 月度自动管线 ──
python -m ml.pipeline.monthly_pipeline

# ── COS 备份 ──
python -m ml.pipeline.cos_backup models predictions reports

# ── 回测 ──
python ml/pipeline/run_backtest.py
```

## 当前模型指标

| 指标 | 值 | 说明 |
|------|:---:|------|
| 训练集 | 362万行 (1063 保单) | Parquet 缓存 |
| 验证集 | 210万行 | 时序安全 |
| val_l1 | 3,182 | Tweedie 损失 |
| R² | 0.306 | P1 特征后持续优化中 |
| Gini | 0.564 | 排序能力 |
| large_auc | 0.852 | L1-B 候选特征有效 |
| total_error_pct | 66.7% | 需 L1-B + calibrator 联合校正 |
| 缓存体积 | 54 MB | 12.16 GB → 54 MB (225x) |

## 依赖

```
lightgbm, optuna, polars, numpy, scikit-learn, scipy, pyyaml, qcloud_cos
```

## 工程基础设施

| 能力 | 实现 | 状态 |
|------|------|:---:|
| 断点续传 | Optuna SQLite + pickle checkpoint | ✅ P0-1 |
| 模型版本管理 | `models/versions/v{N}_{ts}/` + `latest/` + registry.json | ✅ P0-2 |
| COS 云备份 | `ml/pipeline/cos_backup.py` | ✅ P0-3 |
| 实验管理 | `--exp-name` + `experiments/` 归档 | ✅ P2-9 |
| 月度自动管线 | `ml/pipeline/monthly_pipeline.py` | ✅ P2-10 |
| Parquet 缓存 | 一趟扫描 13GB→54MB | ✅ v2.0 |

## 与深度学习项目对比

| 维度 | 控费 (LightGBM) | Kronos (PyTorch) |
|------|----------------|-----------------|
| 训练速度 | 10min | 27s~2h |
| 可解释性 | ✅ 特征重要性 | ❌ 黑盒 |
| 数据要求 | 结构化表格 | 时序滑动窗口 |
| 超参优化 | ✅ Optuna + 断点续传 | ✅ 手动扫描 |
| 断点续传 | ✅ SQLite + pickle | ✅ training_state.pt |
| 版本管理 | ✅ registry.json | ✅ |
| 云备份 | ✅ COS | ✅ COS |
| 数据缓存 | ✅ Parquet 225x | — |
| 双模型 | ✅ L1-A 回归 + L1-B 分类 | ✅ LSTM 方向 + 波动率 |

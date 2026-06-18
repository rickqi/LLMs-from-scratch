# 团险控费 ML — 详细技术文档

## 项目定位

基于 LightGBM + Optuna 的保险理赔金额预测系统。结构化数据 ML 的最佳实践：Tweedie 损失 + 贝叶斯超参优化 + Walk-forward 回测。

## 技术架构

```
13GB CSV (71列)                    特征工程                   模型层
c001_ghb_poicy_clm_duty_d.csv → polars chunked loading → target_encoding
                                                           ↓
                                    categorical encoding → LightGBM
                                                           ↓ Tweedie loss
                                    feature filtering → Optuna tuning
                                                           ↓
                                                      Walk-forward回测
                                                           ↓
                                                     MAE/MAPE/RMSE
```

## 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 数据加载 | `ml/data/loader.py` | polars CSV chunked loading |
| 特征工程 | `ml/data/feature_store.py` | Target encoding + categorical + global stats |
| 数据分割 | `ml/data/split.py` | 时序split (train/val/test) |
| 模型核心 | `ml/models/amount_predictor.py` | LightGBM + Tweedie/Gamma/Regression |
| 训练流水线 | `ml/pipeline/train.py` | 完整训练CLI |
| 超参优化 | `ml/pipeline/tune.py` | Optuna 贝叶斯优化 |
| 预测推断 | `ml/pipeline/predict.py` | 批量预测 |
| 回测 | `ml/pipeline/run_backtest.py` | Walk-forward 时序回测 |
| 评估 | `ml/evaluate/metrics.py` | MAE/MAPE/RMSE |
| 报告 | `ml/report/generator.py` | 自动报告生成 |

## 模型配置

```yaml
# AmountPredictor 默认参数
objective: tweedie          # Tweedie损失(零膨胀+长尾)
tweedie_power: 1.5          # 1=Poisson, 2=Gamma
n_estimators: 800
num_leaves: 63
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
early_stopping_rounds: 50
log_transform: True         # 对数变换目标变量
```

## 特征列表 (71 → 20+)

| 类别 | 特征 |
|------|------|
| 保单信息 | 险种、保单组、销售渠道、续保标志 |
| 人员信息 | 性别、年龄、与主被保人关系 |
| 就诊信息 | 医院等级、就诊类型、费用项目类型 |
| 目标编码 | 险种组平均金额、籍贯地平均金额 |

## 训练命令

```bash
# 完整训练
python -m ml.pipeline.train --config ml/config_ml.yaml

# 超参优化
python -m ml.pipeline.tune --config ml/config_ml.yaml --trials 30

# 预测
python -m ml.pipeline.predict --model models/l1a_amount.txt --input data/

# 回测
python -m ml.pipeline.run_backtest --config ml/config_ml.yaml
```

## 依赖

```
lightgbm, optuna, polars, numpy, scikit-learn, pyyaml
```

## 与深度学习项目对比

| 维度 | 控费 (LightGBM) | Kronos (PyTorch) |
|------|----------------|-----------------|
| 训练速度 | 5min | 27s~2h |
| 可解释性 | ✅ 特征重要性 | ❌ 黑盒 |
| 数据要求 | 结构化表格 | 时序滑动窗口 |
| 超参优化 | ✅ Optuna自动 | ❌ 手动扫描 |
| 断点续传 | ❌ 无 | ✅ training_state.pt |

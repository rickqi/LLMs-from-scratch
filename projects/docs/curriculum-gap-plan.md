# Kronos Stock — 教程覆盖率补充方案

> Ch6 分类微调 (🔴 0%) + 其他未 100% 章节 (🟡)

---

## 一、Ch6 分类微调 🔴 — 唯一完全缺失

### 为什么缺失

Kronos 项目以**回归**（预测连续收益率）为核心，从未实现分类微调模式。这是教程 7 章中唯一完全无实践覆盖的章节。

### Ch6 教程知识点

| 知识点 | 说明 |
|--------|------|
| 替换输出头 | 把 GPT 的 LM head 换成分类 Linear |
| 冻结策略 | 冻结预训练层，仅训练分类头 |
| 微调训练 | 在下游标注数据上 fine-tune |
| 分类评估 | Accuracy / Precision / Recall / F1 |
| 垃圾邮件检测 | 二分类实战案例 |

### 映射到 Kronos 的实现方案

| 教程概念 | Kronos 映射 |
|---------|------------|
| GPT预训练模型 | LSTM(128h,2l) 回归模型 (已训练, RankIC+0.205) |
| 替换输出头 | 将 `head(32→1)` 替换为 `classifier(32→2)` |
| 冻结骨干 | `param.requires_grad = False` 冻结 LSTM + 前两层 |
| 下游任务 | 68只半导体：未来10日涨跌 (UP/DOWN) |
| 二分类 | BCEWithLogitsLoss |
| 评估 | Accuracy / F1 / Confusion Matrix |

### 实施步骤

```python
# Step 1: 加载预训练回归模型
model = LSTMModel(hidden=128, num_layers=2)
model.load_state_dict(torch.load('outputs/production_model.pt'))

# Step 2: 冻结 LSTM 骨干
for name, param in model.named_parameters():
    if 'lstm' in name or 'head' not in name:
        param.requires_grad = False

# Step 3: 替换输出头 (Ch6 核心)
model.head = nn.Sequential(
    nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 2)  # 2分类: UP/DOWN
)

# Step 4: 微调训练
loss_fn = nn.BCEWithLogitsLoss()
# 数据集: StockDataset → ClassifyDataset (label=0/1)
# 训练: 仅更新 head 参数

# Step 5: 评估
# Accuracy = 正确预测数 / 总数
# F1 = 2 * P * R / (P + R)
# Confusion Matrix: [[TN, FP], [FN, TP]]
```

### 完整实施脚本

```bash
# 新增文件: scripts/ch6_classification.py
python scripts/ch6_classification.py \
    --model_path outputs/production_model.pt \
    --data_dir data/processed_real \
    --freeze_layers all \  # 冻结所有非head层
    --epochs 5 \
    --lr 1e-3
```

### 预期结果

| 指标 | 预期值 |
|------|--------|
| Accuracy | 55-60% (基线50%) |
| F1 (UP) | 0.55-0.60 |
| 可训参数 | ~2K (仅classifier head) |

### 工作量: **2 小时**

---

## 二、Ch7 指令微调 🟡 60% — 为什么未 100%

### 已覆盖 (60%)

| 覆盖 | 位置 |
|------|------|
| ✅ Alpaca/ChatML 格式 | 医疗项目 `InstructionDataset` |
| ✅ Response-only loss | 医疗项目 `loss masking` |
| ✅ 指令数据生成 | 医疗项目 `med_qa_generator.py` |

### 未覆盖 (40%)

| 缺失 | 说明 |
|------|------|
| ❌ 自动化评估 | 医疗项目仅人工看生成质量 |
| ❌ DPO 偏好对齐 | 无 RLHF/DPO 实践 |

### 原因

1. **自动化评估**: 文本生成质量评估（BLEU/ROUGE/LLM-as-Judge）在教程中未详细展开，是进阶话题
2. **DPO**: 教程 Ch7 介绍了 DPO 概念但无完整代码示例，属于推荐扩展

### 建议

| 补充 | 工作量 | 优先级 |
|------|--------|--------|
| 医疗项目添加 BLEU/ROUGE 自动评分 | 2h | P2 |
| Kronos 波动率+方向双信号（类似指令微调的 conditional output） | 已完成 | ✅ |

---

## 三、App D 训练增强 🟢 80% — 缺失 LR Finder

### 已覆盖 (80%)

| 覆盖 | 位置 |
|------|------|
| ✅ LR 调度器 | Kronos Cosine+Warmup |
| ✅ 梯度裁剪 | Kronos clip_grad_norm_(5.0) |
| ✅ Checkpoint 保存 | Kronos 每 epoch 保存 training_state.pt |

### 未覆盖 (20%): LR Finder

**原因**: LR Finder 是可选工具，教程将其放在附录 D 作为"锦上添花"。Kronos 已通过手动扫描找到最优 LR (1e-3)。

### 建议: 低优先级，可不补充。手动扫描已足够。

---

## 四、App E LoRA 🟡 50% — 为什么未 100%

### 已覆盖 (50%)

| 覆盖 | 位置 |
|------|------|
| ✅ LoRA 微调 | 医疗项目 PEFT r=8 |
| ✅ 参数效率 | 0.7M / 600M = 0.12% |

### 未覆盖 (50%): Kronos 未用 LoRA

**原因**: Kronos 模型仅 0.5M 参数，全参数训练即可。LoRA 的价值在 ≥100M 参数模型上体现，对小模型无实际意义。

### 建议: 不补充。Kronos 全参数训练 (0.5M) 已是合理选择。如需演示 LoRA，可在医疗项目中展示。

---

## 五、优先级总结

| 优先级 | 章节 | 内容 | 项目 | 工作量 |
|--------|------|------|------|--------|
| **P0** 🔥 | Ch6 | 分类微调 (涨跌二分类) | Kronos | 2h |
| P2 | Ch7 | 自动化评估 (BLEU/ROUGE) | 医疗 | 2h |
| P3 | App D | LR Finder | Kronos | 1h |
| — | App E | LoRA on Kronos | — | 不需要 |

### Ch6 执行后覆盖率: **83% → 93%**

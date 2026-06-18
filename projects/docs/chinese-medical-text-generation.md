# 中文医学文本生成 — 详细技术文档

## 项目定位

基于 Qwen3-0.6B + LoRA 的中文医学诊疗指南文本生成系统。两阶段训练：续写微调(领域适应) → 指令微调(问答能力)。

## 技术架构

```
数据层              训练层                 推理层
412篇MD文档 ──→ data_prep.py → 清洗/分句
                                    ↓
DeepSeek API ──→ med_qa_generator → 449 QA对
                                    ↓
                          Qwen3-0.6B + LoRA
                           ↓           ↓
                     续写模型      指令模型
                           ↓           ↓
                      generate.py (批量/交互)
```

## 模型选型

| 方案 | 模型 | 参数量 | 中文能力 | 结论 |
|------|------|--------|---------|------|
| A | GPT-2 124M | 124M | 乱码 | ❌ |
| B | Qwen3-0.6B | 600M | 通顺 | ✅ |

Qwen3-0.6B 是 ≤1B 参数范围内唯一具备实用中文生成能力的模型。

## 训练配置

| 参数 | 续写微调 | 指令微调 |
|------|---------|---------|
| 数据 | 412篇文档 | 449 QA对 |
| 格式 | 纯文本续写 | ChatML |
| Epochs | 5 | 3 |
| LoRA rank | 8 | 8 |
| 学习率 | 5e-4 | 5e-5 |
| Loss | CrossEntropy | response-only |

## 核心模块

| 模块 | 行数 | 说明 |
|------|------|------|
| `train_qwen_lora.py` | 423 | 训练主脚本 |
| `data_prep.py` | 289 | 数据清洗+样本生成 |
| `generate.py` | — | 推理(3模式) |
| `scripts/med_qa_generator.py` | — | QA对生成(3档) |
| `scripts/cos_backup.py` | 415 | COS云备份 |

## 数据流

```
原始MD(107MB, 412篇)
  → clean_medical_text() 清洗
  → 分句/分段
  → 80% train / 20% val
  → output_full/ (续写)
  → output_inst_v1/ (指令)
```

## Checkpoint 机制

```python
checkpoint_dir / "training_state.pt"
# 保存: model(PeftModel) + optimizer + scheduler + epoch + loss_logs
# 恢复: 自动检测, 从断点继续
```

## 依赖

```
torch, transformers, peft, datasets, accelerate,
sentencepiece, qcloud_cos
```

## 启动命令

```bash
# 续写微调
python train_qwen_lora.py --data_dir ./data_full --output_dir ./output_full --epochs 5

# 指令微调
python train_qwen_lora.py --resume_from ./output_full/best_model \
  --instruction_data ./docs/med_instruction_chatml.json --output_dir ./output_inst_v1

# 推理
python generate.py --model_dir ./output_inst_v1/best_model --interactive
```

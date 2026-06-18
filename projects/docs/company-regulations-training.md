# 公司法规文本生成 — 详细技术文档

## 项目定位

基于 Qwen3-0.6B + LoRA 的公司规章制度文本生成。四项目中最简实现 (329 行训练代码)。

## 技术架构

```
公司制度MD文档 → data_prep.py → Qwen3-0.6B+LoRA → generate.py
```

## 与医疗项目的差异

| 维度 | 医疗 | 法规 |
|------|------|------|
| 代码量 | 423行 | **329行** (最简) |
| 数据量 | 412篇 | ~50篇 |
| 指令微调 | ✅ 449 QA | ❌ 仅续写 |
| 数据生成 | ✅ DeepSeek API | ❌ 无 |
| COS备份 | ✅ | ✅ |
| 断点续传 | ✅ | ✅ |

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Epochs | 5 |
| 学习率 | 5e-4 |
| Batch size | 4 |
| Max length | 512 |

## 核心模块

| 模块 | 行数 | 说明 |
|------|------|------|
| `train_qwen_lora.py` | 329 | 训练(最简) |
| `data_prep.py` | — | 数据准备 |
| `generate.py` | — | 推理 |
| `scripts/cos_backup.py` | — | COS备份 |

## 设计哲学

法规项目采用"最小可用"策略：
- 仅保留续写训练（无指令微调）
- 直接复用医疗项目的训练框架
- 329 行代码覆盖完整训练+推理+备份

## 依赖

```
torch, transformers, peft, datasets, accelerate
```

## 启动命令

```bash
python train_qwen_lora.py --data_dir ./data --output_dir ./output --epochs 5
python generate.py --model_dir ./output/best_model
```

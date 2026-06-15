# AGENTS.md — 中文医学文本生成项目

> AI Agent 工作流文档。描述项目结构、关键文件、开发约定和操作入口。

## 项目概述

基于 Qwen3-0.6B + LoRA 的中文医学文本生成，两阶段训练：

1. **阶段1**: 纯续写微调 — 412 篇医学文档 → 领域语言适应 (训练中)
2. **阶段2**: 指令微调 — 449 条 ChatML QA 对 → 问答能力 (待执行)

## 关键文件索引

| 文件 | 用途 | 何时读取 |
|------|------|---------|
| `README.md` | 项目概览、快速开始 | 首次接触项目 |
| `ANALYSIS.md` | 模型选型、数据来源、超参、风险 | 理解设计决策 |
| `OPERATIONS.md` | 训练/备份/数据生成 操作命令 | 执行具体操作 |
| `docs/instruction_ft_plan.md` | 指令微调实现方案 | 实施阶段2 |
| `scripts/med_qa_generator.py` | QA 数据生成 (3 档) | 生成指令数据 |
| `scripts/cos_backup.py` | COS 云备份 | 备份 LoRA/数据 |
| `train_qwen_lora.py` | 训练脚本 | 执行训练 |
| `generate.py` | 推理脚本 | 模型推理 |

## 当前状态

| 组件 | 状态 | 位置 |
|------|------|------|
| 阶段1 训练 | 🔄 进行中 | `output_full/` (Epoch 1/5) |
| 指令数据 | ✅ 449 对 | `docs/med_qa_cases.json` |
| COS 备份 | ✅ 已备份 | `LLMs-from-scratch/projects/chinese-medical-text-generation/` |
| 阶段2 训练 | ⏳ 待执行 | 等阶段1 完成后 |

## 开发约定

- 所有数据和模型产出加入 `.gitignore`，只提交代码和文档
- LoRA adapter 跨阶段续训 (`--resume_from`)，不重新初始化
- QA 生成使用 DeepSeek V3 API，temperature=0.2
- COS 备份到 `ins-kq6zz7wo-1313469539` (ap-guangzhou)
- 训练日志写入 `train_full.log`，JSON 日志写入 `output_xxx/training_log.json`

## 操作入口

**训练相关**:
```bash
python train_qwen_lora.py --data_dir ./data_full --output_dir ./output_full --epochs 5
```

**指令微调**:
```bash
# 等阶段1完成后执行
python scripts/cos_backup.py pre-ft-backup
python train_qwen_lora.py --resume_from ./output_full/best_model --instruction_data ./docs/med_instruction_chatml.json --output_dir ./output_inst_v1 --epochs 3 --lr 5e-5
```

**数据生成**:
```bash
python scripts/med_qa_generator.py generate --tier medium
python scripts/med_qa_generator.py export --fmt chatml
```

**备份**:
```bash
python scripts/cos_backup.py pre-ft-backup
python scripts/cos_backup.py backup --incremental
```

## 外部依赖

| 依赖 | 说明 |
|------|------|
| `/home/raw/medica/` | 原始医学文档 (412 篇 .md, 107MB) |
| DeepSeek API | QA 生成 (`deepseek-chat`) |
| COS SDK (`qcloud_cos`) | 云备份 (credential 从 medica-handbook/.env 读取) |
| HuggingFace Mirror | 模型下载 (`https://hf-mirror.com`) |

## 关联项目

- `medica-handbook` — 医学文档下载与管理 (数据来源)
- `doc-search` — 全文搜索引擎 (QA 生成脚本参考)

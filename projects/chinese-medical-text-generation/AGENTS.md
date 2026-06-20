# AGENTS.md — 中文医学文本生成项目

> AI Agent 工作流文档。描述项目结构、关键文件、开发约定和操作入口。

## 项目概述

基于 Qwen3-0.6B/1.7B + LoRA 的中文医学文本生成，两阶段训练：

1. **阶段1**: 纯续写微调 — 412 篇医学文档 → 领域语言适应 ✅ (0.6B + **1.7B** 均完成)
2. **阶段2**: 指令微调 — 607 条 ChatML QA 对 → 问答能力 ✅ (1.7B v2: val=1.8879, 含早停+过拟合检测)

## 关键文件索引

| 文件 | 用途 | 何时读取 |
|------|------|---------|
| `README.md` | 项目概览、快速开始 | 首次接触项目 |
| `ANALYSIS.md` | 模型选型、数据来源、超参、风险 | 理解设计决策 |
| `OPERATIONS.md` | 训练/备份/数据生成 操作命令 | 执行具体操作 |
| `docs/instruction_ft_plan.md` | 指令微调实现方案 | 实施阶段2 |
| `docs/swir_integration.md` | SwiReasoning 推理增强方案 | 阶段2完成后 |
| `scripts/med_qa_generator.py` | QA 数据生成 (3 档) | 生成指令数据 |
| `scripts/cos_backup.py` | COS 云备份 | 备份 LoRA/数据 |
| `train_qwen_lora.py` | 训练脚本 | 执行训练 |
| `generate.py` | 推理脚本 | 模型推理 |

## 当前状态

| 组件 | 状态 | 位置 | 关键指标 |
|------|------|------|---------|
| 0.6B 阶段1 续写 | ✅ 完成 | `output_full/` | val_loss=2.4033, 5 epoch |
| **1.7B 阶段1 续写** | ✅ 完成 | `output_17b_full/` | **val_loss=2.1135**, 5 epoch, 19h |
| 0.6B 阶段2 指令 | ✅ 完成 | `output_inst_v3/` | val_loss=2.3938, 4 epoch |
| **1.7B 阶段2 指令 v2** | ✅ 完成 | `output_17b_inst_v2/` | **val_loss=1.8879**, 11min(早停), 640步 |
| 指令数据 | ✅ 607 对 | `docs/med_instruction_chatml.json` | train=557, val=50 (hold-out) |
| COS 备份 | ✅ 已备份 | `ins-kq6zz7wo-1313469539` | — |
| 训练标准 | ✅ 已植入 | `train_qwen_lora.py` | 早停+过拟合检测+独立验证集 |
| DPO 偏好对齐 | ⏳ 待执行 | — | 覆盖率 98%→100% 的关键缺口 |
| SwiReasoning | ⏳ 待执行 | `docs/swir_integration.md` | 推理增强 |

## 最佳模型

```
Qwen3-1.7B
  └── 阶段1: 续写微调 ✅  val=2.1135, 19h
        └── 阶段2: 指令微调 v2 ✅  val=1.8879, 11min
              └── 产出: output_17b_inst_v2/best_model ← 当前最佳
```

## 行动路线图

### 当前步骤：DPO 偏好对齐 (教程覆盖 98% → 100%)

```bash
cd projects/chinese-medical-text-generation
# Step 1: 构造偏好数据
python scripts/prepare_dpo_data.py
# Step 2: DPO 训练 (基于最佳指令模型)
python train_dpo.py --base_model ./output_17b_inst_v2/best_model --preference_data ./data/dpo_pairs.json
# Step 3: 评估
python scripts/eval_compare.py --model_before ./output_17b_inst_v2 --model_after ./output_dpo
```

### P0 之后：可选增强

按优先级排序：

```
DPO 完成 (覆盖率 100%)
        │
        ├── [P1] SwiReasoning 推理增强
        │   - 验证 Qwen3 兼容性
        │   - 对比评估: 有/无 SwiReasoning
        │
        ├── [P2] 数据扩展
        │   python scripts/med_qa_generator.py generate --tier large
        │   607 对 → 目标 700+ 对
        │
        └── [P3] 质量基准评估
            建立标准测试集 + 多模型对比
```

## 开发约定

- 所有数据和模型产出加入 `.gitignore`，只提交代码和文档
- LoRA adapter 跨阶段续训 (`--resume_from`)，不重新初始化
- QA 生成使用 DeepSeek V3 API，temperature=0.2
- COS 备份到 `ins-kq6zz7wo-1313469539` (ap-guangzhou)
- 训练日志写入 `train_full.log`，JSON 日志写入 `output_xxx/training_log.json`
- **指令微调完成前，不执行 SwiReasoning 集成**

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
- `sdc17/SwiReasoning` — 动态推理框架 (阶段2 完成后集成)

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
| `test_questions.py` | 统一测试问题定义（单一真实来源） | 添加/修改测试问题时 |
| `scripts/eval_compare.py` | 多模型评测对比 | 对比基线 vs 目标模型 |

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
| **1.7B DPO v2** | ✅ 完成 | `output_17b_dpo_v2/` | val=0.644, β=0.05, 191对, 3ep, 5.6min |
| DPO 偏好对齐 | ✅ 完成 | 191对过滤数据, 0坍塌 | 教程覆盖 100% |
| SwiReasoning | ⏳ 待执行 | `docs/swir_integration.md` | 推理增强 |

## 最佳模型

```
Qwen3-1.7B
  ├── 阶段1: 续写微调 ✅  val=2.114, 19h
  ├── 阶段2: 指令微调 v1 ✅  val=2.046, 384min (过拟合，已淘汰)
  ├── 阶段2: 指令微调 v2 ✅  val=1.833, 14.9min ← 当前最佳
  │     └── 767 QA (607+160 targeted), lr=1e-5, ratio=0.4, epochs=1
  └── DPO v1 ✅  基于 v1, batch=1, β0.05, 0坍塌

Qwen3-0.6B
  ├── 阶段2: 指令微调 v3 ✅  val=2.404
  └── DPO v3 ✅  380对, β0.05, e1, 0坍塌
```

## 模型排名

| 排名 | 模型 | val | tok/9题 | 速度 | 偏短 |
|------|------|-----|---------|------|------|
| 🥇 | 1.7B Inst-V2 | 1.833 | 2700 | 13.7s | 0 |
| 🥈 | 1.7B DPO v1 | — | 2290 | 11.5s | 0 |
| 🥉 | 0.6B DPO v3 | — | 2175 | 10.2s | 0 |
| 4 | 0.6B Inst-V3 | 2.404 | 2580 | 47.6s | 0 |

## 行动路线图

### 当前步骤：Inst-V2 DPO 完整管线 (知识注入 → 偏好对齐)

```bash
cd projects/chinese-medical-text-generation

# 当前最佳: 1.7B Inst-V2 (val=1.833)
# 下一步: DPO on Inst-V2 完成知识→对齐管线
python train_dpo.py \
    --model_name /home/models/ms_cache/Qwen/Qwen3-1___7B \
    --base_model ./output_17b_inst_v2/best_model \
    --preference_data ./data/dpo_pairs_cleaned_v2.json \
    --output_dir ./output_17b_dpo_v2 \
    --epochs 1 --beta 0.05 --batch_size 1

# 评测
python scripts/eval_compare.py \
    --model_dir ./output_17b_dpo_v2 \
    --base_model /home/models/ms_cache/Qwen/Qwen3-1___7B \
    --output ./evals/eval_17b_dpo_v2.json
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

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
| `docs/swir_integration.md` | SwiReasoning 推理增强方案 | 阶段2完成后 |
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

## 行动路线图

### 当前步骤：等待阶段1 完成

```
阶段1 训练完成 (output_full/best_model 生成)
        │
        ▼
[ ] Step 1: COS 备份阶段1 LoRA 权重
    python scripts/cos_backup.py pre-ft-backup
        │
        ▼
[ ] Step 2: 改造训练脚本 (按 instruction_ft_plan.md §四 实施)
    - 新增 InstructionDataset (ChatML + loss masking)
    - 新增 MixedDataset (80% 指令 + 20% 续写)
    - 新增 --resume_from 参数 (PeftModel.from_pretrained)
    - 新增 --instruction_data / --instruction_ratio 参数
        │
        ▼
[ ] Step 3: 执行指令微调训练
    python train_qwen_lora.py \
        --resume_from ./output_full/best_model \
        --data_dir ./data_full \
        --instruction_data ./docs/med_instruction_chatml.json \
        --output_dir ./output_inst_v1 \
        --epochs 3 --lr 5e-5 --max_length 768 \
        --instruction_ratio 0.8
        │
        ▼
[ ] Step 4: 推理验证
    python generate.py --model_dir ./output_inst_v1/best_model --interactive
```

### 阶段2 完成后：下一步行动

按优先级排序，无需顺序执行：

```
阶段2 完成 (output_inst_v1/best_model)
        │
        ├── [P1] SwiReasoning 推理增强
        │   详见 docs/swir_integration.md
        │   - git clone https://github.com/sdc17/SwiReasoning.git external/SwiReasoning
        │   - 验证 Qwen3-0.6B 兼容性 (hidden_dim=1024 需实测)
        │   - 创建 scripts/generate_swir.py (Wrapper 模式)
        │   - 对比评估: 8 个标准 prompt 有/无 SwiReasoning
        │   - 预期: hard 推理 +2-3%, easy 问题 Token -40-60%
        │
        ├── [P2] 数据扩展 (按需)
        │   python scripts/med_qa_generator.py generate --tier large
        │   当前 449 对 → 目标 560+ 对
        │   适用场景: 指令微调效果不足，需要更多样化数据
        │
        ├── [P3] 质量基准评估
        │   建立标准测试集: 50 个医学 QA (含标准答案)
        │   评分维度: 准确性 / 完整性 / 格式规范性 / 可追溯性
        │   对比: 纯续写模型 vs 指令微调模型 vs +SwiReasoning
        │
        └── [P4] DPO 偏好对齐 (可选)
            流程: 收集好/坏回答对 → DPO 训练 → 输出质量优化
            前提: 指令微调效果满意，需要进一步对齐
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

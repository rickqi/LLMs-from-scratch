# 中文医学诊疗指南文本生成

基于 **Qwen3-0.6B + LoRA** 的中文医学文本生成项目，两阶段训练：纯续写领域适应 → 指令微调。

## 快速开始

```bash
# 环境
pip install torch transformers peft datasets accelerate

# 一键运行 (样本数据 → 训练 → 推理)
bash run.sh

# 自定义数据
python data_prep.py --data_dir /home/raw/medica --output_dir ./data_full
python train_qwen_lora.py --data_dir ./data_full --output_dir ./output_full --epochs 5
python generate.py --model_dir ./output_full/best_model
```

## 训练流程

```
阶段1: 纯续写微调                      阶段2: 指令微调
═══════════════════                   ═══════════════════
Qwen3-0.6B + LoRA                    加载阶段1 LoRA 权重
data_full (52MB, 29K样本)            ChatML 格式 (449 QA对)
5 epochs, lr=1e-4                    3 epochs, lr=5e-5
↓                                    ↓
output_full/best_model               output_inst_v1/best_model
(医学领域语言能力)                    (+ 指令跟随问答能力)
```

## 项目文件

| 文件 | 说明 |
|------|------|
| `run.sh` | 一键执行 (环境检查 → 数据 → 训练 → 推理) |
| `data_prep.py` | 数据准备: 清洗 / 分割 / `===SEP===` 格式 |
| `train_qwen_lora.py` | 训练脚本: Qwen3-0.6B + LoRA, checkpoint 断点续训 |
| `train_qwen_lora.ipynb` | Jupyter 交互式训练笔记本 |
| `generate.py` | 推理: 批量评估 / 单次生成 / 交互模式 |
| `ANALYSIS.md` | 全面评估: 模型选型 / 数据来源 / 超参 / 风险 |
| `OPERATIONS.md` | 运维手册: 备份 / 数据生成 / 训练操作 |
| `requirements.txt` | Python 依赖 |

| 目录 | 说明 |
|------|------|
| `scripts/` | 工具脚本: QA 生成器 / COS 备份 |
| `docs/` | 文档与数据: 方案 / 报告 / QA 数据 / ChatML 导出 |
| `data_full/` | (gitignore) 全量训练数据, 52MB |
| `output_full/` | (gitignore) 阶段1 LoRA 权重 (训练中) |
| `output_inst_v1/` | (gitignore) 阶段2 指令微调产出 (待创建) |

## 训练数据来源

412 篇中文医学文档 (107MB), 4 大分类:

| 来源 | 数量 | 内容 |
|------|------|------|
| L1 卫健委官方规范 | 170 篇 | 肿瘤诊疗指南 / CDC 传染病 / WHO 中文指南 |
| L2 卫生行业标准 WS | 58 篇 | 临床检验 / 医院感染控制 / 基层医疗 / 老年健康 |
| L3 临床诊疗指南丛书 | 140 篇 | CACA 整合指南 / CMA 专家共识 / 36 分册 / 专科指南 |
| L4 临床技术操作规范 | 44 篇 | 49 个分册涵盖内外妇儿各专科操作标准 |

详见 [ANALYSIS.md §二](./ANALYSIS.md)。

## 指令微调数据

使用 `scripts/med_qa_generator.py` 从医学文档自动生成 QA 对:

```bash
# 三档生成
python scripts/med_qa_generator.py generate --tier small    # ~250 对
python scripts/med_qa_generator.py generate --tier medium   # ~450 对 (当前)
python scripts/med_qa_generator.py generate --tier large    # ~560 对

# 导出训练格式
python scripts/med_qa_generator.py export --fmt chatml
python scripts/med_qa_generator.py export --fmt alpaca

# 质量分析
python scripts/med_qa_generator.py stats
```

当前: **449 条 QA 对**, 98 篇文档, easy/medium/hard = 35%/42%/22%

## COS 云备份

```bash
# 指令微调前完整备份
python scripts/cos_backup.py pre-ft-backup

# 增量备份
python scripts/cos_backup.py backup --incremental

# 查看远程文件
python scripts/cos_backup.py list
```

桶: `ins-kq6zz7wo-1313469539` (ap-guangzhou) / 前缀: `LLMs-from-scratch/projects/chinese-medical-text-generation/`

## 模型选型

≤1B 参数范围内, Qwen3-0.6B 是唯一具备实用中文生成能力的模型:
- 预训练 36T tokens (中英双语), 28 层 + GQA 注意力
- 中文 tokenizer 效率 1.6 字/token (GPT-2 仅 0.7)
- LoRA 仅训练 0.12% 参数 (~0.7M), 消费级 GPU 可运行
- Dr. Qwen 论文已验证医学微调有效性

## 目录结构

```
chinese-medical-text-generation/
├── run.sh                     # 一键执行
├── data_prep.py               # 数据预处理
├── train_qwen_lora.py         # 训练脚本 (checkpoint 续训)
├── train_qwen_lora.ipynb      # 交互式训练
├── generate.py                # 推理脚本
├── README.md                  # 本文档
├── ANALYSIS.md                # 全面评估与方案
├── OPERATIONS.md              # 运维操作手册
├── AGENTS.md                  # AI Agent 工作流
├── requirements.txt           # Python 依赖
├── .gitignore                 # 排除数据/模型/产出
├── scripts/
│   ├── med_qa_generator.py    # 指令微调 QA 生成器
│   └── cos_backup.py          # COS 云备份
├── docs/
│   ├── instruction_ft_plan.md # 指令微调实现方案
│   ├── med_qa_report.md       # QA 数据质量报告
│   ├── med_qa_cases.json      # 449 条 QA 原始数据
│   ├── med_instruction_chatml.json  # ChatML 训练格式
│   └── med_instruction_alpaca.json  # Alpaca 训练格式
├── data_full/                 # (gitignore) 训练数据 52MB
├── output_full/               # (gitignore) 阶段1 LoRA
└── output_inst_v1/            # (gitignore) 阶段2 指令微调
```

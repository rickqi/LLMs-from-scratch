# 中文医学诊疗指南文本生成

基于 **Qwen3-0.6B + LoRA** 的完整中文医学文本生成项目。

## 快速开始

```bash
# 一键运行 (自动生成样本数据 + 训练 + 推理)
bash run.sh

# 或分步执行
python data_prep.py --sample --output_dir ./data
python train_qwen_lora.py --data_dir ./data --output_dir ./output --epochs 5
python generate.py --model_dir ./output/best_model
```

## 项目文件

| 文件 | 说明 |
|------|------|
| `run.sh` | 一键执行脚本 (环境检查 → 数据 → 训练 → 推理) |
| `data_prep.py` | 数据准备: 支持真实数据 / 自动生成样本数据 |
| `train_qwen_lora.py` | 训练脚本: Qwen3-0.6B + LoRA 微调 |
| `train_qwen_lora.ipynb` | Jupyter 笔记本: 完整交互式训练流程 |
| `generate.py` | 推理脚本: 批量评估 / 单次生成 / 交互模式 |
| `ANALYSIS.md` | 全面评估: 模型选型分析、执行计划、风险评估 |
| `requirements.txt` | 依赖清单 |
| `.gitignore` | 排除数据/模型产出 (项目可独立复制使用) |

## 方案总览

| 维度 | 方案 A: GPT-2 124M | 方案 B: Qwen3-0.6B |
|------|-------------------|---------------------|
| Tokenizer | GPT-2 BPE (0.7 中文/token) | Qwen BPE (1.6 中文/token, 2.3×) |
| 参数 | 124M (12层) | 600M (28层, GQA) |
| 预训练 | 英文为主, 未知中文 | 36T tokens, 中英双语 |
| 微调 | 全量训练 | LoRA (0.12% 参数, ~0.7M) |
| 训练前中文输出 | `临床表现：＝＝＝＝＝` 乱码 | 通顺中文医学文本 |
| 推荐度 | ❌ | ✅ |

## 模型选型理由

≤1B 参数范围内, Qwen 系列是**唯一**具备实用中文生成能力的模型。Qwen3-0.6B 相比 Qwen2.5-0.5B:
- 预训练数据翻倍 (36T vs 18T tokens)
- 28层 + GQA 注意力, 更深的表示能力
- 已有 "Dr. Qwen" 论文验证医学微调有效性
- 仅 +21% 参数换取显著更强的中文能力

详见 [ANALYSIS.md](./ANALYSIS.md)。

## 目录结构

```
chinese-medical-text-generation/
├── run.sh                 # 一键执行
├── data_prep.py           # 数据准备
├── train_qwen_lora.py     # 训练脚本
├── train_qwen_lora.ipynb  # 训练笔记本
├── generate.py            # 推理脚本
├── README.md              # 本文档
├── ANALYSIS.md            # 全面评估
├── requirements.txt       # 依赖
├── .gitignore             # 独立 gitignore
├── raw_data/              # (gitignore) 原始 .md 文件
├── data/                  # (gitignore) 预处理数据
└── output/                # (gitignore) LoRA 权重 + 训练日志
```

## 扩展路径

| 阶段 | 行动 | 预期提升 |
|------|------|---------|
| 数据增强 | 使用全部 77 册 (15M→50M+ 字符) | 领域覆盖更全 |
| 增加 Epochs | 5 → 10 → 20 | 更充分的领域适应 |
| LoRA 调参 | r=8 → r=16, alpha=16 → 32 | 更强表达能力 |
| 指令微调 | 构建医学 QA 数据集 | 问答助手能力 |
| RAG 增强 | 医学知识库 + 向量检索 | 知识准确性 |
| 更大模型 | Qwen3-1.7B / 4B / 8B | 生成质量飞跃 |
| 评估体系 | 执业医师资格考试 | 可量化评估 |

## 依赖

```bash
pip install torch transformers peft datasets accelerate
```

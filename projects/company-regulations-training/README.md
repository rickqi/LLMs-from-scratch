# 公司规章制度文本生成

基于 **Qwen3-0.6B + LoRA** 的公司规章制度文本生成项目。数据/模型不提交 Git，通过 COS 云备份。

## 当前状态: Epoch 1 训练中 (92%)

```
模型:   Qwen3-0.6B + LoRA (r=8)
数据:   3023 篇公司制度 (53.2 MB, 17.7M tokens)
GPU:    AMD Radeon 890M (ROCm 7.2, bf16, VRAM 6.1 GB)
进度:   Ep 1/3 | Batch 16,300/17,684 (92.3%)
Loss:   Train 3.12→1.91 (↓39%) | Val 2.91→1.72 (↓41%)
备份:   COS 已备份 (best_model + 数据, 74.8 MB)
```

## 快速开始

```bash
# 环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export COS_SECRET_ID="your_id"
export COS_SECRET_KEY="your_key"

# 1. 数据准备
python data_prep.py --data_dir "/home/docs/raw/公司规章制度" --output_dir ./data

# 2. 训练 (AMD GPU, 推荐配置: seq=256)
python train_qwen_lora.py --data_dir ./data --output_dir ./output --epochs 3 --batch_size 4 --max_length 256 --lr 2e-4

# 3. 推理
python generate.py --model_dir ./output/best_model

# 4. 交互模式
python generate.py --model_dir ./output/best_model --interactive

# 5. COS 备份
python scripts/cos_backup.py backup
```

## 训练配置 (实际使用)

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen3-0.6B (596M) | 中英双语预训练, 28 层 |
| LoRA rank | 8 | 低秩适配 |
| LoRA target | q/k/v/o_proj | 28 层 × 4 模块 = 112 矩阵 |
| LoRA 可训练参数 | 2.3M (0.38%) | 基座完全冻结 |
| Batch size | 4 | ×4 梯度累积 = 有效 16 |
| Max length | **256** (优化后) | 原默认 512 太慢 |
| Epochs | 3 | |
| Learning rate | 2e-4 | Warmup 5% + Linear decay |
| Gradient clipping | max_norm=1.0 | |
| Precision | bfloat16 | AMD 890M 原生支持 |

## AMD 890M 实测性能

| batch | seq | 每 batch | VRAM | 每 epoch |
|-------|-----|---------|------|---------|
| **4** | **256** | **~2.7s** | **6.1 GB** | **~13h (实际使用)** |
| 4 | 512 | 5.5s | 12.0 GB | ~27h |
| 2 | 512 | 2.5s | 6.6 GB | ~12h |
| 8 | 512 | 10.6s | 22.8 GB | ~52h |

## 训练数据

| 指标 | 值 |
|------|-----|
| 数据源 | `/home/docs/raw/公司规章制度/` |
| 原始文件 | 4252 个 .md |
| 清洗后训练 | 3023 篇 (53.2 MB) |
| 验证集 | 160 篇 (1.8 MB) |
| 总 tokens | 17.7M (Qwen tokenizer) |
| 训练样本 (seq=256) | 70,738 |
| 验证样本 | 2,273 |

## LoRA 权重影响分析

通过 `merge_and_unload()` 对比基座与微调后权重:

| 指标 | 值 |
|------|-----|
| LoRA 训练参数 | 2.3M (0.38%) |
| 影响基座参数 | 176.2M (**29.6%**) |
| 放大倍数 | 76.8x |
| 平均变化幅度 | 0.0008 |
| 最大变化幅度 | 0.030 (o_proj) |

影响范围: 仅 Attention 的 q/k/v/o_proj (28 层), FFN 和 Embedding 完全冻结。

## 生成示例 (Epoch 1 中途, val_loss=1.72)

```
【制度名称：】《招商信诺人寿保险有限公司
  保险合同条款变更审批制度》
  1. 申请变更保险合同条款变更的，应当在合同条款变更
  申请表上填写变更内容...

【审批权限：】总精算师
  （四）产品定价规则...
```

模型已学会公司制度领域术语和格式 (招商信诺/保险合同/总精算师等)。

## COS 备份

数据/模型不提交 Git，通过 COS 云备份:

```bash
export COS_SECRET_ID="your_id"
export COS_SECRET_KEY="your_key"

python scripts/cos_backup.py backup                # 全量
python scripts/cos_backup.py backup --data-only     # 仅数据
python scripts/cos_backup.py backup --model-only    # 仅模型
python scripts/cos_backup.py backup --incremental    # 增量
python scripts/cos_backup.py list                    # 查看远程
```

- 桶: `ins-kq6zz7wo-1313469539` (ap-guangzhou)
- 前缀: `LLMs-from-scratch/projects/company-regulations-training/`
- 最近备份: 2026-06-16, 9 文件, 74.8 MB

## 文件结构

```
company-regulations-training/
├── data_prep.py              # 数据准备 (清洗/分割/格式化)
├── train_qwen_lora.py        # 训练脚本 (LoRA + 断点续训 + 进度日志)
├── generate.py               # 推理脚本 (批量/交互模式)
├── run.sh                    # 一键执行
├── requirements.txt          # Python 依赖
├── .gitignore                # 排除数据/模型
├── README.md                 # 本文档
├── CHANGELOG.md              # 更新记录
├── AGENTS.md                 # AI Agent 工作流文档
├── scripts/
│   └── cos_backup.py         # COS 云备份 (独立脚本)
├── docs/
│   └── 训练方案.md            # 完整训练方案 + 实测数据
├── data/                     # (gitignore) 训练数据 → COS
└── output/                   # (gitignore) 模型输出 → COS
    ├── best_model/           # 最佳 LoRA 权重 (val_loss 最低)
    ├── final_model/          # 最终 LoRA 权重
    ├── checkpoint/           # 断点检查点
    └── training_log.json     # 训练曲线
```

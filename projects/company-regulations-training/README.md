# 公司规章制度文本生成

基于 **Qwen3-0.6B + LoRA** 的公司规章制度文本生成项目。数据/模型不提交 Git，通过 COS 云备份。

## 快速开始

```bash
# 1. 数据准备
python data_prep.py --data_dir "/home/docs/raw/公司规章制度" --output_dir ./data

# 2. 训练 (AMD GPU)
python train_qwen_lora.py --data_dir ./data --output_dir ./output --epochs 5

# 3. 推理
python generate.py --model_dir ./output/best_model

# 4. 交互模式
python generate.py --model_dir ./output/best_model --interactive

# 5. COS 备份 (数据+模型)
python scripts/cos_backup.py backup
```

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen3-0.6B | 中英双语预训练 |
| LoRA rank | 8 | 低秩适配 |
| LoRA target | q/k/v/o_proj | 注意力层 |
| Batch size | 4 | ×4 梯度累积 = 有效 16 |
| Max length | 512 | 上下文窗口 |
| Epochs | 5 | 训练轮数 |
| Learning rate | 2e-4 | 带 warmup |
| 可训练参数 | ~2.3M (0.38%) | LoRA 高效 |

## AMD 890M 性能基准 (实测)

| batch | seq | 每 batch 耗时 | VRAM 峰值 |
|-------|-----|-------------|----------|
| 4 | 512 | 5.5s | 12.0 GB |
| 2 | 512 | 2.5s | 6.6 GB |
| 4 | 256 | 1.8s | 6.2 GB |
| 8 | 512 | 10.6s | 22.8 GB |

### 训练时间估算 (batch=4, seq=512)

| 指标 | 值 |
|------|-----|
| 总训练 tokens | 17,726,200 |
| 训练样本 (seq=512) | 34,621 |
| 每 epoch 批次数 | 8,655 |
| 每 epoch 耗时 | ~13 小时 |
| 5 epochs 耗时 | ~58 小时 |

> 注意: 全量数据训练时间较长。建议先用子集测试，或减小 seq=256 (每 epoch ~4 小时)。

## 数据

- 来源: `/home/docs/raw/公司规章制度/`
- 原始: 4252 个 .md 文件
- 清洗后: 3023 篇训练 (53.2 MB) + 160 篇验证 (1.8 MB)
- 清洗规则: 去除 HTML/图片/OCR/Markdown 标记

## COS 备份 (数据/模型不提交 Git)

```bash
# 全量备份 (数据+模型+日志)
python scripts/cos_backup.py backup

# 仅备份数据
python scripts/cos_backup.py backup --data-only

# 仅备份模型权重
python scripts/cos_backup.py backup --model-only

# 增量备份 (跳过未修改文件)
python scripts/cos_backup.py backup --incremental

# 预览 (不实际上传)
python scripts/cos_backup.py backup --dry-run

# 查看远程文件
python scripts/cos_backup.py list
```

COS 配置:
- 桶: `ins-kq6zz7wo-1313469539` (ap-guangzhou)
- 前缀: `LLMs-from-scratch/projects/company-regulations-training/`
- 凭证: 环境变量 `COS_SECRET_ID` / `COS_SECRET_KEY`

## 文件结构

```
company-regulations-training/
├── data_prep.py            # 数据准备 (清洗/分割/格式化)
├── train_qwen_lora.py      # 训练脚本 (LoRA + 断点续训)
├── generate.py             # 推理脚本 (批量/交互模式)
├── run.sh                  # 一键执行
├── requirements.txt        # Python 依赖
├── .gitignore              # 排除数据/模型
├── README.md               # 本文档
├── scripts/
│   └── cos_backup.py       # COS 云备份 (独立脚本)
├── docs/
│   └── 训练方案.md          # 完整训练方案
├── data/                   # (gitignore) 训练数据 → COS 备份
└── output/                 # (gitignore) 模型输出 → COS 备份
```

# 运维操作手册

> 中文医学文本生成项目 — 训练 / 备份 / 数据生成 操作指南

## 环境检查

```bash
# GPU
python -c "import torch; print(torch.cuda.is_available())"

# HF 镜像
curl -I https://hf-mirror.com

# COS SDK
python -c "from qcloud_cos import CosS3Client; print('OK')"

# DeepSeek API (QA 生成)
python -c "from openai import OpenAI; c=OpenAI(api_key='$DEEPSEEK_API_KEY',base_url='https://api.deepseek.com'); print(c.models.list().data[0].id)"
```

## 训练操作

### 阶段1: 纯续写微调

```bash
# 数据准备 (如首次)
python data_prep.py --data_dir /home/raw/medica --output_dir ./data_full

# 开始训练
python train_qwen_lora.py \
    --data_dir ./data_full \
    --output_dir ./output_full \
    --epochs 5 \
    --batch_size 4 \
    --max_length 512 \
    --lr 1e-4

# 监控
tail -f train_full.log
watch -n 10 'ls -lh output_full/best_model/adapter_model.safetensors'
```

### 阶段2: 指令微调

```bash
# 前置: 确保 QA 数据已生成
python scripts/med_qa_generator.py stats

# 备份阶段1 权重
python scripts/cos_backup.py pre-ft-backup

# 指令微调训练 (在阶段1 LoRA 基础上)
python train_qwen_lora.py \
    --resume_from ./output_full/best_model \
    --data_dir ./data_full \
    --instruction_data ./docs/med_instruction_chatml.json \
    --output_dir ./output_inst_v1 \
    --epochs 3 \
    --lr 5e-5 \
    --max_length 768 \
    --instruction_ratio 0.8

# 推理验证
python generate.py --model_dir ./output_inst_v1/best_model --interactive
```

### 中断恢复

训练脚本支持 checkpoint 续训，中断后直接重新运行相同命令即可。checkpoint 保存在 `output_xxx/checkpoint/training_state.pt`。

## COS 备份

### 指令微调前完整备份

```bash
python scripts/cos_backup.py pre-ft-backup
```

备份内容: `output_full/` (LoRA 权重 + checkpoint) + `docs/` (QA 数据) + `scripts/` + 训练日志

### 日常增量备份

```bash
python scripts/cos_backup.py backup --incremental
```

仅上传有变化的文件，大小一致的跳过。

### 仅备份特定内容

```bash
python scripts/cos_backup.py backup --lora-only      # 仅 LoRA 权重
python scripts/cos_backup.py backup --data-only      # 仅指令数据
```

### 预览

```bash
python scripts/cos_backup.py backup --dry-run        # 不实际上传
python scripts/cos_backup.py list                    # 查看远程文件
```

COS 桶: `ins-kq6zz7wo-1313469539` (ap-guangzhou)
前缀: `LLMs-from-scratch/projects/chinese-medical-text-generation/`

## 指令微调 QA 数据生成

### 三档生成

```bash
# Small: 快速实验 (~250 对, ~1.5h 训练)
python scripts/med_qa_generator.py generate --tier small

# Medium: 正式训练 (~450 对, ~3h 训练)  ← 当前
python scripts/med_qa_generator.py generate --tier medium

# Large: 最大覆盖 (~560 对, ~5h 训练)
python scripts/med_qa_generator.py generate --tier large
```

增量模式，已有文档自动跳过不重复生成。

### 导出与统计

```bash
python scripts/med_qa_generator.py export --fmt chatml   # Qwen3 训练格式
python scripts/med_qa_generator.py export --fmt alpaca   # 通用格式
python scripts/med_qa_generator.py stats                 # 质量分析报告
```

### 预览文档内容

```bash
python scripts/med_qa_generator.py sample L1_卫健委官方规范
```

## 推理

```bash
# 批量评估 (8 个预设 prompt)
python generate.py --model_dir ./output_full/best_model

# 交互模式
python generate.py --model_dir ./output_full/best_model --interactive

# 单次生成
python generate.py --model_dir ./output_full/best_model --prompt "胃癌的临床表现："
```

## 常用诊断命令

```bash
# 训练状态
tail -20 train_full.log
ps aux | grep train_qwen_lora

# GPU 状态
nvidia-smi

# 模型大小
ls -lh output_full/best_model/adapter_model.safetensors

# 数据统计
wc -l data_full/train.txt
python scripts/med_qa_generator.py stats
```

## 数据流程图

```
/home/raw/medica/ (412篇 .md)
    │
    ├── data_prep.py → data_full/ (52MB, 29K样本)
    │                        │
    │                        └── train_qwen_lora.py → output_full/ (LoRA)
    │                                                      │
    │                              ┌───────────────────────┘
    │                              │
    ├── med_qa_generator.py → docs/med_qa_cases.json (449 QA对)
    │                              │
    │                              ├── export → med_instruction_chatml.json
    │                              │
    │                              └── train_qwen_lora.py --resume_from
    │                                      → output_inst_v1/ (指令微调)
    │
    └── cos_backup.py → COS 云存储 (ins-kq6zz7wo-1313469539)
```

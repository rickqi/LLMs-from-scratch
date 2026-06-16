# 更新记录

## 2026-06-16: 训练启动 + LoRA 影响分析

### 训练状态
- Epoch 1/3 进行中: Batch 14,250 / 17,684 (80.5%)
- Train loss: 3.12 → 1.95 (↓ 37%)
- Val loss: 2.91 → 1.75 (↓ 40%, best_model 持续更新)
- 运行时间: ~20 小时 (含系统休眠中断)
- GPU: AMD Radeon 890M, bf16, VRAM 6.1 GB

### 代码变更 (commit 54d8242)
- `train_qwen_lora.py`:
  - `evaluate(max_batches=30)`: 限制评估到 30 batch, 避免全量 eval 19 分钟卡顿
  - eval 频率从每 10 步改为每 20 步
  - 新增每 50 batch 实时进度日志
- `docs/训练方案.md`:
  - 新增第十一节: 实测性能、Loss 趋势、LoRA 权重影响分析
  - 脚本优化记录

### LoRA 权重影响分析
- LoRA 训练参数: 2.3M (0.38%)
- 实际影响基座参数: 176.2M (29.6%)
- 放大倍数: 76.8x
- 变化幅度: 平均 0.0008, 最大 0.030
- 影响范围: 仅 q/k/v/o_proj (28 层 × 4 模块 = 112 个权重矩阵)

## 2026-06-15: 项目创建 (commit 8545411)

### 初始提交
- `data_prep.py`: 公司制度数据清洗 (4252 .md → 3023 篇有效文本)
- `train_qwen_lora.py`: Qwen3-0.6B + LoRA 训练脚本 (断点续训)
- `generate.py`: 推理脚本 (批量 + 交互模式)
- `scripts/cos_backup.py`: 独立 COS 备份 (数据/模型不进 Git)
- `docs/训练方案.md`: 完整训练方案
- `run.sh`: 一键执行

### 数据准备
- 来源: `/home/docs/raw/公司规章制度/`
- 清洗后: 3023 篇训练 (53.2 MB) + 160 篇验证 (1.8 MB)
- Token 数: 17.7M (Qwen tokenizer)

### AMD 890M 基准测试
- bs=4, seq=256: 2.7s/batch, VRAM 6.1 GB
- bs=4, seq=512: 5.5s/batch, VRAM 12.0 GB
- bs=2, seq=512: 2.5s/batch, VRAM 6.6 GB
- bs=8, seq=512: 10.6s/batch, VRAM 22.8 GB

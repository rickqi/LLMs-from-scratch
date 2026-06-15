#!/bin/bash
set -euo pipefail

# ============================================================
# 公司规章制度 Qwen3-0.6B + LoRA 训练一键脚本
# 环境: AMD Radeon 890M + ROCm 7.2 + PyTorch 2.9
# ============================================================

cd "$(dirname "$0")"
echo "============================================"
echo "  公司规章制度文本生成训练"
echo "  Model: Qwen3-0.6B + LoRA"
echo "  GPU:   $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null || echo CPU)"
echo "============================================"
echo ""

# Step 1: 数据准备
DATA_SRC="${DATA_SRC:-/home/docs/raw/公司规章制度}"
if [ ! -f ./data/train.txt ]; then
    echo "[1/3] 数据准备..."
    python3 data_prep.py --data_dir "$DATA_SRC" --output_dir ./data
else
    echo "[1/3] 数据已存在, 跳过"
fi
echo ""

# Step 2: 训练
echo "[2/3] 开始训练..."
python3 train_qwen_lora.py \
    --data_dir ./data \
    --output_dir ./output \
    --epochs 5 \
    --batch_size 4 \
    --lr 2e-4 \
    --max_length 512
echo ""

# Step 3: 推理
echo "[3/3] 生成测试..."
python3 generate.py --model_dir ./output/best_model
echo ""
echo "完成! 模型保存在 output/best_model/"
echo "交互模式: python3 generate.py --model_dir ./output/best_model --interactive"

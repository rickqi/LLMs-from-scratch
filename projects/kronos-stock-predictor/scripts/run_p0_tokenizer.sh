#!/bin/bash
# P0 Tokenizer 训练启动脚本
# 用法: bash scripts/run_p0_tokenizer.sh
# 中断恢复: bash scripts/run_p0_tokenizer.sh --resume checkpoint_epoch5

cd "$(dirname "$0")/.."

RESUME=""
if [ "$1" = "--resume" ] && [ -n "$2" ]; then
    RESUME="--resume ./outputs/tokenizer_p0/$2"
    echo "Resuming from: $RESUME"
fi

mkdir -p outputs/logs

python -u train/train_tokenizer.py \
    --data_dir ./data/processed_real \
    --output_dir ./outputs/tokenizer_p0 \
    --model_size mini \
    --epochs 10 \
    --batch_size 16 \
    --lookback 180 \
    $RESUME \
    2>&1 | tee outputs/logs/tokenizer_p0.log

echo "=== Tokenizer training complete ==="

# Auto-launch predictor if tokenizer succeeded
if [ -f ./outputs/tokenizer_p0/best_model.pt ]; then
    echo "Launching Predictor training..."
    bash scripts/run_p0_predictor.sh
fi

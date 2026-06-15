#!/bin/bash
# P0 Predictor 训练启动脚本
# 用法: bash scripts/run_p0_predictor.sh
# 中断恢复: bash scripts/run_p0_predictor.sh --resume checkpoint_epoch5

cd "$(dirname "$0")/.."

RESUME=""
if [ "$1" = "--resume" ] && [ -n "$2" ]; then
    RESUME="--resume ./outputs/predictor_p0/$2"
    echo "Resuming from: $RESUME"
fi

mkdir -p outputs/logs

python -u train/train_predictor.py \
    --tokenizer_path ./outputs/tokenizer_p0/best_model.pt \
    --data_dir ./data/processed_real \
    --output_dir ./outputs/predictor_p0 \
    --model_size mini \
    --epochs 30 \
    --batch_size 16 \
    --lookback 180 \
    $RESUME \
    2>&1 | tee outputs/logs/predictor_p0.log

echo "=== Predictor training complete ==="

# Auto-run backtest
if [ -f ./outputs/predictor_p0/best_model.pt ]; then
    echo "Running enhanced backtest..."
    python -u inference/backtest_enhanced.py \
        --tokenizer_path ./outputs/tokenizer_p0/best_model.pt \
        --predictor_path ./outputs/predictor_p0/best_model.pt \
        --data_dir ./data/processed_real \
        --model_size mini \
        --pred_len 5 --top_k 30 \
        2>&1 | tee outputs/logs/backtest_p0.log
fi

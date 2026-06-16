#!/bin/bash
# P0/P1 Predictor 训练 + 回测启动脚本
# 用法: bash scripts/run_p0_predictor.sh [--model_size small] [--resume checkpoint_epoch5]

cd "$(dirname "$0")/.."

MODEL_SIZE="mini"
RESUME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --resume) RESUME="--resume ./outputs/predictor_${MODEL_SIZE}/$2"; shift 2 ;;
        *) shift ;;
    esac
done

TOK_DIR="./outputs/tokenizer_${MODEL_SIZE}"
PRED_DIR="./outputs/predictor_${MODEL_SIZE}"
LOG_DIR="./outputs/logs"

mkdir -p "$LOG_DIR"

echo "=== P1 Predictor: model=$MODEL_SIZE ==="

python -u train/train_predictor.py \
    --tokenizer_path "$TOK_DIR/best_model.pt" \
    --data_dir ./data/processed_real \
    --output_dir "$PRED_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs 30 \
    --batch_size 8 \
    --lookback 180 \
    $RESUME \
    2>&1 | tee "$LOG_DIR/predictor_${MODEL_SIZE}.log"

echo "=== Predictor ($MODEL_SIZE) complete ==="

if [ -f "$PRED_DIR/best_model.pt" ]; then
    echo "Running enhanced backtest..."
    python -u inference/backtest_enhanced.py \
        --tokenizer_path "$TOK_DIR/best_model.pt" \
        --predictor_path "$PRED_DIR/best_model.pt" \
        --data_dir ./data/processed_real \
        --model_size "$MODEL_SIZE" \
        --pred_len 5 --top_k 30 \
        2>&1 | tee "$LOG_DIR/backtest_${MODEL_SIZE}.log"
fi

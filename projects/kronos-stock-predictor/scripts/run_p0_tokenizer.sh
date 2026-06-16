#!/bin/bash
# P0/P1 Tokenizer 训练启动脚本
# 用法: bash scripts/run_p0_tokenizer.sh [--model_size small] [--resume checkpoint_epoch5]

cd "$(dirname "$0")/.."

MODEL_SIZE="mini"
RESUME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --resume) RESUME="--resume ./outputs/tokenizer_${MODEL_SIZE}/$2"; shift 2 ;;
        *) shift ;;
    esac
done

TOK_DIR="./outputs/tokenizer_${MODEL_SIZE}"
LOG_DIR="./outputs/logs"
mkdir -p "$LOG_DIR" "$TOK_DIR"

echo "=== P1 Tokenizer: model=$MODEL_SIZE ==="

python -u train/train_tokenizer.py \
    --data_dir ./data/processed_real \
    --output_dir "$TOK_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs 10 \
    --batch_size 16 \
    --lookback 180 \
    $RESUME \
    2>&1 | tee "$LOG_DIR/tokenizer_${MODEL_SIZE}.log"

echo "=== Tokenizer ($MODEL_SIZE) complete ==="

if [ -f "$TOK_DIR/best_model.pt" ]; then
    echo "Launching Predictor ($MODEL_SIZE)..."
    bash scripts/run_p0_predictor.sh --model_size "$MODEL_SIZE"
fi

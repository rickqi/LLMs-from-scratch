#!/bin/bash
# ============================================================
# Kronos Stock Predictor - 一键执行脚本
# ============================================================
# 用法:
#   bash run.sh                    # 完整流程 (mini 模型, 样本数据)
#   bash run.sh --model small      # 使用 small 模型
#   bash run.sh --skip-download    # 跳过数据下载
# ============================================================

set -euo pipefail

MODEL_SIZE="mini"
SKIP_DOWNLOAD=false
DATA_DIR="./data/processed"
OUTPUT_DIR="./outputs"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  Kronos Stock Predictor"
echo "  Model: $MODEL_SIZE"
echo "============================================"

# ── 环境检查 ──
echo ""
echo "[1/5] Environment check..."

PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

$PYTHON_CMD -c "import torch; print(f'  PyTorch: {torch.__version__}')" || {
    echo "  ERROR: PyTorch not found. Install: pip install torch"
    exit 1
}

$PYTHON_CMD -c "import pandas; import numpy; print(f'  pandas: {pandas.__version__}, numpy: {numpy.__version__}')" || {
    echo "  ERROR: pandas/numpy not found."
    exit 1
}

# ── 数据准备 ──
echo ""
echo "[2/5] Data preparation..."

mkdir -p $DATA_DIR

if [ "$SKIP_DOWNLOAD" = false ]; then
    # 检查 Tushare token
    if [ -z "${TUSHARE_TOKEN:-}" ]; then
        echo "  WARNING: TUSHARE_TOKEN not set. Trying akshare fallback..."
        echo "  Set with: export TUSHARE_TOKEN=<your_token>"
    fi

    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from data.downloader import StockDataDownloader
from data.preprocessor import build_dataset
from config.default_config import Config
from data.symbols import get_default_symbols

config = Config()
symbols = get_default_symbols()

downloader = StockDataDownloader(config)

# Try download few symbols as sample
sample_symbols = symbols[:5]
print(f'  Downloading {len(sample_symbols)} sample symbols...')

try:
    data = downloader.download_batch(sample_symbols, '2023-01-01', '2024-12-31')
    if data:
        build_dataset(data, config, '$DATA_DIR')
        print(f'  Dataset built: $DATA_DIR')
    else:
        print('  WARNING: No data downloaded. Using synthetic sample for demo.')
except Exception as e:
    print(f'  WARNING: Data download failed: {e}')

# Generate synthetic sample if no real data
import os
if not os.path.exists(f'$DATA_DIR/train_data.pkl'):
    print('  Generating synthetic sample data...')
    from data.preprocessor import generate_sample_data
    generate_sample_data(config, '$DATA_DIR', n_symbols=3, n_days=500)
" || echo "  Data preparation skipped (will use existing data or generate sample)"
else
    echo "  Skipping data download (--skip-download)"
fi

# ── Tokenizer 训练 ──
echo ""
echo "[3/5] Training Tokenizer..."

$PYTHON_CMD train/train_tokenizer.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/tokenizer \
    --model_size $MODEL_SIZE \
    --epochs 5 \
    --batch_size 32 || {
    echo "  WARNING: Tokenizer training failed, skipping to predictor."
}

# ── Predictor 训练 ──
echo ""
echo "[4/5] Training Predictor..."

TOKENIZER_PATH="$OUTPUT_DIR/tokenizer/best_model.pt"
if [ -f "$TOKENIZER_PATH" ]; then
    $PYTHON_CMD train/train_predictor.py \
        --tokenizer_path $TOKENIZER_PATH \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR/predictor \
        --model_size $MODEL_SIZE \
        --epochs 5 \
        --batch_size 32 || {
        echo "  WARNING: Predictor training failed."
    }
else
    echo "  Skipping: Tokenizer not found at $TOKENIZER_PATH"
fi

# ── 推理测试 ──
echo ""
echo "[5/5] Running inference test..."

PREDICTOR_PATH="$OUTPUT_DIR/predictor/best_model.pt"
if [ -f "$PREDICTOR_PATH" ] && [ -f "$TOKENIZER_PATH" ]; then
    $PYTHON_CMD inference/generate.py \
        --mode predict \
        --tokenizer_path $TOKENIZER_PATH \
        --predictor_path $PREDICTOR_PATH \
        --model_size $MODEL_SIZE \
        --data /dev/null 2>/dev/null || echo "  (Inference test requires real data)"
else
    echo "  Skipping: Models not found."
fi

echo ""
echo "============================================"
echo "  Done! Outputs: $OUTPUT_DIR"
echo "============================================"

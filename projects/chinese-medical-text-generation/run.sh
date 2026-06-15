#!/bin/bash
# ================================================================
#  中文医学诊疗指南文本生成 — 一键执行脚本
# ================================================================
#  用法:
#    bash run.sh                    # 使用样本数据, 全流程
#    bash run.sh --real ./raw_data  # 使用真实数据
#    bash run.sh --steps prepare    # 只执行数据准备
#    bash run.sh --steps train      # 只执行训练
#    bash run.sh --steps infer      # 只执行推理
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
}

check_deps() {
    echo "检查 Python 环境..."
    python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || {
        echo -e "${RED}未安装 PyTorch, 请运行: pip install -r requirements.txt${NC}"; exit 1; }
    for lib in transformers peft datasets; do
        python -c "import $lib" 2>/dev/null || {
            echo -e "${YELLOW}缺少 $lib, 尝试安装...${NC}"; pip install $lib; }
    done
    echo -e "${GREEN}依赖检查完成${NC}"
}

DATA_DIR="./data"
OUTPUT_DIR="./output"
EPOCHS=5

# -------------------------------------------------------------------
#  步骤 1: 数据准备
# -------------------------------------------------------------------
step_prepare() {
    banner "步骤 1/3: 数据准备"

    if [ -f "$DATA_DIR/train.txt" ] && [ -f "$DATA_DIR/val.txt" ]; then
        echo -e "${YELLOW}数据已存在, 跳过数据准备${NC}"
        echo "  train.txt: $(wc -c < "$DATA_DIR/train.txt") 字节"
        echo "  val.txt:   $(wc -c < "$DATA_DIR/val.txt") 字节"
        return
    fi

    if [ "${USE_REAL_DATA:-0}" = "1" ] && [ -n "${REAL_DATA_DIR:-}" ]; then
        echo "使用真实数据: $REAL_DATA_DIR"
        python data_prep.py --data_dir "$REAL_DATA_DIR" --output_dir "$DATA_DIR"
    else
        echo -e "${YELLOW}未提供真实数据, 使用自动生成的样本数据 (100篇)${NC}"
        python data_prep.py --sample --num_samples 100 --output_dir "$DATA_DIR"
    fi
}

# -------------------------------------------------------------------
#  步骤 2: 训练
# -------------------------------------------------------------------
step_train() {
    banner "步骤 2/3: Qwen3-0.6B + LoRA 微调"

    if [ ! -f "$DATA_DIR/train.txt" ]; then
        echo -e "${RED}训练数据不存在, 请先运行数据准备步骤: bash run.sh --steps prepare${NC}"
        exit 1
    fi

    echo "训练配置:"
    echo "  模型:        Qwen/Qwen3-0.6B"
    echo "  LoRA rank:   8"
    echo "  Epochs:      $EPOCHS"
    echo "  Batch size:  4"
    echo "  Max length:  512"
    echo ""

    python train_qwen_lora.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS"
}

# -------------------------------------------------------------------
#  步骤 3: 推理评估
# -------------------------------------------------------------------
step_infer() {
    banner "步骤 3/3: 推理评估"

    MODEL_DIR="$OUTPUT_DIR/best_model"
    if [ ! -d "$MODEL_DIR" ]; then
        echo -e "${RED}模型不存在 ($MODEL_DIR), 请先运行训练步骤${NC}"
        exit 1
    fi

    echo "加载模型: $MODEL_DIR"
    python generate.py --model_dir "$MODEL_DIR"
}

# ===================================================================
#  入口
# ===================================================================

STEPS="all"
USE_REAL_DATA=0
REAL_DATA_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            STEPS="$2"; shift 2 ;;
        --real)
            USE_REAL_DATA=1; REAL_DATA_DIR="$2"; shift 2 ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --help)
            echo "用法: bash run.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --steps prepare|train|infer   只执行指定步骤 (默认: all)"
            echo "  --real <dir>                  使用真实数据目录"
            echo "  --epochs <n>                  训练轮数 (默认: 5)"
            echo "  --help                        显示帮助"
            exit 0 ;;
        *) shift ;;
    esac
done

check_deps

case "$STEPS" in
    all)
        step_prepare
        step_train
        step_infer
        ;;
    prepare)
        step_prepare ;;
    train)
        step_train ;;
    infer)
        step_infer ;;
    *)
        echo -e "${RED}未知步骤: $STEPS${NC}"; exit 1 ;;
esac

banner "完成! 检查产出:"
echo "  data/train.txt, data/val.txt     — 数据文件"
echo "  output/best_model/               — 最佳 LoRA 权重"
echo "  output/training_log.json         — 训练日志"

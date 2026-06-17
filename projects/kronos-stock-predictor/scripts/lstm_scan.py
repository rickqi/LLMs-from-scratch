"""
LSTM 超参扫描 + A/B 对比 Kronos

扫描 lookback × pred_len 组合，对比 LSTM vs Kronos mini。
"""

import pickle, logging, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from model.lstm_model import LSTMModel, train_lstm, evaluate_lstm, FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "./data/processed_real"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load data
data = {}
for mode in ["train", "val", "test"]:
    with open(f"{DATA_DIR}/{mode}_data.pkl", "rb") as f:
        data[mode] = pickle.load(f)
logger.info(f"Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")

# ── Kronos baseline (from previous experiments) ──
KRONOS_RESULTS = {
    "model": "Kronos mini (BSQ+Transformer)",
    "rankic": -0.016, "direction_acc": 0.567,
    "params": "0.5M", "train_time": "2h",
}

# ── LSTM scan ──
configs = [
    {"lookback": 90, "pred_len": 3},
    {"lookback": 180, "pred_len": 3},
    {"lookback": 180, "pred_len": 5},
    {"lookback": 180, "pred_len": 10},
    {"lookback": 360, "pred_len": 3},
]

lstm_results = []
for cfg in configs:
    lb, pl = cfg["lookback"], cfg["pred_len"]
    logger.info(f"\n=== LSTM lookback={lb}, pred_len={pl} ===")

    t0 = time.time()
    model = train_lstm(data["train"], data["val"],
                       lookback=lb, pred_len=pl,
                       epochs=15, batch_size=64, lr=1e-3, device=DEVICE)
    train_time = time.time() - t0

    metrics = evaluate_lstm(model, data["test"],
                            lookback=lb, pred_len=pl,
                            device=DEVICE, n_stocks=30)
    metrics["lookback"] = lb
    metrics["pred_len"] = pl
    metrics["train_time"] = f"{train_time:.0f}s"
    lstm_results.append(metrics)

    logger.info(f"  RankIC={metrics['rankic']:.4f}, DirAcc={metrics['direction_accuracy']:.2%}, Time={train_time:.0f}s")

# ── Summary ──
print("\n" + "=" * 70)
print("A/B Comparison: LSTM vs Kronos (BSQ+Transformer)")
print("=" * 70)
print(f"{'Model':<30} {'RankIC':>8} {'DirAcc':>8} {'Params':>8} {'Time':>8}")
print("-" * 70)
print(f"{KRONOS_RESULTS['model']:<30} {KRONOS_RESULTS['rankic']:>8.4f} {KRONOS_RESULTS['direction_acc']:>7.1%} {KRONOS_RESULTS['params']:>8} {KRONOS_RESULTS['train_time']:>8}")
for r in lstm_results:
    name = f"LSTM (lb={r['lookback']}, pl={r['pred_len']})"
    print(f"{name:<30} {r['rankic']:>8.4f} {r['direction_accuracy']:>7.1%} {'100K':>8} {r['train_time']:>8}")

# Best LSTM
best = max(lstm_results, key=lambda r: r["rankic"])
print(f"\nBest LSTM: lookback={best['lookback']}, pred_len={best['pred_len']}, RankIC={best['rankic']:.4f}")
print(f"vs Kronos: ΔRankIC = {best['rankic'] - KRONOS_RESULTS['rankic']:+.4f}")

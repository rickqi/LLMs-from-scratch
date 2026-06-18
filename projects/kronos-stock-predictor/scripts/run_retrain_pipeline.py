#!/usr/bin/env python3
"""
后台训练管线 — 183只半导体 LSTM 方向+波动率+信号回测
CPU 训练 (GPU 被医学 LoRA 占用)

用法:
  nohup python -u scripts/run_retrain_pipeline.py > logs/retrain.log 2>&1 &
"""

import pickle, torch, numpy as np, logging, sys, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, train_lstm, evaluate_lstm, FEATURES
from scripts.optimize_signal import backtest_params as run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", stream=sys.stdout)
logging.getLogger("model.lstm_model").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda:0"  # GPU has 4GB free, LSTM only needs 30MB
DATA_V2 = Path("data/semiconductor_v2/processed")
OUTPUT_DIR = Path("outputs/v2")
OUTPUT_DIR.mkdir(exist_ok=True)

# 加载 + 列名适配
logger.info("Loading semiconductor_v2 data...")
train_data = pickle.load(open(DATA_V2 / "train_data.pkl", "rb"))
val_data = pickle.load(open(DATA_V2 / "val_data.pkl", "rb"))
test_data = pickle.load(open(DATA_V2 / "test_data.pkl", "rb"))
for data in [train_data, val_data, test_data]:
    for df in data.values():
        if "amount" in df.columns and "amt" not in df.columns:
            df["amt"] = df["amount"]
logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
logger.info(f"Device: {DEVICE} | Total train rows: {sum(len(df) for df in train_data.values()):,}")

# ── Phase 1: Direction Model ──
logger.info("\n═══ Phase 1: Direction Model ═══")
t0 = time.time()
dir_model = train_lstm(train_data, val_data, lookback=180, pred_len=10,
                       epochs=5, batch_size=64, lr=1e-3, device=DEVICE)
t1 = time.time()
torch.save(dir_model.state_dict(), OUTPUT_DIR / "direction_model.pt")
dir_metrics = evaluate_lstm(dir_model, test_data, lookback=180, pred_len=10, n_stocks=50, device=DEVICE)
logger.info(f"Direction: RankIC={dir_metrics['rankic']:+.4f} Acc={dir_metrics['direction_accuracy']:.3f} Time={t1-t0:.0f}s")

# ── Phase 2: Volatility Model ──
logger.info("\n═══ Phase 2: Volatility Model ═══")
t0 = time.time()
vol_model = train_lstm(train_data, val_data, lookback=180, pred_len=3,
                       epochs=5, batch_size=64, lr=1e-3, device=DEVICE)
t1 = time.time()
torch.save(vol_model.state_dict(), OUTPUT_DIR / "volatility_model.pt")
vol_metrics = evaluate_lstm(vol_model, test_data, lookback=180, pred_len=3, n_stocks=50, device=DEVICE)
logger.info(f"Volatility: RankIC={vol_metrics['rankic']:+.4f} Time={t1-t0:.0f}s")

# ── Phase 3: Signal Backtest ──
logger.info(f"\n═══ Phase 3: Signal Backtest ({len(test_data)} stocks, ~370 days) ═══")
results = []
for tk in [5, 8, 10, 15, 20]:
    for rb in [10, 15, 20]:
        r = run_backtest(test_data, vol_model, tk, rb, lookback=180)
        if r:
            results.append({"top_k": tk, "rebalance": rb, **r})
            logger.info(f"  top_k={tk:>2} reb={rb:>2}: Sharpe={r['sharpe']:+.2f} Ret={r['return']:+.1f}% N={r['n_reb']}")

results.sort(key=lambda x: -x["sharpe"])

# ── Report ──
report = {
    "timestamp": datetime.now().isoformat(),
    "device": DEVICE,
    "data": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
    "direction": dir_metrics,
    "volatility": vol_metrics,
    "signal_best": results[0] if results else {},
    "signal_all": results,
}
with open(OUTPUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2, default=str, ensure_ascii=False)

logger.info(f"\n═══ Complete ═══")
logger.info(f"Direction RankIC: {dir_metrics['rankic']:+.4f}")
logger.info(f"Volatility RankIC: {vol_metrics['rankic']:+.4f}")
if results:
    logger.info(f"Best signal: top_k={results[0]['top_k']} reb={results[0]['rebalance']} Sharpe={results[0]['sharpe']:+.2f}")
logger.info(f"Saved: {OUTPUT_DIR}/")

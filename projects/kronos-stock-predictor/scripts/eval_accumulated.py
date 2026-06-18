"""
累积信号评估 — 2026H2 回看验证 (D任务)
=====================================
当 2026H2 数据可用时运行, 评估每日信号的实际表现。

用法 (2026年12月或之后):
  python scripts/eval_accumulated.py --from 2026-07-01 --to 2026-12-31
"""

import json, pickle, numpy as np, sys, argparse
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import FEATURES
from scipy.stats import spearmanr

SIGNAL_DIR = Path("outputs/daily_signals")
DATA_DIR = Path("data/semiconductor_v2/processed")


def evaluate_signals(from_date: str, to_date: str, top_k: int = 10):
    """评估累积信号的预测准确性"""
    signal_files = sorted(SIGNAL_DIR.glob("*.json"))
    if not signal_files:
        print("No signal files found. Run track_signal.py first.")
        return

    # Filter by date range
    signals = []
    for f in signal_files:
        date_str = f.stem
        if from_date <= date_str <= to_date:
            try:
                signals.append(json.load(open(f)))
            except:
                pass

    if not signals:
        print(f"No signals in range {from_date} ~ {to_date}")
        return

    print(f"Signals: {len(signals)} days in {from_date} ~ {to_date}")

    # Load actual data for verification
    data = pickle.load(open(DATA_DIR / "test_data.pkl", "rb"))

    # Evaluate: predicted low-vol picks vs actual volatility
    preds, actuals = [], []
    for sig in signals:
        picks = {p["symbol"]: p["predicted_volatility"] for p in sig["picks"][:top_k]}
        for sym, pred_vol in picks.items():
            if sym not in data:
                continue
            df = data[sym]
            vals = df[FEATURES].values.astype(np.float32)
            if len(vals) < 10:
                continue
            rets = np.diff(vals[:, 3]) / (vals[:-1, 3] + 1e-5)
            actual_vol = float(np.std(rets))
            preds.append(pred_vol)
            actuals.append(actual_vol)

    if len(preds) >= 10:
        ic, p = spearmanr(preds, actuals)
        print(f"\n=== Evaluation ===")
        print(f"  RankIC:    {ic:+.4f} (p={p:.4f})")
        print(f"  Samples:   {len(preds)}")
        print(f"  Pred mean: {np.mean(preds):.4f}")
        print(f"  Actual:    {np.mean(actuals):.4f}")
        quality = "GOOD" if ic > 0.05 else ("FAIR" if ic > -0.05 else "POOR")
        print(f"  Quality:   {quality}")
    else:
        print("Insufficient samples for evaluation")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate accumulated daily signals")
    p.add_argument("--from", dest="from_date", default="2026-07-01")
    p.add_argument("--to", dest="to_date", default="2026-12-31")
    args = p.parse_args()
    evaluate_signals(args.from_date, args.to_date)

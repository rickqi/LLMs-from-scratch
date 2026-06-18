"""
轻仓信号跟踪 — 每日预测波动率排名并记录
==========================================
不实际交易, 仅记录预测结果供后续评估 (D任务)
输出: outputs/daily_signals/{date}.json

用法:
  python scripts/track_signal.py           # 单次运行
  python scripts/track_signal.py --cron    # crontab 模式 (静默)
"""

import pickle, torch, numpy as np, sys, json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, FEATURES

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT = Path("outputs/daily_signals")
OUTPUT.mkdir(exist_ok=True)
TODAY = datetime.now().strftime("%Y-%m-%d")


def load_model():
    m = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
    m.load_state_dict(torch.load("outputs/v2/volatility_model.pt", map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def main(verbose=True):
    # Load latest data
    data_path = Path("data/semiconductor_v2/processed/test_data.pkl")
    if not data_path.exists():
        if verbose: print("No test data available")
        return

    data = pickle.load(open(data_path, "rb"))
    model = load_model()

    # Predict volatility for all stocks
    rankings = []
    for sym, df in data.items():
        vals = df[FEATURES].values.astype(np.float32)
        if len(vals) < 180:
            continue
        x = vals[-180:]
        m, s = x.mean(axis=0), x.std(axis=0) + 1e-5
        x_norm = torch.FloatTensor((x - m) / s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_vol = float(model(x_norm).item())
        rankings.append({
            "symbol": sym,
            "predicted_volatility": round(pred_vol, 6),
            "last_close": round(float(df.iloc[-1]["close"]), 2),
            "last_date": str(df.index[-1].date()),
        })

    rankings.sort(key=lambda x: x["predicted_volatility"])

    # Top-K low volatility picks
    top_k = 10
    signal = {
        "date": TODAY,
        "model": "v2_volatility",
        "top_k": top_k,
        "total_stocks": len(rankings),
        "picks": rankings[:top_k],
        "all_rankings": rankings[:50],  # top 50 for tracking
    }

    output_file = OUTPUT / f"{TODAY}.json"
    with open(output_file, "w") as f:
        json.dump(signal, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print(f"Signal saved: {output_file}")
        print(f"Top {top_k} low-volatility picks:")
        for i, r in enumerate(rankings[:top_k]):
            print(f"  {i+1}. {r['symbol']} vol={r['predicted_volatility']:.4f} close={r['last_close']}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cron", action="store_true")
    args = p.parse_args()
    main(verbose=not args.cron)

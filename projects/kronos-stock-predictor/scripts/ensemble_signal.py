"""
双信号集成 — 波动率(低) × 方向(涨) → 增强交易信号

信号强度 = (1 - normalized_vol) × direction_score
"""

import pickle, torch, numpy as np, pandas as pd, logging, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, FEATURES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load direction model
dir_model = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
dir_model.load_state_dict(torch.load("outputs/production_model.pt", map_location=DEVICE))
dir_model.eval()

# Load volatility model
vol_model = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
vol_model.load_state_dict(torch.load("outputs/production_vol_model.pt", map_location=DEVICE))
vol_model.eval()

# Load data
data = {m: pickle.load(open(f"data/processed_real/{m}_data.pkl", "rb")) for m in ["test"]}

signals = []
for sym in sorted(data["test"].keys()):
    df = data["test"][sym]
    vals = df[FEATURES].values.astype(np.float32)
    if len(vals) < 180:
        continue
    x = vals[-180:]
    m, s = x.mean(axis=0), x.std(axis=0) + 1e-5
    x_norm = (x - m) / s
    x_t = torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        dir_pred = dir_model(x_t).item()
        vol_pred = vol_model(x_t).item()

    # Dual signal: normalize vol to 0-1 range, multiply with direction
    vol_norm = min(vol_pred / 0.1, 1.0)  # cap at 10% vol
    signal = dir_pred * (1 - vol_norm)  # high vol penalizes signal

    signals.append(
        {
            "symbol": sym,
            "direction": dir_pred,
            "volatility": vol_pred,
            "composite_signal": signal,
            "action": "BUY" if signal > 0.01 else ("SELL" if signal < -0.01 else "HOLD"),
            "last_close": round(float(vals[-1, 3]), 2),
        }
    )

signals.sort(key=lambda x: x["composite_signal"], reverse=True)
df_s = pd.DataFrame(signals)

buy = df_s[df_s["action"] == "BUY"]
sell = df_s[df_s["action"] == "SELL"]
hold = df_s[df_s["action"] == "HOLD"]

logger.info(f"Dual-Signal Integration Results:")
logger.info(f"  BUY:  {len(buy)} stocks")
logger.info(f"  SELL: {len(sell)} stocks")
logger.info(f"  HOLD: {len(hold)} stocks")
logger.info(f"\nTop 5 BUY signals:")
for _, r in buy.head(5).iterrows():
    logger.info(f"  {r.symbol}: sig={r.composite_signal:.4f} (dir={r.direction:+.3f}, vol={r.volatility:.4f})")
logger.info(f"\nTop 5 SELL signals:")
for _, r in sell.head(5).iterrows():
    logger.info(f"  {r.symbol}: sig={r.composite_signal:.4f} (dir={r.direction:+.3f}, vol={r.volatility:.4f})")

df_s.to_json("outputs/dual_signal_forecast.json", orient="records", indent=2)
logger.info("Saved: outputs/dual_signal_forecast.json")

"""
波动率回测验证 — 滚动窗口验证 RankIC=+0.569 在时间序列上的有效性
"""

import pickle, torch, numpy as np, pandas as pd, logging, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy.stats import spearmanr
from model.lstm_model import LSTMModel, FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model
model = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
model.load_state_dict(torch.load("outputs/production_vol_model.pt", map_location=DEVICE))
model.eval()

# Load data
data = {m: pickle.load(open(f"data/processed_real/{m}_data.pkl", "rb")) for m in ["test"]}

# Rolling backtest: predict vol every 20 trading days
results = []
for sym in sorted(data["test"].keys()):
    df = data["test"][sym]
    vals = df[FEATURES].values.astype(np.float32)
    step = 20
    for idx in range(180, len(vals) - 10, step):
        x = vals[idx - 180 : idx]
        m, s = x.mean(axis=0), x.std(axis=0) + 1e-5
        x_norm = (x - m) / s
        with torch.no_grad():
            pred = model(torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)).item()
        rets = np.diff(vals[idx : idx + 10, 3]) / (vals[idx : idx + 9, 3] + 1e-5)
        actual = np.std(rets)
        results.append(
            {
                "symbol": sym,
                "date": str(df.index[idx].date()),
                "predicted_vol": round(pred, 6),
                "actual_vol": round(float(actual), 6),
            }
        )

preds = np.array([r["predicted_vol"] for r in results])
actuals = np.array([r["actual_vol"] for r in results])
ic, p = spearmanr(preds, actuals)

df_r = pd.DataFrame(results)
df_r["error"] = df_r["predicted_vol"] - df_r["actual_vol"]
mae = df_r["error"].abs().mean()

logger.info(f"Volatility Backtest Results:")
logger.info(f"  RankIC:     {ic:.4f} (p={p:.4f})")
logger.info(f"  MAE:        {mae:.6f}")
logger.info(f"  Samples:    {len(results)}")
logger.info(f"  Pred mean:  {preds.mean():.5f}")
logger.info(f"  Actual mean:{actuals.mean():.5f}")
logger.info(f"  Corr:       {np.corrcoef(preds, actuals)[0,1]:.4f}")

# By-stock breakdown
stock_ics = {}
for sym in df_r["symbol"].unique():
    sdf = df_r[df_r["symbol"] == sym]
    if len(sdf) >= 5:
        ic_s, _ = spearmanr(sdf["predicted_vol"], sdf["actual_vol"])
        stock_ics[sym] = ic_s

stock_ics_sorted = sorted(stock_ics.items(), key=lambda x: -x[1])
logger.info(f"\nTop 5 stocks by RankIC:")
for sym, ic_s in stock_ics_sorted[:5]:
    logger.info(f"  {sym}: {ic_s:.4f}")
logger.info(f"Bottom 5:")
for sym, ic_s in stock_ics_sorted[-5:]:
    logger.info(f"  {sym}: {ic_s:.4f}")

df_r.to_json("outputs/volatility_backtest.json", orient="records", indent=2)
logger.info("Saved: outputs/volatility_backtest.json")

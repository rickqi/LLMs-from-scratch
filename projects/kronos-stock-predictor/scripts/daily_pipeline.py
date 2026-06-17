#!/usr/bin/env python3
"""
每日自动化流水线 — Tushare拉最新 → 预测 → 保存 → 输出报告

用法: python scripts/daily_pipeline.py
"""

import os, sys, pickle, json, logging, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch
import tushare as ts
from model.lstm_model import LSTMModel, FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Config ──
TOKEN = os.environ.get("TUSHARE_TOKEN", "260264d1c42c2b5c47262478557e99d7f6a0769523ea19f48e09ed73")
SEMI_SYMBOLS = sorted([f.stem for f in Path("/mnt/d/codes/stock/docs/data/semiconductor/kline_cache").glob("*.csv")])
SYMBOLS = [f"{s}.SH" if int(s) >= 600000 else f"{s}.SZ" for s in SEMI_SYMBOLS[:68]]
OUTPUT_DIR = "outputs/daily"
TODAY = datetime.now().strftime("%Y%m%d")

# ── Step 1: Download latest data ──
logger.info(f"Step 1: Downloading {len(SYMBOLS)} stocks via Tushare...")
ts.set_token(TOKEN)
pro = ts.pro_api()

latest_data = {}
for sym in SYMBOLS:
    try:
        df = pro.daily(ts_code=sym, start_date="20240101", end_date=TODAY)
        time.sleep(0.35)
        if df is not None and len(df) > 0:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date").sort_index()
            df = df[["open", "high", "low", "close", "vol", "amount"]]
            df = df.rename(columns={"vol": "vol", "amount": "amt"})
            df = df[(df["open"] > 0) & (df["close"] > 0)].astype(np.float32)
            if len(df) >= 180:
                latest_data[sym] = df
    except Exception as e:
        logger.debug(f"  {sym}: {e}")

logger.info(f"  Downloaded: {len(latest_data)} stocks")

# ── Step 2: Load models ──
logger.info("Step 2: Loading models...")
dir_model = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
dir_model.load_state_dict(torch.load("outputs/production_model.pt", map_location=DEVICE))
dir_model.eval()

vol_model = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
vol_model.load_state_dict(torch.load("outputs/production_vol_model.pt", map_location=DEVICE))
vol_model.eval()

# ── Step 3: Predict ──
logger.info("Step 3: Predicting...")
forecast = []
for sym, df in latest_data.items():
    vals = df[FEATURES].values.astype(np.float32)[-180:]
    m, s = vals.mean(axis=0), vals.std(axis=0) + 1e-5
    x_norm = (vals - m) / s
    x_t = torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = dir_model(x_t).item()
        v = vol_model(x_t).item()
    forecast.append({
        "symbol": sym,
        "date": TODAY,
        "last_close": round(float(vals[-1, 3]), 2),
        "direction": round(d, 6),
        "volatility_10d": round(v, 6),
        "composite": round(d * (1 - min(v / 0.1, 1.0)), 6),
        "action": "BUY" if d > 0.01 and v < 0.05 else ("SELL" if d < -0.01 else "HOLD"),
    })

# ── Step 4: Save ──
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
date_str = datetime.now().strftime("%Y-%m-%d")
out_path = f"{OUTPUT_DIR}/forecast_{date_str}.json"
with open(out_path, "w") as f:
    json.dump({"generated_at": str(datetime.now()), "n_stocks": len(forecast), "predictions": forecast}, f, indent=2)

# Summary
df = pd.DataFrame(forecast)
logger.info(f"\n{'='*50}")
logger.info(f"Daily Forecast: {date_str}")
logger.info(f"{'='*50}")
logger.info(f"Stocks: {len(forecast)}")
logger.info(f"BUY:  {len(df[df.action=='BUY'])}")
logger.info(f"SELL: {len(df[df.action=='SELL'])}")
logger.info(f"HOLD: {len(df[df.action=='HOLD'])}")
logger.info(f"\nTop 5 signals:")
for _, r in df.nlargest(5, "composite").iterrows():
    logger.info(f"  {r.symbol}: {r.composite:+.4f} ({r.action})")
logger.info(f"\nSaved: {out_path}")

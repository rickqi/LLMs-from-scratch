"""
Tushare CSI300 全量数据下载 + Kronos 格式转换

用法: python scripts/download_tushare.py
前提: export TUSHARE_TOKEN=xxx (或自动从 TradingAgents/.env 读取)
"""

import os, sys, pickle, time, logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.default_config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Token
TOKEN = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_API_KEY")
if not TOKEN:
    env_path = Path("/mnt/d/codes/stock/TradingAgents/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("TUSHARE_API_KEY="):
                TOKEN = line.split("=", 1)[1].strip().strip('"')
if not TOKEN:
    raise RuntimeError("TUSHARE_TOKEN not found")

import tushare as ts
ts.set_token(TOKEN)
pro = ts.pro_api()

OUTPUT_DIR = "./data/processed_tushare"
CSI300_PATH = "/mnt/d/codes/stock/TradingAgents/hs300_tickers.txt"
RATE_LIMIT = 0.35  # seconds between API calls (180/min max)


def get_csi300_symbols() -> list[str]:
    with open(CSI300_PATH) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def download_stock(ts_code: str, start: str = "20150101", end: str = "20260616") -> pd.DataFrame | None:
    """下载单只股票日线"""
    for attempt in range(3):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
            if df is not None and len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.set_index("trade_date").sort_index()
                df = df.rename(columns={"vol": "volume", "amount": "amount"})
                keep = {"open", "high", "low", "close", "volume", "amount"}
                df = df[[c for c in keep if c in df.columns]]
                df = df.rename(columns={"volume": "vol", "amount": "amt"})
                df = df[(df["open"] > 0) & (df["close"] > 0)].astype(np.float32)
                if len(df) >= 100:
                    return df
            time.sleep(RATE_LIMIT)
            return None
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
    return None


def build_dataset(config: Config):
    symbols = get_csi300_symbols()
    logger.info(f"Downloading {len(symbols)} CSI300 stocks via Tushare...")

    train_s = pd.Timestamp(config.train_time_range[0])
    train_e = pd.Timestamp(config.train_time_range[1])
    val_s = pd.Timestamp(config.val_time_range[0])
    val_e = pd.Timestamp(config.val_time_range[1])
    test_s = pd.Timestamp(config.test_time_range[0])
    test_e = pd.Timestamp(config.test_time_range[1])

    train_data, val_data, test_data = {}, {}, {}
    ok, fail = 0, 0

    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(symbols)} (ok={ok}, fail={fail})")

        df = download_stock(sym)
        time.sleep(RATE_LIMIT)

        if df is None:
            fail += 1
            continue
        ok += 1

        t = df[(df.index >= train_s) & (df.index <= train_e)]
        v = df[(df.index >= val_s) & (df.index <= val_e)]
        ts_data = df[(df.index >= test_s) & (df.index <= test_e)]

        if len(t) >= 100: train_data[sym] = t
        if len(v) >= 30: val_data[sym] = v
        if len(ts_data) >= 30: test_data[sym] = ts_data

    logger.info(f"Downloaded: {ok} ok, {fail} failed")

    # Merge existing semiconductor data
    semi_dir = "./data/processed_real"
    for mode, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = f"{semi_dir}/{mode}_data.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                existing = pickle.load(f)
            for sym, df in existing.items():
                if sym not in data:
                    data[sym] = df
            logger.info(f"  +{len(existing)} semiconductor stocks → {mode}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = f"{OUTPUT_DIR}/{name}_data.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        rows = sum(len(v) for v in data.values())
        logger.info(f"  {name}: {len(data)} stocks, {rows} rows")

    logger.info(f"Saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    config = Config()
    build_dataset(config)

"""
CSI300 数据下载 + Kronos 格式转换脚本

1. 从 TradingAgents hs300_tickers.txt 读取成分股
2. 通过 akshare 下载日线数据
3. 转换为 Kronos pickle 格式
4. 与现有半导体数据合并

用法: python scripts/download_csi300.py
"""

import os, sys, pickle, time, logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.default_config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# 读取 CSI300 成分股
HS300_PATH = "/mnt/d/codes/stock/TradingAgents/hs300_tickers.txt"
OUTPUT_DIR = "./data/processed_csi300"
EXISTING_DIR = "./data/processed_real"


def load_csi300_symbols() -> list[str]:
    with open(HS300_PATH) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def download_single(symbol: str) -> pd.DataFrame | None:
    """通过 akshare 下载单只股票日线（带重试）"""
    import akshare as ak
    code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date="20150101", end_date="20260616",
                                    adjust="qfq")
            if df is not None and len(df) >= 100:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                    "成交额": "amount",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                keep = ["open", "high", "low", "close", "volume", "amount"]
                df = df[[c for c in keep if c in df.columns]]
                df = df.rename(columns={"volume": "vol", "amount": "amt"})
                df = df[(df["open"] > 0) & (df["close"] > 0)].astype(np.float32)
                if len(df) >= 100:
                    return df
            if attempt < 2:
                time.sleep(0.5)
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return None


def build_combined_dataset(config: Config):
    """构建 CSI300 + 半导体 合并数据集"""
    symbols = load_csi300_symbols()
    logger.info(f"CSI300 symbols: {len(symbols)}")

    train_s = pd.Timestamp(config.train_time_range[0])
    train_e = pd.Timestamp(config.train_time_range[1])
    val_s = pd.Timestamp(config.val_time_range[0])
    val_e = pd.Timestamp(config.val_time_range[1])
    test_s = pd.Timestamp(config.test_time_range[0])
    test_e = pd.Timestamp(config.test_time_range[1])

    train_data, val_data, test_data = {}, {}, {}
    success, fail = 0, 0

    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(symbols)} (ok={success}, fail={fail})")

        df = download_single(sym)
        if df is None:
            fail += 1
            continue
        success += 1

        t = df[(df.index >= train_s) & (df.index <= train_e)]
        v = df[(df.index >= val_s) & (df.index <= val_e)]
        ts = df[(df.index >= test_s) & (df.index <= test_e)]

        if len(t) >= 100:
            train_data[sym] = t
        if len(v) >= 30:
            val_data[sym] = v
        if len(ts) >= 30:
            test_data[sym] = ts

    logger.info(f"Download complete: {success} ok, {fail} failed")

    # 合并现有半导体数据
    for mode, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        existing_path = f"{EXISTING_DIR}/{mode}_data.pkl"
        if os.path.exists(existing_path):
            with open(existing_path, "rb") as f:
                existing = pickle.load(f)
            for sym, df in existing.items():
                if sym not in data:
                    data[sym] = df
            logger.info(f"  Merged {len(existing)} semiconductor stocks into {mode}")

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = f"{OUTPUT_DIR}/{name}_data.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        rows = sum(len(v) for v in data.values())
        logger.info(f"  {name}_data.pkl: {len(data)} symbols, {rows} rows")

    logger.info(f"Dataset saved: {OUTPUT_DIR}")
    return train_data


if __name__ == "__main__":
    config = Config()
    build_combined_dataset(config)

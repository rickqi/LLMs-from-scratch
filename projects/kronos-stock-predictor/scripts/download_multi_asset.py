#!/usr/bin/env python3
"""
Multi-Asset Data Downloader for Kronos P2 expansion.

Supports:
  - A-shares (沪深300/500/800) — existing
  - HK stocks (港股通) — via akshare
  - US China ADR (中概股) — via yfinance

Usage:
  python scripts/download_multi_asset.py --market hk     # 港股
  python scripts/download_multi_asset.py --market us     # 美股中概
  python scripts/download_multi_asset.py --market all    # 全部
"""

import os, sys, argparse, logging, pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "multi_asset"


# ── Stock lists ─────────────────────────────────────────────────
HK_STOCKS = [
    "00700.HK", "09988.HK", "00941.HK", "00388.HK", "01299.HK",
    "02318.HK", "02628.HK", "01398.HK", "03988.HK", "00939.HK",
    "01810.HK", "02020.HK", "02269.HK", "00175.HK", "02015.HK",
    "09618.HK", "09888.HK", "03690.HK", "01024.HK", "01818.HK",
    "00005.HK", "00011.HK", "02388.HK", "01109.HK", "00688.HK",
    "02382.HK", "02018.HK", "00992.HK", "00823.HK", "00027.HK",
]

US_CHINA_ADRS = [
    "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI",
    "BILI", "TME", "NTES", "TAL", "EDU", "YUMC", "ZTO",
    "BEKE", "VIPS", "HTHT", "DQ", "ATHM", "GDS",
    "TSM", "UMC", "ASX", "AUO",
]

FEATURE_COLS = ["open", "high", "low", "close", "vol", "amt"]


def download_hk(symbols: list[str], start_date: str = "2020-01-01"):
    """Download HK stock data via akshare"""
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare not installed. Run: pip install akshare")
        return {}

    end_date = datetime.now().strftime("%Y%m%d")
    start_fmt = start_date.replace("-", "")
    
    result = {}
    for i, sym in enumerate(symbols):
        code = sym.replace(".HK", "")
        try:
            df = ak.stock_hk_hist(
                symbol=code, period="daily",
                start_date=start_fmt, end_date=end_date, adjust="qfq"
            )
            if df is not None and len(df) > 100:
                df = df.rename(columns={
                    "开盘": "open", "收盘": "close", "最高": "high",
                    "最低": "low", "成交量": "vol", "成交额": "amt",
                })
                df["date"] = pd.to_datetime(df["日期"])
                df = df.set_index("date")[FEATURE_COLS]
                result[sym] = df
                logger.info(f"  [{i+1}/{len(symbols)}] {sym}: {len(df)} rows")
            else:
                logger.warning(f"  [{i+1}/{len(symbols)}] {sym}: no data")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(symbols)}] {sym}: {e}")
    
    return result


def download_us(symbols: list[str], start_date: str = "2020-01-01"):
    """Download US stock data via yfinance"""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return {}

    result = {}
    for i, sym in enumerate(symbols):
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=start_date)
            if df is not None and len(df) > 100:
                df = df.rename(columns={
                    "Open": "open", "Close": "close", "High": "high",
                    "Low": "low", "Volume": "vol",
                })
                df["amt"] = df["close"] * df["vol"]  # approximate amount
                df = df[FEATURE_COLS]
                result[sym] = df
                logger.info(f"  [{i+1}/{len(symbols)}] {sym}: {len(df)} rows")
            else:
                logger.warning(f"  [{i+1}/{len(symbols)}] {sym}: no data")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(symbols)}] {sym}: {e}")
    
    return result


def process_to_arrays(data: dict) -> dict:
    """Convert DataFrames to numpy arrays for Kronos training"""
    result = {}
    for sym, df in data.items():
        arr = df[FEATURE_COLS].values.astype(np.float32)
        # Remove rows with all zeros
        valid = ~np.all(arr == 0, axis=1)
        result[sym] = arr[valid]
    return result


def split_train_val(data: dict, val_ratio: float = 0.15):
    """Split into train/val/test sets"""
    stocks = list(data.keys())
    np.random.seed(42)
    np.random.shuffle(stocks)
    
    n_val = max(1, int(len(stocks) * val_ratio))
    train_stocks = stocks[:len(stocks)-2*n_val]
    val_stocks = stocks[len(stocks)-2*n_val:len(stocks)-n_val]
    test_stocks = stocks[len(stocks)-n_val:]
    
    def make_split(keys):
        return {k: data[k] for k in keys if k in data}
    
    return make_split(train_stocks), make_split(val_stocks), make_split(test_stocks)


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Data Downloader")
    parser.add_argument("--market", default="all", choices=["hk","us","all","cn"])
    parser.add_argument("--output_dir", default=str(DATA_DIR))
    parser.add_argument("--start_date", default="2022-01-01")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = {}
    
    if args.market in ("hk", "all"):
        logger.info(f"Downloading {len(HK_STOCKS)} HK stocks...")
        hk_data = download_hk(HK_STOCKS, args.start_date)
        hk_arr = process_to_arrays(hk_data)
        all_data.update({f"HK_{k}": v for k, v in hk_arr.items()})
        logger.info(f"HK: {len(hk_arr)} stocks with valid data")
    
    if args.market in ("us", "all"):
        logger.info(f"Downloading {len(US_CHINA_ADRS)} US China ADRs...")
        us_data = download_us(US_CHINA_ADRS, args.start_date)
        us_arr = process_to_arrays(us_data)
        all_data.update({f"US_{k}": v for k, v in us_arr.items()})
        logger.info(f"US: {len(us_arr)} stocks with valid data")
    
    if not all_data:
        logger.warning("No data downloaded. Check network/API availability.")
        logger.info("For offline testing, use existing A-share data.")
        return
    
    train, val, test = split_train_val(all_data)
    
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}_data.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"{name}: {len(data)} stocks → {path}")
    
    logger.info(f"Done. Total: {len(all_data)} stocks across markets")


if __name__ == "__main__":
    main()

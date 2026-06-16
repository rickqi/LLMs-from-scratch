"""
技术指标增强脚本 — 为现有数据集添加 MACD/RSI/布林带/ATR 等技术指标

用法: python scripts/add_indicators.py
"""

import pickle, logging, numpy as np, pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = "./data/processed_csi300"
OUTPUT_DIR = "./data/processed_ta"

TA_FEATURES = [
    "macd", "macd_signal", "macd_hist",  # MACD
    "rsi_14",                              # RSI
    "bb_upper", "bb_middle", "bb_lower",   # Bollinger
    "atr_14",                              # ATR
    "vol_chg_5", "vol_chg_20",            # Volume change
    "ret_1", "ret_5", "ret_20",           # Price returns
]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """为 DataFrame 添加技术指标列"""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df.get("vol", df.get("volume", pd.Series(0, index=df.index)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_middle"] = bb_mid
    df["bb_lower"] = bb_mid - 2 * bb_std

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Volume change
    df["vol_chg_5"] = vol.pct_change(5)
    df["vol_chg_20"] = vol.pct_change(20)

    # Price returns
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)

    return df


def main():
    for mode in ["train", "val", "test"]:
        input_path = f"{INPUT_DIR}/{mode}_data.pkl"
        output_path = f"{OUTPUT_DIR}/{mode}_data.pkl"

        with open(input_path, "rb") as f:
            data = pickle.load(f)

        enhanced = {}
        for sym, df in data.items():
            try:
                edf = add_indicators(df)
                # Drop rows with NaN (from rolling windows)
                edf = edf.dropna()
                if len(edf) >= 100:
                    enhanced[sym] = edf
            except Exception as e:
                logger.warning(f"  {sym}: {e}")

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(enhanced, f)

        rows = sum(len(v) for v in enhanced.values())
        logger.info(f"{mode}: {len(enhanced)} stocks, {rows} rows (was {len(data)} stocks)")

    logger.info(f"Done → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

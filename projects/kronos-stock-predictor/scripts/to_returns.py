"""
转换数据集为收益率格式 — 价格替换为日收益率，更平稳

用法: python scripts/to_returns.py
"""

import pickle, logging, numpy as np, pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = "./data/processed_real"
OUTPUT_DIR = "./data/processed_ret"

RETURN_FEATURES = [
    "open",    # open / prev_close - 1  (保持 OHLCV 列名，值替换为收益率)
    "high",    # high / prev_close - 1
    "low",     # low / prev_close - 1
    "close",   # close / prev_close - 1
    "vol",     # volume / prev_volume - 1
    "amt",     # amount / prev_amount - 1
]


def to_returns(df: pd.DataFrame) -> pd.DataFrame:
    """将 OHLCV 价格转为日收益率（列名不变，值替换）"""
    df = df.copy()
    prev_close = df["close"].shift(1)

    df["open"] = df["open"] / prev_close - 1
    df["high"] = df["high"] / prev_close - 1
    df["low"] = df["low"] / prev_close - 1
    df["close"] = df["close"] / prev_close - 1
    df["vol"] = df["vol"].pct_change()
    if "amt" in df.columns:
        df["amt"] = df["amt"].pct_change()

    df = df[["open", "high", "low", "close", "vol", "amt"]].dropna()
    df = df.clip(-0.2, 0.2)  # A股涨跌停 ±20%
    return df.astype(np.float32)


def main():
    for mode in ["train", "val", "test"]:
        input_path = f"{INPUT_DIR}/{mode}_data.pkl"
        output_path = f"{OUTPUT_DIR}/{mode}_data.pkl"

        with open(input_path, "rb") as f:
            data = pickle.load(f)

        ret_data = {}
        for sym, df in data.items():
            try:
                rdf = to_returns(df)
                rdf = rdf.clip(-0.2, 0.2)  # 限制极端值 (A股涨跌停 ±20%)
                if len(rdf) >= 100:
                    ret_data[sym] = rdf
            except Exception:
                pass

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(ret_data, f)

        rows = sum(len(v) for v in ret_data.values())
        logger.info(f"{mode}: {len(ret_data)} stocks, {rows} rows (was {len(data)})")

    logger.info(f"Done → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

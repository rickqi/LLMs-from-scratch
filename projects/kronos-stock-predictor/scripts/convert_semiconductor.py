"""
半导体 K 线数据转换脚本

将 docs/data/semiconductor/kline_cache/*.csv 转为 Kronos 格式的 pickle 文件。

用法:
    python scripts/convert_semiconductor.py --csv_dir /mnt/d/codes/stock/docs/data/semiconductor/kline_cache --output_dir ./data/processed_real
"""

import os, sys, pickle, argparse, logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.default_config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_csv(filepath: str) -> pd.DataFrame | None:
    """加载单个股票 CSV，返回标准化 DataFrame"""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None

    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"{filepath}: missing columns {missing}")
        return None

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").sort_index()

    # 复权价格处理
    if "adj_factor" in df.columns and df["adj_factor"].notna().any():
        factor = df["adj_factor"].fillna(1.0)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * factor

    # 标准化列名
    df = df.rename(columns={"volume": "vol"})
    keep = ["open", "high", "low", "close", "vol", "amount"]
    df = df[[c for c in keep if c in df.columns]]

    # 过滤异常值: 去掉价格为0的行
    df = df[(df["open"] > 0) & (df["close"] > 0)]

    if len(df) < 100:
        logger.warning(f"{filepath}: only {len(df)} rows, skipping")
        return None

    return df.astype(np.float32)


def build_dataset(csv_dir: str, output_dir: str, config: Config) -> dict:
    """构建训练/验证/测试数据集"""
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {csv_dir}")

    train_start = pd.Timestamp(config.train_time_range[0])
    train_end = pd.Timestamp(config.train_time_range[1])
    val_start = pd.Timestamp(config.val_time_range[0])
    val_end = pd.Timestamp(config.val_time_range[1])
    test_start = pd.Timestamp(config.test_time_range[0])
    test_end = pd.Timestamp(config.test_time_range[1])

    train_data, val_data, test_data = {}, {}, {}
    skipped = 0

    for fp in csv_files:
        symbol = fp.stem  # e.g., "002049"
        # 根据文件名推断交易所
        code = int(symbol)
        if code >= 600000:
            ts_code = f"{symbol}.SH"
        elif code >= 300000 or code >= 200000:
            ts_code = f"{symbol}.SZ"
        elif code >= 800000:
            ts_code = f"{symbol}.BJ"
        else:
            ts_code = f"{symbol}.SZ"

        df = load_csv(str(fp))
        if df is None:
            skipped += 1
            continue

        # 划分
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        val_df = df[(df.index >= val_start) & (df.index <= val_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]

        if len(train_df) >= 100:
            train_data[ts_code] = train_df
        if len(val_df) >= 30:
            val_data[ts_code] = val_df
        if len(test_df) >= 30:
            test_data[ts_code] = test_df

    logger.info(f"Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test symbols ({skipped} skipped)")

    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = f"{output_dir}/{name}_data.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        rows = sum(len(v) for v in data.values())
        logger.info(f"  {name}_data.pkl: {len(data)} symbols, {rows} rows")

    return {"train": train_data, "val": val_data, "test": test_data}


def main():
    parser = argparse.ArgumentParser(description="Convert semiconductor CSV data to Kronos format")
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing stock CSV files")
    parser.add_argument("--output_dir", type=str, default="./data/processed_real", help="Output directory")
    parser.add_argument("--train_start", type=str, default="2018-01-01")
    parser.add_argument("--train_end", type=str, default="2023-12-31")
    parser.add_argument("--val_start", type=str, default="2024-01-01")
    parser.add_argument("--val_end", type=str, default="2024-12-31")
    parser.add_argument("--test_start", type=str, default="2025-01-01")
    parser.add_argument("--test_end", type=str, default="2026-06-15")
    args = parser.parse_args()

    config = Config()
    config.train_time_range = [args.train_start, args.train_end]
    config.val_time_range = [args.val_start, args.val_end]
    config.test_time_range = [args.test_start, args.test_end]

    build_dataset(args.csv_dir, args.output_dir, config)
    logger.info(f"Done! Data saved to {args.output_dir}")


if __name__ == "__main__":
    main()

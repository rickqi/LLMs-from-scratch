"""
生成合成样本数据（用于无真实数据时的演示）

在 data/preprocessor.py 中导入使用
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging

logger = logging.getLogger(__name__)


def generate_sample_data(config, output_dir: str, n_symbols: int = 5, n_days: int = 500):
    """
    生成合成 OHLCV 数据用于演示和测试。

    生成逻辑: 几何布朗运动 + 随机波动率 + 周期扰动
    """
    np.random.seed(config.seed)
    symbols = [f"SYNTH_{i:03d}.SH" for i in range(n_symbols)]

    train_data = {}
    val_data = {}
    test_data = {}

    train_start = pd.Timestamp(config.train_time_range[0])
    train_end = pd.Timestamp(config.train_time_range[1])
    val_start = pd.Timestamp(config.val_time_range[0])
    val_end = pd.Timestamp(config.val_time_range[1])
    test_start = pd.Timestamp(config.test_time_range[0])
    test_end = pd.Timestamp(config.test_time_range[1])

    for symbol in symbols:
        # 基础参数
        initial_price = np.random.uniform(10, 100)
        mu = np.random.uniform(0.05, 0.15) / 252  # 日均收益率
        sigma = np.random.uniform(0.01, 0.03) / np.sqrt(252)  # 日波动率

        # 生成日期序列
        date_range = pd.date_range(start=train_start, end=test_end, freq="B")
        T = len(date_range)

        # 几何布朗运动
        returns = np.random.normal(mu, sigma, T)
        prices = initial_price * np.exp(np.cumsum(returns))

        # 添加周期扰动
        t = np.arange(T)
        seasonal = 0.02 * np.sin(2 * np.pi * t / 252)  # 年度周期
        weekly = 0.005 * np.sin(2 * np.pi * t / 5)  # 周度周期
        prices *= (1 + seasonal + weekly)

        # 生成 OHLCV
        daily_volatility = sigma * prices
        df = pd.DataFrame({
            "open": prices * (1 + np.random.normal(0, 0.005, T)),
            "high": prices + np.abs(np.random.normal(0, daily_volatility * 0.5, T)),
            "low": prices - np.abs(np.random.normal(0, daily_volatility * 0.5, T)),
            "close": prices,
            "vol": np.random.lognormal(mean=10, sigma=0.5, size=T),
            "amt": prices * np.random.lognormal(mean=10, sigma=0.5, size=T),
        }, index=date_range)

        # 确保 high >= close/open 和 low <= close/open
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        # 划分
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        val_mask = (df.index >= val_start) & (df.index <= val_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)

        train_data[symbol] = df[train_mask].copy()
        val_data[symbol] = df[val_mask].copy()
        test_data[symbol] = df[test_mask].copy()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = f"{output_dir}/{name}_data.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        total_rows = sum(len(v) for v in data.values())
        logger.info(f"  {name}_data.pkl: {len(data)} symbols, {total_rows} total rows")

    logger.info(f"Synthetic data generated: {output_dir}")

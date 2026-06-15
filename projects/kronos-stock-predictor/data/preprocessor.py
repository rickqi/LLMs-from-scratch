"""
数据预处理模块 — Z-score 归一化 + 数据集构建

流程:
  downloader 输出 ({symbol: DataFrame})
  → preprocess_series() 逐序列归一化
  → build_dataset() 划分 train/val/test 并保存 pickle
  → load_dataset() 加载已处理数据集
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# 逐序列预处理
# ======================================================================


def preprocess_series(
    df: pd.DataFrame,
    feature_cols: list[str],
    time_cols: list[str],
    lookback: int,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score 归一化单只股票的特征序列。

    使用 lookback 窗口的滚动统计进行归一化，避免未来信息泄露。

    Args:
        df: 包含特征列和时间列的 DataFrame。
        feature_cols: 特征列名列表 (如 ['open','high','low','close','vol','amt'])。
        time_cols: 时间特征列名列表 (如 ['minute','hour','weekday','day','month'])。
        lookback: 滚动窗口长度。
        clip: 裁剪阈值。

    Returns:
        (normalized_array, mean_array, std_array)
        - normalized_array: (N, len(feature_cols)) 归一化后的特征。
        - mean_array: (N, len(feature_cols)) 逐行均值。
        - std_array: (N, len(feature_cols)) 逐行标准差。
    """
    values = df[feature_cols].values.astype(np.float32)  # (N, F)
    n_rows, n_feat = values.shape

    mean_arr = np.zeros_like(values)
    std_arr = np.ones_like(values)
    normed = np.zeros_like(values)

    for i in range(n_rows):
        # ── 窗口范围: [max(0, i-lookback+1), i+1] ──
        start = max(0, i - lookback + 1)
        window = values[start : i + 1]  # (W, F)

        mean_i = window.mean(axis=0)  # (F,)
        std_i = window.std(axis=0)  # (F,)
        std_i = np.where(std_i < 1e-8, 1.0, std_i)

        mean_arr[i] = mean_i
        std_arr[i] = std_i
        normed[i] = (values[i] - mean_i) / std_i

    # ── 裁剪 ──
    normed = np.clip(normed, -clip, clip)

    return normed, mean_arr, std_arr


# ======================================================================
# 数据集构建
# ======================================================================


def build_dataset(
    downloader_output: dict[str, pd.DataFrame],
    config,
    output_dir: Optional[str] = None,
) -> None:
    """从下载器输出构建训练/验证/测试数据集并保存为 pickle。

    Args:
        downloader_output: {symbol: DataFrame} 字典，来自 StockDataDownloader。
        config: Config 对象，包含 feature_list, time_feature_list, lookback_window,
                predict_window, clip, train/val/test_time_range 等。
        output_dir: 输出目录，默认使用 config.dataset_path。
    """
    if output_dir is None:
        output_dir = config.dataset_path

    os.makedirs(output_dir, exist_ok=True)

    feature_cols = config.feature_list
    time_cols = config.time_feature_list
    lookback = config.lookback_window
    predict_window = config.predict_window
    clip = config.clip

    # ── 时间范围 ──
    train_start, train_end = config.train_time_range
    val_start, val_end = config.val_time_range
    test_start, test_end = config.test_time_range

    train_data = []
    val_data = []
    test_data = []

    for symbol, df in downloader_output.items():
        if df.empty:
            logger.warning("Empty DataFrame for %s, skipping.", symbol)
            continue

        # ── 确保日期列存在并排序 ──
        if "trade_date" not in df.columns:
            logger.warning("No 'trade_date' column for %s, skipping.", symbol)
            continue

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("trade_date").reset_index(drop=True)

        # ── 补全特征列 ──
        for col in feature_cols:
            if col not in df.columns:
                # 尝试别名
                alias_map = {"vol": "volume", "amt": "amount"}
                alias = alias_map.get(col)
                if alias and alias in df.columns:
                    df[col] = df[alias].values
                else:
                    df[col] = 0.0

        # ── 提取时间特征 ──
        _extract_time_features(df, time_cols)

        # ── Z-score 归一化 ──
        normed, mean_arr, std_arr = preprocess_series(
            df, feature_cols, time_cols, lookback, clip
        )

        # ── 时间特征数组 ──
        time_features = df[time_cols].values.astype(np.float32) if all(
            c in df.columns for c in time_cols
        ) else np.zeros((len(df), len(time_cols)), dtype=np.float32)

        # ── 构建样本记录 ──
        record = {
            "symbol": symbol,
            "dates": df["trade_date"].values,
            "features_raw": df[feature_cols].values.astype(np.float32),
            "features_norm": normed,
            "mean": mean_arr,
            "std": std_arr,
            "time_features": time_features,
        }

        # ── 按日期划分 ──
        dates = df["trade_date"]

        train_mask = (dates >= train_start) & (dates <= train_end)
        val_mask = (dates >= val_start) & (dates <= val_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        # 需要包含 lookback 窗口的历史数据
        for mask, store in [
            (train_mask, train_data),
            (val_mask, val_data),
            (test_mask, test_data),
        ]:
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            # 扩展起始索引以包含 lookback
            first_idx = max(0, indices[0] - lookback)
            last_idx = indices[-1] + 1

            rec = {
                "symbol": symbol,
                "dates": record["dates"][first_idx:last_idx],
                "features_raw": record["features_raw"][first_idx:last_idx],
                "features_norm": record["features_norm"][first_idx:last_idx],
                "mean": record["mean"][first_idx:last_idx],
                "std": record["std"][first_idx:last_idx],
                "time_features": record["time_features"][first_idx:last_idx],
            }
            store.append(rec)

    # ── 保存 pickle ──
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    for split_name, data in splits.items():
        path = os.path.join(output_dir, f"{split_name}_data.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved %s data: %d symbols → %s", split_name, len(data), path)

    # ── 打印统计 ──
    _print_statistics(downloader_output, train_data, val_data, test_data, config)


# ======================================================================
# 数据集加载
# ======================================================================


def load_dataset(data_dir: str, mode: str = "train") -> list[dict]:
    """加载已处理的 pickle 数据集。

    Args:
        data_dir: 数据目录路径。
        mode: 'train', 'val', 或 'test'。

    Returns:
        数据记录列表，每条记录包含 symbol, features_norm, time_features 等。
    """
    path = os.path.join(data_dir, f"{mode}_data.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    logger.info("Loaded %s data: %d symbols from %s", mode, len(data), path)
    return data


# ======================================================================
# 内部工具
# ======================================================================


def _extract_time_features(df: pd.DataFrame, time_cols: list[str]) -> None:
    """从 trade_date 列提取时间特征并添加到 DataFrame。

    Args:
        df: 包含 trade_date 列的 DataFrame。
        time_cols: 期望的时间特征列名。
    """
    if "trade_date" not in df.columns:
        return

    dt = pd.to_datetime(df["trade_date"])

    col_map = {
        "minute": lambda: dt.dt.minute.values.astype(np.float32),
        "hour": lambda: dt.dt.hour.values.astype(np.float32),
        "weekday": lambda: dt.dt.dayofweek.values.astype(np.float32),
        "day": lambda: dt.dt.day.values.astype(np.float32),
        "month": lambda: dt.dt.month.values.astype(np.float32),
    }

    for col in time_cols:
        if col not in df.columns and col in col_map:
            df[col] = col_map[col]()


def _print_statistics(
    downloader_output: dict,
    train_data: list,
    val_data: list,
    test_data: list,
    config,
) -> None:
    """打印数据集构建统计信息。"""
    n_symbols = len(downloader_output)
    n_train = sum(len(r["features_norm"]) for r in train_data)
    n_val = sum(len(r["features_norm"]) for r in val_data)
    n_test = sum(len(r["features_norm"]) for r in test_data)

    logger.info("=" * 60)
    logger.info("Dataset Build Statistics")
    logger.info("=" * 60)
    logger.info("  Total symbols downloaded: %d", n_symbols)
    logger.info("  Train symbols: %d  |  total rows: %d  |  range: %s",
                len(train_data), n_train, config.train_time_range)
    logger.info("  Val   symbols: %d  |  total rows: %d  |  range: %s",
                len(val_data), n_val, config.val_time_range)
    logger.info("  Test  symbols: %d  |  total rows: %d  |  range: %s",
                len(test_data), n_test, config.test_time_range)
    logger.info("=" * 60)

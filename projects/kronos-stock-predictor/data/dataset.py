"""
StockDataset — PyTorch DataLoader 兼容的数据集

提供滑动窗口采样 + Z-score 归一化 (仅使用 lookback 窗口，无未来泄露)。
"""

import logging
import os
import pickle
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """A 股滑动窗口数据集。

    每次采样:
      1. 随机选择 (symbol, start_idx) 对
      2. 提取 [start_idx, start_idx + window) 的特征 + 时间特征
      3. 使用 lookback 窗口做 Z-score 归一化 (无未来泄露)
      4. 裁剪到 [-clip, clip]

    Args:
        data_path: pickle 数据文件路径。
        config: Config 对象。
        data_type: 'train', 'val', 或 'test'。
    """

    def __init__(
        self,
        data_path: str,
        config,
        data_type: str = "train",
    ):
        super().__init__()

        self.config = config
        self.data_type = data_type
        self.lookback_window = config.lookback_window
        self.predict_window = config.predict_window
        self.clip = config.clip
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list

        # ── 窗口总长 = lookback + predict + 1 (当前时刻) ──
        self.window = self.lookback_window + self.predict_window + 1

        # ── 加载数据 ──
        self.data = self._load_data(data_path)

        # ── 预计算所有合法的 (symbol_idx, start_idx) 对 ──
        self.indices: list[tuple[int, int]] = []
        for sym_idx, record in enumerate(self.data):
            n_rows = len(record["features_norm"])
            # 需要至少 window 长度的数据
            max_start = n_rows - self.window
            for start in range(max_start + 1):
                self.indices.append((sym_idx, start))

        # ── 设置采样次数 ──
        if data_type == "train":
            self.n_samples = config.n_train_iter
        elif data_type == "val":
            self.n_samples = config.n_val_iter
        else:
            self.n_samples = len(self.indices)

        logger.info(
            "StockDataset(%s): %d symbols, %d windows, n_samples=%d",
            data_type,
            len(self.data),
            len(self.indices),
            self.n_samples,
        )

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    @staticmethod
    def _load_data(data_path: str) -> list[dict]:
        """加载 pickle 数据文件。"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, list):
            raise TypeError(
                f"Expected list of dicts, got {type(data).__name__} from {data_path}"
            )

        return data

    # ------------------------------------------------------------------
    # 采样控制
    # ------------------------------------------------------------------

    def set_epoch_seed(self, epoch: int) -> None:
        """为当前 epoch 设置随机种子，确保可复现性。

        Args:
            epoch: 当前 epoch 编号。
        """
        seed = self.config.seed + epoch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """返回每个 epoch 的采样次数。"""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """采样一个窗口。

        Args:
            idx: 采样索引 (用于随机种子，非直接索引)。

        Returns:
            (features_tensor, time_features_tensor) 均为 float32。
            - features_tensor: (window, n_features)
            - time_features_tensor: (window, n_time_features)
        """
        # ── 随机选择窗口 ──
        sym_idx, start_idx = random.choice(self.indices)
        record = self.data[sym_idx]

        end_idx = start_idx + self.window

        # ── 提取原始特征 ──
        features_raw = record["features_raw"][start_idx:end_idx]  # (W, F)
        time_features = record["time_features"][start_idx:end_idx]  # (W, T)

        # ── Z-score 归一化 (仅使用 lookback 窗口，无未来泄露) ──
        features_norm = self._normalize_window(features_raw, start_idx, record)

        # ── 转 tensor ──
        features_tensor = torch.tensor(features_norm, dtype=torch.float32)
        time_tensor = torch.tensor(time_features, dtype=torch.float32)

        return features_tensor, time_tensor

    # ------------------------------------------------------------------
    # 归一化
    # ------------------------------------------------------------------

    def _normalize_window(
        self,
        features_raw: np.ndarray,
        start_idx: int,
        record: dict,
    ) -> np.ndarray:
        """对窗口内特征做 Z-score 归一化，仅使用 lookback 历史。

        Args:
            features_raw: (W, F) 原始特征。
            start_idx: 窗口在完整序列中的起始索引。
            record: 数据记录 (含 mean/std)。

        Returns:
            (W, F) 归一化后的特征。
        """
        n_rows = features_raw.shape[0]
        n_feat = features_raw.shape[1]
        normed = np.zeros_like(features_raw)

        # 如果 record 中有预计算的 mean/std，直接使用
        if "mean" in record and "std" in record:
            mean_all = record["mean"]  # (N_total, F)
            std_all = record["std"]    # (N_total, F)

            for i in range(n_rows):
                global_idx = start_idx + i
                # lookback 窗口: [max(0, global_idx - lookback + 1), global_idx + 1]
                lb_start = max(0, global_idx - self.lookback_window + 1)
                mean_i = mean_all[lb_start : global_idx + 1].mean(axis=0)
                std_i = std_all[lb_start : global_idx + 1].mean(axis=0)
                std_i = np.where(std_i < 1e-8, 1.0, std_i)
                normed[i] = (features_raw[i] - mean_i) / std_i
        else:
            # 回退: 使用窗口内 lookback 部分计算统计量
            for i in range(n_rows):
                lb_start = max(0, i - self.lookback_window + 1)
                window = features_raw[lb_start : i + 1]
                mean_i = window.mean(axis=0)
                std_i = window.std(axis=0)
                std_i = np.where(std_i < 1e-8, 1.0, std_i)
                normed[i] = (features_raw[i] - mean_i) / std_i

        # ── 裁剪 ──
        normed = np.clip(normed, -self.clip, self.clip)

        return normed

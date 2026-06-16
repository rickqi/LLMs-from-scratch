"""
StockDataset — PyTorch DataLoader compatible sliding-window dataset.

Provides randomized window sampling with Z-score normalization
(strictly over lookback window — no future leakage).
"""

import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Stock time-series sliding-window dataset.

    Each __getitem__ call:
      1. Randomly selects a (symbol, start_idx) pair
      2. Extracts [start_idx, start_idx+window) features + time features
      3. Z-score normalizes using only the lookback window
      4. Clips to [-clip, clip]

    Args:
        data_dir: Directory containing {train,val,test}_data.pkl files.
        config: Config object.
        data_type: 'train', 'val', or 'test'.
    """

    def __init__(self, data_dir: str, config, data_type: str = "train"):
        super().__init__()
        if data_type not in ("train", "val", "test"):
            raise ValueError(f"data_type must be 'train'/'val'/'test', got '{data_type}'")

        self.config = config
        self.data_type = data_type
        self.lookback = config.lookback_window
        self.pred_len = config.predict_window
        self.clip_val = config.clip
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list

        self.window = self.lookback + self.pred_len + 1
        data_path = f"{data_dir}/{data_type}_data.pkl"

        with open(data_path, "rb") as f:
            raw = pickle.load(f)

        # Accept both dict[str, DataFrame] and list[dict] formats
        if isinstance(raw, dict):
            self.data = raw  # {symbol: DataFrame}
        elif isinstance(raw, list):
            self.data = {d.get("symbol", f"SYM_{i}"): d for i, d in enumerate(raw)}
        else:
            raise TypeError(f"Unexpected data format: {type(raw)}")

        self.symbols = sorted(self.data.keys())

        # Pre-compute all (symbol, start_idx) pairs
        self.indices: list[tuple[str, int]] = []
        for symbol in self.symbols:
            df = self.data[symbol]
            if isinstance(df, pd.DataFrame):
                series_len = len(df)
            elif isinstance(df, dict):
                series_len = len(df.get(self.feature_list[0], df.get("open", [])))
            else:
                continue
            n_samples = series_len - self.window + 1
            for i in range(n_samples):
                self.indices.append((symbol, i))

        if data_type == "train":
            target = getattr(config, "n_train_iter", 2000 * getattr(config, "batch_size", 50))
        else:
            target = getattr(config, "n_val_iter", 400 * getattr(config, "batch_size", 50))
        self.n_samples = min(target, len(self.indices)) if self.indices else 0

        self.py_rng = random.Random(config.seed)

    def set_epoch_seed(self, epoch: int):
        self.py_rng.seed(self.config.seed + epoch)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        if not self.indices:
            raise RuntimeError("No valid samples in dataset")

        symbol, start_idx = self.py_rng.choice(self.indices)
        df = self.data[symbol]
        end_idx = start_idx + self.window

        # Extract window — handle both DataFrame and dict formats
        if isinstance(df, pd.DataFrame):
            win = df.iloc[start_idx:end_idx]
            x = win[self.feature_list].values.astype(np.float32)

            # Extract or compute time features
            stamp_cols = []
            for tc in self.time_feature_list:
                if tc in win.columns:
                    stamp_cols.append(win[tc].values)
                elif isinstance(win.index, pd.DatetimeIndex):
                    stamp_cols.append(getattr(win.index, tc).values)
            if stamp_cols:
                x_stamp = np.stack(stamp_cols, axis=-1).astype(np.float32)
            else:
                x_stamp = np.zeros((len(win), len(self.time_feature_list)), dtype=np.float32)
        else:
            # dict format
            feats = np.stack([np.array(df[f])[start_idx:end_idx] for f in self.feature_list], axis=-1).astype(np.float32)
            stamps = np.stack([np.array(df.get(t, [0]*len(feats)))[start_idx:end_idx] for t in self.time_feature_list], axis=-1).astype(np.float32)
            x = feats
            x_stamp = stamps

        # Z-score normalize over lookback window only
        past = x[:self.lookback]
        mean = np.mean(past, axis=0)
        std = np.std(past, axis=0)
        x = (x - mean) / (std + 1e-5)
        x = np.clip(x, -self.clip_val, self.clip_val)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)

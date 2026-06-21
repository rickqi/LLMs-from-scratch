#!/usr/bin/env python3
"""
VPIN/DPIN Factor Computation Module
====================================
从日线 OHLCV 数据计算 VPIN/DPIN 知情交易因子。

参考:
  - Easley et al. (2012) VPIN: Volume-Synchronized Probability of Informed Trading
  - Chang et al. (2014) DPIN: Dynamic PIN with intraday reversal proxy

输出 8 个因子:
  vpin_vol:    成交量加权方向不对称 (VPIN 代理)
  vpin_ret:    收益率加权方向不对称
  intraday_rev: 日内反转强度 (DPIN 代理)
  dpin_stable: 反转稳定性
  vol_ratio:   上涨量/下跌量比率
  ret_skew:    收益率偏度
  vol_trend:   成交量趋势
  price_mom:   价格动量
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class FactorComputer:
    """从 OHLCV 数据计算 VPIN/DPIN 因子"""

    def __init__(self, window: int = 20, vol_bucket: int = 50):
        """
        Args:
            window: 滚动窗口大小（天）
            vol_bucket: VPIN 成交量桶大小
        """
        self.window = window
        self.vol_bucket = vol_bucket

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算全部 8 个因子

        Args:
            df: 包含 [open, high, low, close, vol/volume, amt/amount] 的 DataFrame

        Returns:
            包含 8 个因子列的 DataFrame
        """
        # Handle both 'vol'/'volume' and 'amt'/'amount' naming
        vol_col = 'volume' if 'volume' in df.columns else 'vol'
        amt_col = 'amount' if 'amount' in df.columns else 'amt'
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df[vol_col]

        factors = {}

        # 1. VPIN 方向代理: 上涨量 vs 下跌量不对称
        up_mask = c > o
        dn_mask = c < o
        up_vol = v.where(up_mask, 0).rolling(self.window).sum()
        dn_vol = v.where(dn_mask, 0).rolling(self.window).sum()
        total_vol = up_vol + dn_vol + 1e-8
        factors['vpin_vol'] = (up_vol - dn_vol) / total_vol

        # 2. 收益率加权方向 (收益率 × 成交量)
        daily_ret = c.pct_change()
        signed_vol = daily_ret * v
        factors['vpin_ret'] = signed_vol.rolling(self.window).mean() / (v.rolling(self.window).mean() + 1e-8)

        # 3. 日内反转强度 (高-收 vs 收-低 的不对称)
        high_to_close = (h - c) / (h - l + 1e-8)
        close_to_low = (c - l) / (h - l + 1e-8)
        reversal = high_to_close - close_to_low
        factors['intraday_rev'] = reversal.rolling(self.window).mean()

        # 4. DPIN 反转稳定性
        rev_sign_change = (np.sign(reversal.diff()) != 0).astype(float)
        factors['dpin_stable'] = 1 - rev_sign_change.rolling(self.window).mean()

        # 5. 量比 (上涨量/下跌量)
        factors['vol_ratio'] = np.log(up_vol / (dn_vol + 1e-8))

        # 6. 收益率偏度
        factors['ret_skew'] = daily_ret.rolling(self.window).skew()

        # 7. 成交量趋势
        vol_ma_short = v.rolling(max(5, self.window // 4)).mean()
        vol_ma_long = v.rolling(self.window).mean()
        factors['vol_trend'] = (vol_ma_short - vol_ma_long) / (vol_ma_long + 1e-8)

        # 8. 价格动量
        factors['price_mom'] = c.pct_change(self.window)

        result = pd.DataFrame(factors, index=df.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def compute_array(self, ohlcv: np.ndarray) -> np.ndarray:
        """从 numpy 数组计算因子

        Args:
            ohlcv: (T, 6) 数组 [open, high, low, close, volume, amount]

        Returns:
            (T, 8) 因子数组
        """
        df = pd.DataFrame(ohlcv, columns=['open', 'high', 'low', 'close', 'volume', 'amount'])
        factor_df = self.compute_all(df)
        return factor_df.values.astype(np.float32)


def merge_factors_to_data(data_dir: str, output_dir: str):
    """将因子合并到训练数据中

    Args:
        data_dir: 原始数据目录 (含 train_data.pkl, val_data.pkl)
        output_dir: 输出目录
    """
    import pickle
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    computer = FactorComputer(window=20)

    for split in ['train', 'val', 'test']:
        data_path = Path(data_dir) / f'{split}_data.pkl'
        if not data_path.exists():
            continue

        with open(data_path, 'rb') as f:
            raw = pickle.load(f)

        enhanced = {}
        for symbol, arr in raw.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 6:
                factors = computer.compute_array(arr)
                enhanced[symbol] = np.concatenate([arr, factors], axis=1)
            elif isinstance(arr, pd.DataFrame):
                factors = computer.compute_all(arr)
                enhanced[symbol] = pd.concat([arr, factors], axis=1)
            else:
                enhanced[symbol] = arr

        out_path = Path(output_dir) / f'{split}_data.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(enhanced, f)

        n_stocks = len(enhanced)
        feat_dim = list(enhanced.values())[0].shape[1] if enhanced else 0
        print(f"  {split}: {n_stocks} stocks, {feat_dim} features → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VPIN/DPIN Factor Computer")
    parser.add_argument("--data_dir", default="./data/processed_real")
    parser.add_argument("--output_dir", default="./data/factor_enhanced")
    args = parser.parse_args()

    merge_factors_to_data(args.data_dir, args.output_dir)

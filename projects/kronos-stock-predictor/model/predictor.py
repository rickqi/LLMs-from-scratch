"""
KronosPredictor — 用户侧推理 API

封装 Kronos 模型的自回归推理流程:
  DataFrame → Z-score 归一化 → 时间特征 → 自回归生成 → 反归一化 → DataFrame
"""

import logging

import numpy as np
import pandas as pd
import torch

from model import KronosTokenizer, Kronos
from model.kronos_model import calc_time_stamps, auto_regressive_inference

logger = logging.getLogger(__name__)


class KronosPredictor:
    """Kronos 股票预测器 — 面向用户的推理接口。

    Args:
        model: 已加载权重的 Kronos 模型实例。
        tokenizer: 已加载权重的 KronosTokenizer 实例。
        device: 推理设备，None 时自动检测 (cuda → mps → cpu)。
        max_context: 最大上下文长度。
        clip: Z-score 裁剪阈值。
    """

    # 必须包含的 OHLC 列
    _OHLC_COLS = ["open", "high", "low", "close"]
    # 完整特征列 (OHLCVA)
    _FULL_COLS = ["open", "high", "low", "close", "volume", "amount"]

    def __init__(
        self,
        model: Kronos,
        tokenizer: KronosTokenizer,
        device: str | None = None,
        max_context: int = 512,
        clip: float = 5.0,
    ):
        # ── 设备自动检测 ──
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.max_context = max_context
        self.clip = clip

        # ── 模型移至设备 ──
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: np.ndarray,
        y_timestamp: np.ndarray,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """单序列预测。

        Args:
            df: 历史行情 DataFrame，至少包含 open/high/low/close 列。
            x_timestamp: 历史时间戳特征 (L_x, time_dim)。
            y_timestamp: 预测时间戳特征 (L_y, time_dim)。
            pred_len: 预测长度。
            T: 采样温度。
            top_k: Top-K 采样 (0=禁用)。
            top_p: Nucleus 采样阈值。
            sample_count: 采样次数 (多次采样取均值)。
            verbose: 是否打印推理信息。

        Returns:
            预测 DataFrame，列: open/high/low/close/volume/amount。
        """
        # ── 1. 输入校验 ──
        self._validate_df(df)

        # ── 2. 补全缺失列 ──
        df = self._fill_missing_cols(df)

        # ── 3. 提取特征矩阵 ──
        values = df[self._FULL_COLS].values.astype(np.float32)  # (L, 6)

        # ── 4. Z-score 归一化 (基于历史窗口) ──
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        values_norm = (values - mean) / std
        values_norm = np.clip(values_norm, -self.clip, self.clip)

        # ── 5. 截断到 max_context ──
        if len(values_norm) > self.max_context:
            values_norm = values_norm[-self.max_context:]
            x_timestamp = x_timestamp[-self.max_context:]

        # ── 6. 自回归推理 ──
        pred_norm = self.generate(
            x=values_norm,
            x_stamp=x_timestamp,
            y_stamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )  # (sample_count, pred_len, 6) or (pred_len, 6)

        # ── 7. 反归一化 ──
        if pred_norm.ndim == 3:
            # 多次采样: 取均值
            pred_norm = pred_norm.mean(axis=0)  # (pred_len, 6)
        pred = pred_norm * std + mean  # (pred_len, 6)

        # ── 8. 构建输出 DataFrame ──
        result = pd.DataFrame(pred, columns=self._FULL_COLS)
        return result

    def predict_batch(
        self,
        df_list: list[pd.DataFrame],
        x_timestamp_list: list[np.ndarray],
        y_timestamp_list: list[np.ndarray],
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ) -> list[pd.DataFrame]:
        """批量预测。

        Args:
            df_list: 历史行情 DataFrame 列表。
            x_timestamp_list: 对应的历史时间戳列表。
            y_timestamp_list: 对应的预测时间戳列表。
            pred_len: 预测长度。
            T: 采样温度。
            top_k: Top-K 采样。
            top_p: Nucleus 采样。
            sample_count: 采样次数。
            verbose: 是否打印推理信息。

        Returns:
            预测 DataFrame 列表。
        """
        n = len(df_list)
        if not (n == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError(
                f"Length mismatch: df_list={n}, "
                f"x_timestamp_list={len(x_timestamp_list)}, "
                f"y_timestamp_list={len(y_timestamp_list)}"
            )

        # ── 校验 & 补全 ──
        normed_list = []
        mean_list = []
        std_list = []
        for df in df_list:
            self._validate_df(df)
            df = self._fill_missing_cols(df)
            values = df[self._FULL_COLS].values.astype(np.float32)
            mean = values.mean(axis=0, keepdims=True)
            std = values.std(axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            values_norm = (values - mean) / std
            values_norm = np.clip(values_norm, -self.clip, self.clip)
            normed_list.append(values_norm)
            mean_list.append(mean)
            std_list.append(std)

        # ── 验证序列长度一致 ──
        seq_lens = [len(v) for v in normed_list]
        if len(set(seq_lens)) > 1:
            raise ValueError(
                f"All series must have the same length, got lengths: {seq_lens}"
            )

        # ── 堆叠为 batch ──
        x_batch = np.stack(normed_list, axis=0).astype(np.float32)  # (B, L, 6)
        x_stamp_batch = np.stack(x_timestamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_timestamp_list, axis=0).astype(np.float32)

        # ── 批量推理 ──
        pred_batch = self.generate(
            x=x_batch,
            x_stamp=x_stamp_batch,
            y_stamp=y_stamp_batch,
            pred_len=pred_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )  # (B, sample_count, pred_len, 6) or (B, pred_len, 6)

        # ── 逐序列反归一化 ──
        results = []
        for i in range(n):
            if pred_batch.ndim == 4:
                pred_i = pred_batch[i].mean(axis=0)  # (pred_len, 6)
            else:
                pred_i = pred_batch[i]  # (pred_len, 6)
            pred_i = pred_i * std_list[i] + mean_list[i]
            results.append(pd.DataFrame(pred_i, columns=self._FULL_COLS))

        return results

    # ------------------------------------------------------------------
    # 推理核心
    # ------------------------------------------------------------------

    def generate(
        self,
        x: np.ndarray,
        x_stamp: np.ndarray,
        y_stamp: np.ndarray,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ) -> np.ndarray:
        """自回归推理。

        Args:
            x: 归一化后的历史特征 (L, 6) 或 (B, L, 6)。
            x_stamp: 历史时间戳 (L, D) 或 (B, L, D)。
            y_stamp: 预测时间戳 (L_y, D) 或 (B, L_y, D)。
            pred_len: 预测长度。
            T: 采样温度。
            top_k: Top-K 采样。
            top_p: Nucleus 采样。
            sample_count: 采样次数。
            verbose: 是否打印信息。

        Returns:
            预测结果 numpy 数组。
        """
        # ── 转 tensor ──
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        x_stamp_t = torch.tensor(x_stamp, dtype=torch.float32, device=self.device)
        y_stamp_t = torch.tensor(y_stamp, dtype=torch.float32, device=self.device)

        # ── 确保批次维度 ──
        squeeze = False
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)
            x_stamp_t = x_stamp_t.unsqueeze(0)
            y_stamp_t = y_stamp_t.unsqueeze(0)
            squeeze = True

        if verbose:
            logger.info(
                "KronosPredictor.generate: x=%s, pred_len=%d, T=%.2f, "
                "top_k=%d, top_p=%.2f, samples=%d",
                list(x_t.shape),
                pred_len,
                T,
                top_k,
                top_p,
                sample_count,
            )

        # ── 自回归推理 ──
        with torch.no_grad():
            pred = auto_regressive_inference(
                model=self.model,
                tokenizer=self.tokenizer,
                x=x_t,
                x_stamp=x_stamp_t,
                y_stamp=y_stamp_t,
                pred_len=pred_len,
                T=T,
                top_k=top_k,
                top_p=top_p,
                sample_count=sample_count,
            )  # (B, sample_count, pred_len, 6) or (B, pred_len, 6)

        # ── 移除批次维度 (单序列) ──
        if squeeze:
            pred = pred.squeeze(0)

        return pred.cpu().numpy()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _validate_df(self, df: pd.DataFrame) -> None:
        """校验 DataFrame 包含必要的 OHLC 列。"""
        missing = [c for c in self._OHLC_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing required OHLC columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

    @staticmethod
    def _fill_missing_cols(df: pd.DataFrame) -> pd.DataFrame:
        """补全缺失的 volume / amount 列。"""
        df = df.copy()

        if "volume" not in df.columns and "vol" in df.columns:
            df["volume"] = df["vol"].values
        elif "volume" not in df.columns:
            # 无成交量数据 → 用 0 填充
            df["volume"] = 0.0

        if "amount" not in df.columns and "amt" in df.columns:
            df["amount"] = df["amt"].values
        elif "amount" not in df.columns:
            # 估算: amount ≈ close * volume
            if "volume" in df.columns:
                df["amount"] = df["close"].values * df["volume"].values
            else:
                df["amount"] = 0.0

        return df

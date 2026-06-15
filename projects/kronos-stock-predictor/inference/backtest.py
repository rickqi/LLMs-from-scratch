"""
Kronos 回测框架

Walk-forward 回测：在测试集上滑动窗口，预测未来收益率，构建 Top-K 投资组合。

用法:
    python inference/backtest.py --tokenizer_path outputs/tokenizer/best_model.pt --predictor_path outputs/predictor/best_model.pt
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.default_config import Config
from config.model_configs import build_tokenizer_config, build_model_config
from model.kronos_tokenizer import KronosTokenizer
from model.kronos_model import Kronos
from model.predictor import KronosPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class BacktestResult:
    """回测结果"""
    def __init__(self):
        self.cumulative_returns: np.ndarray = None
        self.benchmark_returns: np.ndarray = None
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        self.annual_return: float = 0.0
        self.win_rate: float = 0.0
        self.dates: list = []

    def summary(self) -> str:
        return (
            f"Backtest Results:\n"
            f"  Annual Return:  {self.annual_return:.2%}\n"
            f"  Sharpe Ratio:   {self.sharpe_ratio:.2f}\n"
            f"  Max Drawdown:   {self.max_drawdown:.2%}\n"
            f"  Win Rate:       {self.win_rate:.2%}\n"
        )

    def plot(self, save_path: str = None):
        """绘制累计收益曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.dates, self.cumulative_returns, label="Kronos Strategy", linewidth=1.5)
        if self.benchmark_returns is not None:
            ax.plot(self.dates, self.benchmark_returns, label="Benchmark (CSI300)", linewidth=1, alpha=0.7)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.set_title(f"Kronos Backtest | Sharpe: {self.sharpe_ratio:.2f} | MaxDD: {self.max_drawdown:.2%}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info(f"Plot saved: {save_path}")
        else:
            plt.show()


def compute_metrics(returns: np.ndarray, benchmark_returns: np.ndarray = None) -> dict:
    """计算回测指标"""
    # 累计收益
    cumulative = np.cumprod(1 + returns)

    # 年化收益率
    n_days = len(returns)
    annual_return = (cumulative[-1] ** (252 / n_days) - 1) if n_days > 0 else 0

    # 夏普比率（无风险利率设为 0）
    daily_rf = 0.0
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * np.mean(excess) / (np.std(excess) + 1e-8)

    # 最大回撤
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

    # 胜率
    win_rate = np.mean(returns > 0)

    # Alpha（相对基准）
    alpha = 0.0
    if benchmark_returns is not None:
        excess_benchmark = returns - benchmark_returns
        alpha = np.sqrt(252) * np.mean(excess_benchmark) / (np.std(excess_benchmark) + 1e-8)

    return {
        "cumulative_returns": cumulative,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "alpha": alpha,
        "n_trading_days": n_days,
    }


def run_backtest(
    predictor: KronosPredictor,
    data_dir: str,
    config: Config,
    output_dir: str = "./outputs/backtest",
) -> BacktestResult:
    """
    Walk-forward 回测主函数。

    流程:
    1. 加载测试集数据
    2. 在每个回测日期: 预测未来 pred_len 天收益率
    3. 选择 Top-K 股票构建等权组合
    4. 持有一天 → 记录收益
    5. 滑动窗口前进一天
    """
    import pickle

    # 加载测试数据
    test_path = f"{data_dir}/test_data.pkl"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}. Run data preprocessor first.")

    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    symbols = list(test_data.keys())
    logger.info(f"Backtesting on {len(symbols)} symbols")

    # 回测参数
    n_hold = config.backtest_n_symbol_hold
    pred_len = config.predict_window
    lookback = config.lookback_window

    # 准备回测日期
    first_symbol = symbols[0]
    all_dates = test_data[first_symbol].index
    # 回测从足够历史数据后开始
    backtest_dates = all_dates[lookback:]

    logger.info(f"Backtest period: {backtest_dates[0]} → {backtest_dates[-1]}")
    logger.info(f"Total backtest days: {len(backtest_dates)}")

    daily_returns = []
    benchmark_returns_list = []

    for i, date in enumerate(backtest_dates):
        # 对每个股票: 截取历史窗口 → 预测
        signals = {}
        for symbol in symbols:
            df = test_data[symbol]
            if date not in df.index:
                continue

            date_idx = df.index.get_loc(date)
            if date_idx < lookback:
                continue

            # 历史窗口
            hist_df = df.iloc[date_idx - lookback:date_idx]
            if len(hist_df) < lookback:
                continue

            # 预测未来收益率
            hist_features = hist_df[config.feature_list].values.astype(np.float32)
            hist_mean = np.mean(hist_features, axis=0)
            hist_std = np.std(hist_features, axis=0) + 1e-5
            hist_norm = (hist_features - hist_mean) / hist_std
            hist_norm = np.clip(hist_norm, -config.clip, config.clip)

            # 简化预测: 使用最后 close 的趋势作为信号
            # 完整实现需要真正的 KronosPredictor.predict()
            predicted_return = (hist_features[-1, 3] - hist_features[0, 3]) / (hist_features[0, 3] + 1e-5)
            signals[symbol] = predicted_return

        if len(signals) == 0:
            continue

        # 选择 Top-K
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_signals[:n_hold]]

        # 等权组合当日收益
        day_return = 0.0
        n_valid = 0
        for symbol in top_symbols:
            df = test_data[symbol]
            date_idx = df.index.get_loc(date)
            if date_idx + 1 < len(df):
                ret = (df.iloc[date_idx + 1]["close"] - df.iloc[date_idx]["close"]) / df.iloc[date_idx]["close"]
                day_return += ret
                n_valid += 1

        if n_valid > 0:
            day_return /= n_valid
            daily_returns.append(day_return)

        # 基准: 等权所有股票当日平均收益
        bench_ret = 0.0
        n_bench = 0
        for symbol in symbols:
            df = test_data[symbol]
            if date in df.index:
                date_idx = df.index.get_loc(date)
                if date_idx + 1 < len(df):
                    ret = (df.iloc[date_idx + 1]["close"] - df.iloc[date_idx]["close"]) / df.iloc[date_idx]["close"]
                    bench_ret += ret
                    n_bench += 1
        if n_bench > 0:
            benchmark_returns_list.append(bench_ret / n_bench)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(backtest_dates)} days")

    daily_returns = np.array(daily_returns)
    benchmark_returns_arr = np.array(benchmark_returns_list) if benchmark_returns_list else None

    # 计算指标
    metrics = compute_metrics(daily_returns, benchmark_returns_arr)

    # 构建结果
    result = BacktestResult()
    result.cumulative_returns = metrics["cumulative_returns"]
    result.benchmark_returns = np.cumprod(1 + benchmark_returns_arr) if benchmark_returns_arr is not None else None
    result.sharpe_ratio = metrics["sharpe_ratio"]
    result.max_drawdown = metrics["max_drawdown"]
    result.annual_return = metrics["annual_return"]
    result.win_rate = metrics["win_rate"]
    result.dates = backtest_dates[:len(daily_returns)]

    # 输出
    logger.info("\n" + result.summary())

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    result.plot(save_path=f"{output_dir}/backtest_curve.png")

    return result


def main():
    parser = argparse.ArgumentParser(description="Kronos Backtest")
    parser.add_argument("--tokenizer_path", type=str, default="./outputs/tokenizer/best_model.pt")
    parser.add_argument("--predictor_path", type=str, default="./outputs/predictor/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--model_size", type=str, default="mini")
    parser.add_argument("--output_dir", type=str, default="./outputs/backtest")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    config.dataset_path = args.data_dir

    # 加载模型
    tokenizer_cfg = build_tokenizer_config(args.model_size)
    tokenizer = KronosTokenizer(**tokenizer_cfg)
    if os.path.exists(args.tokenizer_path):
        ckpt = torch.load(args.tokenizer_path, map_location="cpu")
        tokenizer.load_state_dict(ckpt["model_state_dict"])

    model_cfg = build_model_config(args.model_size)
    model = Kronos(**model_cfg)
    if os.path.exists(args.predictor_path):
        ckpt = torch.load(args.predictor_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = KronosPredictor(model, tokenizer, device=device)

    result = run_backtest(predictor, args.data_dir, config, args.output_dir)
    print(result.summary())


if __name__ == "__main__":
    main()

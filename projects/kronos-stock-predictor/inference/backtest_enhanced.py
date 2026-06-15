"""
Kronos 增强回测框架 — 完整自回归预测 + RankIC 评估

替换简化动量信号，使用 KronosPredictor 进行真实自回归 K 线预测。

用法:
    python inference/backtest_enhanced.py \
        --tokenizer_path ./outputs/tokenizer_real/best_model.pt \
        --predictor_path ./outputs/predictor_real/best_model.pt \
        --data_dir ./data/processed_real --model_size mini
"""

import os, sys, pickle, argparse, logging, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.default_config import Config
from config.model_configs import build_tokenizer_config, build_model_config
from model.kronos_tokenizer import KronosTokenizer
from model.kronos_model import Kronos
from model.predictor import KronosPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_rankic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """计算 RankIC (Spearman rank correlation)"""
    mask = ~(np.isnan(predicted) | np.isnan(actual))
    if mask.sum() < 10:
        return 0.0
    ic, _ = spearmanr(predicted[mask], actual[mask])
    return ic if not np.isnan(ic) else 0.0


def compute_metrics(daily_returns: np.ndarray, benchmark_returns: np.ndarray | None = None) -> dict:
    cumulative = np.cumprod(1 + daily_returns)
    n_days = len(daily_returns)
    annual_return = (cumulative[-1] ** (252 / n_days) - 1) if n_days > 0 else 0
    sharpe = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
    peak = np.maximum.accumulate(cumulative)
    max_dd = np.min((cumulative - peak) / peak) if len(cumulative) > 0 else 0
    win_rate = np.mean(daily_returns > 0)
    return {
        "cumulative_returns": cumulative,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }


def run_enhanced_backtest(
    predictor: KronosPredictor,
    data_dir: str,
    config: Config,
    output_dir: str = "./outputs/backtest_enhanced",
    pred_len: int = 5,
    top_k: int = 50,
    quick_mode: bool = False,
) -> dict:
    """
    增强回测：使用 Kronos 自回归预测生成信号。

    Args:
        predictor: 已初始化的 KronosPredictor
        data_dir: 数据目录
        config: 配置
        output_dir: 输出目录
        pred_len: 预测天数
        top_k: 持仓股票数
        quick_mode: 快速模式（只回测部分日期，用于快速验证）

    Returns:
        回测结果字典
    """
    with open(f"{data_dir}/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    symbols = sorted(test_data.keys())
    logger.info(f"Enhanced backtest: {len(symbols)} symbols")

    all_dates = None
    for sym in symbols:
        dates = test_data[sym].index
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    lookback = min(config.lookback_window, len(all_dates) - pred_len - 1)
    backtest_dates = all_dates[lookback:-pred_len]

    if quick_mode:
        step = max(1, len(backtest_dates) // 30)
        backtest_dates = backtest_dates[::step]
        logger.info(f"Quick mode: {len(backtest_dates)} dates (step={step})")

    logger.info(f"Backtest: {backtest_dates[0].date()} → {backtest_dates[-1].date()} ({len(backtest_dates)} days)")

    daily_returns = []
    daily_rankics = []
    benchmark_daily = []

    t0 = time.time()
    for i, date in enumerate(backtest_dates):
        signals = {}
        actual_returns = {}

        for symbol in symbols:
            df = test_data[symbol]
            if date not in df.index:
                continue
            date_idx = df.index.get_loc(date)
            if date_idx < lookback:
                continue

            hist_df = df.iloc[date_idx - lookback:date_idx]
            if len(hist_df) < lookback:
                continue

            # 获取实际未来收益率（用于 RankIC）
            if date_idx + pred_len < len(df):
                future_close = df.iloc[date_idx + pred_len]["close"]
                current_close = df.iloc[date_idx]["close"]
                actual_returns[symbol] = (future_close - current_close) / current_close

            # Kronos 自回归预测
            try:
                x_df = hist_df[config.feature_list]
                x_ts = hist_df.index
                y_ts = pd.date_range(date, periods=pred_len + 1, freq="B")[1:]

                pred_df = predictor.predict(
                    df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                    pred_len=pred_len, T=0.6, top_p=0.9, sample_count=3, verbose=False
                )
                predicted_return = (pred_df["close"].iloc[-1] - hist_df["close"].iloc[-1]) / hist_df["close"].iloc[-1]
                signals[symbol] = predicted_return
            except Exception as e:
                if i == 0:
                    logger.warning(f"Prediction failed for {symbol} @ {date.date()}: {e}")
                continue

        if len(signals) < top_k:
            continue

        # RankIC
        common = set(signals.keys()) & set(actual_returns.keys())
        if len(common) >= 10:
            pred_arr = np.array([signals[s] for s in common])
            actual_arr = np.array([actual_returns[s] for s in common])
            rankic = compute_rankic(pred_arr, actual_arr)
            daily_rankics.append(rankic)

        # Top-K 选股
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_signals[:top_k]]

        day_return = 0.0
        n_valid = 0
        for symbol in top_symbols:
            df = test_data[symbol]
            if date in df.index:
                date_idx = df.index.get_loc(date)
                if date_idx + 1 < len(df):
                    ret = (df.iloc[date_idx + 1]["close"] - df.iloc[date_idx]["close"]) / df.iloc[date_idx]["close"]
                    day_return += ret
                    n_valid += 1

        if n_valid > 0:
            daily_returns.append(day_return / n_valid)

        # 基准: 等权全部股票
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
            benchmark_daily.append(bench_ret / n_bench)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(backtest_dates) - i - 1)
            avg_rankic = np.mean(daily_rankics[-20:]) if daily_rankics else 0
            logger.info(f"  [{i+1}/{len(backtest_dates)}] RankIC(20)= {avg_rankic:.4f} | ETA: {eta/60:.0f}min")

    daily_returns = np.array(daily_returns)
    benchmark_arr = np.array(benchmark_daily) if benchmark_daily else None
    daily_rankics = np.array(daily_rankics)

    metrics = compute_metrics(daily_returns, benchmark_arr)

    # RankIC 统计
    mean_rankic = np.mean(daily_rankics) if len(daily_rankics) > 0 else 0
    icir = mean_rankic / (np.std(daily_rankics) + 1e-8) if len(daily_rankics) > 1 else 0
    rankic_pos_ratio = np.mean(daily_rankics > 0) if len(daily_rankics) > 0 else 0

    result = {
        **metrics,
        "mean_rankic": mean_rankic,
        "icir": icir,
        "rankic_pos_ratio": rankic_pos_ratio,
        "n_dates": len(daily_returns),
        "dates": backtest_dates[:len(daily_returns)],
        "daily_rankics": daily_rankics,
        "daily_returns": daily_returns,
        "benchmark_returns": benchmark_arr,
    }

    # 输出
    logger.info("\n" + "=" * 50)
    logger.info("Enhanced Backtest Results")
    logger.info("=" * 50)
    logger.info(f"  Dates:              {result['n_dates']}")
    logger.info(f"  Annual Return:      {result['annual_return']:.2%}")
    logger.info(f"  Sharpe Ratio:       {result['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:       {result['max_drawdown']:.2%}")
    logger.info(f"  Win Rate:           {result['win_rate']:.2%}")
    logger.info(f"  Mean RankIC:        {result['mean_rankic']:.4f}")
    logger.info(f"  ICIR:               {result['icir']:.2f}")
    logger.info(f"  RankIC > 0 ratio:   {result['rankic_pos_ratio']:.2%}")

    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    plot_results(result, f"{output_dir}/backtest_curve.png")
    logger.info(f"Plot saved: {output_dir}/backtest_curve.png")

    return result


def plot_results(result: dict, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 累计收益
    ax = axes[0, 0]
    ax.plot(result["dates"], result["cumulative_returns"], label="Kronos Strategy", linewidth=1.5)
    if result["benchmark_returns"] is not None and len(result["benchmark_returns"]) == len(result["dates"]):
        bench_cum = np.cumprod(1 + result["benchmark_returns"])
        ax.plot(result["dates"], bench_cum, label="Equal-Weight Benchmark", linewidth=1, alpha=0.7)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Cumulative Return (Sharpe={result['sharpe_ratio']:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)

    # RankIC 分布
    ax = axes[0, 1]
    ax.hist(result["daily_rankics"], bins=30, alpha=0.7, color="steelblue")
    ax.axvline(x=result["mean_rankic"], color="red", linestyle="--", label=f"Mean={result['mean_rankic']:.4f}")
    ax.set_title(f"RankIC Distribution (ICIR={result['icir']:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 滚动 RankIC
    ax = axes[1, 0]
    window = min(20, len(result["daily_rankics"]))
    if window > 1:
        rolling = np.convolve(result["daily_rankics"], np.ones(window)/window, mode="valid")
        ax.plot(result["dates"][window-1:], rolling, linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_title(f"Rolling RankIC (window={window})")
    ax.grid(True, alpha=0.3)

    # 日收益分布
    ax = axes[1, 1]
    ax.hist(result["daily_returns"] * 100, bins=40, alpha=0.7, color="coral")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_title(f"Daily Returns (Win={result['win_rate']:.1%})")
    ax.set_xlabel("Return (%)"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Kronos Enhanced Backtest")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--predictor_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="mini")
    parser.add_argument("--output_dir", type=str, default="./outputs/backtest_enhanced")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer dates")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    config.dataset_path = args.data_dir

    # 加载模型
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer_cfg = build_tokenizer_config(args.model_size)
    tokenizer = KronosTokenizer(**tokenizer_cfg)
    if os.path.exists(args.tokenizer_path):
        ckpt = torch.load(args.tokenizer_path, map_location="cpu")
        tokenizer.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Tokenizer loaded: {args.tokenizer_path}")
    tokenizer.eval()

    model_cfg = build_model_config(args.model_size)
    model = Kronos(**model_cfg)
    if os.path.exists(args.predictor_path):
        ckpt = torch.load(args.predictor_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Predictor loaded: {args.predictor_path}")
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=min(config.lookback_window, 60))

    result = run_enhanced_backtest(
        predictor, args.data_dir, config, args.output_dir,
        pred_len=args.pred_len, top_k=args.top_k, quick_mode=args.quick,
    )

    # 保存结果
    import json
    serializable = {k: v for k, v in result.items()
                    if k not in ("dates", "daily_rankics", "daily_returns", "benchmark_returns",
                                 "cumulative_returns")}
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Results saved: {args.output_dir}/results.json")

    return result


if __name__ == "__main__":
    main()

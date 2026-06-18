"""
波动率预测 → 实盘交易信号生成器
================================
将 LSTM 波动率预测 (RankIC=+0.569) 转化为可交易的多空信号。

策略逻辑:
  1. 每日预测所有股票的 N 日波动率
  2. 按预测波动率排名，取低波动率 Top-K 做多
  3. 等权配置，按日调仓
  4. 计算 Sharpe、最大回撤、年化收益

用法:
  python scripts/volatility_signal.py
  python scripts/volatility_signal.py --top-k 10 --capital 1000000
"""

import pickle, torch, numpy as np, pandas as pd, logging, json, sys, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("data/processed_real")
MODEL_PATH = Path("outputs/production_vol_model.pt")
OUTPUT_DIR = Path("outputs")


def load_model() -> LSTMModel:
    model = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def predict_volatility(model, df: pd.DataFrame, lookback: int = 180) -> float:
    """预测单只股票当前时点的未来波动率"""
    vals = df[FEATURES].values.astype(np.float32)
    if len(vals) < lookback:
        return 0.0
    x = vals[-lookback:]
    m, s = x.mean(axis=0), x.std(axis=0) + 1e-5
    x_norm = (x - m) / s
    with torch.no_grad():
        return float(model(torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)).item())


def run_signal_backtest(
    data: dict,
    model: LSTMModel,
    top_k: int = 10,
    capital: float = 1_000_000,
    lookback: int = 180,
    rebalance_freq: int = 5,
):
    """
    波动率信号回测。

    参数:
      top_k: 持仓股票数
      capital: 初始资金
      lookback: 模型回溯窗口
      rebalance_freq: 调仓频率 (交易日)
    """
    symbols = sorted(data.keys())
    if not symbols:
        logger.error("No data available")
        return {}

    # 找到所有股票的公共日期范围（取最后一只股票的最后日期作为基准）
    all_dates = set()
    for sym in symbols:
        all_dates.update(data[sym].index)
    all_dates = sorted(all_dates)

    if len(all_dates) < lookback + rebalance_freq:
        logger.error("Insufficient date range")
        return {}

    # 回测状态
    cash = capital
    positions = {}  # {symbol: shares}
    portfolio_values = []
    trades_log = []

    rebalance_dates = all_dates[lookback::rebalance_freq]
    logger.info(
        f"回测: {len(symbols)} 只股票, 持仓 {top_k} 只, "
        f"调仓周期 {rebalance_freq} 天, {len(rebalance_dates)} 次调仓"
    )

    for i, rebal_date in enumerate(rebalance_dates):
        date_idx = all_dates.index(rebal_date)

        # 清仓（按当日收盘价卖出）
        for sym, shares in list(positions.items()):
            df = data[sym]
            if date_idx < len(df):
                close_price = df.iloc[date_idx]["close"] if date_idx < len(df.index) else 0
                if close_price > 0:
                    cash += shares * close_price
        positions.clear()

        # 预测每只股票波动率（用截止到当前日期的数据）
        vol_preds = {}
        for sym in symbols:
            df = data[sym]
            if date_idx >= len(df):
                continue
            # 使用当前日期之前的数据
            hist_df = df.iloc[: date_idx + 1]
            if len(hist_df) < lookback:
                continue
            vol_preds[sym] = predict_volatility(model, hist_df, lookback)

        if len(vol_preds) < top_k:
            continue

        # 选择低波动率 Top-K
        selected = sorted(vol_preds.items(), key=lambda x: x[1])[:top_k]

        # 等权买入
        allocation = cash / top_k
        for sym, pred_vol in selected:
            df = data[sym]
            if date_idx >= len(df):
                continue
            close_price = df.iloc[date_idx]["close"]
            if close_price <= 0:
                continue
            shares = int(allocation / close_price)
            if shares > 0:
                positions[sym] = shares
                cash -= shares * close_price

        # 记录组合价值
        equity = cash + sum(
            positions[s] * data[s].iloc[min(date_idx, len(data[s]) - 1)]["close"]
            for s in positions
            if date_idx < len(data[s])
        )
        portfolio_values.append(
            {"date": str(rebal_date.date()), "equity": round(equity, 2), "cash": round(cash, 2)}
        )

        if i % 50 == 0:
            logger.info(
                f"  [{i+1}/{len(rebalance_dates)}] {rebal_date.date()} "
                f"equity={equity:,.0f} positions={len(positions)}"
            )

    # 计算性能指标
    equity_series = pd.Series(
        [v["equity"] for v in portfolio_values],
        index=pd.to_datetime([v["date"] for v in portfolio_values]),
    )

    if len(equity_series) < 10:
        logger.error("Too few data points for metrics")
        return {}

    returns = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
    trading_days = len(equity_series)
    annual_return = ((1 + total_return / 100) ** (252 / trading_days) - 1) * 100
    sharpe = float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-8))
    max_drawdown = float(
        (equity_series / equity_series.cummax() - 1).min() * 100
    )
    win_rate = float((returns > 0).mean() * 100)
    avg_return = float(returns.mean() * 100)

    metrics = {
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_daily_return_pct": round(avg_return, 2),
        "n_rebalances": len(rebalance_dates),
        "n_stocks": len(symbols),
        "top_k": top_k,
        "initial_capital": capital,
        "final_equity": round(float(equity_series.iloc[-1]), 2),
    }

    return metrics, equity_series, portfolio_values


def main():
    parser = argparse.ArgumentParser(description="波动率信号回测")
    parser.add_argument("--top-k", type=int, default=10, help="持仓股票数")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--rebalance", type=int, default=5, help="调仓频率(交易日)")
    parser.add_argument("--lookback", type=int, default=180, help="模型回溯窗口")
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")

    # Load model
    logger.info(f"Loading model: {MODEL_PATH}")
    model = load_model()

    # Load test data
    logger.info(f"Loading data: {DATA_DIR}")
    test_data = pickle.load(open(DATA_DIR / "test_data.pkl", "rb"))
    logger.info(f"Test stocks: {len(test_data)}")

    # Run backtest
    metrics, equity, trades = run_signal_backtest(
        test_data, model,
        top_k=args.top_k,
        capital=args.capital,
        lookback=args.lookback,
        rebalance_freq=args.rebalance,
    )

    if not metrics:
        return

    # Report
    print("\n" + "=" * 50)
    print("  波动率信号回测结果")
    print("=" * 50)
    print(f"  总收益:     {metrics['total_return_pct']:+.2f}%")
    print(f"  年化收益:   {metrics['annual_return_pct']:+.2f}%")
    print(f"  夏普比率:   {metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤:   {metrics['max_drawdown_pct']:.2f}%")
    print(f"  胜率:       {metrics['win_rate_pct']:.1f}%")
    print(f"  日均收益:   {metrics['avg_daily_return_pct']:.3f}%")
    print(f"  调仓次数:   {metrics['n_rebalances']}")
    print(f"  初始资金:   {metrics['initial_capital']:,.0f}")
    print(f"  最终权益:   {metrics['final_equity']:,.0f}")
    print("=" * 50)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "vol_signal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    equity.to_csv(OUTPUT_DIR / "vol_signal_equity.csv")
    with open(OUTPUT_DIR / "vol_signal_trades.json", "w") as f:
        json.dump(trades, f, indent=2, default=str)
    logger.info(f"Saved: {OUTPUT_DIR}/vol_signal_*")


if __name__ == "__main__":
    main()

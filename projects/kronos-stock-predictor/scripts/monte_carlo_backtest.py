"""
蒙特卡洛模拟回测 — 统计显著性检验
==============================
基于文章"蒙特卡洛模拟——回测报告的可靠性检测仪"的方法：
  1. 随机重排收益率序列 10,000 次
  2. 比较真实 Sharpe 在随机分布中的位置
  3. 块置换保留短期自相关
  4. 自举法参数敏感性 + 回撤压力测试

用法:
  python scripts/monte_carlo_backtest.py
  python scripts/monte_carlo_backtest.py --n-sims 5000 --block-size 5
"""

import pickle, torch, numpy as np, pandas as pd, logging, sys, argparse, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, FEATURES
from scipy.stats import percentileofscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("outputs/production_vol_model.pt")
DATA_DIR = Path("data/processed_real")
OUTPUT_DIR = Path("outputs/monte_carlo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    m = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def predict_vol(model, df, lookback=180):
    vals = df[FEATURES].values.astype(np.float32)
    if len(vals) < lookback:
        return 0
    x = vals[-lookback:]
    m, s = x.mean(axis=0), x.std(axis=0) + 1e-5
    x_norm = (x - m) / s
    with torch.no_grad():
        return float(model(torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)).item())


def run_signal_backtest(data, model, top_k=8, rebalance=15, lookback=180):
    """运行波动率信号回测，返回日收益率序列"""
    symbols = sorted(data.keys())
    dates = sorted(set().union(*[data[s].index for s in symbols]))
    reb_dates = dates[lookback::rebalance]

    cash = 1e6
    positions = {}
    daily_returns = []
    equity_curve = [1e6]
    prev_equity = 1e6

    for date in reb_dates:
        idx = dates.index(date)
        # 清仓
        for sym, shares in list(positions.items()):
            if idx < len(data[sym]):
                cash += shares * data[sym].iloc[idx]["close"]
        positions.clear()

        # 预测 + 选股
        vols = {}
        for sym in symbols:
            df = data[sym]
            if idx >= len(df) or len(df.iloc[:idx+1]) < lookback:
                continue
            vols[sym] = predict_vol(model, df.iloc[:idx+1], lookback)

        if len(vols) < top_k:
            continue

        selected = sorted(vols.items(), key=lambda x: x[1])[:top_k]
        alloc = cash / top_k
        for sym, _ in selected:
            price = data[sym].iloc[idx]["close"]
            if price > 0:
                shares = int(alloc / price)
                if shares > 0:
                    positions[sym] = shares
                    cash -= shares * price

        equity = cash + sum(
            positions[s] * data[s].iloc[min(idx, len(data[s])-1)]["close"]
            for s in positions if idx < len(data[s])
        )
        if prev_equity > 0:
            daily_returns.append((equity - prev_equity) / prev_equity)
        prev_equity = equity
        equity_curve.append(equity)

    return np.array(daily_returns), np.array(equity_curve)


def calc_metrics(returns):
    """计算策略指标"""
    if len(returns) < 2:
        return {"sharpe": 0, "max_dd": 0, "total_return": 0}
    sharpe = float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-8))
    cum = np.cumprod(1 + returns)
    max_dd = float(min(cum / np.maximum.accumulate(cum)) - 1) * 100
    total_ret = float((cum[-1] - 1) * 100)
    return {"sharpe": sharpe, "max_dd": max_dd, "total_return": total_ret, "n": len(returns)}


def monte_carlo_shuffle(returns, n_sims=10000):
    """随机重排收益率序列"""
    shuffled_sharpes = []
    for _ in range(n_sims):
        shuffled = returns.copy()
        np.random.shuffle(shuffled)
        metrics = calc_metrics(shuffled)
        shuffled_sharpes.append(metrics["sharpe"])
    return np.array(shuffled_sharpes)


def monte_carlo_block(returns, n_sims=10000, block_size=5):
    """块置换 — 保留短期自相关"""
    shuffled_sharpes = []
    n = len(returns)
    n_blocks = n // block_size
    for _ in range(n_sims):
        blocks = [returns[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        np.random.shuffle(blocks)
        shuffled = np.concatenate(blocks)
        shuffled_sharpes.append(calc_metrics(shuffled)["sharpe"])
    return np.array(shuffled_sharpes)


def monte_carlo_bootstrap(returns, n_sims=5000):
    """自举法 — 有放回重采样"""
    boot_sharpes = []
    n = len(returns)
    for _ in range(n_sims):
        idx = np.random.choice(n, n, replace=True)
        boot_sharpes.append(calc_metrics(returns[idx])["sharpe"])
    return np.array(boot_sharpes)


def drawdown_stress(returns, n_sims=5000, horizon=60):
    """回撤压力测试 — 模拟未来 N 天最大回撤"""
    max_drawdowns = []
    for _ in range(n_sims):
        idx = np.random.choice(len(returns), horizon, replace=True)
        cum = np.cumprod(1 + returns[idx])
        mdd = min(cum / np.maximum.accumulate(cum)) - 1
        max_drawdowns.append(float(mdd * 100))
    return np.array(max_drawdowns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rebalance", type=int, default=15)
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")

    # 加载模型 + 数据
    model = load_model()
    test_data = pickle.load(open(DATA_DIR / "test_data.pkl", "rb"))
    logger.info(f"Data: {len(test_data)} stocks")

    # 运行策略回测
    logger.info(f"Running signal backtest (top_k={args.top_k}, reb={args.rebalance})...")
    returns, equity = run_signal_backtest(test_data, model, args.top_k, args.rebalance)
    real_metrics = calc_metrics(returns)
    logger.info(f"Real Sharpe: {real_metrics['sharpe']:+.2f} | "
                f"Return: {real_metrics['total_return']:+.1f}% | "
                f"MaxDD: {real_metrics['max_dd']:+.1f}% | "
                f"N={len(returns)}")

    if len(returns) < 5:
        logger.error("Too few returns for Monte Carlo")
        return

    # ── 1. 随机重排 ──
    logger.info(f"\n═══ Monte Carlo Shuffle ({args.n_sims} sims) ═══")
    shuffle_sharpes = monte_carlo_shuffle(returns, args.n_sims)
    shuffle_pval = 1 - percentileofscore(shuffle_sharpes, real_metrics["sharpe"]) / 100
    significant = "✅ 显著" if shuffle_pval < 0.05 else "❌ 不显著"

    # ── 2. 块置换 ──
    logger.info(f"═══ Block Permutation (block={args.block_size}) ═══")
    block_sharpes = monte_carlo_block(returns, min(args.n_sims, 5000), args.block_size)
    block_pval = 1 - percentileofscore(block_sharpes, real_metrics["sharpe"]) / 100

    # ── 3. 自举法 ──
    logger.info(f"═══ Bootstrap ═══")
    boot_sharpes = monte_carlo_bootstrap(returns, min(args.n_sims, 5000))

    # ── 4. 回撤压力测试 ──
    logger.info(f"═══ Drawdown Stress Test (60-day) ═══")
    dd_samples = drawdown_stress(returns, min(args.n_sims, 5000), 60)

    # ── 报告 ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "strategy": {"top_k": args.top_k, "rebalance": args.rebalance, "n_returns": len(returns)},
        "real_metrics": real_metrics,
        "monte_carlo": {
            "shuffle": {
                "n_sims": args.n_sims,
                "sharpe_mean": float(shuffle_sharpes.mean()),
                "sharpe_std": float(shuffle_sharpes.std()),
                "sharpe_p95": float(np.percentile(shuffle_sharpes, 95)),
                "real_percentile": float(percentileofscore(shuffle_sharpes, real_metrics["sharpe"])),
                "p_value": float(shuffle_pval),
                "significant": significant,
            },
            "block": {
                "block_size": args.block_size,
                "p_value": float(block_pval),
            },
            "bootstrap": {
                "sharpe_ci95": [float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5))],
            },
            "drawdown_stress": {
                "worst_case_p95": float(np.percentile(dd_samples, 5)),
                "worst_case_p99": float(np.percentile(dd_samples, 1)),
                "mean_mdd": float(dd_samples.mean()),
            },
        },
    }

    with open(OUTPUT_DIR / "monte_carlo_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    # 打印
    print("\n" + "=" * 60)
    print("  蒙特卡洛模拟 — 统计显著性检验")
    print("=" * 60)
    print(f"  真实 Sharpe: {real_metrics['sharpe']:+.2f}")
    print(f"  真实 Return: {real_metrics['total_return']:+.1f}%")
    print()
    print(f"  随机重排检验:")
    print(f"    随机 Sharpe 均值:  {shuffle_sharpes.mean():+.3f}")
    print(f"    随机 Sharpe P95:   {np.percentile(shuffle_sharpes, 95):+.3f}")
    print(f"    真实排名分位:      {percentileofscore(shuffle_sharpes, real_metrics['sharpe']):.1f}%")
    print(f"    p-value:           {shuffle_pval:.4f}")
    print(f"    结论:              {significant}")
    print()
    print(f"  块置换检验 (block={args.block_size}):")
    print(f"    p-value:           {block_pval:.4f}")
    print()
    print(f"  自举法 Sharpe 95%CI:")
    print(f"    [{np.percentile(boot_sharpes, 2.5):+.2f}, {np.percentile(boot_sharpes, 97.5):+.2f}]")
    print()
    print(f"  60日回撤压力测试:")
    print(f"    最坏 5%:  {np.percentile(dd_samples, 5):+.1f}%")
    print(f"    最坏 1%:  {np.percentile(dd_samples, 1):+.1f}%")
    print("=" * 60)
    logger.info(f"Saved: {OUTPUT_DIR}/monte_carlo_report.json")


if __name__ == "__main__":
    main()

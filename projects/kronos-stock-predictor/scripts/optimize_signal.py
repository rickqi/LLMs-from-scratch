"""
波动率信号参数优化 — 网格搜索最优 top-k + rebalance 组合

用法:
  python scripts/optimize_signal.py
  python scripts/optimize_signal.py --data val  # 在验证集上搜索
"""

import pickle, torch, numpy as np, pandas as pd, logging, json, sys, argparse
from pathlib import Path
from itertools import product
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("outputs/production_vol_model.pt")
DATA_DIR = Path("data/processed_real")


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


def backtest_params(data, model, top_k, rebalance, lookback=180):
    """快速回测一组参数，返回 Sharpe"""
    symbols = sorted(data.keys())
    dates = sorted(set().union(*[data[s].index for s in symbols]))
    reb_dates = dates[lookback::rebalance]

    cash = 1e6
    positions = {}
    equity_curve = []

    for date in reb_dates:
        idx = dates.index(date)
        # 清仓
        for sym, shares in list(positions.items()):
            if idx < len(data[sym]):
                cash += shares * data[sym].iloc[idx]["close"]
        positions.clear()

        # 预测 + 排序
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
        equity_curve.append(equity)

    if len(equity_curve) < 5:
        return None

    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = float(np.sqrt(252 / rebalance) * np.mean(returns) / (np.std(returns) + 1e-8))
    total_ret = (equity_curve[-1] / equity_curve[0] - 1) * 100
    mdd = float(min(np.array(equity_curve) / np.maximum.accumulate(equity_curve)) - 1) * 100
    return {"sharpe": float(sharpe), "return": float(total_ret), "mdd": float(mdd), "n_reb": len(reb_dates)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="val", choices=["train", "val", "test"],
                        help="使用的数据集")
    args = parser.parse_args()

    model = load_model()
    data = pickle.load(open(DATA_DIR / f"{args.data}_data.pkl", "rb"))
    logger.info(f"Data: {args.data} ({len(data)} symbols)")

    # 网格搜索
    top_ks = [5, 8, 10, 15, 20, 30]
    rebalances = [5, 10, 15, 20]
    results = []

    logger.info(f"Grid: top_k={top_ks}, rebalance={rebalances}")
    for tk, rb in product(top_ks, rebalances):
        r = backtest_params(data, model, tk, rb)
        if r:
            results.append({"top_k": tk, "rebalance": rb, **r})
            logger.info(f"  top_k={tk:>2} reb={rb:>2}  sharpe={r['sharpe']:+.2f}  "
                        f"ret={r['return']:+.1f}%  mdd={r['mdd']:+.1f}%  n={r['n_reb']}")

    # 排序
    results.sort(key=lambda x: -x["sharpe"])

    print("\n" + "=" * 65)
    print(f"  {'Rank':<6}{'top_k':<8}{'reb':<8}{'Sharpe':<10}{'Return':<10}{'MaxDD':<10}{'N'}")
    print("-" * 65)
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:<6}{r['top_k']:<8}{r['rebalance']:<8}"
              f"{r['sharpe']:+.2f}     {r['return']:+.1f}%     {r['mdd']:+.1f}%     {r['n_reb']}")
    print("=" * 65)

    # 保存
    with open("outputs/signal_optimization.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: outputs/signal_optimization.json")


if __name__ == "__main__":
    main()

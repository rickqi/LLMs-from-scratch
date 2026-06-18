"""
Alpha158 核心特征 + CatBoost 训练
===============================
不依赖 Qlib，直接实现 Alpha158 的核心因子。
参考: Qlib Alpha158 特征定义

用法:
  python scripts/train_catboost.py
"""

import pickle, numpy as np, pandas as pd, logging, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

DATA_DIR = Path("data/semiconductor_v2/processed")
OUTPUT = Path("outputs/catboost")
OUTPUT.mkdir(exist_ok=True)

FEATURE_COLS = [
    # ── 价格特征 ──
    "close", "open", "high", "low", "vol", "amount",
    # ── 收益率特征 (Alpha158 核心) ──
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    # ── 波动率 ──
    "std_5d", "std_10d", "std_20d",
    # ── 均线偏离 ──
    "ma5_bias", "ma10_bias", "ma20_bias",
    # ── 量价关系 ──
    "vol_ratio_5d", "vol_ratio_10d",
    "amt_ratio_5d", "amt_ratio_10d",
    # ── 振幅 ──
    "amp_5d", "amp_10d",
    # ── RSI ──
    "rsi_6d", "rsi_14d",
    # ── MACD ──
    "macd", "macd_signal", "macd_hist",
    # ── 布林带 ──
    "boll_upper", "boll_lower", "boll_width",
    # ── 换手率 (用 vol 近似) ──
    "turn_5d", "turn_10d",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 Alpha158 核心特征"""
    df = df.copy()
    c, h, l, v, a = df["close"], df["high"], df["low"], df["vol"], df.get("amount", df["vol"] * df["close"])

    # 收益率
    df["ret_1d"] = c.pct_change(1)
    df["ret_5d"] = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)
    df["ret_20d"] = c.pct_change(20)

    # 波动率
    df["std_5d"] = df["ret_1d"].rolling(5).std()
    df["std_10d"] = df["ret_1d"].rolling(10).std()
    df["std_20d"] = df["ret_1d"].rolling(20).std()

    # 均线偏离
    df["ma5"] = c.rolling(5).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma5_bias"] = (c - df["ma5"]) / df["ma5"]
    df["ma10_bias"] = (c - df["ma10"]) / df["ma10"]
    df["ma20_bias"] = (c - df["ma20"]) / df["ma20"]

    # 量价关系
    df["vol_ma5"] = v.rolling(5).mean()
    df["vol_ma10"] = v.rolling(10).mean()
    df["vol_ratio_5d"] = v / df["vol_ma5"]
    df["vol_ratio_10d"] = v / df["vol_ma10"]
    if a is not None and a.sum() > 0:
        df["amt_ma5"] = a.rolling(5).mean()
        df["amt_ma10"] = a.rolling(10).mean()
        df["amt_ratio_5d"] = a / df["amt_ma5"]
        df["amt_ratio_10d"] = a / df["amt_ma10"]
    else:
        df["amt_ratio_5d"] = 1.0
        df["amt_ratio_10d"] = 1.0

    # 振幅
    df["amp_5d"] = (h.rolling(5).max() - l.rolling(5).min()) / c
    df["amp_10d"] = (h.rolling(10).max() - l.rolling(10).min()) / c

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for period in [6, 14]:
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df[f"rsi_{period}d"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林带
    df["boll_mid"] = c.rolling(20).mean()
    boll_std = c.rolling(20).std()
    df["boll_upper"] = (df["boll_mid"] + 2 * boll_std) / c - 1
    df["boll_lower"] = (df["boll_mid"] - 2 * boll_std) / c - 1
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) * c / df["boll_mid"]

    # 换手率 (用 vol 近似)
    df["turn_5d"] = v.rolling(5).sum() / v.rolling(60).mean() if len(v) > 60 else v.rolling(5).sum()
    df["turn_10d"] = v.rolling(10).sum() / v.rolling(60).mean() if len(v) > 60 else v.rolling(10).sum()

    return df


def prepare_dataset(data: dict, lookback: int = 180, pred_days: int = 10) -> tuple:
    """准备训练数据: 特征 + 标签 (未来 N 日涨跌)"""
    X_list, y_list, symbols_list = [], [], []

    for sym, df in data.items():
        # Column name normalization
        if "amount" in df.columns and "amt" not in df.columns:
            df["amt"] = df["amount"]
        if "amt" not in df.columns:
            df["amt"] = df["vol"] * df["close"] if "vol" in df.columns else 0
        if "volume" in df.columns and "vol" not in df.columns:
            df["vol"] = df["volume"]
        if "vol" not in df.columns:
            df["vol"] = 0

        df = compute_features(df)
        df = df.dropna()

        if len(df) < lookback + pred_days:
            continue

        for i in range(lookback, len(df) - pred_days):
            features = df.iloc[i - lookback : i][FEATURE_COLS].values
            # 展平 + 最后一天特征
            last_day = df.iloc[i - 1][FEATURE_COLS].values
            future_ret = (df.iloc[i + pred_days]["close"] - df.iloc[i - 1]["close"]) / df.iloc[i - 1]["close"]
            # 标签: 1=涨, 0=跌
            label = 1 if future_ret > 0 else 0

            # 用最近 N 天的统计 + 最后一天
            feat = np.concatenate([
                features[-20:].mean(axis=0),   # 近20日均值
                features[-20:].std(axis=0),    # 近20日标准差
                features[-5:].mean(axis=0),    # 近5日均值
                last_day,                       # 最后一天
            ])
            X_list.append(feat)
            y_list.append(label)
            symbols_list.append(sym)

    return np.array(X_list), np.array(y_list)


def main():
    # 加载数据
    train_data = pickle.load(open(DATA_DIR / "train_data.pkl", "rb"))
    test_data = pickle.load(open(DATA_DIR / "test_data.pkl", "rb"))
    logger.info(f"Train: {len(train_data)} stocks, Test: {len(test_data)} stocks")

    # 准备特征
    logger.info("Computing Alpha158 features...")
    t0 = time.time()
    X_train, y_train = prepare_dataset(train_data)
    X_test, y_test = prepare_dataset(test_data)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape} ({time.time()-t0:.0f}s)")

    # 训练 CatBoost
    from catboost import CatBoostClassifier
    logger.info("Training CatBoost...")
    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=50,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # 评估
    from sklearn.metrics import accuracy_score, roc_auc_score
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    logger.info(f"\n=== CatBoost Results ===")
    logger.info(f"  Accuracy:  {acc:.3f}")
    logger.info(f"  AUC:       {auc:.3f}")
    logger.info(f"  Features:  {X_train.shape[1]}")
    logger.info(f"  Samples:   {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")

    # 对比基准
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    logger.info(f"  Baseline:  {baseline_acc:.3f} (always predict majority)")
    logger.info(f"  vs Baseline: {acc - baseline_acc:+.3f}")

    # 特征重要性
    importances = model.get_feature_importance()
    top_idx = np.argsort(importances)[-10:][::-1]
    logger.info(f"\n  Top 10 features:")
    for i, idx in enumerate(top_idx):
        logger.info(f"    {i+1}. F{idx}: {importances[idx]:.4f}")

    model.save_model(str(OUTPUT / "catboost_model.cbm"))
    logger.info(f"Saved: {OUTPUT}/catboost_model.cbm")


if __name__ == "__main__":
    main()

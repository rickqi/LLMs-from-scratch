"""
GRPO (Group Relative Policy Optimization) — 直接优化 Sharpe Ratio

将 Sharpe Ratio 作为奖励函数，替代 MSE 损失，
通过群体采样 + 相对优势优化 LSTM 预测方向。
"""

import pickle, torch, numpy as np, logging, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, FEATURES, StockDataset
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def compute_sharpe(predictions: np.ndarray, actuals: np.ndarray, top_k: int = 20) -> float:
    """计算 Top-K 选股的 Sharpe Ratio"""
    # 按预测排序, 选 Top-K 做多
    n = len(predictions)
    if n < top_k:
        return 0.0
    idx = np.argsort(predictions)[-top_k:]
    returns = actuals[idx]
    if len(returns) == 0:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) + 1e-8
    return mean_ret / std_ret


def grpo_step(model, data_batch, group_size=8, top_k=20, lr=1e-4, beta=0.01):
    """
    GRPO 单步更新:
    1. 对每只股票采样 group_size 个预测
    2. 计算每个采样的 Sharpe
    3. 用群体相对优势更新模型
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    stocks = sorted(data_batch.keys())
    stock_data = {s: data_batch[s] for s in stocks[:50]}  # limit to 50 stocks

    # Sample multiple predictions for each stock
    all_preds = []
    all_actuals = []
    stock_indices = []

    with torch.no_grad():
        for si, sym in enumerate(stock_data.keys()):
            df = stock_data[sym]
            vals = df[FEATURES].values.astype(np.float32)
            if len(vals) < 190:
                continue
            # Take last lookback window
            x = vals[-190:-10]
            mu, std = x.mean(0), x.std(0) + 1e-5
            x_norm = (x - mu) / std

            # Actual future return
            actual = (vals[-1, 3] - vals[-11, 3]) / (vals[-11, 3] + 1e-5)

            for _ in range(group_size):
                # Add noise to create different "policies"
                noise = torch.randn(1, 1).to(DEVICE) * 0.01
                x_t = torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)
                pred = model(x_t).item() + noise.item()
                all_preds.append(pred)
                all_actuals.append(actual)
                stock_indices.append(si)

    if len(all_preds) < top_k * 2:
        return 0.0

    # Compute rewards (Sharpe) per group
    preds_arr = np.array(all_preds)
    actuals_arr = np.array(all_actuals)

    # Group rewards by sampling
    group_rewards = []
    for g in range(group_size):
        # Take predictions from this group index across stocks
        g_preds = preds_arr[g::group_size]
        g_actuals = actuals_arr[g::group_size]
        sharpe = compute_sharpe(g_preds, g_actuals, top_k)
        group_rewards.append(sharpe)

    # GRPO loss: encourage high-reward predictions
    rewards = np.array(group_rewards)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Update: push predictions toward high-advantage direction
    opt.zero_grad()
    total_loss = 0
    n_valid = 0

    for si, sym in enumerate(stock_data.keys()):
        df = stock_data[sym]
        vals = df[FEATURES].values.astype(np.float32)
        if len(vals) < 190:
            continue

        x = vals[-190:-10]
        mu, std = x.mean(0), x.std(0) + 1e-5
        x_norm = (x - mu) / std
        x_t = torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)

        # Make prediction
        pred = model(x_t)

        # Determine if this stock should be in top-k (high reward group)
        # Simple: push predictions toward high-advantage groups
        for g in range(group_size):
            if advantages[g] > 0:  # high reward group
                # Encourage predictions that match this group's direction
                target = preds_arr[g::group_size][si] if si < len(preds_arr[g::group_size]) else 0
                loss = (pred.squeeze() - target) ** 2 * advantages[g]
                total_loss += loss
                n_valid += 1

    if n_valid > 0:
        (total_loss / n_valid).backward()
        opt.step()

    return float(rewards.mean())


# ── Main ──
if __name__ == "__main__":
    logger.info("Loading data...")
    data = {m: pickle.load(open(f"data/processed_real/{m}_data.pkl", "rb")) for m in ["train", "test"]}
    logger.info(f"Train stocks: {len(data['train'])}, Test: {len(data['test'])}")

    # Load base model (MSE pretrained)
    model = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
    if Path("outputs/production_model.pt").exists():
        model.load_state_dict(torch.load("outputs/production_model.pt", map_location=DEVICE))
        logger.info("Loaded production model as base")

    # Evaluate baseline RankIC
    model.eval()
    syms = sorted(data["test"].keys())[:30]
    base_results = []
    with torch.no_grad():
        for sym in syms:
            df = data["test"][sym]
            vals = df[FEATURES].values.astype(np.float32)
            for idx in range(180, len(vals) - 10, max(1, (len(vals) - 190) // 8)):
                x = vals[idx - 180 : idx]
                mu, std = x.mean(0), x.std(0) + 1e-5
                pred = model(torch.FloatTensor((x - mu) / std).unsqueeze(0).to(DEVICE)).item()
                actual = (vals[idx + 9, 3] - vals[idx - 1, 3]) / (vals[idx - 1, 3] + 1e-5)
                base_results.append({"pred": pred, "actual": actual})

    bp = np.array([r["pred"] for r in base_results])
    ba = np.array([r["actual"] for r in base_results])
    base_ic, _ = spearmanr(bp, ba)
    base_sharpe = compute_sharpe(bp, ba)
    logger.info(f"Baseline (MSE): RankIC={base_ic:.4f}, Sharpe={base_sharpe:.4f}")

    # GRPO training
    logger.info("\nGRPO training (20 steps)...")
    t0 = time.time()
    grpo_rewards = []

    for step in range(20):
        reward = grpo_step(model, data["train"], group_size=8, top_k=20, lr=1e-4)
        grpo_rewards.append(reward)
        if step % 5 == 0:
            logger.info(f"  Step {step+1}/20: reward={reward:.4f}")

    elapsed = time.time() - t0
    logger.info(f"GRPO complete: {elapsed:.0f}s")

    # Evaluate GRPO model
    model.eval()
    grpo_results = []
    with torch.no_grad():
        for sym in syms:
            df = data["test"][sym]
            vals = df[FEATURES].values.astype(np.float32)
            for idx in range(180, len(vals) - 10, max(1, (len(vals) - 190) // 8)):
                x = vals[idx - 180 : idx]
                mu, std = x.mean(0), x.std(0) + 1e-5
                pred = model(torch.FloatTensor((x - mu) / std).unsqueeze(0).to(DEVICE)).item()
                actual = (vals[idx + 9, 3] - vals[idx - 1, 3]) / (vals[idx - 1, 3] + 1e-5)
                grpo_results.append({"pred": pred, "actual": actual})

    gp = np.array([r["pred"] for r in grpo_results])
    ga = np.array([r["actual"] for r in grpo_results])
    grpo_ic, _ = spearmanr(gp, ga)
    grpo_sharpe = compute_sharpe(gp, ga)

    logger.info(f"\n{'='*50}")
    logger.info(f"GRPO vs Baseline Comparison")
    logger.info(f"{'='*50}")
    logger.info(f"{'Metric':<15} {'Baseline(MSE)':>15} {'GRPO':>15} {'Change':>15}")
    logger.info(f"-" * 60)
    logger.info(f"{'RankIC':<15} {base_ic:>15.4f} {grpo_ic:>15.4f} {grpo_ic-base_ic:>+15.4f}")
    logger.info(f"{'Sharpe':<15} {base_sharpe:>15.4f} {grpo_sharpe:>15.4f} {grpo_sharpe-base_sharpe:>+15.4f}")
    logger.info(f"{'Reward mean':<15} {'-':>15} {np.mean(grpo_rewards):>15.4f} {'-':>15}")

    # Save
    torch.save(model.state_dict(), "outputs/grpo_model.pt")
    logger.info("Saved: outputs/grpo_model.pt")

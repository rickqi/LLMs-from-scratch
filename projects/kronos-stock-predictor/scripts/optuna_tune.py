"""
Optuna 超参优化 for Kronos LSTM — 替代手动扫描

用法: python scripts/optuna_tune.py --trials 30
"""

import pickle, torch, numpy as np, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, StockDataset, FEATURES
from torch.utils.data import DataLoader
import optuna

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load data once
data = {m: pickle.load(open(f"data/processed_real/{m}_data.pkl", "rb")) for m in ["train", "val"]}


def objective(trial):
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    layers = trial.suggest_int("layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    ds_train = StockDataset(data["train"], 180, 10, max_samples=5000)
    ds_val = StockDataset(data["val"], 180, 10, max_samples=1000)
    tl = DataLoader(ds_train, batch_size, True, drop_last=True)
    vl = DataLoader(ds_val, batch_size)

    model = LSTMModel(hidden=hidden, num_layers=layers, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")

    for e in range(5):  # fewer epochs for tuning speed
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        vloss = sum(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item() for x, y in vl) / len(vl)
        if vloss < best_val:
            best_val = vloss

    return best_val


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="kronos_lstm")
    args = parser.parse_args()

    logger.info(f"Optuna tuning ({args.trials} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=f"sqlite:///outputs/optuna_{args.study_name}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)

    elapsed = time.time() - t0
    logger.info(f"Tuning complete: {elapsed:.0f}s, {len(study.trials)} trials")
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best val: {study.best_value:.6f}")

    # Save results
    import json
    results = {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "elapsed": elapsed,
    }
    with open("outputs/optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: outputs/optuna_results.json")

"""
LSTM 基线模型 — 直接价格预测，跳过 BSQ tokenizer

与 Kronos 对比，验证 BSQ 离散化是否是 RankIC 瓶颈。
"""

import pickle, logging, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURES = ["open", "high", "low", "close", "vol", "amt"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StockPriceDataset(Dataset):
    def __init__(self, data, lookback=180, pred_len=3):
        self.samples = []
        for sym, df in data.items():
            values = df[FEATURES].values.astype(np.float32)
            for i in range(len(values) - lookback - pred_len):
                x = values[i:i + lookback]
                # Z-score normalize
                mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
                x_norm = (x - mean) / std
                # Target: future close return
                future_close = values[i + lookback + pred_len - 1, 3]  # close
                current_close = values[i + lookback - 1, 3]
                y = (future_close - current_close) / (current_close + 1e-5)
                if abs(y) < 0.2:  # filter extreme moves
                    self.samples.append((x_norm, y))
        self.n = min(len(self.samples), 20000)  # cap for speed

    def __len__(self): return self.n

    def __getitem__(self, i):
        x, y = self.samples[i % len(self.samples)]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_lstm(train_data, val_data, epochs=10, batch_size=64):
    train_ds = StockPriceDataset(train_data)
    val_ds = StockPriceDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LSTMPredictor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")

    for ep in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += nn.MSELoss()(model(x), y).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "/tmp/lstm_best.pt")

        if ep % 2 == 0:
            logger.info(f"  Epoch {ep+1}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

    model.load_state_dict(torch.load("/tmp/lstm_best.pt"))
    return model


def evaluate_rankic(model, test_data, lookback=180, pred_len=3, n_stocks=30):
    from collections import defaultdict
    model.eval()
    symbols = sorted(test_data.keys())[:n_stocks]
    results = []

    with torch.no_grad():
        for sym in symbols:
            df = test_data[sym]
            values = df[FEATURES].values.astype(np.float32)
            for idx in range(lookback, len(values) - pred_len, len(values) // 5):
                x = values[idx - lookback:idx]
                mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
                x_norm = (x - mean) / std
                pred = model(torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)).item()

                future_close = values[idx + pred_len - 1, 3]
                current_close = values[idx - 1, 3]
                actual = (future_close - current_close) / (current_close + 1e-5)
                results.append({"pred": pred, "actual": actual, "sym": sym})

    preds = np.array([r["pred"] for r in results])
    actuals = np.array([r["actual"] for r in results])
    ic, p = spearmanr(preds, actuals)
    return ic, p, len(results)


if __name__ == "__main__":
    DATA_DIR = "./data/processed_real"
    for mode in ["train", "val", "test"]:
        with open(f"{DATA_DIR}/{mode}_data.pkl", "rb") as f:
            globals()[f"{mode}_data"] = pickle.load(f)

    logger.info(f"Train: {len(train_data)} stocks, Val: {len(val_data)}, Test: {len(test_data)}")

    model = train_lstm(train_data, val_data, epochs=15, batch_size=64)
    ic, pval, n = evaluate_rankic(model, test_data)
    logger.info(f"\n{'='*50}")
    logger.info(f"LSTM Baseline Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Samples: {n}")
    logger.info(f"  RankIC:  {ic:.4f} (p={pval:.4f})")
    logger.info(f"  vs Kronos mini: RankIC=-0.016")

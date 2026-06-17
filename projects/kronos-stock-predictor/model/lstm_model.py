"""
LSTM 价格预测模型 — 直接回归，作为 Kronos BSQ+Transformer 的 A/B 对照

用法:
    from model.lstm_model import LSTMPredictor, train_lstm, evaluate_lstm
"""

import torch, torch.nn as nn, numpy as np, pickle, logging
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

FEATURES = ["open", "high", "low", "close", "vol", "amt"]


class StockDataset(Dataset):
    """LSTM 滑动窗口数据集 — 直接回归未来收益率"""
    def __init__(self, data: dict, lookback: int = 180, pred_len: int = 3, max_samples: int = 20000):
        self.samples = []
        for sym, df in data.items():
            vals = df[FEATURES].values.astype(np.float32)
            n = len(vals)
            for i in range(n - lookback - pred_len):
                x = vals[i:i + lookback]
                mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
                x_norm = (x - mean) / std
                y = (vals[i + lookback + pred_len - 1, 3] - vals[i + lookback - 1, 3]) / (vals[i + lookback - 1, 3] + 1e-5)
                if abs(y) < 0.2:
                    self.samples.append((x_norm, y))
        self.n = min(len(self.samples), max_samples)

    def __len__(self): return self.n

    def __getitem__(self, i):
        x, y = self.samples[i % len(self.samples)]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMModel(nn.Module):
    """双层 LSTM + MLP head"""
    def __init__(self, input_dim: int = 6, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_lstm(train_data: dict, val_data: dict, model=None,
               lookback: int = 180, pred_len: int = 3,
               epochs: int = 20, batch_size: int = 64, lr: float = 1e-3,
               device: str = "cuda:0") -> LSTMModel:
    """训练 LSTM 模型"""
    if model is None:
        model = LSTMModel().to(device)

    train_ds = StockDataset(train_data, lookback, pred_len)
    val_ds = StockDataset(val_data, lookback, pred_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
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
                x, y = x.to(device), y.to(device)
                val_loss += nn.MSELoss()(model(x), y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate_lstm(model: LSTMModel, test_data: dict,
                  lookback: int = 180, pred_len: int = 3,
                  device: str = "cuda:0", n_stocks: int = 30) -> dict:
    """评估 LSTM 模型，返回 RankIC + 方向准确率"""
    model.eval()
    symbols = sorted(test_data.keys())[:n_stocks]
    results = []

    with torch.no_grad():
        for sym in symbols:
            df = test_data[sym]
            vals = df[FEATURES].values.astype(np.float32)
            step = max(1, (len(vals) - lookback - pred_len) // 10)
            for idx in range(lookback, len(vals) - pred_len, step):
                x = vals[idx - lookback:idx]
                mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
                x_norm = (x - mean) / std
                pred = model(torch.FloatTensor(x_norm).unsqueeze(0).to(device)).item()
                actual = (vals[idx + pred_len - 1, 3] - vals[idx - 1, 3]) / (vals[idx - 1, 3] + 1e-5)
                results.append({"pred": pred, "actual": actual})

    preds = np.array([r["pred"] for r in results])
    actuals = np.array([r["actual"] for r in results])
    ic, pval = spearmanr(preds, actuals)
    direction_acc = np.mean(np.sign(preds) == np.sign(actuals))

    return {
        "rankic": ic, "p_value": pval,
        "direction_accuracy": direction_acc,
        "n_samples": len(results),
        "pred_mean": float(preds.mean()), "pred_std": float(preds.std()),
        "actual_mean": float(actuals.mean()), "actual_std": float(actuals.std()),
    }

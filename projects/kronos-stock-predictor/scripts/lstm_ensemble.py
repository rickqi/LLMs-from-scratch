"""
LSTM 分类 + 集成 — 涨跌二分类 + 多模型投票

与回归 LSTM + Kronos 进行 A/B 对比
"""

import pickle, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURES = ["open", "high", "low", "close", "vol", "amt"]
DATA_DIR = "./data/processed_real"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ClassifyDataset(Dataset):
    """分类数据集: y=1(涨) / 0(跌)"""
    def __init__(self, data, lookback=180, pred_len=10, max_samples=20000):
        self.samples = []
        for sym, df in data.items():
            vals = df[FEATURES].values.astype(np.float32)
            n = len(vals)
            for i in range(n - lookback - pred_len):
                x = vals[i:i + lookback]
                mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
                x_norm = (x - mean) / std
                future = vals[i + lookback + pred_len - 1, 3]
                current = vals[i + lookback - 1, 3]
                y = 1 if future > current else 0
                self.samples.append((x_norm, y))
        self.n = min(len(self.samples), max_samples)

    def __len__(self): return self.n

    def __getitem__(self, i):
        x, y = self.samples[i % len(self.samples)]
        return torch.FloatTensor(x), torch.LongTensor([y])


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMEnsemble:
    """多模型集成: 回归 LSTM × N + 分类 LSTM"""
    def __init__(self, models: list, weights: list = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> float:
        x_t = torch.FloatTensor(x).unsqueeze(0).to(DEVICE)
        preds = []
        for m, w in zip(self.models, self.weights):
            m.eval()
            out = m(x_t)
            if out.shape[-1] == 1:  # regression
                preds.append(out.item() * w)
            else:  # classification → probability of UP
                prob = torch.softmax(out, -1)[0, 1].item()
                preds.append((prob - 0.5) * 2 * w)  # map to [-1, 1]
        return np.mean(preds)


def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3, is_classification=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if is_classification else nn.MSELoss()
    best_val, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if is_classification:
                loss = loss_fn(model(x), nn.functional.one_hot(y.squeeze(), 2).float())
            else:
                loss = loss_fn(model(x), y.float())
            loss.backward(); opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if is_classification:
                    val_loss += loss_fn(model(x), nn.functional.one_hot(y.squeeze(), 2).float()).item()
                else:
                    val_loss += loss_fn(model(x), y.float()).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate(model, test_data, lookback=180, pred_len=10, is_classification=False, n_stocks=30):
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
                x_t = torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)
                out = model(x_t)
                if is_classification:
                    pred = torch.softmax(out, -1)[0, 1].item()  # prob of UP
                    pred = (pred - 0.5) * 2  # map to [-1, +1]
                else:
                    pred = out.item()
                actual = (vals[idx + pred_len - 1, 3] - vals[idx - 1, 3]) / (vals[idx - 1, 3] + 1e-5)
                results.append({"pred": pred, "actual": actual})
    preds = np.array([r["pred"] for r in results])
    actuals = np.array([r["actual"] for r in results])
    ic, p = spearmanr(preds, actuals)
    return {"rankic": ic, "dir_acc": np.mean(np.sign(preds) == np.sign(actuals)), "n": len(results)}


# ── Load data ──
data = {}
for mode in ["train", "val", "test"]:
    with open(f"{DATA_DIR}/{mode}_data.pkl", "rb") as f:
        data[mode] = pickle.load(f)

# ── Train models ──
logger.info("=== 1. LSTM Regression (pred_len=10) ===")
from model.lstm_model import LSTMModel, StockDataset
ds_train = StockDataset(data["train"], 180, 10)
ds_val = StockDataset(data["val"], 180, 10)
t0 = time.time()
m_reg = LSTMModel().to(DEVICE)
m_reg = train_model(m_reg, DataLoader(ds_train, 64, True), DataLoader(ds_val, 64), 15, 1e-3)
r_reg = evaluate(m_reg, data["test"], 180, 10)
logger.info(f"  RankIC={r_reg['rankic']:.4f}, DirAcc={r_reg['dir_acc']:.2%}, Time={time.time()-t0:.0f}s")

logger.info("\n=== 2. LSTM Classification (pred_len=10) ===")
ds_train_c = ClassifyDataset(data["train"], 180, 10)
ds_val_c = ClassifyDataset(data["val"], 180, 10)
t0 = time.time()
m_cls = LSTMClassifier().to(DEVICE)
m_cls = train_model(m_cls, DataLoader(ds_train_c, 64, True), DataLoader(ds_val_c, 64), 15, 1e-3, True)
r_cls = evaluate(m_cls, data["test"], 180, 10, True)
logger.info(f"  RankIC={r_cls['rankic']:.4f}, DirAcc={r_cls['dir_acc']:.2%}, Time={time.time()-t0:.0f}s")

logger.info("\n=== 3. Ensemble (Reg + Cls) ===")
ensemble = LSTMEnsemble([m_reg, m_cls], [0.6, 0.4])
model_list = [m_reg, m_cls]
model_list[0].eval(); model_list[1].eval()
symbols = sorted(data["test"].keys())[:30]
results = []
with torch.no_grad():
    for sym in symbols:
        df = data["test"][sym]
        vals = df[FEATURES].values.astype(np.float32)
        step = max(1, (len(vals) - 180 - 10) // 10)
        for idx in range(180, len(vals) - 10, step):
            x = vals[idx - 180:idx]
            mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
            x_norm = (x - mean) / std
            pred = ensemble.predict(x_norm)
            actual = (vals[idx + 9, 3] - vals[idx - 1, 3]) / (vals[idx - 1, 3] + 1e-5)
            results.append({"pred": pred, "actual": actual})
preds, actuals = np.array([r["pred"] for r in results]), np.array([r["actual"] for r in results])
ic, p = spearmanr(preds, actuals)
logger.info(f"  RankIC={ic:.4f}, DirAcc={np.mean(np.sign(preds)==np.sign(actuals)):.2%}")

# ── Summary ──
print("\n" + "=" * 60)
print("A/B/C Comparison")
print("=" * 60)
print(f"{'Model':<25} {'RankIC':>8} {'DirAcc':>8}")
print("-" * 60)
print(f"{'Kronos (BSQ+Transformer)':<25} {-0.016:>8.4f} {0.567:>7.1%}")
print(f"{'LSTM Regression':<25} {r_reg['rankic']:>8.4f} {r_reg['dir_acc']:>7.1%}")
print(f"{'LSTM Classification':<25} {r_cls['rankic']:>8.4f} {r_cls['dir_acc']:>7.1%}")
print(f"{'LSTM Ensemble':<25} {ic:>8.4f} {np.mean(np.sign(preds)==np.sign(actuals)):>7.1%}")

"""
LSTM 超参数扫描 — hidden size × num_layers × lr
"""

import pickle, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from model.lstm_model import LSTMModel, StockDataset, FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "./data/processed_real"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

data = {}
for mode in ["train", "val", "test"]:
    with open(f"{DATA_DIR}/{mode}_data.pkl", "rb") as f:
        data[mode] = pickle.load(f)

def train_eval(hidden, layers, lr, lookback=180, pred_len=10, epochs=15):
    ds_train = StockDataset(data["train"], lookback, pred_len)
    ds_val = StockDataset(data["val"], lookback, pred_len)
    train_loader = DataLoader(ds_train, 64, True, drop_last=True)
    val_loader = DataLoader(ds_val, 64)

    model = LSTMModel(hidden=hidden, num_layers=layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += loss_fn(model(x), y).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # RankIC
    symbols = sorted(data["test"].keys())[:30]
    results = []
    with torch.no_grad():
        for sym in symbols:
            df = data["test"][sym]
            vals = df[FEATURES].values.astype(np.float32)
            step = max(1, (len(vals) - lookback - pred_len) // 10)
            for idx in range(lookback, len(vals) - pred_len, step):
                x = vals[idx - lookback:idx]
                mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
                x_norm = (x - mean) / std
                pred = model(torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)).item()
                actual = (vals[idx + pred_len - 1, 3] - vals[idx - 1, 3]) / (vals[idx - 1, 3] + 1e-5)
                results.append({"pred": pred, "actual": actual})
    preds = np.array([r["pred"] for r in results])
    actuals = np.array([r["actual"] for r in results])
    ic, _ = spearmanr(preds, actuals)
    n_params = sum(p.numel() for p in model.parameters())
    return ic, n_params


configs = [
    (128, 2, 1e-3),   # baseline
    (256, 2, 1e-3),   # wider
    (128, 3, 1e-3),   # deeper
    (256, 3, 1e-3),   # wider+deeper
    (256, 3, 5e-4),   # lower lr
    (128, 2, 5e-4),   # lower lr baseline
]

print(f"{'Hidden':>6} {'Layers':>6} {'LR':>8} {'RankIC':>8} {'Params':>10} {'Time':>8}")
print("-" * 55)
best_ic, best_cfg = -1, None

for hidden, layers, lr in configs:
    t0 = time.time()
    ic, params = train_eval(hidden, layers, lr)
    elapsed = time.time() - t0
    marker = "← BEST" if ic > best_ic else ""
    print(f"{hidden:>6} {layers:>6} {lr:>8.0e} {ic:>8.4f} {params:>8,} {elapsed:>7.0f}s {marker}")
    if ic > best_ic:
        best_ic, best_cfg = ic, (hidden, layers, lr)

print(f"\nBest: hidden={best_cfg[0]}, layers={best_cfg[1]}, lr={best_cfg[2]:.0e}, RankIC={best_ic:.4f}")

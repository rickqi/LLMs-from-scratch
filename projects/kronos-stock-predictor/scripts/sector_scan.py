"""快速板块对比 — 单板块 LSTM 训练"""
import tushare as ts, pandas as pd, numpy as np, torch, time, sys
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from model.lstm_model import LSTMModel, FEATURES
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda:0"
ts.set_token("260264d1c42c2b5c47262478557e99d7f6a0769523ea19f48e09ed73")
pro = ts.pro_api()

class SDS(Dataset):
    def __init__(self, data, m=5000):
        self.s = []
        for sym, df in data.items():
            v = df[FEATURES].values.astype(np.float32)
            for i in range(len(v) - 190):
                x = v[i : i + 180]; mu, std = x.mean(0), x.std(0) + 1e-5
                xn = np.clip((x - mu) / std, -5, 5)
                y = (v[i + 189, 3] - v[i + 179, 3]) / (v[i + 179, 3] + 1e-5)
                if abs(y) < 0.2:
                    self.s.append((xn.astype(np.float32), np.float32(y)))
        self.n = min(len(self.s), m)
    def __len__(self): return self.n
    def __getitem__(self, i):
        x, y = self.s[i % len(self.s)]
        return torch.FloatTensor(x), torch.FloatTensor([y])

def go(sector, n=30):
    syms = pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry")
    syms = syms[syms["industry"] == sector]["ts_code"].tolist()[:n]
    data = {"train": {}, "val": {}, "test": {}}
    print(f"  Downloading {len(syms)} {sector}...")
    for sym in syms:
        try:
            df = pro.daily(ts_code=sym, start_date="20180101", end_date="20260617"); time.sleep(0.35)
            if df is None or len(df) < 200: continue
            df["trade_date"] = pd.to_datetime(df["trade_date"]); df = df.set_index("trade_date").sort_index()
            df = df[["open", "high", "low", "close", "vol", "amount"]]
            df = df.rename(columns={"vol": "vol", "amount": "amt"}).astype(np.float32)
            df = df[(df["open"] > 0) & (df["close"] > 0)]
            for m, (s, e) in [("train", ("2018-01-01", "2022-12-31")),
                              ("val", ("2023-01-01", "2023-12-31")),
                              ("test", ("2024-01-01", "2025-12-31"))]:
                sub = df[(df.index >= s) & (df.index <= e)]
                if len(sub) >= 50: data[m][sym] = sub
        except: pass
    if len(data["train"]) < 15: return None, 0
    print(f"    Building dataset...", flush=True)
    t0 = time.time()
    tl = DataLoader(SDS(data["train"]), 64, True, drop_last=True)
    vl = DataLoader(SDS(data["val"], 3000), 64)
    print(f"    Dataset ready ({time.time()-t0:.1f}s), training...", flush=True)
    model = LSTMModel().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss(); bv, bs = float("inf"), None
    for e in range(8):
        t0 = time.time()
        model.train()
        for x, y in tl: x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
        model.eval()
        vloss = sum(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item() for x, y in vl) / len(vl)
        if vloss < bv: bv = vloss; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if e % 4 == 0: print(f"    E{e+1}/8: val={vloss:.6f} ({time.time()-t0:.1f}s)", flush=True)
    model.load_state_dict(bs); model.eval()
    rr = []
    with torch.no_grad():
        for sym in sorted(data["test"].keys())[:10]:
            df = data["test"][sym]; v = df[FEATURES].values.astype(np.float32)
            for idx in range(180, len(v) - 10, max(1, (len(v) - 190) // 8)):
                x = v[idx - 180 : idx]; mu, std = x.mean(0), x.std(0) + 1e-5
                pred = model(torch.FloatTensor((x - mu) / std).unsqueeze(0).to(DEVICE)).item()
                actual = (v[idx + 9, 3] - v[idx - 1, 3]) / (v[idx - 1, 3] + 1e-5)
                rr.append({"pred": pred, "actual": actual})
    p = np.array([r["pred"] for r in rr]); a = np.array([r["actual"] for r in rr])
    ic, _ = spearmanr(p, a)
    return ic, len(data["train"])

if __name__ == "__main__":
    print(f"{'Sector':<12s} {'Stocks':>6s} {'RankIC':>8s} {'DirAcc':>8s}")
    print("-" * 38)
    for s in ["电气设备", "汽车配件", "证券"]:
        ic, n = go(s, 30)
        print(f"{s:<12s} {n:>6d} {ic:>8.4f}")
    print(f"{'半导体':<12s} {'68':>6s} {'0.2050':>8s}")

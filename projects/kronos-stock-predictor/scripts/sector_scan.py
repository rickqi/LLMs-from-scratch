"""
分行业波动率扫描 — 识别最优板块
用法: python scripts/sector_scan.py --sector 金融 --epochs 20
"""

import tushare as ts
import torch, pickle, numpy as np, logging, sys, time, json, argparse, pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, train_lstm, FEATURES
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

_TOKEN = open(Path(__file__).resolve().parent.parent/".env").read().split("TUSHARE_TOKEN=")[1].split("\n")[0].strip()
ts.set_token(_TOKEN); pro = ts.pro_api()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TODAY = datetime.now().strftime("%Y%m%d")
OUTPUT = Path("outputs/sector_scan"); OUTPUT.mkdir(exist_ok=True)

SECTORS = {
    "金融":     ["保险","银行","证券"],    "半导体":   ["半导体","芯片"],
    "医药":     ["医药","医疗"],           "消费":     ["白酒","食品","饮料","家电"],
    "汽车":     ["汽车"],                  "地产":     ["地产","房产"],
    "软件":     ["软件","互联网"],          "通信":     ["通信"],
    "电力":     ["电力"],                  "化工":     ["化工"],
}

def download_sector(symbols, start="20180101"):
    data = {}
    for i, ts_code in enumerate(symbols):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=TODAY,
                          fields="trade_date,open,high,low,close,vol,amount")
            if df is not None and len(df) >= 500:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").set_index("trade_date")
                data[ts_code] = df[["open","high","low","close","vol","amount"]].rename(columns={"amount":"amt"})
            if i%20==0 and i>0: time.sleep(3)
            time.sleep(0.2)
        except: pass
    return data

def walk_forward(data, epochs=20):
    quarters = [(f'{y}-{m:02d}-01',) for y in [2025,2026] for m in [1,4,7,10]][:6]
    results = []
    for (q_start,) in quarters:
        qs = pd.Timestamp(q_start); qe = qs + pd.DateOffset(months=3) - pd.DateOffset(days=1)
        train_d, test_d = {}, {}
        for sym, df in data.items():
            t = df[df.index < qs]; ts = df[(df.index >= qs) & (df.index <= qe)]
            if len(t) >= 300: train_d[sym] = t
            if len(ts) >= 5: test_d[sym] = ts
        if len(train_d) < 10 or len(test_d) < 10: continue
        try:
            val_sym = list(train_d.keys())[0]
            model = train_lstm(train_d, {val_sym: train_d[val_sym]}, lookback=180, pred_len=3,
                              epochs=epochs, batch_size=128, lr=1e-3, device=DEVICE)
        except: continue
        preds, actuals = [], []
        for sym, df in test_d.items():
            hist = data[sym][data[sym].index <= qe]
            if len(hist) < 180: continue
            vals = hist[FEATURES].values.astype(np.float32)
            x = vals[-180:]; m, s = x.mean(axis=0), x.std(axis=0)+1e-5
            with torch.no_grad():
                pred = model(torch.FloatTensor((x-m)/s).unsqueeze(0).to(DEVICE)).item()
            qv = df[FEATURES].values.astype(np.float32)
            if len(qv) < 5: continue
            rets = np.diff(qv[:,3])/(qv[:-1,3]+1e-5)
            actuals.append(float(np.std(rets))); preds.append(pred)
        if len(preds) >= 10:
            ic, _ = spearmanr(preds, actuals)
            results.append({'quarter': q_start[:7], 'rankic': float(ic), 'n': len(preds)})
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sector", default=None); p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--skip-download", action="store_true"); p.add_argument("--all", action="store_true")
    args = p.parse_args()

    to_scan = SECTORS if (args.all or not args.sector) else {args.sector: SECTORS[args.sector]}
    summary = {}

    for name, keywords in to_scan.items():
        logger.info(f"\n=== {name} ({keywords}) ===")
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        pattern = '|'.join(keywords)
        symbols = sorted(df[df['industry'].str.contains(pattern, na=False)]['ts_code'].tolist())
        logger.info(f"Stocks: {len(symbols)}")

        if args.skip_download:
            data = pickle.load(open(OUTPUT/f"{name}_data.pkl","rb"))
        else:
            data = download_sector(symbols)
            pickle.dump(data, open(OUTPUT/f"{name}_data.pkl","wb"))
        logger.info(f"Data: {len(data)} stocks, {sum(len(d) for d in data.values()):,} rows")

        if len(data) < 10: continue

        results = walk_forward(data, args.epochs)
        ics = [r['rankic'] for r in results]
        summary[name] = {'stocks': len(data), 'rows': sum(len(d) for d in data.values()),
                        'mean_ic': float(np.mean(ics)) if ics else 0,
                        'pos': f"{sum(1 for ic in ics if ic>0)}/{len(results)}",
                        'details': results}
        logger.info(f"Mean IC: {summary[name]['mean_ic']:+.4f} ({summary[name]['pos']})")

    print(f"\n{'='*65}")
    print(f"  Sector Scan Summary")
    print(f"{'='*65}")
    for name, r in sorted(summary.items(), key=lambda x: -x[1]['mean_ic']):
        print(f"  {name:<10} {r['stocks']:<6}stocks  {r['rows']:>8,}rows  RankIC={r['mean_ic']:+.4f}  {r['pos']}")
    print(f"{'='*65}")

    with open(OUTPUT/"summary.json","w") as f: json.dump(summary, f, indent=2, default=str)

if __name__ == "__main__": main()

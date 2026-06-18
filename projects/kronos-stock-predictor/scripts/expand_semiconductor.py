"""
扩展半导体板块数据 — Tushare 下载全行业半导体股票日线

用法:
  python scripts/expand_semiconductor.py
  python scripts/expand_semiconductor.py --output data/semiconductor_all
"""

import tushare as ts
import pandas as pd
import pickle, logging, sys, argparse, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

# Tushare token
_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
if not _TOKEN:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    _TOKEN = os.environ.get("TUSHARE_TOKEN", "")

ts.set_token(_TOKEN)
pro = ts.pro_api()

FEATURES = ["open", "high", "low", "close", "vol", "amount"]
TODAY = datetime.now().strftime("%Y%m%d")


def get_semiconductor_symbols() -> list[str]:
    """获取 A 股半导体板块所有股票 (通过 stock_basic 行业筛选)"""
    try:
        df = pro.stock_basic(exchange='', list_status='L',
                            fields='ts_code,name,industry')
        semi = df[df['industry'].str.contains('半导体|芯片|集成电路', na=False)]
        result = sorted(semi['ts_code'].tolist())
        logger.info(f"半导体行业: {len(result)} 只")
        for _, row in semi.head(5).iterrows():
            logger.info(f"  {row['ts_code']} {row['name']}")
        return result
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return []


def download_stock(ts_code: str, start_date: str = "20180101") -> pd.DataFrame | None:
    """下载单只股票日线数据"""
    import time
    for attempt in range(3):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=TODAY,
                           fields="trade_date,open,high,low,close,vol,amount")
            if df is None or len(df) == 0:
                return None

            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").set_index("trade_date")
            return df[["open", "high", "low", "close", "vol", "amount"]]

        except Exception as e:
            msg = str(e)
            if "每分钟最多访问" in msg or "200次" in msg:
                logger.debug(f"  限频, 等待 60s...")
                time.sleep(60)
            elif attempt < 2:
                time.sleep(1)
            else:
                logger.debug(f"  {ts_code}: {e}")
                return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/semiconductor_all")
    parser.add_argument("--start", default="20180101", help="起始日期")
    parser.add_argument("--min-days", type=int, default=500, help="最少交易日")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取股票列表
    symbols = get_semiconductor_symbols()
    if not symbols:
        logger.error("未获取到半导体股票列表")
        return

    # 下载
    import time
    stock_data = {}
    failed = []
    for i, ts_code in enumerate(symbols):
        if i % 20 == 0 and i > 0:
            time.sleep(5)  # 每 20 只休息 5 秒，避免限频
        time.sleep(0.3)    # 每次请求间隔 0.3s
        logger.info(f"[{i+1}/{len(symbols)}] {ts_code}...")
        df = download_stock(ts_code, args.start)
        if df is not None and len(df) >= args.min_days:
            stock_data[ts_code] = df
            logger.info(f"  ✅ {len(df)} days ({df.index[0].date()} ~ {df.index[-1].date()})")
        else:
            failed.append(ts_code)
            logger.warning(f"  ❌ {'无数据' if df is None else f'仅{len(df)}天'}")

    logger.info(f"\n成功: {len(stock_data)} 只, 失败: {len(failed)} 只")

    # 保存
    with open(output_dir / "all_stocks.pkl", "wb") as f:
        pickle.dump(stock_data, f)

    # 统计
    symbols_list = sorted(stock_data.keys())
    total_rows = sum(len(df) for df in stock_data.values())
    date_ranges = [(s, df.index[0], df.index[-1], len(df)) for s, df in stock_data.items()]
    avg_days = total_rows / max(len(stock_data), 1)

    summary = {
        "n_symbols": len(stock_data),
        "total_rows": total_rows,
        "avg_days": round(avg_days, 1),
        "date_range": f"{min(r[1] for r in date_ranges).date()} ~ {max(r[2] for r in date_ranges).date()}",
        "failed": failed,
        "symbols": symbols_list,
    }

    with open(output_dir / "summary.json", "w") as f:
        import json
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    logger.info(f"\n保存: {output_dir}/all_stocks.pkl")
    logger.info(f"统计: {len(stock_data)} 只, {total_rows:,} 行, 均 {avg_days:.0f} 天")
    logger.info(f"时间: {summary['date_range']}")


if __name__ == "__main__":
    main()

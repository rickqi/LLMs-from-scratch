"""
扩展半导体板块数据 — 断点续传 + 数据隔离

功能:
  1. 从 Tushare 下载 194 只半导体股票日线
  2. 每只股票单独保存为 .pkl，支持断点续传
  3. 已完成股票自动跳过，中断后重跑不重复下载
  4. 数据与原有 processed_real 完全隔离

输出结构:
  data/semiconductor_v2/
    raw/                    # 原始下载数据 (每只股票一个 .pkl)
      002049.SZ.pkl
      002077.SZ.pkl
      ...
    processed/              # 预处理后数据 (train/val/test split)
      train_data.pkl
      val_data.pkl
      test_data.pkl
    checkpoint.json         # 断点续传状态
    summary.json            # 统计信息

用法:
  python scripts/expand_semiconductor.py                         # 断点续传下载
  python scripts/expand_semiconductor.py --preprocess             # 下载+预处理
  python scripts/expand_semiconductor.py --retrain                # 下载+预处理+训练
"""

import tushare as ts
import pandas as pd
import pickle, logging, sys, argparse, os, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

# ── Tushare 配置 ──
_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
if not _TOKEN:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    _TOKEN = os.environ.get("TUSHARE_TOKEN", "")
ts.set_token(_TOKEN)
pro = ts.pro_api()

TODAY = datetime.now().strftime("%Y%m%d")
FEATURES = ["open", "high", "low", "close", "vol", "amount"]

# ── 输出路径 (与原始数据隔离) ──
OUTPUT_BASE = Path("data/semiconductor_v2")
RAW_DIR = OUTPUT_BASE / "raw"
CHECKPOINT_FILE = OUTPUT_BASE / "checkpoint.json"
SUMMARY_FILE = OUTPUT_BASE / "summary.json"


# ═══════════════════════════════════════════════════════════════
# 股票列表
# ═══════════════════════════════════════════════════════════════

def get_semiconductor_symbols() -> list[str]:
    """通过 Tushare stock_basic 行业筛选获取半导体板块"""
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    semi = df[df['industry'].str.contains('半导体|芯片|集成电路', na=False)]
    result = sorted(semi['ts_code'].tolist())
    logger.info(f"半导体行业: {len(result)} 只")
    return result


# ═══════════════════════════════════════════════════════════════
# 断点续传下载
# ═══════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    """加载断点续传状态"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "total": 0, "started_at": "", "updated_at": ""}


def save_checkpoint(completed: list, failed: list):
    """保存断点续传状态"""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    ckpt = {
        "completed": sorted(completed),
        "failed": sorted(failed),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "started_at": load_checkpoint()["started_at"] or now,
        "updated_at": now,
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2, ensure_ascii=False)


def download_stock(ts_code: str, start_date: str) -> pd.DataFrame | None:
    """下载单只股票，带重试和限频处理"""
    for attempt in range(3):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=TODAY,
                           fields="trade_date,open,high,low,close,vol,amount")
            if df is None or len(df) == 0:
                return None
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").set_index("trade_date")
            return df[FEATURES]
        except Exception as e:
            msg = str(e)
            if "每分钟最多访问" in msg or "200次" in msg:
                logger.info("  触发限频，等待 65s...")
                time.sleep(65)
            elif attempt < 2:
                time.sleep(2)
            else:
                logger.debug(f"  {ts_code} 失败: {e}")
                return None
    return None


def cmd_download(start_date: str = "20180101", min_days: int = 300):
    """断点续传下载所有半导体股票"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    symbols = get_semiconductor_symbols()
    if not symbols:
        logger.error("未获取到股票列表")
        return

    ckpt = load_checkpoint()
    completed = set(ckpt["completed"])
    failed = set(ckpt["failed"])

    # 检查已有文件 (比 checkpoint 更可靠)
    for f in RAW_DIR.glob("*.pkl"):
        completed.add(f.stem)

    pending = [s for s in symbols if s not in completed and s not in failed]
    if not pending:
        logger.info(f"全部完成: {len(completed)} 只")
        return

    logger.info(f"待下载: {len(pending)} 只 (已完成 {len(completed)}, 失败 {len(failed)})")

    ok = set()
    err = set()
    for i, ts_code in enumerate(pending):
        if i % 20 == 0 and i > 0:
            time.sleep(5)
        time.sleep(0.3)

        logger.info(f"[{len(completed)+len(ok)+1}/{len(symbols)}] {ts_code}...")
        df = download_stock(ts_code, start_date)

        if df is not None and len(df) >= min_days:
            with open(RAW_DIR / f"{ts_code}.pkl", "wb") as f:
                pickle.dump(df, f)
            ok.add(ts_code)
            logger.info(f"  ✅ {len(df)}d ({df.index[0].date()} ~ {df.index[-1].date()})")
        else:
            err.add(ts_code)
            logger.warning(f"  ❌ {'无数据' if df is None else f'仅{len(df)}d'}")

        # 每 5 只保存一次断点
        if (i + 1) % 5 == 0:
            save_checkpoint(list(completed | ok), list(failed | err))

    # 最终保存
    all_completed = completed | ok
    all_failed = failed | err
    save_checkpoint(list(all_completed), list(all_failed))

    # 汇总
    generate_summary(list(all_completed))
    logger.info(f"\n✅ 完成: {len(all_completed)} 只, ❌ 失败: {len(all_failed)} 只")


def generate_summary(symbols: list[str]):
    """生成统计摘要"""
    total_rows = 0
    date_ranges = []
    for ts_code in symbols:
        fpath = RAW_DIR / f"{ts_code}.pkl"
        if fpath.exists():
            df = pickle.load(open(fpath, "rb"))
            total_rows += len(df)
            date_ranges.append((ts_code, df.index[0], df.index[-1], len(df)))

    if not date_ranges:
        return

    summary = {
        "n_symbols": len(symbols),
        "total_rows": total_rows,
        "avg_days": round(total_rows / len(symbols), 1) if symbols else 0,
        "date_range": f"{min(r[1] for r in date_ranges).date()} ~ {max(r[2] for r in date_ranges).date()}",
        "updated_at": datetime.now().isoformat(),
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"统计: {summary['n_symbols']} 只, {summary['total_rows']:,} 行")


# ═══════════════════════════════════════════════════════════════
# 数据预处理 + 隔离
# ═══════════════════════════════════════════════════════════════

def cmd_preprocess():
    """将 raw 数据分割为 train/val/test (时间序列严格划分)"""
    from data.preprocessor import preprocess_series

    raw_files = sorted(RAW_DIR.glob("*.pkl"))
    if not raw_files:
        logger.error(f"无原始数据, 请先运行 --download")
        return

    # 加载所有股票
    all_data = {}
    for f in raw_files:
        all_data[f.stem] = pickle.load(open(f, "rb"))

    logger.info(f"加载 {len(all_data)} 只, {sum(len(df) for df in all_data.values()):,} 行")

    # 时间划分 (扩展测试集到 2023 → 700天/35+调仓点)
    train_data, val_data, test_data = {}, {}, {}
    train_end = "2021-12-31"
    val_end = "2022-12-31"

    for sym, df in all_data.items():
        train = df[df.index <= train_end]
        val = df[(df.index > train_end) & (df.index <= val_end)]
        test = df[df.index > val_end]

        if len(train) >= 500:
            train_data[sym] = train
        if len(val) >= 100:
            val_data[sym] = val
        if len(test) >= 100:
            test_data[sym] = test

    processed_dir = OUTPUT_BASE / "processed"
    processed_dir.mkdir(exist_ok=True)

    with open(processed_dir / "train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(processed_dir / "val_data.pkl", "wb") as f:
        pickle.dump(val_data, f)
    with open(processed_dir / "test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    logger.info(f"保存: {processed_dir}/")
    logger.info(f"  train: {len(train_data)} 只 ({min(len(df) for df in train_data.values())}-{max(len(df) for df in train_data.values())} 天)")
    logger.info(f"  val:   {len(val_data)} 只")
    logger.info(f"  test:  {len(test_data)} 只")
    logger.info(f"  test 日期范围: 2025-01 ~ 2026-06 (18个月, 比原数据多 10个月!)")

    # 数据隔离对比
    logger.info(f"\n═══ 数据隔离对比 ═══")
    logger.info(f"  原始 (processed_real): train 68只 2018-2023, test 248天")
    logger.info(f"  新版 (semiconductor_v2): train {len(train_data)}只 2018-2023, test ~370天")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="半导体数据扩展 (断点续传)")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--min-days", type=int, default=300)
    parser.add_argument("--preprocess", action="store_true", help="下载后自动预处理")
    args = parser.parse_args()

    logger.info(f"输出目录: {OUTPUT_BASE} (与 processed_real 隔离)")
    cmd_download(args.start, args.min_days)

    if args.preprocess:
        cmd_preprocess()


if __name__ == "__main__":
    main()

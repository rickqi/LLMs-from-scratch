"""
StockDataDownloader — A 股行情数据下载器

主数据源: Tushare Pro API (需要 token)
备选数据源: akshare (无需 token，但数据字段有限)
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

# ── 可选依赖 ──
try:
    import tushare as ts

    _HAS_TUSHARE = True
except ImportError:
    ts = None  # type: ignore[assignment]
    _HAS_TUSHARE = False

try:
    import akshare as ak

    _HAS_AKSHARE = True
except ImportError:
    ak = None  # type: ignore[assignment]
    _HAS_AKSHARE = False

from data.symbols import tushare_to_qlib, qlib_to_tushare

logger = logging.getLogger(__name__)


class StockDataDownloader:
    """A 股行情数据下载器。

    Args:
        config: Config 对象，用于读取数据路径等配置。
    """

    # Tushare 标准输出列
    _TUSHARE_COLUMNS = [
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "adj_factor",
    ]

    def __init__(self, config):
        self.config = config
        self._tushare_token = os.environ.get("TUSHARE_TOKEN", "")
        self._pro = None

        if _HAS_TUSHARE and self._tushare_token:
            try:
                ts.set_token(self._tushare_token)
                self._pro = ts.pro_api()
                logger.info("Tushare Pro API initialized successfully.")
            except Exception as e:
                logger.warning("Tushare init failed: %s", e)
                self._pro = None
        else:
            if not _HAS_TUSHARE:
                logger.warning("tushare package not installed; Tushare source unavailable.")
            elif not self._tushare_token:
                logger.warning(
                    "TUSHARE_TOKEN env var not set; Tushare source unavailable. "
                    "Set it via: export TUSHARE_TOKEN=your_token"
                )

    # ------------------------------------------------------------------
    # Tushare 数据源
    # ------------------------------------------------------------------

    def download_tushare(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """通过 Tushare Pro 下载单只股票日线行情。

        Args:
            ts_code: Tushare 代码，如 '600036.SH'。
            start_date: 起始日期 'YYYYMMDD'。
            end_date: 结束日期 'YYYYMMDD'。

        Returns:
            DataFrame，列: trade_date, open, high, low, close, vol, amount, adj_factor。
            失败时返回空 DataFrame。
        """
        if self._pro is None:
            logger.warning("Tushare API not available; returning empty DataFrame.")
            return pd.DataFrame(columns=self._TUSHARE_COLUMNS)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # ── 日线行情 ──
                df_daily = self._pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )

                # ── 复权因子 ──
                df_adj = self._pro.adj_factor(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )

                if df_daily is None or df_daily.empty:
                    logger.warning("Tushare returned empty data for %s.", ts_code)
                    return pd.DataFrame(columns=self._TUSHARE_COLUMNS)

                # ── 合并 ──
                df = df_daily.merge(df_adj, on="ts_code", how="left")
                if "adj_factor" not in df.columns:
                    df["adj_factor"] = 1.0

                # ── 计算复权收盘价 ──
                df["adjclose"] = df["close"] * df["adj_factor"]

                # ── 选取标准列 ──
                col_map = {
                    "vol": "vol",
                    "amount": "amount",
                }
                df = df.rename(columns=col_map)

                # 确保所有目标列存在
                for col in self._TUSHARE_COLUMNS:
                    if col not in df.columns:
                        df[col] = np.nan

                df = df[self._TUSHARE_COLUMNS].copy()
                df = df.sort_values("trade_date").reset_index(drop=True)

                logger.info(
                    "Tushare: %s  %s~%s  %d rows",
                    ts_code,
                    start_date,
                    end_date,
                    len(df),
                )
                return df

            except Exception as e:
                logger.warning(
                    "Tushare download attempt %d/%d failed for %s: %s",
                    attempt,
                    max_retries,
                    ts_code,
                    e,
                )
                if attempt < max_retries:
                    time.sleep(1)

        logger.error("Tushare download failed after %d retries for %s.", max_retries, ts_code)
        return pd.DataFrame(columns=self._TUSHARE_COLUMNS)

    # ------------------------------------------------------------------
    # akshare 数据源 (备选)
    # ------------------------------------------------------------------

    def download_akshare(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """通过 akshare 下载单只股票日线行情 (备选数据源)。

        Args:
            symbol: 股票代码，如 '600036' 或 'sh600036'。
            start_date: 起始日期 'YYYYMMDD' 或 'YYYY-MM-DD'。
            end_date: 结束日期 'YYYYMMDD' 或 'YYYY-MM-DD'。

        Returns:
            DataFrame，列: trade_date, open, high, low, close, vol, amount, adj_factor。
            失败时返回空 DataFrame。
        """
        if not _HAS_AKSHARE:
            logger.warning("akshare package not installed; returning empty DataFrame.")
            return pd.DataFrame(columns=self._TUSHARE_COLUMNS)

        try:
            # ── 标准化代码: 去掉交易所后缀 ──
            code = symbol.split(".")[0] if "." in symbol else symbol
            # 去掉可能的前缀 (sh/sz)
            if code.startswith(("sh", "sz", "bj")):
                code = code[2:]

            # ── 标准化日期格式 ──
            s_date = start_date.replace("-", "")
            e_date = end_date.replace("-", "")

            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=s_date,
                end_date=e_date,
                adjust="qfq",
            )

            if df is None or df.empty:
                logger.warning("akshare returned empty data for %s.", symbol)
                return pd.DataFrame(columns=self._TUSHARE_COLUMNS)

            # ── 列名映射 ──
            akshare_col_map = {
                "日期": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "vol",
                "成交额": "amount",
            }
            df = df.rename(columns=akshare_col_map)

            # ── 补全缺失列 ──
            if "adj_factor" not in df.columns:
                df["adj_factor"] = 1.0

            # ── 日期格式统一 ──
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")

            # ── 选取标准列 ──
            for col in self._TUSHARE_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[self._TUSHARE_COLUMNS].copy()
            df = df.sort_values("trade_date").reset_index(drop=True)

            logger.info(
                "akshare: %s  %s~%s  %d rows",
                symbol,
                start_date,
                end_date,
                len(df),
            )
            return df

        except Exception as e:
            logger.error("akshare download failed for %s: %s", symbol, e)
            return pd.DataFrame(columns=self._TUSHARE_COLUMNS)

    # ------------------------------------------------------------------
    # 批量下载
    # ------------------------------------------------------------------

    def download_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        source: str = "tushare",
    ) -> dict[str, pd.DataFrame]:
        """批量下载多只股票行情。

        Args:
            symbols: 股票代码列表。
            start_date: 起始日期 'YYYYMMDD'。
            end_date: 结束日期 'YYYYMMDD'。
            source: 数据源 'tushare' 或 'akshare'。

        Returns:
            {ts_code: DataFrame} 字典，失败的股票不包含在结果中。
        """
        results: dict[str, pd.DataFrame] = {}
        total = len(symbols)

        for idx, sym in enumerate(symbols, 1):
            logger.info("Downloading [%d/%d] %s ...", idx, total, sym)
            try:
                if source == "tushare":
                    df = self.download_tushare(sym, start_date, end_date)
                elif source == "akshare":
                    df = self.download_akshare(sym, start_date, end_date)
                else:
                    logger.warning("Unknown source '%s', skipping %s.", source, sym)
                    continue

                if not df.empty:
                    results[sym] = df
                else:
                    logger.warning("Empty data for %s, skipping.", sym)

            except Exception as e:
                logger.error("Error downloading %s: %s", sym, e)

        logger.info(
            "Batch download complete: %d/%d symbols succeeded.",
            len(results),
            total,
        )
        return results

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def get_trade_calendar(
        self,
        start_date: str,
        end_date: str,
    ) -> list[str]:
        """获取交易日历。

        Args:
            start_date: 起始日期 'YYYYMMDD'。
            end_date: 结束日期 'YYYYMMDD'。

        Returns:
            交易日列表 ['YYYYMMDD', ...]。
        """
        if self._pro is None:
            logger.warning("Tushare API not available; returning empty calendar.")
            return []

        try:
            df_cal = self._pro.trade_cal(
                exchange="SSE",
                start_date=start_date,
                end_date=end_date,
            )
            if df_cal is None or df_cal.empty:
                return []

            # 筛选开市日
            trading_days = df_cal[df_cal["is_open"] == 1]["cal_date"].tolist()
            return trading_days

        except Exception as e:
            logger.error("Failed to get trade calendar: %s", e)
            return []

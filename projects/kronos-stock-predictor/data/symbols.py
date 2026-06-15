"""
股票池配置 — 参考 TradingAgents/hs300_tickers.txt + TradingAgents/default_config.py benchmark_map
"""

# 沪深300 核心成分股（精选代表）
CSI300_SYMBOLS = [
    # ── 金融 ──
    "600036.SH",  # 招商银行
    "601318.SH",  # 中国平安
    "601398.SH",  # 工商银行
    "600030.SH",  # 中信证券
    "601166.SH",  # 兴业银行
    "000001.SZ",  # 平安银行
    "601688.SH",  # 华泰证券
    "600016.SH",  # 民生银行
    # ── 消费 ──
    "600519.SH",  # 贵州茅台
    "000858.SZ",  # 五粮液
    "000568.SZ",  # 泸州老窖
    "002304.SZ",  # 洋河股份
    "600887.SH",  # 伊利股份
    "000333.SZ",  # 美的集团
    "000651.SZ",  # 格力电器
    # ── 科技 ──
    "002415.SZ",  # 海康威视
    "300750.SZ",  # 宁德时代
    "002475.SZ",  # 立讯精密
    "300059.SZ",  # 东方财富
    "688981.SH",  # 中芯国际
    # ── 医药 ──
    "600276.SH",  # 恒瑞医药
    "300760.SZ",  # 迈瑞医疗
    "000538.SZ",  # 云南白药
    # ── 能源/材料 ──
    "601857.SH",  # 中国石油
    "600028.SH",  # 中国石化
    "601899.SH",  # 紫金矿业
    "600585.SH",  # 海螺水泥
    # ── 地产/基建 ──
    "000002.SZ",  # 万科A
    "601668.SH",  # 中国建筑
    # ── 交通/公用 ──
    "601111.SH",  # 中国国航
    "600900.SH",  # 长江电力
]


def get_default_symbols() -> list:
    """获取默认股票池（可通过环境变量覆盖）"""
    import os
    symbols_env = os.environ.get("KRONOS_SYMBOLS", "")
    if symbols_env:
        return [s.strip() for s in symbols_env.split(",") if s.strip()]
    return CSI300_SYMBOLS


# ── 基准指数映射（参考 TradingAgents default_config.py:138-147） ──

BENCHMARK_MAP = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
    "csi800": "000906.SH",
    "csi1000": "000852.SH",
}

# ── 交易所后缀映射 ──

EXCHANGE_MAPPING = {
    "SH": "sh",
    "SZ": "sz",
    "BJ": "bj",
}


def tushare_to_qlib(ts_code: str) -> str:
    """600000.SH → sh600000"""
    parts = ts_code.split(".")
    if len(parts) != 2:
        return ts_code
    code, exchange = parts
    prefix = EXCHANGE_MAPPING.get(exchange, exchange.lower())
    return prefix + code


def qlib_to_tushare(sym: str) -> str:
    """sh600000 → 600000.SH"""
    for orig, prefix in EXCHANGE_MAPPING.items():
        if sym.startswith(prefix):
            return sym[len(prefix):] + "." + orig
    return sym

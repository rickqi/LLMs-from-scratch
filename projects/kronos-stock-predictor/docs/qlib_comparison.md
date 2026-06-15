# Kronos vs QLIB 数据格式对比分析

> **独立对比，不直接集成** — 分析两种数据管道差异

---

## 数据格式对比

### QLIB 格式 (investment_data)

```
~/.qlib/qlib_data/cn_data/
├── calendars/day.txt               # 交易日历
├── instruments/all.txt             # 全品种列表
├── features/
│   └── sh600000/
│       ├── open.day.bin            # 开盘价 (二进制格式)
│       ├── high.day.bin
│       ├── low.day.bin
│       ├── close.day.bin
│       ├── volume.day.bin
│       ├── amount.day.bin
│       ├── adjclose.day.bin
│       ├── vwap.day.bin
│       ├── change.day.bin
│       └── factor.day.bin          # 复权因子
```

**特点**:
- 二进制 `.bin` 文件，按字段分文件
- 归一化坐标系（第0行为占位符）
- 复权价格 = raw_close × adj_factor
- 专为 QLIB 库优化

### Kronos 格式 (本项目)

```
data/processed/
├── train_data.pkl                  # {symbol: DataFrame}
├── val_data.pkl
└── test_data.pkl
```

**特点**:
- Pickle 序列化的 `dict[str, DataFrame]`
- OHLCV 6列（open/high/low/close/vol/amt）
- 原始价格，训练时 Z-score 归一化
- PyTorch DataLoader 兼容

---

## 关键差异

| 维度 | QLIB | Kronos 本项目 |
|------|------|-------------|
| **存储格式** | 二进制 (.bin) | Pickle (DataFrame) |
| **字段组织** | 按字段分文件 | 按品种分 DataFrame |
| **价格类型** | 复权价格 | 原始价格 |
| **归一化** | 全局归一化 | 按窗口 Z-score |
| **时间特征** | 无 (QLIB 不提供) | minute/hour/weekday/day/month |
| **数据更新** | incremental_update.py | downloader.py (Tushare/akshare) |
| **数据来源** | 多源融合 (Wind/Tushare/akshare/baostock) | Tushare + akshare |

---

## 兼容性方案（不集成，独立使用）

如需对比 QLIB 格式数据，可通过以下独立脚本转换：

```python
# scripts/qlib_to_kronos.py (独立转换脚本)
"""
将 QLIB bin 格式转为 Kronos pickle 格式，用于对比分析。
运行: python scripts/qlib_to_kronos.py --qlib_dir ~/.qlib/qlib_data/cn_data --output ./data/qlib_converted
"""
import struct, os, pickle, pandas as pd, numpy as np
from pathlib import Path

def read_qlib_bin(bin_path: str) -> np.ndarray:
    """读取 QLIB .bin 文件"""
    with open(bin_path, 'rb') as f:
        data = []
        while True:
            chunk = f.read(4)
            if not chunk: break
            data.append(struct.unpack('f', chunk)[0])
    return np.array(data[1:])  # 跳过第0行占位符

def convert_symbol(qlib_dir: str, symbol: str) -> pd.DataFrame:
    """转换单个品种"""
    base = Path(qlib_dir) / 'features' / symbol
    fields = ['open','high','low','close','volume','amount','adjclose']
    data = {}
    for f in fields:
        bin_path = base / f'{f}.day.bin'
        if bin_path.exists():
            data[f] = read_qlib_bin(str(bin_path))
    # 用 adjclose 替换 close
    if 'adjclose' in data:
        data['close'] = data.pop('adjclose')
    return pd.DataFrame(data)
```

---

## 建议

1. **训练用本项目格式** — 更灵活，支持时间特征提取
2. **回测对比用 QLIB** — 利用 QLIB 内置的风险模型和组合优化
3. **数据质量对比** — 两套数据交叉验证，识别异常值

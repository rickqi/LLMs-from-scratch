# CSI300 数据获取方案分析

> 2026-06-16

## 现状

| 数据源 | 品种数 | 时间范围 | 格式 | 状态 |
|--------|--------|---------|------|------|
| 半导体 kline_cache | 69 | 2018-2026 | CSV (OHLCV) | ✅ 已用 |
| Kronos/data/ | 30 | 2005-2026 | CSV (中文列名) | ⚠️ 23只新增 |
| akshare (已下载) | 14 | 2015-2026 | DataFrame | ✅ 已合并 |
| **已合并** | **82** | | | |
| akshare (剩余) | 286 | — | — | ❌ API不可用 |

## 方案评估

### 方案 A: akshare 重试优化
- **状态**: ❌ 不可行
- **原因**: 连续10次请求全部失败，非限速而是API底层不可用（版本过旧或数据源变更）
- **升级 akshare**: `pip install akshare --upgrade` 可能修复，但需网络

### 方案 B: 合并 Kronos/data/ 新增23只 ✅ 推荐
- **效果**: 82 → ~99 只（去重后）
- **优势**: 数据已在本地，即刻可用，含 2005-2026 长历史
- **劣势**: 仍需转换格式（中文列名）
- **执行时间**: 5分钟

### 方案 C: Tushare Pro
- **状态**: ⚠️ 需 Token
- **效果**: 可获取全部 CSI300
- **Token 来源**: investment_data 项目使用过 Tushare
- **安装**: `pip install tushare`

### 方案 D: 升级 akshare
- `pip install akshare --upgrade`
- 可能修复 API 问题
- 需网络

## 推荐执行顺序

1. **立即执行**: 方案 B — 合并 Kronos/data/ 23只 → 扩展至 ~99只
2. **如有网络**: 方案 D — 升级 akshare → 重试下载
3. **如有 Token**: 方案 C — Tushare Pro 全量下载

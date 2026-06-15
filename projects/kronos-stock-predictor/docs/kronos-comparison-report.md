# Kronos 实现对比分析报告

> 本项目 `projects/kronos-stock-predictor` vs 真实 Kronos `D:\codes\stock\Kronos`  
> 分析日期：2026-06-15

---

## 一、总体评估

| 维度 | 本项目 | 真实 Kronos | 差距 |
|------|--------|-----------|------|
| **代码量** | ~3500 行 | ~2500 行 (核心) | 本项目更完整（含全套训练/推理/测试） |
| **BSQ 实现** | 简化版（直接二值化+commitment loss） | 完整版（熵正则化+码本管理+分组量化） | ⚠️ 关键差异 |
| **Tokenizer API** | encode→(s1,s2), decode(s1,s2) | encode→tuple, decode([s1,s2]) | ⚠️ 接口不同 |
| **自回归推理** | 期望已 tokenized 的 dict 输入 | 接受原始特征，内部 tokenize | ⚠️ 接口不同 |
| **训练管道** | 独立 loss 函数 | DualHead.compute_loss() | 🔵 小差异 |
| **DDP 支持** | 基础实现 | 完整 DDP + Comet ML 日志 | 🔵 功能差异 |
| **端到端可用** | ✅ 合成+真实数据均跑通 | ✅ | 持平 |

---

## 二、逐模块差异分析

### 2.1 BSQ 量化器 — ⚠️ CRITICAL

| 特性 | 本项目 (model/modules.py) | 真实 Kronos (model/module.py) | 影响 |
|------|--------------------------|------------------------------|------|
| **量化方式** | 直接二值化 `sign(z)` | 同 | 相同 |
| **熵正则化** | ❌ 无 | ✅ 完整 soft/hard entropy | **高**：真实版防止码本退化 |
| **分组量化** | 仅 group_size 参数用于整除检查 | ✅ 完整 subgroup codebook | **中**：真实版更高效的码本利用率 |
| **码本管理** | ❌ 无 used_codes 追踪 | ✅ codes_to_indexes, codes_to_group_indexes | **低**：调试用 |
| **BSQ loss** | 仅 commitment loss (MSE) | commitment + entropy penalty + zeta scaling | **高**：真实版训练更稳定 |
| **STE 实现** | `z + (zhat - z).detach()` | `z + (zhat - z).detach()` | 相同 |
| **forward 返回** | `(loss, quantized, indices)` | `(quantized, loss, metrics_dict)` | **中**：返回值顺序不同 |

**建议**：补充熵正则化是最高优先级改进。

### 2.2 KronosTokenizer — ⚠️ CRITICAL

| 特性 | 本项目 | 真实 Kronos | 影响 |
|------|--------|-----------|------|
| **encode 返回** | `(s1_ids, s2_ids)` tuple | `(s1_ids, s2_ids)` tuple (half=True) | 相同 ✅ |
| **decode 签名** | `decode(s1_ids, s2_ids)` | `decode([s1_ids, s2_ids], half=True)` | **中**：接口不同 |
| **indices_to_bits** | 手动位运算 | 手动位运算 | 相同 |
| **重建损失** | `L_coarse + L_fine + λ*L_quant` | 同（forward 中实现） | 相同 ✅ |
| **encoder/decoder 层数** | n_enc_layers-1, n_dec_layers-1 | 同 | 相同 ✅ |
| **post_quant_embed** | 两个独立 Linear | 同 | 相同 ✅ |

### 2.3 Kronos Model — 🔵 MINOR

| 特性 | 本项目 | 真实 Kronos | 影响 |
|------|--------|-----------|------|
| **teacher forcing** | 前向用 s1_logits 采样 | 同 | 相同 ✅ |
| **DependencyAwareLayer** | LayerNorm + MLP fusion | 同 | 相同 ✅ |
| **DualHead** | `head(x)` + `cond_forward(x)` | `head(x)` + `cond_forward(x)` | 相同 ✅ |
| **compute_loss** | 独立 `predictor_loss()` 函数 | `head.compute_loss()` 方法 | 🔵 小差异 |
| **RMSNorm** | 手动实现 | 同 | 相同 ✅ |
| **TemporalEmbedding** | 5 个 Embedding 求和 | 同 | 相同 ✅ |

### 2.4 自回归推理 — ⚠️ CRITICAL

| 特性 | 本项目 | 真实 Kronos | 影响 |
|------|--------|-----------|------|
| **输入格式** | `x = {"s1_ids": ..., "s2_ids": ...}` | `x = raw_features (B,T,6)` | **高**：接口完全不同 |
| **tokenize 时机** | 调用方预先 tokenize | 函数内部 tokenize | **高**：本项目需在 predict 中额外处理 |
| **sample_count 处理** | repeat 在外部 | `unsqueeze(1).repeat(1, sample_count,...)` | 相同 |
| **滑动窗口** | 同 | pre_buffer/post_buffer + torch.roll | 相同 ✅ |
| **decode 调用** | `tokenizer.decode(s1[i], s2[i])` | `tokenizer.decode([s1, s2], half=True)` | **中** |
| **s1→s2 条件** | decode_s1 → sample → decode_s2 | 同 | 相同 ✅ |

### 2.5 训练管道 — 🔵 MINOR

| 特性 | 本项目 | 真实 Kronos | 影响 |
|------|--------|-----------|------|
| **DDP** | 基础 `DistributedDataParallel` | 完整 DDP + `torchrun` | 🔵 |
| **损失计算** | 独立 `predictor_loss()` | `model.module.head.compute_loss()` | 🔵 |
| **梯度裁剪** | `clip_grad_norm_(clip=5.0)` | `clip_grad_norm_(max_norm=3.0)` | 🔵 |
| **日志** | Python logging | Comet ML + print | 🔵 |

### 2.6 Dataset — 相同 ✅

| 特性 | 本项目 | 真实 Kronos | 影响 |
|------|--------|-----------|------|
| **索引预计算** | `(symbol, start_idx)` 对 | 同 | ✅ |
| **归一化** | Z-score on lookback window | 同 | ✅ |
| **随机采样** | `py_rng.choice()` | `py_rng.randint()` | ✅ |
| **时间特征** | minute/hour/weekday/day/month | 同 | ✅ |

---

## 三、差异影响评估

| 差异类别 | 数量 | 严重度 | 对效果的影响 |
|---------|------|--------|------------|
| BSQ 熵正则化缺失 | 1 | 🔴 高 | 可能导致码本退化，token 利用率不均 |
| 自回归推理接口不同 | 2 | 🔴 高 | 当前推理路径绕过了真实 Kronos 的滑动窗口优化 |
| decode 签名不同 | 1 | 🟡 中 | 需维护两个调用约定 |
| loss 计算位置不同 | 1 | 🔵 低 | 数学等价，仅代码组织不同 |
| 日志/监控缺失 | 1 | 🔵 低 | 不影响模型效果 |

---

## 四、修复优先级

| 优先级 | 修复项 | 工作量 | 预期效果 |
|--------|--------|--------|---------|
| **P0** | 回测接入完整自回归预测 | 2h | 回测指标真实反映模型能力 |
| **P0** | BSQ 添加熵正则化 | 3h | 码本利用率提升，训练更稳定 |
| **P1** | 统一 decode 接口 | 0.5h | 与上游 Kronos 生态兼容 |
| **P2** | DualHead.compute_loss() | 1h | 代码一致性 |
| **P3** | Comet ML / WandB 日志 | 2h | 训练监控 |

---

## 五、实测数据对比

| 指标 | 本项目 (mini, 半导体68只) | 真实 Kronos (base, 45交易所) | 说明 |
|------|--------------------------|---------------------------|------|
| 参数量 | 0.5M | 103M | 200× 差距 |
| 预训练数据 | 68 只半导体（2018-2026） | 45+ 交易所，12B+ K线 | 规模不可比 |
| Tokenizer val | 0.136 | 未公开 | — |
| Predictor val | 4.337 | 未公开 | — |
| 回测 Sharpe (简化动量) | 1.96 | — | 简化信号，高估 |
| **回测 Sharpe (自回归预测)** | **-0.30** | 论文未单独报告 | 完整 Kronos 预测 |
| **平均 RankIC** | **-0.0028** | 论文报告 93% 提升 | mini 模型未达预测能力 |
| **RankIC > 0 比例** | **60.0%** | — | 微正 |
| **回测胜率** | **53.3%** | — | 略高于随机 |
| 回测日期数 | 30 (quick) / 248 (full) | — | — |

> **关键发现**：完整 Kronos 自回归预测的 RankIC 接近零（-0.0028），表明 mini 模型（0.5M 参数，3 epochs）在当前设置下尚未学习到有效的预测信号。这与预期一致——真实 Kronos 需要 103M 参数 + 12B+ K线预训练才能达到论文报告的 93% RankIC 提升。当前结果是合理的起点。

---

## 六、预估完成时间

| 任务 | 预估时间 |
|------|---------|
| 回测接入完整自回归预测 | 2 小时 |
| 添加 RankIC 指标 | 1 小时 |
| BSQ 熵正则化 | 3 小时 |
| 统一 decode 接口 | 0.5 小时 |
| **总计** | **6.5 小时** |

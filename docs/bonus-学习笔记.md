# Bonus 深入：KV Cache + GQA + MoE + GPT→Llama — 学习笔记

---

## 1. KV Cache (推理加速)

### 问题
自回归生成时，每生成一个新 token 就重新计算所有 K/V → O(n²) 二次增长。

### 解决
缓存已计算的 K/V，每步只算新 token 的 K/V → O(n) 线性增长。

```
无缓存:                        有缓存:
Step 2: 输入 [A,B] 重算 A 的 K  Step 2: 输入 [B] 只算 B 的 K
Step 3: 输入 [A,B,C] 重算 A,B   Step 3: 输入 [C] 只算 C 的 K
```

### 实现要点
```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)

# forward 中:
if use_cache:
    cache_k = torch.cat([cache_k, keys_new], dim=1)
    cache_v = torch.cat([cache_v, values_new], dim=1)
```

### 效果
| 实现 | tokens/sec (Mac M4) |
|------|---------------------|
| 无缓存 | 27 |
| 有缓存 | 144 (5.3×) |

---

## 2. GQA: Grouped-Query Attention (内存优化)

### 问题
标准 MHA 每个头有独立 K/V → KV Cache 占用大量内存。

### 解决
多个 Q 头共享一组 K/V → 减少 K/V 数量。

```
MHA:  12 个 Q 头, 12 组 K/V
GQA:  12 个 Q 头, 4 组 K/V (每 3 个头共享一组)
MQA:  12 个 Q 头, 1 组 K/V (所有头共享)
```

### KV Cache 内存对比 (bf16)

| 模型 | seq_len | MHA | GQA | 节省 |
|------|---------|-----|-----|------|
| Llama-3 8B | 8192 | 4.0 GB | 1.0 GB | 75% |
| Llama-3 70B | 8192 | 20.0 GB | 2.5 GB | 88% |
| Llama-3 8B | 32768 | 16.0 GB | 4.0 GB | 75% |

### 实现差异
```python
# MHA: W_key 投影到全部维度
self.W_key = nn.Linear(emb_dim, emb_dim)  # 12 个 K 头

# GQA: W_key 投影到更小维度
self.W_key = nn.Linear(emb_dim, head_dim * n_kv_heads)  # 只有 4 个 K 头
# 然后 repeat_interleave 给所有 Q 头使用
```

---

## 3. MoE: Mixture of Experts (稀疏激活)

### 问题
FFN 占模型 2/3 参数，但推理时每个 token 用全部 FFN。

### 解决
用 N 个小 FFN (专家) 替代 1 个大 FFN，每个 token 只激活 top-k 个。

### 参数对比

| 配置 | 总参数 | 活跃参数 | 活跃比 |
|------|--------|---------|--------|
| Dense FFN (768→3072) | 4.7M | 4.7M | 100% |
| MoE (8 experts, top-2) | 4.7M | 1.2M | 25% |
| DeepSeek-V3 | 671B | 37B | 5.5% |

### 实现
```python
class MoELayer(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_experts, top_k):
        self.router = nn.Linear(emb_dim, num_experts)
        self.experts = nn.ModuleList([FFN(...) for _ in range(num_experts)])

    def forward(self, x):
        logits = self.router(x)  # 每个 token 的专家打分
        top_k_weights = softmax(top_k(logits))
        output = sum(weight * expert(x) for each selected expert)
```

### 权衡
- ✓ 总容量大（更多知识）+ 稀疏推理（更少计算）
- ✗ 所有专家权重都要加载到 GPU（显存大）
- ✗ 需要负载均衡（防止只用几个专家）

---

## 4. GPT → Llama 架构转换

### 6 大改进

| 组件 | GPT-2 | Llama 2/3 | 改进原因 |
|------|-------|-----------|---------|
| Normalization | LayerNorm | **RMSNorm** | 简化，更快 |
| 位置编码 | 可学习绝对 | **RoPE** | 可外推，编码相对位置 |
| Attention | MHA | **GQA** | 省内存 |
| 激活函数 | GELU | **SwiGLU** | 门控，更低 loss |
| Bias | 有 | **无** | 简化 |
| 上下文长度 | 1024 | **8192-128k** | 更长 |

### RMSNorm vs LayerNorm
```
LayerNorm: (x - mean) / std × γ + β   → 2×emb_dim 参数
RMSNorm:   x / RMS × γ                → emb_dim 参数 (减半)
```

### RoPE (旋转位置编码)
```python
# 不给输入加位置向量，而是旋转 Q 和 K
q_rotated = q * cos(θ) + rotate_half(q) * sin(θ)
# θ_i = position / 10000^(2i/d)
# 好处: 编码相对距离，可外推，无额外参数
```

### SwiGLU vs GELU
```
GELU FFN:   Linear → GELU → Linear          (2 个 Linear)
SwiGLU FFN: SiLU(W_gate·x) × (W_up·x) → W_down  (3 个 Linear + 门控)
```

---

## 现代 LLM 标准配方

```
RoPE + RMSNorm + SwiGLU + GQA (+ optional MoE)

  Llama 3:     RoPE + RMSNorm + SwiGLU + GQA
  Qwen 3:      RoPE + RMSNorm + SwiGLU + GQA (+ MoE)
  Gemma 3:     RoPE + RMSNorm + GeGLU + GQA + 跨层 KV 共享
  DeepSeek:    RoPE + RMSNorm + SwiGLU + MLA + MoE
```

### 技术总结

| 技术 | 优化目标 | 效果 |
|------|---------|------|
| KV Cache | 推理速度 | O(n²)→O(n), ~5× |
| GQA | KV Cache 内存 | 减少 75-88% |
| MoE | 计算量 vs 容量 | 671B 参数只用 37B |
| RoPE | 位置编码泛化 | 可外推到更长序列 |
| RMSNorm | 训练速度 | 更少参数和计算 |
| SwiGLU | 模型质量 | 门控机制，更低 loss |

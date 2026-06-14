# 附录 A/D/E：PyTorch 入门 + 训练增强 + LoRA — 学习笔记

---

## 附录 A: PyTorch 入门

### Tensor 基础

```python
# 创建
torch.tensor([1, 2, 3])        # 从列表
torch.zeros(2, 3)               # 全零
torch.randn(2, 3)               # 标准正态分布

# 操作
a + b    # 逐元素加
a * b    # 逐元素乘
a @ b    # 矩阵乘 (dot for 1D)
x.view(3, 4)  # reshape

# 索引 (同 NumPy)
mat[0]       # 第 0 行
mat[:, 0]    # 第 0 列
mat[0:2, 1:] # 切片

# 广播 (同 NumPy)
A(2,3) + v(3,) → v 自动广播到 (2,3)
```

### Autograd (自动微分)

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2      # 前向: 构建计算图
y.backward()    # 反向: 自动计算梯度
print(x.grad)   # dy/dx = 6.0
```

**核心**：设置 `requires_grad=True` → PyTorch 自动追踪所有操作 → `backward()` 反向计算梯度

### nn.Module (神经网络)

```python
class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 30), nn.ReLU(),
            nn.Linear(30, n_out))
    def forward(self, x):
        return self.layers(x)
```

### 训练循环 5 步 (最重要!)

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        logits = model(X_batch)              # 1. 前向传播
        loss = F.cross_entropy(logits, y_batch)  # 2. 计算 loss
        optimizer.zero_grad()                # 3. 清零梯度 (必须!)
        loss.backward()                      # 4. 反向传播
        optimizer.step()                     # 5. 更新参数
```

**为什么 `zero_grad()`？** PyTorch 梯度是累加的，不清零会叠加历史梯度。

### GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)   # 模型移到 GPU
data.to(device)    # 数据移到 GPU (必须同一设备!)
```

---

## 附录 D: 训练循环增强

### ch05 基础 → 附录D 增强

| 技术 | 作用 | 何时用 |
|------|------|--------|
| 学习率预热 | 防止训练初期梯度爆炸 | 所有 LLM 训练 |
| 余弦退火 | 后期精细收敛 | 所有 LLM 训练 |
| 梯度裁剪 | 防止单步梯度爆炸 | 大模型 / 长序列 |

### 1. 学习率预热 (Warmup)

```
前 N 步: lr 从 0 线性增长到 max_lr
  lr = max_lr × (step + 1) / warmup_steps

原因: 初始权重随机, 大 lr 会"弹飞"模型
```

### 2. 余弦退火 (Cosine Annealing)

```
预热后: lr 按余弦曲线衰减到 min_lr

  lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))

  lr 曲线:  ╭╮ (peak)
           ╱  ╲
          ╱    ╲___ (min)
```

### 3. 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 如果 ||grad|| > 1.0, 缩放到 1.0
```

### 现代 LLM 训练标准配方

```
AdamW + Warmup(2000步) + Cosine Decay + Grad Clip(1.0) + Weight Decay(0.1)
```

---

## 附录 E: LoRA (Low-Rank Adaptation)

### 核心思想

```
全量微调:  W_new = W_old + ΔW         ΔW 是 d×d 大矩阵
LoRA:      ΔW = B × A                  A 是 r×d, B 是 d×r, r << d

  d=768, r=8:
    全量: 768×768 = 589,824 参数
    LoRA: 8×768 + 768×8 = 12,288 参数  ← 48× 压缩!
```

### PyTorch 实现

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B=0 → 初始 ΔW=0 → 不改变模型行为

    def forward(self, x):
        return x @ self.A.t() @ self.B.t()  # B(Ax)

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.linear = linear  # 冻结原始权重
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank)

    def forward(self, x):
        return self.linear(x) + self.lora(x)  # 原始 + LoRA
```

### GPT-2 124M 上的参数对比

| 方法 | 可训练参数 | 压缩比 | GPU 内存 |
|------|-----------|--------|---------|
| 全量微调 | 124M | 1× | ~2GB |
| LoRA (r=8, Q+K) | ~295K | **420×** | ~0.5GB |
| ch06 冻结策略 | ~7.1M | 18× | ~1GB |

### LoRA 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| rank (r) | 8-64 | 表达能力; r=8 轻量, r=64 重型 |
| alpha | 2×rank | 缩放因子 |
| target_modules | Q+K 或 Q+K+V+O | 哪些层加 LoRA |

### LoRA 四大优势

1. **内存高效**：只存 A+B，不存完整 ΔW
2. **任务切换**：不同任务换 LoRA 权重 (几 MB)，基础模型不变
3. **无推理延迟**：训练后 W_new = W + B@A 融合，推理速度不变
4. **效果接近全量**：95%+ 性能，1% 参数

---

## 全书学习完成总结

| 章节 | 核心知识点 | 状态 |
|------|-----------|------|
| ch01 | LLM 三阶段训练 (预训练→SFT→RLHF) | ✅ |
| ch02 | BPE 分词 + Embedding + 滑动窗口 + DataLoader | ✅ |
| ch03 | Self-Attention + Q/K/V + Causal Mask + MultiHead | ✅ |
| ch04 | LayerNorm + GELU FFN + 残差 + 12 层 = GPT-2 124M | ✅ |
| ch05 | Cross-Entropy + 训练循环 + 加载 OpenAI 权重 | ✅ |
| ch06 | 替换输出头 + 冻结 + 最后 token 分类 | ✅ |
| ch07 | Alpaca 格式 + -100 masking + 全量微调 | ✅ |
| 附录A | Tensor + Autograd + nn.Module + 训练循环 | ✅ |
| 附录D | Warmup + Cosine Decay + Grad Clipping | ✅ |
| 附录E | LoRA 低秩分解 + 参数高效微调 | ✅ |

**你已完整学完全书所有章节和附录！** 🎉

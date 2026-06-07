# 第 09 篇：PyTorch 入门（附录 A）

> 📖 基于《Build a Large Language Model (From Scratch)》附录 A
> 💡 本附录是全书代码的 PyTorch 前置知识，如果你已有 PyTorch 经验可以快速浏览。

---

## 一、本篇目标

**一句话**：掌握本书其余章节所需的 PyTorch 核心基础，从 Tensor 到完整的 GPU 训练流程。

读完本篇，你将：
- 理解 Tensor 的创建、运算和数据类型
- 掌握计算图和自动微分（autograd）
- 使用 nn.Module 和 nn.Sequential 构建神经网络
- 实现 DataLoader 进行批量数据加载
- 编写完整的训练循环
- 保存和加载模型
- 在 GPU 上训练（单卡和多卡）

> ⏱️ 预计用时：2-3 小时

---

## 二、核心知识点

### 2.1 Tensor 基础（Section A.2）

Tensor 是 PyTorch 中最基本的数据结构，类似于 NumPy 的 ndarray，但可以在 GPU 上运行并支持自动微分。

#### 创建 Tensor

```python
import torch
import numpy as np

# 从 Python 标量创建 0D Tensor（标量）
tensor0d = torch.tensor(1)

# 从 Python 列表创建 1D Tensor（向量）
tensor1d = torch.tensor([1, 2, 3])

# 从嵌套列表创建 2D Tensor（矩阵）
tensor2d = torch.tensor([[1, 2],
                         [3, 4]])

# 从嵌套列表创建 3D Tensor
tensor3d_1 = torch.tensor([[[1, 2], [3, 4]],
                           [[5, 6], [7, 8]]])

# 从 NumPy 数组创建 3D Tensor
ary3d = np.array([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]])
tensor3d_2 = torch.tensor(ary3d)      # 复制数据（安全）
tensor3d_3 = torch.from_numpy(ary3d)  # 共享内存（高效）
```

**关键区别**：`torch.tensor()` 创建数据副本，修改原始数组不会影响 Tensor；`torch.from_numpy()` 与 NumPy 数组共享内存，修改一方会影响另一方：

```python
ary3d[0, 0, 0] = 999
print(tensor3d_2)  # 不变，仍然是 1
print(tensor3d_3)  # 变为 999（共享内存）
```

#### 数据类型

PyTorch 会自动推断数据类型。整数列表默认 `torch.int64`，浮点数列表默认 `torch.float32`：

```python
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)   # torch.int64

floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)   # torch.float32

# 类型转换
floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)   # torch.float32
```

#### 常用操作

```python
tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])

tensor2d.shape          # torch.Size([2, 3])
tensor2d.reshape(3, 2)  # 重塑为 3x2
tensor2d.view(3, 2)     # 同上（共享内存）
tensor2d.T              # 转置 -> 3x2

# 矩阵乘法
tensor2d.matmul(tensor2d.T)  # 等价于
tensor2d @ tensor2d.T        # @ 运算符
```

> `.view()` 和 `.reshape()` 的区别见 [常见问题](#四常见问题)。

---

### 2.2 计算图与自动微分（Sections A.3-A.4）

#### 计算图（Computational Graph）

PyTorch 通过**计算图**追踪 Tensor 上的所有运算。前向传播时自动构建计算图，反向传播时利用它计算梯度。

```python
import torch.nn.functional as F

y = torch.tensor([1.0])   # 真实标签
x1 = torch.tensor([1.1])  # 输入特征
w1 = torch.tensor([2.2])  # 权重参数
b = torch.tensor([0.0])   # 偏置

z = x1 * w1 + b           # 线性组合（net input）
a = torch.sigmoid(z)      # 激活函数（sigmoid 输出）

loss = F.binary_cross_entropy(a, y)  # 二分类交叉熵损失
print(loss)  # tensor(0.0852)
```

这段代码的前向传播路径：`x1 * w1 + b -> sigmoid -> BCE loss`，形成一条计算链。

#### 自动微分（autograd）

在需要计算梯度的 Tensor 上设置 `requires_grad=True`，PyTorch 会自动追踪所有运算：

```python
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)  # 追踪梯度
b = torch.tensor([0.0], requires_grad=True)    # 追踪梯度

z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
```

**方式一**：`torch.autograd.grad()` 手动计算梯度

```python
grad_L_w1 = grad(loss, w1, retain_graph=True)  # 保留计算图
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)  # (tensor([-0.0898]),)
print(grad_L_b)   # (tensor([-0.0817]),)
```

**方式二**：`loss.backward()` 自动反向传播（更常用）

```python
loss.backward()

print(w1.grad)  # tensor([-0.0898])
print(b.grad)   # tensor([-0.0817])
```

**关键要点**：
- `requires_grad=True`：标记需要计算梯度的参数
- `loss.backward()`：执行反向传播，计算所有梯度
- `.grad` 属性：存储计算好的梯度值
- `retain_graph=True`：保留计算图以进行多次 `backward()` 调用
- `torch.autograd.grad()`：手动获取特定参数的梯度

---

### 2.3 构建神经网络（Section A.5）

PyTorch 提供两种方式构建神经网络：

#### 方式一：nn.Sequential（简单堆叠）

`nn.Sequential` 将多个层按顺序串联，适用于前一层输出直接作为下一层输入的情况：

```python
import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
```

注意：虽然使用了 `nn.Sequential`，但这里仍然是 `nn.Module` 子类。`nn.Sequential` 也可以单独使用：

```python
model = torch.nn.Sequential(
    torch.nn.Linear(50, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 3),
)
```

#### 方式二：nn.Module 子类化（灵活）

当需要残差连接、条件分支等复杂前向逻辑时，必须使用 `nn.Module` 子类化并手写 `forward()` 方法。本书的 GPT 模型就使用了这种方式。

#### 参数统计与可复现性

```python
torch.manual_seed(123)  # 设置随机种子，确保结果可复现
model = NeuralNetwork(50, 3)

# 统计可训练参数数量
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
# Total number of trainable model parameters: 2213
```

查看模型结构和参数形状：

```python
print(model)
# NeuralNetwork(
#   (layers): Sequential(
#     (0): Linear(in_features=50, out_features=30, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=30, out_features=20, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=20, out_features=3, bias=True)
#   )
# )

print(model.layers[0].weight.shape)  # torch.Size([30, 50])
```

#### 推理模式

训练和推理的行为不同。推理时使用 `torch.no_grad()` 禁用梯度计算以节省内存：

```python
# 带梯度追踪的前向传播（训练时）
X = torch.rand((1, 50))
out = model(X)
print(out)  # tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)

# 不带梯度追踪的前向传播（推理时）
with torch.no_grad():
    out = model(X)
print(out)  # tensor([[-0.1262,  0.1080, -0.1792]])  -- 没有 grad_fn

# 获取预测概率
with torch.no_grad():
    probas = torch.softmax(model(X), dim=1)
print(probas)  # tensor([[0.3113, 0.3934, 0.2952]])
```

---

### 2.4 DataLoader（Section A.6）

DataLoader 负责将数据集分成小批量（mini-batch）并提供给训练循环。

#### 自定义 Dataset

继承 `torch.utils.data.Dataset` 并实现三个方法：

```python
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]
```

#### 创建 DataLoader

```python
from torch.utils.data import DataLoader

# 准备数据
X_train = torch.tensor([
    [-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6],
    [2.3, -1.1], [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
y_test = torch.tensor([0, 1])

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True  # 丢弃最后一个不完整的 batch
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)
```

#### DataLoader 参数说明

| 参数 | 说明 |
|------|------|
| `batch_size` | 每个 batch 的样本数量 |
| `shuffle` | 是否在每个 epoch 打乱数据顺序（训练集=True，测试集=False） |
| `num_workers` | 数据加载的子进程数（0=主进程加载） |
| `drop_last` | 是否丢弃最后一个不完整的 batch |

遍历 DataLoader：

```python
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
# Batch 1: tensor([[ 2.3, -1.1], [-0.9,  2.9]]) tensor([1, 0])
# Batch 2: tensor([[-1.2,  3.1], [-0.5,  2.6]]) tensor([0, 0])
```

---

### 2.5 训练循环（Section A.7）

这是 PyTorch 中最核心的模式，本书从第 5 章开始反复使用：

```python
import torch.nn.functional as F

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()  # 设置为训练模式
    for batch_idx, (features, labels) in enumerate(train_loader):

        # ---- 五步训练循环 ----

        # 1. 清除上一步的梯度
        optimizer.zero_grad()

        # 2. 前向传播
        logits = model(features)

        # 3. 计算损失
        loss = F.cross_entropy(logits, labels)

        # 4. 反向传播（计算梯度）
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        # ---- 日志 ----
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()  # 设置为评估模式
```

输出：
```
Epoch: 001/003 | Batch 001/002 | Train/Val Loss: 0.75
Epoch: 001/003 | Batch 002/002 | Train/Val Loss: 0.65
Epoch: 002/003 | Batch 001/002 | Train/Val Loss: 0.44
Epoch: 002/003 | Batch 002/002 | Train/Val Loss: 0.13
Epoch: 003/003 | Batch 001/002 | Train/Val Loss: 0.03
Epoch: 003/003 | Batch 002/002 | Train/Val Loss: 0.00
```

**五步详解**：

| 步骤 | 代码 | 作用 |
|------|------|------|
| 1 | `optimizer.zero_grad()` | 清除上一步残留的梯度，避免梯度累加 |
| 2 | `logits = model(features)` | 前向传播，计算模型输出 |
| 3 | `loss = F.cross_entropy(logits, labels)` | 用损失函数评估预测与真实的差距 |
| 4 | `loss.backward()` | 反向传播，自动计算所有参数的梯度 |
| 5 | `optimizer.step()` | 根据梯度更新模型参数 |

#### 评估准确率

```python
def compute_accuracy(model, dataloader):

    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():           # 不计算梯度
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)  # 取最大值的索引
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

print(compute_accuracy(model, train_loader))  # 1.0
print(compute_accuracy(model, test_loader))   # 1.0
```

---

### 2.6 保存与加载模型（Section A.8）

PyTorch 推荐只保存模型的 `state_dict`（参数字典），而不是整个模型对象：

```python
# 保存模型参数
torch.save(model.state_dict(), "model.pth")

# 加载模型参数（需要先创建相同结构的模型）
model = NeuralNetwork(2, 2)  # 结构必须与原始模型完全一致
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

`state_dict` 本质上是一个 Python 字典，键是参数名，值是 Tensor：

```python
print(model.state_dict().keys())
# odict_keys(['layers.0.weight', 'layers.0.bias',
#             'layers.2.weight', 'layers.2.bias',
#             'layers.4.weight', 'layers.4.bias'])
```

> `weights_only=True` 是安全选项，防止加载恶意 pickle 数据。本书代码使用此选项。

---

### 2.7 GPU 训练（Section A.9）

#### 检查 GPU 可用性

```python
import torch

print(torch.__version__)          # e.g., '2.4.0+cu121'
print(torch.cuda.is_available())  # True/False
```

#### Tensor 在 GPU 上的运算

```python
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

# CPU 上运算
print(tensor_1 + tensor_2)  # tensor([5., 7., 9.])

# 移到 GPU
tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda")
print(tensor_1 + tensor_2)  # tensor([5., 7., 9.], device='cuda:0')
```

**重要**：不同设备上的 Tensor 不能直接运算，必须先移到同一设备：

```python
tensor_1 = tensor_1.to("cpu")
print(tensor_1 + tensor_2)
# RuntimeError: Expected all tensors to be on the same device,
# but found at least two devices, cuda:0 and cpu!
```

#### 单卡训练

只需在 CPU 训练代码基础上添加三处 `.to(device)`：

```python
# 1. 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 模型移到 GPU
model = NeuralNetwork(num_inputs=2, num_outputs=2)
model.to(device)

# 3. 每个 batch 的数据移到 GPU
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        features, labels = features.to(device), labels.to(device)  # 数据移到 GPU

        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
```

评估时同样需要将数据移到 GPU：

```python
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)  # 数据移到 GPU

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()
```

#### 多卡训练（DDP）

当单卡内存不够或需要加速训练时，使用 `DistributedDataParallel`（DDP）在多张 GPU 上并行训练。

核心思路：每张 GPU 运行一个独立的进程，各自持有模型副本，处理不同的数据子集，然后同步梯度。

```python
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# 初始化进程组（每个 GPU 一个进程）
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    # Linux 使用 nccl 后端，Windows 使用 gloo
    if platform.system() == "Windows":
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
```

训练函数：

```python
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # 初始化进程组

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # 用 DDP 包装模型
    model = DDP(model, device_ids=[rank])

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)  # 每个 epoch 不同的数据划分

        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()   # DDP 自动同步梯度
            optimizer.step()

    model.eval()
    # ... 评估 ...
    destroy_process_group()  # 清理
```

启动多进程：

```python
if __name__ == "__main__":
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size：为每张 GPU 启动一个进程
```

也可以使用 `torchrun` 启动（更简洁，不需要手动管理进程）：

```bash
torchrun --nproc_per_node=2 DDP-script-torchrun.py
```

DDP 关键组件总结：

| 组件 | 作用 |
|------|------|
| `init_process_group()` | 初始化进程间通信 |
| `DistributedSampler` | 将数据集均匀划分到各 GPU |
| `DDP(model)` | 包装模型，自动同步梯度 |
| `mp.spawn()` | 为每张 GPU 启动独立进程 |
| `destroy_process_group()` | 清理分布式环境 |

---

## 三、动手练习：跑通附录 A 全流程

### 步骤 1：打开 notebook

```bash
cd appendix-A/01_main-chapter-code/
jupyter notebook code-part1.ipynb
```

### 步骤 2：创建和操作 Tensor

```python
import torch
import numpy as np

# 创建各种维度的 Tensor
scalar = torch.tensor(42)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1, 2], [3, 4]])

# 基本运算
print(matrix.shape)       # torch.Size([2, 2])
print(matrix.reshape(4))  # tensor([1, 2, 3, 4])
print(matrix @ matrix.T)  # 矩阵乘法
```

### 步骤 3：构建神经网络

```python
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)

torch.manual_seed(123)
model = NeuralNetwork(2, 3)
print(model)
```

### 步骤 4：实现 DataLoader 和训练循环

```python
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 准备数据
X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6],
                        [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])

train_ds = ToyDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_ds, batch_size=2,
                          shuffle=True, drop_last=True)

# 训练
model = NeuralNetwork(2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(3):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
```

### 步骤 5：保存和重新加载模型

```python
# 保存
torch.save(model.state_dict(), "model.pth")

# 加载
model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
```

### 步骤 6：（如有 GPU）运行 GPU 训练

```bash
jupyter notebook code-part2.ipynb
```

或运行 DDP 脚本（需要 2+ GPU）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py
```

---

## 四、常见问题

**Q: `.view()` 和 `.reshape()` 有什么区别？**

A: `.view()` 返回的是与原 Tensor 共享内存的新视图，要求原 Tensor 在内存中是连续的（contiguous）。`.reshape()` 更灵活：如果数据连续就返回共享内存的视图，否则自动复制数据。简单记忆：**优先用 `.reshape()`，除非你明确需要共享内存**。

---

**Q: 什么时候用 `torch.no_grad()`？**

A: 在**推理阶段**（不训练的时候）使用。它禁用梯度追踪，好处有两个：(1) 节省内存（不存储中间激活值）；(2) 加速计算。训练时**不要**使用，否则无法计算梯度。

---

**Q: DataLoader 的 `num_workers` 设多少？**

A: 一般设为 2-8。`num_workers=0` 表示在主进程中加载数据，可能成为训练瓶颈。增大 `num_workers` 可以并行加载数据，但会占用更多内存。建议从小数字开始逐步增加，观察 GPU 利用率是否提升。在 Windows 上可能需要设为 0。

---

**Q: `state_dict` 和完整模型保存有什么区别？**

A: `torch.save(model.state_dict(), ...)` 只保存参数（权重和偏置），文件小且安全。`torch.save(model, ...)` 保存整个模型对象（包括类定义），文件更大，且存在 pickle 安全风险。PyTorch 官方推荐使用 `state_dict` 方式。

---

**Q: 单卡训练够用吗？什么时候需要 DDP？**

A: 对于本书的教学目的（训练小模型），单卡完全够用。需要 DDP 的典型场景：(1) 模型太大，单卡内存放不下；(2) 数据集太大，训练时间不可接受；(3) 需要使用更大的 batch size 来提高训练稳定性。

---

**Q: 为什么用 PyTorch 而不是 TensorFlow？**

A: 两者都很优秀。本书选择 PyTorch 的原因：(1) 更 Pythonic 的 API，代码更易读；(2) 动态计算图，调试更方便；(3) 学术界和 LLM 社区的主导框架（Hugging Face、OpenAI 等都基于 PyTorch）；(4) 拥有最活跃的开源生态。

---

## 五、考试卡片

<details>
<summary>Card 1：torch.tensor() 和 torch.from_numpy() 的区别？</summary>

**torch.tensor()** 创建数据的副本（安全），修改原始数组不会影响 Tensor。

**torch.from_numpy()** 与 NumPy 数组共享内存（高效），修改一方会影响另一方。

```python
ary = np.array([1, 2, 3])
t1 = torch.tensor(ary)       # 复制
t2 = torch.from_numpy(ary)   # 共享内存
ary[0] = 999
print(t1)  # tensor([1, 2, 3])，不受影响
print(t2)  # tensor([999, 2, 3])，已改变
```

</details>

<details>
<summary>Card 2：autograd 如何工作？</summary>

设置 `requires_grad=True` 后，PyTorch 自动追踪所有运算构建计算图。调用 `loss.backward()` 时，自动反向传播计算所有参数的梯度，存储在 `.grad` 属性中。

```python
w = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = x1 * w + b
loss = F.binary_cross_entropy(torch.sigmoid(z), y)
loss.backward()       # 自动计算 w.grad 和 b.grad
print(w.grad)         # tensor([-0.0898])
```

</details>

<details>
<summary>Card 3：nn.Sequential 和 nn.Module 子类化的区别？</summary>

**nn.Sequential** 适用于简单的层堆叠（前一层输出直接连接下一层输入），不需要手写 `forward()`。

**nn.Module 子类化** 更灵活，可以在 `forward()` 中定义任意复杂的计算逻辑（残差连接、条件分支等）。

本书的 GPT 模型使用 `nn.Module` 子类化，因为 Transformer Block 内部有复杂的注意力连接模式。

</details>

<details>
<summary>Card 4：训练循环的五个关键步骤是什么？</summary>

1. **`optimizer.zero_grad()`** — 清除上一步的梯度
2. **`outputs = model(inputs)`** — 前向传播
3. **`loss = loss_fn(outputs, labels)`** — 计算损失
4. **`loss.backward()`** — 反向传播计算梯度
5. **`optimizer.step()`** — 用梯度更新参数

这五步是所有 PyTorch 训练的标准模式，本书从第 5 章开始反复使用。

</details>

---

## 六、关键文件速查

| 想看什么 | 打开 |
|---------|------|
| PyTorch 核心基础 (A.1-A.8) | `appendix-A/01_main-chapter-code/code-part1.ipynb` |
| GPU 训练 (A.9) | `appendix-A/01_main-chapter-code/code-part2.ipynb` |
| 多卡 DDP 训练 | `appendix-A/01_main-chapter-code/DDP-script.py` |
| DDP (torchrun 版) | `appendix-A/01_main-chapter-code/DDP-script-torchrun.py` |
| 练习答案 | `appendix-A/01_main-chapter-code/exercise-solutions.ipynb` |

---

## 七、补充资源

附录 A 没有配套视频，内容全部来自代码 notebook。

延伸学习：
- PyTorch 官方教程：https://pytorch.org/tutorials/
- 本书环境配置指南：`setup/README.md`
- 同一作者的深度 PyTorch 教程：[PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/)

---

## 上一篇

第 08 篇：指令微调 -> [08-指令微调.md](08-指令微调.md)

## 下一篇

第 10 篇：LoRA 参数高效微调 -> [10-LoRA参数高效微调.md](10-LoRA参数高效微调.md)

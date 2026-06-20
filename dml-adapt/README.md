# dml-adapt — 在 AMD GPU 上训练本书的 GPT(DirectML 适配)

把第 4/5 章的 GPT 训练搬到 **AMD Radeon GPU** 上,走 **torch-directml**(DirectX 12),
适用于那些**不在 ROCm WSL 官方支持矩阵**的 AMD GPU——例如 Radeon RX 7600M XT(Navi 33)。

> 核心适配:把原书代码里的 `torch.device("cuda")` 换成 `dml_device.pick_device()`。

---

## 为什么需要这个目录

很多 AMD 笔记本/掌机的 GPU(如 RX 7600 系列、Radeon 780M 核显)**不在 ROCm 的 WSL
支持矩阵**里(ROCm WSL 官方支持的 RDNA3 最低到 RX 7700 XT)。但它们都是 DX12 GPU,
可以用微软的 **DirectML** 跑 PyTorch——不挑型号,只要是 DX12 就行。

所以"不能跑 ROCm" ≠ "不能用这块 AMD GPU 训练"。DirectML 就是这条替代路径。

---

## 文件

| 文件 | 作用 |
|---|---|
| `dml_device.py` | 设备选型:优先选独显 AMD Radeon,回退首个 DirectML 设备 |
| `train_gpt_ch04.py` | ch04 GPT 快速训练验证(5 epoch),引用仓库自有的 `ch04/.../gpt.py` |
| `train_pretrain_ch05.py` | ch05 风格预训练(10 epoch + 周期生成样本) |
| `train_classify_ch06.py` | ch06 分类微调(GPT-2 124M 做 SMS 垃圾短信分类),需预缓存权重 |
| `train_instruction_ch07.py` | ch07 指令微调(GPT-2 124M 全参数微调,Alpaca 指令数据),复用 ch06 权重缓存 |
| `run.sh` | WSL→Windows 桥接运行器(模板,路径需按机器改) |

**不重复代码**:`train_*.py` 通过 `sys.path` 直接 import 本仓库 `ch04/01_main-chapter-code/gpt.py`
的 `GPTModel`,数据直接用 `ch02/01_main-chapter-code/the-verdict.txt`。

---

## 一次性环境准备(Windows 端)

torch-directml 是 Windows-only,**不能装进 WSL 的 python**。在 Windows 上建一个
conda 环境:

```powershell
conda create -n dml python=3.11 -y
conda activate dml
# CPU 版 torch 即可(DirectML 不用 CUDA,真正算力走 DX12)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
pip install --no-deps torch-directml
pip install tiktoken
pip install pandas             # ch06 分类微调需要
# 注意:torch-directml 强绑定 torch==2.4.1,切勿 pip install -U torch
```

### ch06 的额外一步:预缓存 GPT-2 权重(解耦 tensorflow)

`ch06/01_main-chapter-code/gpt_download.py` 需要 **tensorflow** 来解析 OpenAI 的 GPT-2
权重 checkpoint。为避免在 torch-directml 环境里装 tensorflow(共存有风险),改用**任何
装有 tensorflow 的 python**(如 WSL 的 python、或单独的 conda 环境)**一次性**下载并缓存
解析后的权重为单个 `.pt` 文件,之后 dml 训练脚本直接 `torch.load` 加载(无需 tensorflow):

```python
# 在装有 tensorflow 的环境里跑一次(WSL python 通常已有)
import sys, torch
sys.path.insert(0, "ch06/01_main-chapter-code")
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
torch.save({"settings": settings, "params": params}, "dml-adapt/gpt2-124M-params.pt")
```

`train_classify_ch06.py` 会从脚本同目录读取 `gpt2-124M-params.pt`。缓存文件约 745MB,**不要
提交到 git**(已在 `.gitignore` 规则覆盖范围内的话忽略;否则手动忽略)。

---

## 运行

### 方式 A:Windows 原生(Anaconda Prompt)
```powershell
conda activate dml
cd <repo>\dml-adapt
python dml_device.py            # 看选了哪块 GPU
python train_gpt_ch04.py        # ch04 训练,~85s(7600M XT)
python train_pretrain_ch05.py   # ch05 预训练,~180s
```

### 方式 B:从 WSL 经 run.sh 桥接
先改 `run.sh` 顶部的 `CONDA_DML` 和 `WIN_LOCAL` 两个路径,然后:
```bash
./run.sh train_gpt_ch04.py
./run.sh train_pretrain_ch05.py
```

---

## 已验证的实测结果

设备:`AMD Radeon RX 7600M XT (privateuseone:0)`,GPT 162.4M 参数,batch=2,ctx=256。

**ch04 训练**(`train_gpt_ch04.py`,5 epoch,84.9s):
```
train_loss 6.99 -> 4.34 | val_loss 7.47 -> 6.23 | ~16s/epoch
```

**ch05 预训练**(`train_pretrain_ch05.py`,10 epoch,180.2s)—— 文本从乱码变连贯:
```
epoch 1 : train 7.20  (乱码)
epoch 4 : -> "and I had been the picture--and--and"          (碎片)
epoch 6 : -> "it was not to the picture--I looked up"        (半通顺)
epoch 8 : -> "know," was one of the picture for nothing"      (基本通顺)
epoch 10: train 0.66 -> "Yes--quite insensible to the irony.  (连贯句子 ✓)
                       She wanted him vindicated--and by me!"
```
末句是 Edith Wharton《The Verdict》的真实句子,证明模型真正学到了。
train_loss 降到 0.66、val_loss 在 ~6.1 平台 = 小数据集过拟合(原书也讨论此现象)。

**ch06 分类微调**(`train_classify_ch06.py`,GPT-2 124M,5 epoch,2.83 min):
```
Ep 1: Train loss 0.62→0.52 | Train acc 70.00% | Val acc 72.50%
Ep 3:                       | Train acc 90.00% | Val acc 90.00%
Ep 5: Train loss 0.08       | Train acc 100.00% | Val acc 97.50%
Test accuracy: 95.67%
```
- 加载预训练 GPT-2 124M 权重(从缓存),冻结主体、只微调最后一层 transformer block + final_norm + 新的 2 类输出头
- **测试准确率 95.67%**,与原书 ch06 结果(~95-97%)一致
- ~33s/epoch,全程无 OOM

**ch07 指令微调**(`train_instruction_ch07.py`,GPT-2 **124M** 全参数微调,2 epoch,16.4 min):
```
Initial:  Train loss 6.77 | Val loss 6.64
Ep 2 末:  Train loss ~0.37 | Val loss ~0.68
```
- 用 124M 而非原书的 355M —— **355M 全参数微调(权重+梯度+Adam 状态 fp32 ≈ 6GB)+ 激活在 8GB 显存上放不下**;124M 在 ≥16GB 卡上可切回 355M
- **显存调优**:batch=4(非书的 8)、max_length=512(非 1024);batch=8 会在 ~30 步后 OOM
- 训练中样本生成证明**学会指令跟随**:主动→被动 "The meal every day is cooked by the chef." ✓
- 测试生成:明喻 "as fast as a bullet"(与预期 "as fast as lightning" 均合法 simile)✓;部分答案错误属 124M 容量限制(书用 355M)
- 复用 ch06 的 `gpt2-124M-params.pt` 权重缓存(同一份 124M 权重)

---

## 已知良性现象

- `aten::lerp.Scalar_out ... fall back to CPU` 警告:Adam 优化器一个算子在 DirectML
  未实现,回退 CPU,变慢但不报错。
- torch-directml 非全量 PyTorch,极少数算子会回退 CPU(变慢,不报错)。
- 162M 全模型在 8GB 显存上 batch=2 可跑,无 OOM。

---

## 适用范围与局限

- ✅ 中小模型训练、教学/原型验证、推理部署
- ⚠️ DirectML 比 CUDA/ROCm 原生栈慢(经 DX12 抽象层),不追求极致吞吐
- ❌ 不适合大型分布式训练(那需要裸金属 Linux + ROCm + 受支持型号)

换到 **RX 7700 XT / 7800 XT / 7900 系列**或 **Ryzen AI Max(Strix)** 这类受 ROCm WSL
支持的机器后,可改走 ROCm 7.2.1 + ROCDXG(librocdxg)原生路径,比 DirectML 更快、算子更全。

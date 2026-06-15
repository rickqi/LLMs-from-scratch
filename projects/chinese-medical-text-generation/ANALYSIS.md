# 中文医学文本生成项目 — 全面评估与执行计划

## 一、模型选型分析

### 1.1 候选模型筛选

在 ≤1B 参数范围内搜索支持中文的因果语言模型：

| 模型 | 参数 | 词表大小 | 中文效率 | 预训练数据 | 是否适合 |
|------|------|---------|---------|-----------|---------|
| **Qwen3-0.6B** | 600M | 151,643 | ~1.6 字/token | 36T (中英双语) | ✅ **最佳** |
| Qwen2.5-0.5B | 494M | 151,643 | ~1.6 字/token | 18T (中英双语) | ✅ 可接受 |
| SmolLM2-360M | 360M | 49,600 | ~0.5 字/token | ~4T (英文为主) | ❌ 基本不识中文 |
| Gemma-3-1B | 1,000M | 256,000 | ~0.8 字/token | ~2T (多语，中文弱) | ❌ 中文偏弱 |
| Llama-3.2-1B | 1,000M | 128,000 | ~0.5 字/token | ~9T (英文为主) | ❌ 几乎不识中文 |
| MiniMind | 26-64M | 6,400 | ~1.0 字/token | 少量中文 | ❌ 玩具模型 |

**结论**：≤1B 范围内，只有 Qwen 系列具备真正的中文生成能力。其他厂商不关注 ≤1B 的中文模型。

### 1.2 Qwen3-0.6B vs Qwen2.5-0.5B 深度对比

```
┌────────────────── Qwen2.5-0.5B ────── Qwen3-0.6B ──────────────────┐
│                                                                      │
│  参数规模:       494M ─────────────── 600M (+21%)                    │
│  层数:           24 ───────────────── 28 (+17%)                      │
│  注意力:         14头 MHA ─────────── 16Q+8KV GQA (更高效)          │
│  隐藏维:         896 ──────────────── 1024 (+14%)                    │
│  词表:           151,643 ──────────── 151,643 (相同)                 │
│  上下文:         32K ──────────────── 32K (相同)                     │
│                                                                      │
│  ──────── 训练配置 ────────                                          │
│  预训练数据:     18T tokens ───────── 36T tokens (翻倍)             │
│  中文占比:       ~15% ─────────────── ~18% (估算)                    │
│  训练阶段:       2阶段 ────────────── 3阶段 (含长文训练)             │
│                                                                      │
│  ──────── 医学微调潜力 ──────                                        │
│  Dr. Qwen 论文:  无专门验证 ───────── 已有论文验证医学微调有效        │
│  LoRA adapter:   约0.5M可训练 ─────── 约0.7M可训练                   │
│  微调数据需求:   相同 ─────────────── 相同 (5K+ 条医学文本)          │
│                                                                      │
│  ──────── 硬件需求 ────────                                          │
│  推理显存(fp16):  ~1.2GB ──────────── ~1.5GB                         │
│  训练显存(LoRA):  ~3.5GB ──────────── ~4.0GB                         │
│  消费级可行:      ✅ 4GB GPU即可 ─── ✅ 4GB GPU即可                  │
│                                                                      │
│  推荐度:          ⭐⭐⭐⭐ ──────────── ⭐⭐⭐⭐⭐                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 为什么是唯一最优解

| 原因 | 说明 |
|------|------|
| **中文 tokenizer 不可替代** | 英文模型的 BPE 词表几乎不包含中文字词，1个中文词被拆成 3-5 个 token，等价损失 3-5 倍模型容量 |
| **预训练数据的语言分布** | Qwen 的中文预训练占比 15-18%，其他小型模型基本为 0% |
| **参数效率天花板** | 0.5-0.6B 是中文 LLM 的最小可用规模，低于此的模型不具备可用的中文生成能力 |
| **LoRA 兼容性** | Qwen3 使用标准 Qwen2 架构，与 PEFT 库完美兼容，LoRA 注入无坑 |

---

## 二、技术架构

### 2.1 整体 Pipeline

```
                    ┌──────────────┐
                    │  原始医学数据  │  77 个 .md 文件, ~15M 字符
                    │  raw_data/   │
                    └──────┬───────┘
                           │ data_prep.py
                           ▼
                    ┌──────────────┐
                    │  清洗后数据    │  去HTML/ISBN/OCR → train.txt + val.txt
                    │   data/      │  95% : 5% 分割
                    └──────┬───────┘
                           │ train_qwen_lora.py
                           ▼
            ┌──────────────────────────────┐
            │  hf-mirror.com               │
            │  Qwen/Qwen3-0.6B (600M)      │
            │  + LoRA (rank=8, 0.7M train) │
            └──────────────┬───────────────┘
                           │ 训练 5 epochs
                           ▼
                    ┌──────────────┐
                    │  LoRA 权重    │  output/best_model/
                    │               │  output/final_model/
                    └──────┬───────┘
                           │ generate.py
                           ▼
                    ┌──────────────┐
                    │  医学文本生成  │  批量评估 / 交互模式
                    │               │  记录 training_log.json
                    └──────────────┘
```

### 2.2 数据管理策略

```
projects/chinese-medical-text-generation/
├── data_prep.py          # 数据预处理
├── train_qwen_lora.py    # 训练脚本
├── generate.py           # 推理脚本
├── README.md             # 使用指南
├── ANALYSIS.md           # 本文档
├── requirements.txt      # 依赖
│
├── raw_data/             # (gitignore) 原始 77 个 .md 文件
│   ├── 肿瘤分册-01.md
│   ├── 肿瘤分册-02.md
│   └── ...
│
├── data/                 # (gitignore) 预处理后的数据
│   ├── train.txt         # 训练集 (95%)  ≈14.25M 字符
│   └── val.txt           # 验证集 (5%)   ≈0.75M 字符
│
└── output/               # (gitignore) 训练产出
    ├── best_model/       # 最佳验证 loss 的 LoRA 权重
    ├── final_model/      # 最终 epoch 的 LoRA 权重
    └── training_log.json # 训练日志
```

**数据隔离策略**：`raw_data/`、`data/`、`output/` 均加入 `.gitignore`，只提交代码和文档。

### 2.3 LoRA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `r` (rank) | 8 | 低秩矩阵的秩 |
| `lora_alpha` | 16 | 缩放因子 (alpha/r = 2.0) |
| `lora_dropout` | 0.05 | 正则化 |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj` | 只对注意力层做 LoRA |
| 可训练参数 | ~0.7M / 600M = 0.12% | 极高效的微调 |

选择 `q_proj, k_proj, v_proj, o_proj` 而非全部 Linear 的原因：
- 注意力层负责"理解"，FFN 层负责"记忆"
- 微调医学文本主要是让模型学习新领域的理解方式，不需要改变 FFN 的记忆
- 只对 Attention 做 LoRA 可以在 4GB 显存下训练

### 2.4 训练超参数

| 参数 | 值 | 调整建议 |
|------|-----|---------|
| batch_size | 4 | 内存受限可降到 2 |
| gradient_accumulation | 4 | 等效 batch=16 |
| learning_rate | 2e-4 | LoRA 标准范围 1e-4 ~ 5e-4 |
| warmup | 5% | 前 5% 步数线性预热 |
| scheduler | linear decay | warmup后线性衰减到0 |
| epochs | 5 | 小数据可增至10-20 |
| max_length | 512 | 医学段落通常 <500 字 |

---

## 三、执行清单

### 阶段 1: 环境准备

- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 验证 GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 测试 HF 镜像: `curl -I https://hf-mirror.com`
- [ ] 克隆仓库: `git clone https://github.com/rickqi/LLMs-from-scratch.git`

### 阶段 2: 数据准备

- [ ] 获取原始数据: 中华医学会《临床诊疗指南 — 肿瘤分册》77 个 .md 文件
- [ ] 放入 `raw_data/` 目录
- [ ] 执行: `python data_prep.py --data_dir ./raw_data --output_dir ./data`
- [ ] 验证输出:
  - `data/train.txt` 存在且 > 10MB
  - `data/val.txt` 存在且 > 500KB
  - 脚本输出显示 77 个文件全部处理成功

### 阶段 3: 训练

- [ ] 第一次运行 (测试): `python train_qwen_lora.py --epochs 1`
  - 验证模型可正常下载
  - 验证 DataLoader 工作正常
  - 确认 loss 在下降
- [ ] 正式训练: `python train_qwen_lora.py --epochs 5`
  - 监控 `output/training_log.json`
  - 观察每个 epoch 的生成样本质量
  - 记录训练耗时
- [ ] 检查产出:
  - `output/best_model/adapter_model.safetensors` 存在
  - `output/training_log.json` 包含完整记录

### 阶段 4: 评估

- [ ] 运行基准评估:
  ```bash
  python generate.py --model_dir ./output/best_model
  ```
- [ ] 观察 8 个预设 prompt 的生成质量
- [ ] 与文档中方案 A (GPT-2) 的结果对比
- [ ] (可选) 人工评分:
  - 流畅度 (1-5)
  - 医学准确性 (1-5)
  - 格式规范性 (1-5)

### 阶段 5: 迭代优化 (后续)

- [ ] 增加 epochs: 5 → 10 → 20
- [ ] 调整 LoRA rank: r=8 → r=16
- [ ] 增加训练数据: 仅肿瘤分册 → 全部 77 册 (15M → 50M+ 字符)
- [ ] 尝试更大模型: Qwen3-1.7B (需要更多显存)
- [ ] 构建指令微调数据集: 提取诊断-症状-治疗三元组
- [ ] 集成 RAG 检索: 医学知识库 + 向量搜索 + 生成

---

## 四、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 原始数据不可获取 | 中 | 高 | 使用公开中文医学数据集替代 (Huatuo-26M/ChiMed) |
| HF 镜像不稳定 | 中 | 中 | 先手动下载模型到本地, 本地加载 |
| 训练 loss 不收敛 | 低 | 高 | 降低 lr 到 1e-4, 增加 warmup 比例 |
| 4GB 显存不足 | 低 | 中 | batch_size=1, max_length=256, 使用 QLoRA (4bit) |
| 生成样本医学不准确 | 中 | 中 | 增加训练数据量, 或改用 RAG 方案补充知识 |

---

## 五、附录: Tokenizer 效率对比验证代码

```python
from transformers import AutoTokenizer

# 测试文本
test_texts = [
    "临床表现：患者出现持续性腹痛，伴有恶心呕吐。",
    "诊断标准：根据病理学检查结果，确诊为恶性肿瘤。",
    "治疗方法：首选手术切除，术后辅以化疗。",
    "预后判断：五年生存率约为60%，需定期复查。",
]

# 加载 Qwen3 tokenizer
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 加载 GPT-2 tokenizer (对照组)
import tiktoken
gpt2_tok = tiktoken.get_encoding("gpt2")

print(f"{'文本':<20} {'GPT-2':>8} {'Qwen3':>8} {'压缩比':>8}")
print("-" * 50)
for text in test_texts:
    gpt_len = len(gpt2_tok.encode(text))
    qwen_len = len(qwen_tok.encode(text))
    ratio = gpt_len / qwen_len
    print(f"{text[:16]}...  {gpt_len:>6}  {qwen_len:>6}  {ratio:>6.1f}×")
```

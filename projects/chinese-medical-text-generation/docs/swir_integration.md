# SwiReasoning 集成方案

> 在指令微调完成后，推理时叠加 SwiReasoning 动态推理框架，零成本提升复杂医学推理质量并降低 Token 消耗。

## 一、SwiReasoning 简介

**仓库**: https://github.com/sdc17/SwiReasoning | **协议**: BSD 3-Clause

训练无关（training-free）的动态双推理框架。在 LLM 解码阶段实时监控输出熵值，自动在两种模式间切换：

- **显式 CoT 推理**：熵持续下降（模型思路收敛）→ 输出完整文字推理步骤
- **隐空间潜推理**：熵持续上升（模型不确定性高）→ 停止 Token 输出，仅在 hidden state 空间做多路径推演

**实测数据**（论文，Qwen3-8B）：

| 指标 | 基线 | +SwiReasoning | 提升 |
|------|------|---------------|------|
| 平均准确率 | — | +1.8-3.1% | ↑ |
| Token 效率 | 100% | 143-179% | **57-79% Token 节省** |

---

## 二、与当前方案的定位

```
训练阶段                            推理阶段
════════                            ════════
阶段1: 纯续写微调                    generate.py
阶段2: 指令微调 (ChatML)         →  + SwiReasoning 外挂
  ↓                                  ↓
  给模型注入医学知识+问答格式          让模型在回答时"思考得更聪明"
  (改 LoRA 权重)                    (不改任何权重)
```

**不冲突、不互斥、可叠加**。指令微调解决「知不知道」，SwiReasoning 解决「想不想得深」。

---

## 三、对医学 QA 的预期收益

### 3.1 受益最大的问题类型

| 类型 | 当前问题 | SwiReasoning 效果 |
|------|---------|-----------------|
| **hard/diagnostic** (22%) | 推理深度不足，答案过于简略 | ✅ 熵高→自动隐推理→多路径推演→更完整答案 |
| **comparative** (21%) | 需要对比多方案，容易遗漏 | ✅ 隐空间推演多方案→显式输出对比结果 |
| **easy/factual** (35%) | 简单查询也输出长推理链 | ✅ 熵低→跳过 CoT→直接输出，减少 Token 浪费 |
| **procedural** (27%) | 步骤顺序可能混乱 | ✅ 隐推理消解歧义→输出正确顺序 |

### 3.2 预期指标提升

基于 SwiReasoning 在 Qwen3 系列上的论文数据，保守预估：

| 指标 | 当前 (指令微调) | +SwiReasoning | 说明 |
|------|---------------|---------------|------|
| hard 推理准确率 | — | +2-3% | 隐推理弥补深度不足 |
| 输出 Token 数 | 基准 | -40~60% | easy 问题跳过无意义 CoT |
| 推理延迟 | 基准 | +10-20% | 隐推理步骤额外计算，但 Token 减少抵消 |

---

## 四、技术实现路径

### 4.1 环境依赖

我们已有的依赖与 SwiReasoning 完全重叠：

```
对比:
  SwiReasoning 需要:  transformers, torch, accelerate, datasets, tqdm
  我们已有:           transformers, torch, accelerate, datasets, tqdm

  SwiReasoning 额外需要: sympy, wandb (评估用，非运行时必需)
```

安装：
```bash
cd /home/LLMs-from-scratch
git clone https://github.com/sdc17/SwiReasoning.git external/SwiReasoning
cd external/SwiReasoning
pip install sympy  # 唯一缺失的依赖
```

### 4.2 集成方式：Wrapper 模式

不修改 `generate.py`，新增 `generate_swir.py` 作为增强版推理入口：

```python
# generate_swir.py — SwiReasoning 增强推理
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from swi_reasoning import SwiReasoningDecoder  # SwiReasoning 核心

BASE_MODEL = "Qwen/Qwen3-0.6B"
LORA_PATH = "./output_inst_v1/best_model"

def load_model_with_swir():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # 包裹 SwiReasoning 解码器
    decoder = SwiReasoningDecoder(
        model=model,
        tokenizer=tokenizer,
        alpha=0.5,           # 熵阈值 (0-1, 越低越倾向隐推理)
        max_switch_count=2,  # 最大切换次数
        block_size=6,        # 每块 Token 数
    )
    return decoder, tokenizer

def generate(decoder, prompt: str, max_new_tokens=512):
    return decoder.generate(
        prompt, 
        max_new_tokens=max_new_tokens,
        temperature=0.7, top_p=0.9
    )
```

### 4.3 熵阈值调优

`alpha` 是最关键的参数，需要根据医学 QA 场景调优：

```
alpha=0.3  过于激进 → 几乎所有问题都走隐推理 → 延迟高、不可审计
alpha=0.5  默认值   → 官方推荐起点
alpha=0.7  保守     → 仅高不确定性时隐推理 → 更适合医疗（需要可审计输出）
```

建议医学场景使用 **alpha=0.6-0.7**，优先保证输出可追溯。

### 4.4 最大切换次数

`max_switch_count` 控制推理链路复杂度：

```
max_switch_count=1  → 最多一次模式切换 (显→隐 或 隐→显)
max_switch_count=2  → 可两次切换 (如: 显→隐→显)
max_switch_count=3  → 充分推理，但延迟增加
```

建议 **max_switch_count=2**，平衡深度与效率。

---

## 五、完整管线

### 当前阶段

```
Raw Docs (412篇) → data_prep.py → data_full (52MB)
    │                                    ↓
    │                          train_qwen_lora.py
    │                          (阶段1: 纯续写, 5 epochs)
    │                                    ↓
    │                          output_full/best_model
    │                                    ↓
    │                          cos_backup.py pre-ft-backup
    │                                    ↓
    │                          train_qwen_lora.py --resume_from
    │                          (阶段2: 指令微调, 3 epochs, 449 QA)
    │                                    ↓
    │                          output_inst_v1/best_model
    │                                    ↓
    │                          generate_swir.py  ← [SwiReasoning 集成]
    │                          generate.py       ← [原有推理保留]
```

### SwiReasoning 集成步骤

```bash
# Step 1: 克隆框架
git clone https://github.com/sdc17/SwiReasoning.git external/SwiReasoning
pip install sympy

# Step 2: 验证小模型兼容性
# Qwen3-0.6B 的 hidden_dim=1024，SwiReasoning 论文验证了 Qwen3-8B (hidden_dim=4096)
# 0.6B 可能效果衰减，需实测确认
python external/SwiReasoning/run_chat.py \
    --model_name Qwen/Qwen3-0.6B \
    --method swir --max_switch_count 2

# Step 3: 创建增强推理入口
# scripts/generate_swir.py (见 §4.2)

# Step 4: 对比评估
python generate.py --model_dir ./output_inst_v1/best_model \
    --prompt "胃癌的临床表现有哪些？"          # 基线
python scripts/generate_swir.py --model_dir ./output_inst_v1/best_model \
    --prompt "胃癌的临床表现有哪些？"          # +SwiReasoning

# Step 5: 批量对比 (8 个标准 prompt, 相同问题跑两遍)
python scripts/compare_swir.py  # 输出: docs/swir_comparison.md
```

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Qwen3-0.6B 太小，hidden_dim=1024 隐推理效果差 | 中 | 高 | 先用 `run_chat.py` 快速验证，不行则仅用于 7B+ 模型 |
| 框架 API 不稳定（新项目） | 中 | 中 | 封装 Wrapper 层，API 变化改一处即可 |
| 医疗场景需要可审计输出 | — | 高 | 使用 `alpha=0.7` + `max_switch_count=1`，优先显式推理 |
| 隐推理路径无法追溯 | 低 | 中 | 医疗关键问题关闭隐推理（alpha=1.0, 纯显式 CoT） |

---

## 七、对比：有无 SwiReasoning

| 场景 | 无 SwiReasoning | 有 SwiReasoning |
|------|----------------|-----------------|
| "胃癌的临床表现？" (easy) | 续写 150 tokens，含冗余推理 | 熵低→直接列出症状，~40 tokens |
| "患者56岁男，上腹痛+体重下降，鉴别诊断？" (hard) | 浅层回答，可能遗漏关键鉴别点 | 熵高→隐推理→多路径推演→完整鉴别诊断 |
| "对比CT和MRI在肿瘤分期中的优劣" (comparative) | 可能偏向某一种模态 | 隐空间同时推演两种方案→均衡对比 |
| API 计费 | 全部 Token 计费 | easy 问题 Token 减少 50%+，总成本降低 |

---

## 八、文件清单

```
projects/chinese-medical-text-generation/
├── scripts/
│   ├── generate_swir.py       ← [新建] SwiReasoning 增强推理入口
│   └── compare_swir.py        ← [新建] 有无 SwiReasoning 对比评估
├── docs/
│   └── swir_integration.md    ← 本文档
├── external/
│   └── SwiReasoning/          ← [克隆] 官方开源框架
└── generate.py                 ← [保留] 原有推理入口
```

---

## 九、时间线

```
当前 ──────────────→ 阶段1 完成 ──→ 阶段2 指令微调 ──→ SwiReasoning 集成
                         ↓                ↓                   ↓
                    约 2-3 小时       约 1.5-3 小时         约 30 分钟
                                                          (clone + 测试)
```

SwiReasoning 集成在指令微调完成后执行，作为独立的增强层，不影响任何现有流程。

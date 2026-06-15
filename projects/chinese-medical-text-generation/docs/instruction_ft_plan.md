# 中文医学文本生成 — 指令微调实现方案

> 创建: 2026-06-15 | 基于 ANALYSIS.md §六 展开

---

## 一、目标

将当前**纯续写模型**（输入"临床表现："→ 续写文档）升级为**指令跟随模型**（输入"胃癌的临床表现是什么？"→ 结构化回答），同时保留领域语言能力。

| 当前 | 目标 |
|------|------|
| 训练: raw text continuation | 训练: ChatML instruction following |
| 推理: prompt → 续写 | 推理: user question → assistant answer |
| 可控性: 低 (输出不可预测) | 可控性: 高 (按问题类型结构化输出) |
| 交互: 仅支持续写 prompt | 交互: 问答/总结/推理/对比/操作指导 |

---

## 二、数据状态

### 2.1 当前数据

| 数据集 | 格式 | 数量 | 用途 |
|--------|------|------|------|
| `data_full/train.txt` | raw text (===SEP=== segmented) | 29,650 样本 | 领域语言适应 (已完成) |
| `docs/med_qa_cases.json` | QA pairs with metadata | 172 条 | 指令数据 R1 |
| `docs/med_instruction_chatml.json` | ChatML messages | 172 条 | 可直接用于训练 |

### 2.2 数据扩展计划

```
R1 (已完成): 172 条, 4 领域, 质量验证 ✅
    ↓ 批量扩展: +300~800 条 (同脚本, 增加 --sample-count)
R2 (目标): 500-1000 条
    ↓ 数据增强: 同义改写/反向问答/多文档综合
R3 (目标): 2000-5000 条
```

扩展命令：
```bash
# 修改 DOMAIN_SAMPLES 的 sample_count 后重新运行
python scripts/med_qa_generator.py generate
```

### 2.3 数据混合策略

训练时混合两类数据防止灾难性遗忘：

| 数据类型 | 占比 | 作用 |
|---------|------|------|
| ChatML 指令数据 | 80% | 学习指令跟随 |
| 纯续写数据 (data_full) | 20% | 保留领域语言能力 |

---

## 三、训练策略：续训，不重训

**在现有 LoRA 权重基础上继续训练，而非重新初始化。**

```
Qwen3-0.6B (frozen)
     +
LoRA adapter ──→ 阶段1: 纯续写微调 (data_full, 5 epochs) ──→ output_full/best_model
                                                                    │
                                                    PeftModel.from_pretrained()
                                                                    │
                                                                    ▼
                                              LoRA adapter (加载 output_full 权重)
                                                                    │
                                              阶段2: 指令微调 (ChatML, 3 epochs)
                                                                    │
                                                                    ▼
                                                         output_inst_v1/best_model
                                                    (一个 adapter, 两种能力)
```

**为什么不重训？**

| 原因 | 说明 |
|------|------|
| 医学知识保留 | 5 epochs 学到的医学术语、标准格式、领域表达不会丢失 |
| LoRA 权重连续演化 | 同一组 r=8 低秩矩阵在已有基础上微调，非重新初始化 |
| 推理时一个 adapter | 加载 `output_inst_v1` 即同时具备医学知识 + 指令跟随 |
| PEFT 原生支持 | `PeftModel.from_pretrained()` 加载后 `.train()` 就是续训 |

**唯一例外**：如果当前 continuation 训练 val_loss 不收敛或生成乱码，才需要方案 B（base model + 全新 LoRA + 仅指令数据）。当前 val_loss 从 2.87→2.62 持续下降，状态健康，无需走方案 B。

---

## 四、实现路径

### 阶段 A: 训练脚本改造

#### 3A.1 新增 InstructionDataset

在 `train_qwen_lora.py` 中新增数据集类：

```python
class InstructionDataset(Dataset):
    """加载 ChatML 格式的指令微调数据"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        self.examples = []
        for item in raw["data"]:
            messages = item["messages"]
            # 使用 Qwen3 原生 chat_template
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokenized = tokenizer(
                text, truncation=True, max_length=max_length, 
                return_tensors=None
            )
            input_ids = tokenized["input_ids"]
            
            # Loss masking: 只对 assistant 部分计算 loss
            labels = input_ids.copy()
            assistant_start = self._find_assistant_start(input_ids)
            for i in range(assistant_start):
                labels[i] = -100
            
            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
            })
    
    def _find_assistant_start(self, input_ids: list[int]) -> int:
        """找到 assistant 回复的起始位置"""
        # assistant 标记: <|im_start|>assistant\n
        assistant_token = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        for i in range(len(input_ids) - len(assistant_token) + 1):
            if input_ids[i:i+len(assistant_token)] == assistant_token:
                return i + len(assistant_token)
        return 0  # fallback: 全部计算 loss
```

#### 3A.2 数据混合加载

```python
class MixedDataset(Dataset):
    """混合指令数据和纯续写数据"""
    
    def __init__(self, instruction_path: str, continuation_path: str, 
                 tokenizer, max_length: int = 512, 
                 instruction_ratio: float = 0.8):
        self.inst_dataset = InstructionDataset(instruction_path, tokenizer, max_length)
        self.cont_dataset = MedicalTextDataset(continuation_path, tokenizer, max_length)
        self.inst_len = len(self.inst_dataset)
        self.cont_len = len(self.cont_dataset)
        self.inst_ratio = instruction_ratio
        self.total = self.inst_len + self.cont_len
    
    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        # 按比例采样
        if random.random() < self.inst_ratio:
            return self.inst_dataset[random.randint(0, self.inst_len - 1)]
        else:
            return self.cont_dataset[random.randint(0, self.cont_len - 1)]
```

#### 3A.3 训练参数调整

| 参数 | 纯续写 (当前) | 指令微调 (建议) | 原因 |
|------|-------------|---------------|------|
| learning_rate | 1e-4 | 5e-5 | 指令数据量更小，需要更低 lr |
| epochs | 5 | 3 | 防止过拟合小数据集 |
| max_length | 512 | 768 | ChatML 格式开销 ~100 tokens |
| LoRA r | 8 | 8 | 保持不变 |
| data | data_full (52MB) | mixed (inst + cont) | 混合防遗忘 |

### 阶段 B: 训练执行

```bash
# 1. 确保 ChatML 数据已导出
python scripts/med_qa_generator.py export --fmt chatml

# 2. 指令微调训练（在 output_full LoRA 权重基础上续训）
python train_qwen_lora.py \
    --resume_from ./output_full/best_model \
    --data_dir ./data_full \
    --instruction_data ./docs/med_instruction_chatml.json \
    --output_dir ./output_inst_v1 \
    --epochs 3 \
    --lr 5e-5 \
    --max_length 768 \
    --instruction_ratio 0.8

# 3. 评估
python generate.py --model_dir ./output_inst_v1/best_model --interactive
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--resume_from` | `./output_full/best_model` | 加载 continuation 阶段的 LoRA 权重作为起点 |
| `--lr` | `5e-5` | 已有基础的模型用更低学习率（原 1e-4 的一半） |
| `--epochs` | `3` | 指令数据量小（172-500条），少轮次防过拟合 |
| `--max_length` | `768` | ChatML 格式比纯文本多 ~100 token 开销 |
| `--instruction_ratio` | `0.8` | 80% 指令 + 20% 纯续写，防灾难性遗忘 |

### 阶段 B: 质量评估

### 评估维度

| 维度 | 方法 | 指标 |
|------|------|------|
| 指令跟随 | 人工评分 (1-5) | 是否按要求格式/内容回答 |
| 医学准确性 | 原文对照 | 回答是否有原文依据 |
| 流畅度 | 人工评分 (1-5) | 中文表达是否自然 |
| 抗遗忘 | 纯续写测试 | "临床表现："续写质量是否退化 |

### 对比基线

```
纯续写模型 (output_full)  vs  指令微调模型 (output_inst_v1)
         ↓                              ↓
  prompt: "胃癌的临床表现？"       prompt: "胃癌的临床表现？"
  output: 续写文档后续段落        output: 结构化列举临床表现
```

---

## 五、下一步建议

### 优先级排序

```
[高] 1. 训练脚本改造 → 支持 InstructionDataset + MixedDataset
         └─ 预计工作量: 2-3 小时
         └─ 关键风险: tokenizer.apply_chat_template 验证

[高] 2. 数据扩展到 500+ 条
         └─ 命令: python scripts/med_qa_generator.py generate
         └─ 修改 DOMAIN_SAMPLES.sample_count *= 1.5

[中] 3. 指令微调实验 (3 epochs, lr=5e-5)
         └─ 验证: loss 下降 + 生成样本评估
         └─ 预计时间: ~1h (172 条) / ~2h (500 条)

[中] 4. 质量对比评估
         └─ 8 个标准 prompt 的前后对比
         └─ 人工评分: 指令跟随 + 医学准确性 + 流畅度

[低] 5. DPO 偏好对齐
         └─ 收集好/坏回答对
         └─ DPO 训练提升输出质量

[低] 6. 多任务扩展
         └─ 增加 summarization/extraction/reasoning 类型
         └─ 扩展 data_prep.py 支持自动构造
```

### 详细行动计划

#### Step 1: 训练脚本改造 (建议立即开始)

文件: `train_qwen_lora.py`

改动清单:
- [ ] 新增 `InstructionDataset` 类
- [ ] 新增 `MixedDataset` 类
- [ ] `collate_fn` 适配：instruction 和 continuation 共享同一 collate
- [ ] `main()` 新增 `--instruction_data` 和 `--instruction_ratio` 参数
- [ ] 训练日志区分 instruction loss 和 continuation loss
- [ ] 生成样本时使用 ChatML 格式的 prompt（如需要）

#### Step 2: 验证 chat_template

```python
# 快速验证脚本
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
msgs = [
    {"role": "system", "content": "你是医学助手"},
    {"role": "user", "content": "胃癌的症状？"},
    {"role": "assistant", "content": "胃癌症状包括..."},
]
text = tok.apply_chat_template(msgs, tokenize=False)
print(text)
# 预期输出格式:
# <|im_start|>system\n你是医学助手<|im_end|>\n<|im_start|>user\n...
```

#### Step 3: 数据扩展

修改 `scripts/med_qa_generator.py` 中的 `DOMAIN_SAMPLES` 配置：

```python
DOMAIN_SAMPLES = {
    "L1_卫健委官方规范": {"sample_count": 18, ...},    # 12 → 18
    "L2_卫生行业标准_WS": {"sample_count": 15, ...},    # 10 → 15
    "L3_中华医学会临床诊疗指南丛书": {"sample_count": 18, ...},  # 12 → 18
    "L4_中华医学会临床技术操作规范": {"sample_count": 12, ...},  # 8 → 12
}
```

#### Step 4: 训练 + 评估

```bash
# 训练
python train_qwen_lora.py \
    --data_dir ./data_full \
    --instruction_data ./docs/med_instruction_chatml.json \
    --output_dir ./output_inst_v1 \
    --epochs 3 --lr 5e-5 --max_length 768

# 评估
python generate.py --model_dir ./output_inst_v1/best_model --interactive
```

---

## 六、风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Qwen3 chat_template 不匹配 | 低 | 先用 `apply_chat_template` 验证输出格式 |
| 172 条数据不够激活指令能力 | 中 | 扩展到 500+ 再训练；或先用 LoRA rank=16 |
| 灾难性遗忘（纯续写能力退化） | 中 | 混合 20% 纯续写数据；监控 continuation loss |
| 指令数据过拟合 | 低 | 3 epochs + lr=5e-5 保守策略 |
| 生成回答过长/过短 | 低 | 调整 max_new_tokens + temperature |
| DeepSeek API 不可用（数据扩展时） | 低 | 切换到 GLM API (`LLM_PROVIDER=glm`) |

---

## 七、文件清单

```
projects/chinese-medical-text-generation/
├── train_qwen_lora.py              # [待改造] +resume_from, +InstructionDataset
├── scripts/med_qa_generator.py     # [已完成] QA 生成脚本
├── docs/
│   ├── med_qa_cases.json           # [已完成] 172 条 QA
│   ├── med_instruction_chatml.json # [已完成] ChatML 格式
│   ├── med_instruction_alpaca.json # [已完成] Alpaca 格式
│   ├── med_qa_report.md            # [已完成] 质量报告
│   └── instruction_ft_plan.md      # 本文档
├── data_full/                      # 纯续写数据 (20% 混合用)
├── output_full/                    # [训练中] 阶段1 LoRA 权重
└── output_inst_v1/                 # [待创建] 阶段2 指令微调产出
```

---

## 附录 A: 验证脚本

```python
# validate_chat_template.py — 验证 Qwen3 chat_template 输出
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

data = json.load(open("docs/med_instruction_chatml.json"))
sample = data["data"][0]

text = tok.apply_chat_template(sample["messages"], tokenize=False)
print("=== ChatML 格式输出 ===")
print(text)
print()

encoded = tok(text, return_tensors="pt")
print(f"Token 数: {encoded['input_ids'].shape[1]}")
print(f"是否 <= 768: {encoded['input_ids'].shape[1] <= 768}")
```

## 附录 C: resume_from 实现要点

```python
# train_qwen_lora.py main() 中新增逻辑

def main():
    parser.add_argument("--resume_from", type=str, default=None,
                        help="已有 LoRA adapter 路径 (继续训练)")

    # ... 加载 base model ...

    if args.resume_from:
        # 加载已有 LoRA adapter 作为起点
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        logger.info(f"从已有 LoRA 权重续训: {args.resume_from}")
    else:
        # 全新 LoRA 初始化
        lora_config = LoraConfig(...)
        model = get_peft_model(model, lora_config)

    # 之后的训练循环完全不变
```

**注意事项**：
- `is_trainable=True` 确保 LoRA 参数保持 requires_grad=True
- LoRA config (r, alpha, target_modules) 必须与 `resume_from` 的 adapter 匹配

## 附录 D: 评估 prompts

```python
TEST_INSTRUCTION_PROMPTS = [
    "胃癌的典型临床表现有哪些？请列举。",
    "WS/T 862-2025 中导尿管相关尿路感染的预防措施是什么？",
    "请对比 CT 和 MRI 在肿瘤分期中的优缺点。",
    "一位56岁男性，上腹痛伴体重下降，应考虑哪些鉴别诊断？",
    "请描述气管插管的标准操作步骤。",
    "肺癌的TNM分期标准是什么？",
    "手术后需要观察哪些并发症？请列出。",
    "请根据指南，给出胃癌术后化疗的推荐方案。",
]
```

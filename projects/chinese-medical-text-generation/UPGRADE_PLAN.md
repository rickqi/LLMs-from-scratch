# 多模型训练扩展设计方案

> 基于现有 Qwen3-0.6B 训练基础设施，扩展支持 Qwen3.5-2B 和 Qwen3-1.7B

---

## 一、现有程序分析

### 1.1 已具备的能力

| 能力 | 实现位置 | 状态 |
|------|---------|------|
| `--model_name` 切换模型 | `train_qwen_lora.py:197` | ✅ |
| `--output_dir` 隔离产出 | `train_qwen_lora.py` | ✅ |
| `--base_model` 推理指定 | `generate.py:104` | ✅ 刚修复 |
| `--resume_from` 续训 | `train_qwen_lora.py` | ✅ |
| `--instruction_data` 指令微调 | `train_qwen_lora.py` | ✅ |
| LoRA 自动配置 | `train_qwen_lora.py` | ✅ (target_modules 通用) |
| Checkpoint 断点续训 | `train_qwen_lora.py` | ✅ |
| 滑动窗口数据加载 | `MedicalTextDataset` | ✅ |
| ChatML 指令数据 | `InstructionDataset` | ✅ |

### 1.2 缺少的能力

| 能力 | 缺失影响 | 优先级 |
|------|---------|--------|
| VRAM 自适应 batch_size | 不同模型需手动调 batch | 🟡 中 |
| 训练进度汇总面板 | 无法一眼看所有模型状态 | 🟢 低 |
| 输出目录自动命名 | 需手动指定 `--output_dir` | 🟢 低 |

### 1.3 已完成的探索

| 模型 | 数据集 | 状态 | val_loss | 耗时 |
|------|--------|------|---------|------|
| Qwen3-0.6B | 374 篇 | ✅ | 2.403 | 11h |
| Qwen3-0.6B | 10 CACA | ✅ | 3.148 | 25min |
| Qwen3.5-2B | 10 CACA | ✅ | 2.457 | 10h (3epoch) |
| Qwen3-1.7B | — | 🔲 | — | — |

---

## 二、Qwen3.5-2B 和 Qwen3-1.7B 对比

### 2.1 模型规格

| 规格 | Qwen3-0.6B | Qwen3-1.7B | Qwen3.5-2B |
|------|-----------|-----------|-----------|
| 参数 | 600M | 1.7B | 2B |
| 架构 | Qwen3 | Qwen3 | Qwen3.5 |
| 层数 | 28 | 28 | ~32 |
| GQA | 16Q/8KV | 16Q/8KV | 16Q/8KV |
| 缓存路径 | `~/.cache/huggingface/` | `~/.cache/modelscope/` | `/home/models/ms_cache/` |
| 缓存大小 | 1.5GB | 待下载(666MB partial) | 4.3GB |
| tokenizer | BPE 151K | BPE 151K | BPE 151K |

### 2.2 VRAM 预估 (batch=4, max_len=512)

| 模型 | 模型大小(bf16) | LoRA | 激活+优化器 | 总计 | RTX5080 16GB |
|------|--------------|------|-----------|------|-------------|
| 0.6B | 1.2GB | 0.5GB | 2.5GB | ~4GB | ✅ 充裕 |
| 1.7B | 3.4GB | 0.5GB | 5GB | ~9GB | ✅ 安全 |
| 2B | 4GB | 0.5GB | 11GB | ~15.5GB | ⚠️ 紧张 |

### 2.3 训练速度预估 (374篇, 29,650样本, batch=4)

| 模型 | 每 batch 耗时 | 每 epoch | 5 epochs 总耗时 | 可行？ |
|------|-------------|----------|---------------|--------|
| 0.6B | ~1.1s | 2.2h | 11h | ✅ |
| 1.7B | ~4s (估) | ~8h | **~40h** | ⚠️ 需2天 |
| 2B | ~18-21s (实测) | ~37h | **~185h** | ❌ 不可行 |

> **关键发现**: Qwen3.5-2B 在 RTX 5080 16GB 上训练极慢（实测 19x 慢于 0.6B），原因是 VRAM 接近满负荷导致内存管理开销巨大。**推荐仅用 1.7B 做全量训练**，2B 仅做小规模对比实验。

---

## 三、数据隔离方案

### 3.1 目录隔离（零冲突保证）

```
projects/chinese-medical-text-generation/
├── data_full/              ← 共享训练数据 (只读)
│   ├── train.txt           (52MB, OCR清洗)
│   └── val.txt             (2.4MB)
│
├── output_full/            ← Qwen3-0.6B, 纯续写
├── output_inst_v3/         ← Qwen3-0.6B, 指令微调
├── output_oncology/        ← Qwen3-0.6B, 10篇对比基线
│
├── output_2b_full/         ← Qwen3.5-2B, 纯续写 (新建)
├── output_2b_inst/         ← Qwen3.5-2B, 指令微调 (新建)
├── output_17b_full/        ← Qwen3-1.7B, 纯续写 (新建)
├── output_17b_inst/        ← Qwen3-1.7B, 指令微调 (新建)
```

**隔离原则**:
1. 每个模型用独立 `--output_dir`，绝对无交叉
2. 训练数据 `data_full/` 只读共享
3. LoRA adapter 各自保存，互不干扰
4. 日志文件按模型命名 (`train_2b_full.log`, `train_17b_full.log`)

### 3.2 GPU 隔离

```
同一时间只允许一个训练任务占用 GPU。
启动前检查: nvidia-smi → 确认无其他训练进程
```

---

## 四、代码改动清单

### 4.1 零改动项（现有能力直接覆盖）

无需任何代码修改即可启动：

```bash
# 1.7B 全量训练
python train_qwen_lora.py \
  --model_name Qwen/Qwen3-1.7B \
  --data_dir ./data_full \
  --output_dir ./output_17b_full \
  --epochs 5 --batch_size 4 --max_length 512 --lr 1e-4 \
  > train_17b_full.log 2>&1 &

# 2B 指令微调 (在 2B LoRA 基础上)
python train_qwen_lora.py \
  --model_name /home/models/ms_cache/Qwen/Qwen3___5-2B \
  --data_dir ./data_full \
  --instruction_data ./docs/med_instruction_chatml.json \
  --resume_from ./output_2b_full/best_model \
  --output_dir ./output_2b_inst \
  --epochs 3 --batch_size 2 --max_length 512 --lr 5e-5

# 推理
python generate.py \
  --model_dir ./output_2b_full/best_model \
  --base_model /home/models/ms_cache/Qwen/Qwen3___5-2B \
  --prompt "临床表现："
```

### 4.2 建议改动（可选，提升体验）

#### A. VRAM 自适应降级

```python
# train_qwen_lora.py main() 中新增
try:
    model = ...  # 原始加载
except torch.cuda.OutOfMemoryError:
    logger.warning("OOM! 降级 batch_size 重试")
    args.batch_size = max(1, args.batch_size // 2)
    model = ...  # 重新加载
```

#### B. 训练清单脚本 `train_all.sh`

```bash
#!/bin/bash
# 顺序训练所有模型 (避免 VRAM 冲突)
MODELS=(
  "Qwen/Qwen3-1.7B:output_17b_full:5:4"
)
for entry in "${MODELS[@]}"; do
  IFS=':' read model out epochs batch <<< "$entry"
  echo "=== 训练 $model → $out ==="
  python train_qwen_lora.py \
    --model_name "$model" --data_dir ./data_full \
    --output_dir "./$out" --epochs "$epochs" \
    --batch_size "$batch" --lr 1e-4
  echo "完成! 等待 GPU 冷却..."
  sleep 60
done
```

---

## 五、执行路线

### Phase 1: 1.7B 全量训练（优先）

```bash
# 先确保 1.7B 模型已下载
python train_qwen_lora.py \
  --model_name Qwen/Qwen3-1.7B \
  --data_dir ./data_full \
  --output_dir ./output_17b_full \
  --epochs 5 --batch_size 4 --lr 1e-4 \
  > train_17b_full.log 2>&1 &
```

**预估**: ~40h (1.7天)，VRAM ~9GB，可后台运行

### Phase 2: 1.7B 指令微调

```bash
python train_qwen_lora.py \
  --model_name Qwen/Qwen3-1.7B \
  --data_dir ./data_full \
  --instruction_data ./docs/med_instruction_chatml.json \
  --resume_from ./output_17b_full/best_model \
  --output_dir ./output_17b_inst \
  --epochs 3 --batch_size 4 --lr 5e-5
```

**预估**: ~24h (1天)

### Phase 3: 2B 小规模实验（已基本完成）

仅用于 10 篇对比实验，不做全量训练。

---

## 六、风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 1.7B 下载缓慢 | 中 | 使用 ModelScope 国内镜像 |
| 1.7B OOM | 低 | batch=4 安全，可降至 2 |
| 训练中断 | 中 | checkpoint 自动续训 |
| 40h 训练期间断电 | 低 | checkpoint + 建议配 UPS |
| 日志文件过大 | 低 | 定期 `> train_17b_full.log` |

---

## 七、产出物总览

| 模型 | 纯续写 | 指令微调 | 推理命令 |
|------|--------|---------|---------|
| 0.6B | `output_full/` (8.8MB) | `output_inst_v3/` (8.8MB) | `generate.py --model_dir ./output_full/best_model` |
| 1.7B | `output_17b_full/` | `output_17b_inst/` | `generate.py --model_dir ./output_17b_full/best_model --base_model Qwen/Qwen3-1.7B` |
| 2B | `output_2b_full/` | `output_2b_inst/` | `generate.py --model_dir ./output_2b_full/best_model --base_model /home/models/ms_cache/Qwen/Qwen3___5-2B` |

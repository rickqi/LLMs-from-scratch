# AGENTS.md — 公司规章制度文本生成项目

> AI Agent 工作流文档。描述项目结构、当前状态、关键文件和操作入口。

## 项目概述

基于 Qwen3-0.6B + LoRA 的公司规章制度领域文本生成，在 AMD Radeon 890M GPU 上训练。

- 数据: 3023 篇公司制度文档 (53.2 MB, 17.7M tokens)
- 模型: Qwen3-0.6B + LoRA (r=8, 2.3M 可训练参数)
- 硬件: AMD 890M (ROCm 7.2, bf16, VRAM 6.1 GB)

## 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| 数据准备 | ✅ 完成 | 3023 篇训练 + 160 篇验证 |
| Epoch 1 | 🔄 训练中 (92%) | val_loss 2.91→1.72 |
| Epoch 2-3 | ⏳ 待执行 | 断点续训自动衔接 |
| COS 备份 | ✅ 已备份 | 74.8 MB (best_model + 数据) |
| 模型评估 | ⏳ 待执行 | 等 3 epochs 完成后 |

## 关键文件索引

| 文件 | 用途 | 何时读取 |
|------|------|---------|
| `README.md` | 项目概览、快速开始、实测数据 | 首次接触项目 |
| `CHANGELOG.md` | 更新记录 | 了解变更历史 |
| `docs/训练方案.md` | 完整训练方案 + LoRA 影响分析 | 理解设计决策 |
| `data_prep.py` | 数据清洗与分割 | 准备数据 |
| `train_qwen_lora.py` | 训练脚本 | 执行训练 |
| `generate.py` | 推理脚本 | 模型评估 |
| `scripts/cos_backup.py` | COS 云备份 | 数据/模型备份 |

## 操作入口

### 训练

```bash
cd projects/company-regulations-training

# 正常训练 (断点自动恢复)
python train_qwen_lora.py \
    --data_dir ./data \
    --output_dir ./output \
    --epochs 3 \
    --batch_size 4 \
    --max_length 256 \
    --lr 2e-4
```

### 推理

```bash
# 批量评估
python generate.py --model_dir ./output/best_model

# 交互模式
python generate.py --model_dir ./output/best_model --interactive

# 单次生成
python generate.py --model_dir ./output/best_model --prompt "制度名称："
```

### COS 备份

```bash
export COS_SECRET_ID="your_id"
export COS_SECRET_KEY="your_key"

python scripts/cos_backup.py backup              # 全量
python scripts/cos_backup.py backup --incremental # 增量
python scripts/cos_backup.py list                 # 查看远程
```

### 监控训练

```bash
# 实时日志
tail -f train.log

# 查看进度
grep -E "(Step|Batch)" train.log | tail -10

# tmux 会话
tmux attach -t company-train
```

## 行动路线图

### 当前: 等待 Epoch 1 完成

```
Epoch 1 完成 (~2h)
    │
    ▼
[ ] Step 1: COS 增量备份 (epoch 1 checkpoint)
    python scripts/cos_backup.py backup --incremental
    │
    ▼
[ ] Step 2: Epoch 2-3 自动续训 (断点恢复)
    │
    ▼
[ ] Step 3: 最终评估
    python generate.py --model_dir ./output/best_model
    │
    ▼
[ ] Step 4: COS 完整备份
    python scripts/cos_backup.py backup
```

### 训练完成后

```
3 epochs 完成
    │
    ├── [P1] 温度采样生成评估
    │   对比 greedy vs temp=0.7 vs temp=0.7+top_k=5
    │
    ├── [P2] 指令微调 (阶段 2, 可选)
    │   构建公司制度 Q&A 对 → ChatML 格式 → 指令微调
    │   参考医学项目 docs/instruction_ft_plan.md
    │
    ├── [P3] RAG 方案 (可选)
    │   不微调, 用 Qwen + 向量检索 + 制度知识库
    │   更适合知识密集型查询
    │
    └── [P4] 定量评估
        建立标准测试集: 50 个制度 QA
        评分: 准确性 / 完整性 / 格式规范性
```

## 开发约定

- 数据和模型产出加入 `.gitignore`，只提交代码和文档
- COS 备份到 `ins-kq6zz7wo-1313469539` (ap-guangzhou)
- 训练日志写入 `train.log`
- 断点检查点写入 `output/checkpoint/`
- HuggingFace 模型通过 `hf-mirror.com` 下载
- AMD 890M 使用 bf16 精度, `device_map={"": device}` 加载

## 环境依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| PyTorch | 2.9.1+rocm7.2 | AMD GPU 支持 |
| transformers | 5.12+ | Qwen3 模型加载 |
| peft | 0.19+ | LoRA 微调 |
| cos-python-sdk-v5 | 1.9+ | COS 备份 |
| tiktoken | (仅 GPT-2 对比用) | 效率对比 |

## 关联项目

- `chinese-medical-text-generation` — 医学文本生成 (参考项目, 同一训练方案)
- `LLMs-from-scratch` — 主仓库 (学习笔记 + CN notebooks)

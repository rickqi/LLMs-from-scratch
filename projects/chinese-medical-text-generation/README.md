# 中文医学诊疗指南文本生成

使用 Qwen2.5-0.5B + LoRA 对中文医学诊疗指南进行文本生成微调。

## 方案对比

| 维度 | 方案 A: GPT-2 124M | 方案 B: Qwen2.5-0.5B |
|------|-------------------|---------------------|
| Tokenizer | GPT-2 BPE (50257 vocab, 0.7 中文/token) | Qwen BPE (151643 vocab, 1.6 中文/token) |
| 预训练 | 英文为主 | 中英双语 |
| 微调方式 | 全量训练 | LoRA (0.1% 参数) |
| 训练前中文 | 完全乱码 | 通顺中文 |
| 推荐度 | ❌ 不适合中文 | ✅ 中文项目首选 |

## 使用流程

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将 77 个 `.md` 医学指南文件放入 `raw_data/` 目录后执行:

```bash
python data_prep.py --data_dir ./raw_data --output_dir ./data
```

> 如果没有原始数据, 脚本会提示。数据来源: 中华医学会《临床诊疗指南 — 肿瘤分册》。

### 3. 训练

```bash
python train_qwen_lora.py --data_dir ./data --output_dir ./output --epochs 5
```

关键参数:
- `--batch_size`: 批次大小 (默认 4)
- `--lr`: 学习率 (默认 2e-4)
- `--max_length`: 最大序列长度 (默认 512)

训练过程会:
- 自动从 hf-mirror.com 下载 Qwen2.5-0.5B
- 应用 LoRA (rank=8, target: q_proj/k_proj/v_proj/o_proj)
- 每 10 步评估验证集 loss
- 每个 epoch 生成样本文本
- 保存最佳模型到 `output/best_model`
- 记录训练日志到 `output/training_log.json`

### 4. 生成

```bash
# 批量评估
python generate.py --model_dir ./output/best_model

# 单次生成
python generate.py --model_dir ./output/best_model --prompt "临床表现："

# 交互模式
python generate.py --model_dir ./output/best_model --interactive
```

### 5. 结果记录

训练完成后检查 `output/training_log.json`, 包含:
- train_loss / val_loss 曲线
- 最佳验证 loss
- 总耗时
- 超参数配置

## 扩展方向

1. **指令微调**: 构建医学 Q&A 对, 做问答助手
2. **RAG 方案**: 检索增强生成 + 医学知识库
3. **更大模型**: Qwen2.5-7B 或 Qwen3-8B (需要 GPU ≥16GB)
4. **多模态**: 加入医学影像 (X光/CT)
5. **评估**: 用执业医师资格考试题评估

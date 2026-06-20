# AGENTS.md — LLMs-from-scratch 项目群

> 书籍配套代码 + 4个独立实战项目。中文优先。2026-06-19 更新。

## 仓库结构（两个独立层）

```
LLMs-from-scratch/
├── ch01/ ~ ch07/ + appendix-*/   ← 书籍代码 (PyTorch, 独立运行)
├── pkg/llms_from_scratch/        ← PyPI 包 (pip install llms-from-scratch)
├── projects/                     ← 4个独立项目 (各自有不同的依赖/环境)
│   ├── kronos-stock-predictor/
│   ├── chinese-medical-text-generation/
│   ├── company-regulations-training/
│   └── gbcost-insurance-ml/
└── projects/docs/                 ← 跨项目文档
```

**关键规则**：4 个项目相互独立。不要在项目间混用依赖或假设共享环境。

## 环境与依赖

- **环境管理**: pixi (conda-forge) 或 uv/pip。`pyproject.toml` 定义书籍代码依赖。
- **HF 镜像**: **所有 HuggingFace 下载必须用 `hf-mirror.com`**。训练脚本已内置 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`。
- **Python 要求**: 书籍代码 3.10-3.12。项目各自独立（医疗需要 3.10+，控费需要 3.11+）。
- **GPU 情况**: NVIDIA RTX 5080 (15.9GB VRAM) 用于医疗项目。AMD 890M (6.1GB, ROCm 7.2) 用于法规项目。Kronos 支持 CPU 训练。

## 测试

```bash
# 书籍代码测试（根目录）
python -m pytest pkg/llms_from_scratch/tests/ -v

# 各项目独立测试
cd projects/kronos-stock-predictor && python -m pytest tests/ -v
cd projects/gbcost-insurance-ml && python -m pytest tests/ -v   # 4549 用例
```

## 四个项目速查

### 1. Kronos Stock（金融时序预测）

- **核心发现**: OHLCV-only 无效。VPIN/DPIN 知情交易因子是关键（因子工程 > 模型复杂度）。
- **最佳模型**: LSTM 波动率预测，RankIC=+0.579。
- **训练**: `bash run.sh` 一键执行。或分步：`python data/preprocessor.py` → `train/train_tokenizer.py` → `train/train_predictor.py` → `inference/generate.py`。
- **教程覆盖**: Ch6 分类微调（`scripts/ch6_classification.py`，涨跌二分类 Acc=55%）。

### 2. 医疗文本生成（中文医学 LLM）

- **当前最佳模型**: `output_17b_dpo_v1` (Qwen3-1.7B + LoRA + DPO, 11.5s/题, 0 坍塌)。
- **模型排名**: 🥇1.7B DPO > 🥈1.7B Inst-V1 > 🥉0.6B DPO v3 > 0.6B Inst-V3。
- **DPO 关键陷阱**: chosen/rejected 长度差 >50% 会导致模式坍塌。必须过滤长度偏差。beta=0.05 + 1 epoch 足矣。1.7B DPO 需 batch_size=1（双模型 OOM）。
- **训练阶段**: 必须先续写微调（阶段1）再指令微调（阶段2）。`--resume_from` 加载阶段1 LoRA 权重。
- **训练脚本已内置早停和过拟合检测**：`--early_stopping_patience 30 --min_delta 0.001 --overfit_gap_threshold 0.5`。
- **指令微调推荐参数**: `--lr 1e-5 --instruction_ratio 0.4 --epochs 1 --max_length 512`。
- **独立验证集**: `docs/med_instruction_val_chatml.json` (50条 hold-out QA)。
- **关键陷阱**: lr=5e-5 + ratio=0.8 会导致严重过拟合（gap>1.0）。必须用低 lr + 低 ratio。
- **推理**: `python generate.py --model_dir ./output_17b_inst_v2/best_model --instruct --prompt "问题"`。
- **1.7B 基座模型路径**: `/home/models/ms_cache/Qwen/Qwen3-1___7B`（已缓存，3.8GB）。

### 3. 法规文本生成（公司制度）

- **硬件**: AMD 890M (ROCm 7.2, bf16)。**不是 NVIDIA GPU**，不要用 CUDA 参数。
- **关键优化**: max_length=256（非默认 512），否则训练极慢。
- **当前状态**: Epoch 1 训练中（val_loss 2.91→1.72）。
- **数据**: 3023 篇公司制度（53.2MB），COS 备份到 `ins-kq6zz7wo-1313469539`。

### 4. 控费 ML（保险理赔）

- **这是一个独立的大项目**（593行 AGENTS.md），有自己完整的 CLI + TUI + ML 管线。
- **命令名陷阱**: 全局命令是 `ghb-cost-control`（**连字符**），不是 `ghb_cost_control`（下划线）。
- **等效命令**: `python -m src.cli` 可替代全局命令。
- **三 LLM 提供商**: Provider A (GLM-5-Turbo, 报告), Provider B (DeepSeek-V4-Pro, 分析), Provider C (DeepSeek, 备用)。
- **Config 单例污染**: Phase 1 的 `Config.set_llm_enabled(False)` 会持久修改全局 Config。TUI 通过 `_restore_llm_config()` 解决。
- **conftest 自动禁用 LLM**: 测试 fixture 会删除 API key 环境变量 + 设置 `rule_primary` 模式。
- **prompt_toolkit 3.0.52+**: `Keys.Space` 已移除，空格键必须用字符串 `" "`。
- **Polars 类别编码**: 训练和预测必须用相同的 category_mappings.json，否则 R² 从 0.91 崩到 0.02。
- **ML 预测**: 独立子系统 `python -m ml.pipeline.train/predict/evaluate`。

## COS 云备份（3/4 项目共用）

- 桶: `ins-kq6zz7wo-1313469539` (ap-guangzhou)
- 凭证: 从环境变量 `COS_SECRET_ID` / `COS_SECRET_KEY` 读取
- 路径前缀: `LLMs-from-scratch/projects/{project_name}/`

## 教程覆盖度追踪

- 总览文档: `projects/docs/curriculum-coverage-analysis.md`（当前 98%）
- 缺口: Ch7 DPO 偏好对齐（完成后 100%）
- 更新覆盖度时同步更新此文件，不要另建新文件

## 通用陷阱

1. **不要假设项目间共享 Python 环境** — 每个项目有独立的 requirements.txt。
2. **医疗项目 1.7B 基座路径是绝对路径** — `/home/models/ms_cache/Qwen/Qwen3-1___7B`，不是 HuggingFace 模型名。
3. **训练脚本默认 batch_size=4** — 这在 0.6B 模型上 OK，但 1.7B 需要 batch_size=2。
4. **所有 generate.py 脚本** — 默认是批量评估模式。需要 `--prompt` 或 `--interactive` 才能交互使用。
5. **不要删除或修改 `output_*/` 下的 best_model** — 这是多轮训练的中间产物。
6. **书籍代码和项目代码风格不同** — 书籍代码遵循教学清晰性，项目代码遵循工程实践。不要统一风格。

# LLMs-from-Scratch 四项目全面分析研究报告

> 生成日期: 2026-06-21 | 更新: P0+P1 Kronos 优化完成 | 基于实际训练结果 + GitHub OSS 对比分析

---

## 一、执行摘要

本报告对 LLMs-from-Scratch 仓库下的 4 个独立项目进行全面的技术分析，涵盖：

1. **Kronos Stock Predictor** — 金融时序预测（自定义 Transformer）
2. **Chinese Medical Text Generation** — 中文医学文本生成（Qwen3 + LoRA + DPO）
3. **Company Regulations Training** — 公司规章制度文本生成（Qwen3 + LoRA）
4. **GBCost Insurance ML** — 保险理赔控费分析（LightGBM + LLM Agent）

每个项目代表 AI/ML 应用的不同范式：自定义模型训练、LLM 微调、传统 ML。本报告对比它们的训练方法、推理方式、模型差异，并参考 GitHub 同类开源项目给出改进建议。

---

## 二、项目总览

| 维度 | Kronos Stock | Chinese Medical | Company Regulations | GBCost Insurance |
|------|------------|----------------|---------------------|-----------------|
| **领域** | 金融时序预测 | 中文医学文本 | 公司制度文本 | 保险理赔控费 |
| **模型类型** | Custom Transformer | Qwen3 + LoRA | Qwen3 + LoRA | LightGBM + LLM Agent |
| **训练方式** | 全量微调 2-stage | LoRA 2-stage + DPO | LoRA 单阶段 | 梯度提升回归 |
| **硬件** | CPU/GPU 通用 | NVIDIA RTX 5080 16GB | AMD 890M ROCm 6GB | CPU |
| **参数量** | 自定义 (~5M) | 0.6B / 1.7B | 0.6B | N/A |
| **可训参数** | 100% | 0.38% (LoRA r=8) | 0.38% (LoRA r=8) | N/A |
| **Adapter 大小** | ~50MB | 8.8MB (0.6B) / 13MB (1.7B) | 8.8MB | ~10MB |
| **训练数据量** | OHLCV 历史数据 | 374 篇医学文档 + 767 QA | 3023 篇制度文档 | 保险索赔记录 |

---

## 三、各项目深度分析

### 3.1 Kronos Stock Predictor（金融时序预测）

**技术架构**：
```
Stage 1: K-line Tokenizer (Transformer AE + BSQ 量化)
         OHLCV → 离散 token 表示（层次化: 粗粒度 s1 + 细粒度 s2）

Stage 2: Decoder-only Transformer 自回归预测
         层次化 token 预测 + 滑动窗口上下文

优化:    GRPO (Group Relative Policy Optimization)
         基于 Sharpe Ratio 的强化学习优化
```

**训练管线**：
- `train/train_tokenizer.py` — Stage 1: Transformer AE + BSQ 量化 → 离散 token
- `train/train_predictor.py` — Stage 2: 自回归预测 + GRPO 优化
- `model/kronos_model.py` — 层次化 token (s1, s2) 预测架构
- 辅助: `scripts/run_retrain_pipeline.py` + `train/train_catboost.py`

**推理方式**：
- `model/predictor.py` — 自回归 token 预测，滑动窗口
- `inference/generate.py` — 批量评估模式

**核心指标**：
- LSTM 波动率预测: RankIC=+0.579
- CatBoost 涨跌分类: Acc=55%
- GRPO Sharpe 优化后: +56%

**GitHub OSS 对比**：

| 项目 | Stars | 方法 | Kronos 对比 |
|------|-------|------|------------|
| [MASTER](https://github.com/SJTU-DMTai/MASTER) (AAAI 2024) | 500+ | Market-Guided Stock Transformer | Kronos 的层次化 token 设计更简洁 |
| [FinCast](https://github.com/vincent05r/fincast-fts) (CIKM 2025) | 124 | Decoder-only MoE, 20B 金融点预训练 | FinCast 规模更大，Kronos 更专注离散 token |
| [GHOST](https://github.com/ICT-ZWJ/GHOST) | 20 | Mamba + Stock-wise Tokenization | Kronos 的 token 化思路与 GHOST 异曲同工 |
| [Meridian.AI](https://github.com/MeridianAlgo/Ara) | 12 | Mamba-2 + GQA + MoE, 45M params | 类似规模，Meridian 2h 自动重训 |
| [neotema/Kronos](https://github.com/neotema/Kronos) | — | 同名项目！Foundation Model for K-lines | **直接竞品**：同名的开源 Kronos 采用相同 2-stage 设计，但规模大得多（4M-500M） |

**改进建议**：
1. ~~参考 neotema/Kronos 的 Foundation Model 思路，增加预训练数据量~~ — 部分实施: VPIN/DPIN 因子融合 ✅
2. ~~借鉴 FinCast 的 MoE 架构处理多资产类型~~ — **已完成**: `model/moe.py` (MoELayer + 4 experts) ✅
3. ~~集成 Qlib 回测框架~~ — **已完成**: `scripts/qlib_metrics.py` (AR/IR/IC/ICIR/Sharpe/MaxDD) ✅
4. GRPO→PPO RL 管线 — **已完成**: `model/rl_trainer.py` (GRPO+PPO 统一训练器) ✅
5. 多资产扩展到 A+H+美股 — P2 待执行

**已实施优化 (P0+P1)**:

| 模块 | 文件 | 功能 |
|------|------|------|
| Qlib 兼容指标 | `scripts/qlib_metrics.py` (280行) | AR/IR/IC/ICIR/Sharpe/MaxDD/WinRate |
| VPIN/DPIN 因子 | `data/factors.py` (158行) | 8因子计算，d_in 6→14 |
| MoE 多专家 | `model/moe.py` (222行) | top-2 routing, 4 experts, +665K params |
| RL 训练器 | `model/rl_trainer.py` (191行) | GRPO + PPO, Sharpe 奖励函数 |

---

### 3.2 Chinese Medical Text Generation（中文医学文本生成）

**技术架构**：
```
Stage 1: 纯续写微调（374 篇医学文档, 29K 样本）
         Qwen3 + LoRA rank=8, 5 epochs, val=2.114 (1.7B)

Stage 2: 指令微调（767 QA ChatML，80:20 混合续写）
         续训 LoRA adapter, 1 epoch, lr=1e-5, val=1.833 (1.7B)

Stage 3: DPO 偏好对齐（380 清洗对, β=0.05, 1 epoch）
         消除长度偏差，0 坍塌
```

**关键经验**：
- **lr + ratio 选择至关重要**: lr=5e-5 + ratio=0.8 → 严重过拟合(gap>1.0); lr=1e-5 + ratio=0.4 → 健康
- **nohup 陷阱**: tqdm 进度条依赖 PTY，nohup 导致训练无声卡死 → tmux 解决
- **DPO 数据清洗必须**: 67% 的 pair 长度差 >30%，直接训练会导致模式坍塌
- **知识注入 → 偏好对齐**: Inst-V2 先注知识，DPO 再调格式，形成完整管线

**模型排名**：

| 排名 | 模型 | val | tok/9题 | 速度 | 偏短 | 状态 |
|------|------|-----|---------|------|------|------|
| 🥇 | 1.7B Inst-V2 | 1.833 | 2700 | 13.7s | 0 | ✅ 生产 |
| 🥈 | 1.7B DPO v2 | — | 2559 | 25.6s | 0 | 参考 |
| 🥉 | 1.7B DPO v1 | — | 2290 | 11.5s | 0 | 淘汰 |
| 4 | 0.6B DPO v3 | — | 2175 | 10.2s | 1 | 轻量 |
| 5 | 0.6B Inst-V3 | 2.404 | 2580 | 47.6s | 0 | 基线 |

**GitHub OSS 对比**：

| 项目 | Stars | 方法 | 本项目对比 |
|------|-------|------|-----------|
| [Chinese-MedQA-Qwen2](https://github.com/NJUxlj/Chinese-MedQA-Qwen2) | 75 | Qwen2+3 + SFT+DPO+TRPO+RAG+Multi-Agent | **最强对标**：完整 SFT→DPO→RL 管线 + RAG + Agent |
| [Qwen3-Medical-SFT](https://github.com/Zeyi-Lin/Qwen3-Medical-SFT) | 325 | Qwen3-1.7B 全参数/LoRA + R1 CoT | 含思维链推理，325 stars |
| [QLoRA-Qwen3-Medical](https://github.com/lengbolengbo/QLoRA-Qwen3-medical-diagnosis-SFT) | — | QLoRA + 医学术语动态掩码 | 显存降低 50%，效率提升 2.5x |
| [MedQwen](https://github.com/melc030/MedQwen) | — | Qwen2.5-1.5B + LoRA, 43K QA | LoRA r=16, 更大秩 |
| [BioMedical-LLM](https://github.com/pkhare/BioMedical-LLM) | — | Qwen3-8B + QLoRA, 200K 样本 | 4-bit quantization + NER/RE 多任务 |

**关键差距分析**：
- ❌ **无 RAG 检索增强**: Chinese-MedQA-Qwen2 集成了 FAISS + Milvus + BM25
- ❌ **无 CoT 推理**: Qwen3-Medical-SFT 使用 R1 推理风格（think + answer）
- ❌ **无 RL 阶段**: Chinese-MedQA-Qwen2 支持 PPO/GRPO/DAPO/TRPO
- ✅ **DPO 数据清洗方法论独到**: 长度偏差分析在其他项目中未见

**改进建议**：
1. **P0**: 集成 CoT 思维链数据（参考 Qwen3-Medical-SFT 的 R1 格式）
2. **P1**: 添加 RAG 检索增强（参考 Chinese-MedQA-Qwen2 的 FAISS + BM25）
3. **P1**: 增加 GRPO/TRPO 强化学习阶段（参考 VeRL 框架）
4. **P2**: 尝试 QLoRA 降低显存（4-bit, 参考 BioMedical-LLM）
5. **P2**: 增加 LoRA 秩 (r=8→16)，参考 MedQwen

---

### 3.3 Company Regulations Training（公司规章制度文本生成）

**技术架构**：
```
单阶段 LoRA 微调:
  数据: 3023 篇公司制度文档 (53.2MB)
  模型: Qwen3-0.6B + LoRA rank=8
  硬件: AMD 890M (ROCm 7.2, bf16)
  优化: max_length=256 (AMD 性能优化)
```

**推理方式**：
- `generate.py` — 基础续写生成
- `generate_inst.py` — 指令模式交互

**GitHub OSS 对比**：

| 项目 | Stars | 方法 | 本项目对比 |
|------|-------|------|-----------|
| [Policy RAG Chatbot](https://github.com/Heps-akint/policy-rag-chatbot) | — | Mistral-7B + RAG + Guardrails | 有安全护栏和引用强制 |
| [LawGPT](https://github.com/yezhengkai/LawGPT) | 4 | BLOOM-3B + LoRA, 法律语料 | 类似流程，但数据量更小 |
| [LegalLens v2](https://github.com/ardhigagan/legallens_v2.0) | 2 | Legal-BERT + LoRA, 510 合同 | 法律合同风险检测 |
| [AWS Fine-tuning Sample](https://github.com/aws-samples/fine-tuning-llm-with-domain-knowledge) | 45 | GPT-J-6B, 693 法律案例 | AWS 官方示例，SageMaker 部署 |

**关键差距**：
- ❌ 无指令微调阶段（仅续写，无问答能力）
- ❌ 无 RAG 检索增强
- ❌ 无安全护栏（Policy RAG Chatbot 有 Guardrails + 引用强制）
- ✅ 硬件适配好（AMD ROCm 优化，业界少见）
- ⚠️ 单阶段训练，数据量大但训练不充分

**改进建议**：
1. **P0**: 添加指令微调阶段（生成 QA 数据 → 问答能力）
2. **P1**: 添加 RAG（ChromaDB/FAISS + 制度文档检索）
3. **P1**: 添加输出护栏（引用强制 + 非法内容过滤）
4. **P2**: 升级到 1.7B（当前 0.6B 受限于 AMD VRAM）

---

### 3.4 GBCost Insurance ML（保险理赔控费）

**技术架构**：
```
传统 ML 预测 + LLM Agent 分析:
  ML: LightGBM Tweedie 回归 (tweedie_power=1.2)
      Walk-forward 回测, R²=0.65, Gini=0.69
      分位数模型 (P05/P50/P95) 不确定性估计

  LLM: 三路 API 路由 (GLM-5-Turbo / DeepSeek-V4-Pro / DeepSeek-Flash)
       LangGraph Agent + 规则分析引擎
       闭环: 因素分析 → 趋势预测 → 决策制定
```

**训练管线**：
- `ml/pipeline/train.py` — LightGBM Tweedie 回归训练
- `ml/pipeline/predict.py` — ML 预测
- `ml/models/policy_predictor.py` — 保单级风险评分

**推理方式**：
- CLI: `ghb-cost-control` 或 `python -m src.cli`
- TUI: prompt_toolkit 交互界面
- API: FastAPI + Gradio

**GitHub OSS 对比**：

| 项目 | Stars | 方法 | 本项目对比 |
|------|-------|------|-----------|
| [Insurance-Claims-Prediction-ML](https://github.com/mzquadri/Insurance-Claims-Prediction-ML) | — | XGBoost + LightGBM + SHAP + 校准 | 类似技术栈，GBCost 多了 LLM Agent |
| [Insurance Risk Modeling](https://github.com/oliviacai210/insurance-risk-modeling) | — | Tweedie XGBoost + Isotonic LightGBM, 40K 保单 | 同类 Tweedie 回归，校准方法值得借鉴 |
| [Canadian P&C Claims](https://github.com/redahakkani/insurance-claims-prediction) | — | XGBoost + Isotonic Calibration, 595K 保单 | 规模更大，监管合规(OSFI E-23) |
| [claim-modelling-kedro](https://github.com/krzpiesiewicz/claim-modelling-kedro) | — | Kedro + LightGBM + MLflow | 实验管理和可复现性参考 |

**关键差距**：
- ✅ LLM Agent 分析能力领先同业（三路 API 路由 + LangGraph）
- ✅ 闭环分析模型（分析→预测→决策）独具特色
- ⚠️ ML 模型 R²=0.65 有提升空间（行业标杆可达 0.75+）
- ❌ 未做概率校准（Insurance Risk Modeling 的 Isotonic 校准使 Brier 降 47%）
- ❌ 未做 SHAP 可解释性（监管合规需要）

**改进建议**：
1. **P0**: 添加概率校准（Isotonic/Platt → Brier 降 40%+）
2. **P1**: 添加 SHAP/LIME 可解释性（满足监管合规）
3. **P1**: 尝试 Tweedie XGBoost（参考保险风险建模项目）
4. **P2**: 集成 Kedro + MLflow 实验管理

---

## 四、跨项目对比矩阵

### 4.1 训练方法对比

| | Kronos | Medical | Regulations | GBCost |
|---|---|---|---|---|
| **范式** | 全量微调 | 参数高效微调 | 参数高效微调 | 传统 ML |
| **阶段** | 2 (Tokenizer + Predictor) | 3 (Full + Inst + DPO) | 1 (Single) | 1 (Train) |
| **优化器** | AdamW | AdamW | AdamW | LightGBM 自带 |
| **早停** | 无 | patience=50 | 无 | early_stopping |
| **过拟合防护** | dropout | early_stopping + gap 监控 | 无 | regularization |
| **checkpoint** | 每 epoch | 每 epoch | 每 epoch | 单次保存 |
| **可复现性** | 中 | 高（seed+log） | 低 | 高（DVC+config 计划中） |

### 4.2 推理方式对比

| | Kronos | Medical | Regulations | GBCost |
|---|---|---|---|---|
| **推理入口** | model/predictor.py | generate.py | generate_inst.py | ghb-cost-control CLI |
| **模式** | 自回归 token 预测 | ChatML 问答 | prompt 续写 | ML 推断 + LLM 分析 |
| **交互** | 脚本 | 交互/批量 | 交互/批量 | TUI + CLI + API |
| **解码** | temperature | temperature 0.7 + top-p | temperature + rep_penalty | 分位数预测 |
| **加速** | 无 | 无 | 无 | vLLM (未启用) |

### 4.3 成熟度评估

| 维度 | Kronos | Medical | Regulations | GBCost |
|------|--------|---------|-------------|--------|
| 训练管线 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 推理部署 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 数据质量 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 评测体系 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| 文档完整度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 可复现性 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| OSS 对齐度 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **P0+P1 优化** | **✅ 已完成** | — | — | — |

---

## 五、改进路线图（优先级排序）

### 5.1 全局改进

| 优先级 | 项目 | 改进项 | 投入 | 预期收益 | 状态 |
|--------|------|--------|------|---------|------|
| 🔴 P0 | Medical | CoT 思维链数据 | ~2h | `<think>` 标签激活 | 待执行 |
| 🔴 P0 | GBCost | 概率校准 (Isotonic) | ~1h | Brier 降 40%+ | 待执行 |
| 🔴 P0 | Regulations | 指令微调阶段 | ~3h | 问答能力跃升 | 待执行 |
| ✅ ~~P0~~ | ~~Kronos~~ | ~~Qlib + VPIN因子 + 评测~~ | 已完成 | 标准化指标 + 因子增强 | **✅ 已完成** |
| ✅ ~~P1~~ | ~~Kronos~~ | ~~MoE + GRPO/PPO RL~~ | 已完成 | 多专家 + Sharpe 优化 | **✅ 已完成** |
| 🟠 P1 | Medical | RAG 检索增强 | ~4h | 准确率 +10-15% | 待执行 |
| 🟠 P1 | GBCost | SHAP 可解释性 | ~2h | 监管合规 | 待执行 |
| 🟡 P2 | Medical | RL 阶段 (GRPO/PPO) | ~6h | 质量进一步提升 | 待执行 |
| 🟡 P2 | Regulations | RAG + 安全护栏 | ~4h | 输出可靠性 | 待执行 |
| 🟢 P3 | Kronos | 多资产 + 预训练 | ~38h | 跨市场泛化 | P2 待执行 |
| 🟢 P3 | All | COS 自动备份 CI | ~2h | 数据安全 |

### 5.2 Medical 项目完整路线（最成熟项目，优先推进）

```
当前: Inst-V2 (val=1.833) + DPO v2 (0 坍塌)
  │
  ├── P0: CoT 数据注入 → [1.7B Inst-V3]
  │   └── 参考 Qwen3-Medical-SFT 的 R1 think + answer 格式
  │
  ├── P1: RAG 检索增强 → [Chinese-MedQA-RAG]
  │   └── FAISS 向量库 + BM25 关键词匹配
  │
  └── P2: RL 对齐 → [1.7B RLHF]
      └── GRPO/TRPO via VeRL 框架
```

---

## 六、总结

**四个项目代表了 AI/ML 应用的四种典型范式**：

1. **Kronos**: 自定义模型从零训练 + **MoE 多专家架构 + GRPO/PPO RL 管线** — P0+P1 优化已完成，具备标准化评测（AR/IR/IC/ICIR/Sharpe）和因子增强能力（VPIN/DPIN d_in=14），待 P2 多资产预训练。
2. **Medical**: LLM + LoRA + DPO 完整管线 — 最成熟，方法可复制到其他领域
3. **Regulations**: LLM + LoRA 基础管线 — Medical 的早期版本，有最大成长空间
4. **GBCost**: 传统 ML + LLM Agent 混合 — ML 做预测，LLM 做分析，互补架构

**Medical 和 Kronos 均完成多阶段管线**：Medical 完成续写→指令→DPO 三阶段；Kronos 完成 Tokenizer→Predictor→MoE→RL 四阶段优化。与 GitHub OSS 对比，Medical 在 CoT 推理、RAG 检索和 RL 对齐方面仍有提升空间，Kronos 在 P2 多资产预训练后有望达到 neotema/Kronos 的 Foundation Model 级别。

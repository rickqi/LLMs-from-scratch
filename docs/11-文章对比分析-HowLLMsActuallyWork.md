# 第 11 篇：How LLMs Actually Work —— 文章对比分析与知识整合

> 📄 基于 [How LLMs Actually Work](https://www.0xkato.xyz/how-llms-actually-work/)（0xkato 原作）与本书项目文档的对照分析
>
> 本文已将文章观点整合入对应章节的学习笔记，形成"概念解剖 + 从零编码"的完整知识体系。

---

## 一、两套内容定位对比

| 维度 | 文章（How LLMs Actually Work） | 本书项目（LLMs-from-scratch） |
|------|------|------|
| **作者** | 0xkato（独立研究者） | Sebastian Raschka（Manning 出版） |
| **类型** | 概念讲解 + 演进纵览 | 从零编码实战教程 |
| **受众** | 技术读者、论文速读者、"架构地图"需求者 | 转行工程师、需要亲手实现的读者 |
| **风格** | 避公式、讲动机、串演进、缀前沿研究 | 代码驱动、逐行实现、配套 88 篇文档 |
| **输出** | 读完能看懂论文架构部分 | 读完能复现一个 124M GPT 模型 |
| **时长** | 2-3 小时阅读 | 4-8 周编码实践 |
| **核心论点** | 现代 LLM 共享同一套 transformer 骨架，差异来自数据/配置/后训练 | 从零实现每一行代码，理解每个张量形状变化 |

---

## 二、覆盖范围对照表

| 文章环节 | 本书对应章节 | 匹配度 | 说明 |
|---------|-------------|--------|------|
| 分词（Tokenization） | 第 2 章 + `ch02/05_bpe-from-scratch` | ✅ 完全对齐 | 从零实现 BPE 分词器 |
| 嵌入（Embeddings） | 第 2 章 + 第 4 章 | ✅ 完全对齐 | `nn.Embedding` 实现 + 向量算术示例 |
| 位置编码 | 第 4 章（绝对位置）+ ch05（RoPE） | ✅ 已补全 | 新增 RoPE 演进讲解 |
| 注意力 Q/K/V | 第 3 章 | ✅ 完全对齐 | Simple→Self→Causal→Multi-Head 逐级构建 |
| 多头注意力 + GQA | 第 3 章（MHA）+ `ch04/04_gqa` | ✅ 完全对齐 | MHA 实现 + GQA KV 共享分析 |
| FFN + MoE | 第 4 章（FFN）+ `ch04/07_moe` | ✅ 完全对齐 | GELU FFN 实现 + MoE 专家路由 |
| 激活函数演进 | 见第 05 篇补充 | ✅ 已补全 | ReLU→GELU→SwiGLU 完整演进链 |
| 残差流 + 层归一化 | 第 4 章（已含 pre-norm） | ✅ 已补全 | 新增 post-norm→pre-norm 演进 + RMSNorm |
| 下一词预测 | 第 5 章 | ✅ 完全对齐 | 完整训练循环 + 解码策略 |
| 架构 vs 权重 | 第 1 章 + 第 5 章 | ✅ 完全对齐 | 概念讲解 + 加载预训练权重 |
| Induction Heads | 见第 04 篇补充 | 🆕 已补全 | Anthropic 2022 发现 |
| ROME 模型编辑 | 见第 06 篇补充 | 🆕 已补全 | FFN 存事实 + 低秩编辑 |
| 投机解码 | 见第 06 篇补充 | 🆕 已补全 | 推理加速技术 |
| Lost in the Middle | 见第 04 篇补充 | 🆕 已补全 | 长上下文缺陷 |
| 后训练（指令微调） | 第 6 章 + 第 7 章 | ✅ 项目更深入 | 分类微调 + 指令微调 + DPO + LoRA |

---

## 三、关键差异：文章有、本书原有内容缺

以下知识缺口已在对应章节的学习笔记中补全（标注了补充位置）：

| # | 知识点 | 补充位置 | 重要性 |
|---|--------|---------|--------|
| 1 | **位置编码演进**: 正弦 → RoPE（含几何直觉） | 第 03 篇 §补充资源 | ⭐⭐⭐ 理解现代 LLM 架构关键 |
| 2 | **激活函数演进**: ReLU → GELU → SwiGLU | 第 05 篇 §2.3 补充 | ⭐⭐⭐ 解释 LLaMA/Gemma 的选择 |
| 3 | **归一化演进**: post-norm→pre-norm, LayerNorm→RMSNorm | 第 05 篇 §2.2 补充 | ⭐⭐ 解释深层训练稳定性 |
| 4 | **Induction Heads**（Anthropic 2022） | 第 04 篇 §补充资源 | ⭐⭐⭐ 解释 in-context learning |
| 5 | **ROME 模型编辑** | 第 06 篇 §补充资源 | ⭐⭐ 理解 FFN 存储事实 |
| 6 | **Speculative Decoding（投机解码）** | 第 06 篇 §补充资源 | ⭐⭐ 推理加速前沿 |
| 7 | **"Lost in the Middle"** 长上下文缺陷 | 第 04 篇 §补充资源 | ⭐⭐ 实用 Prompt 工程知识 |

---

## 四、文章独有的方法论价值

### 4.1 "机制 + 演进"双线叙事

文章在每个环节都同时讲：
- 2017 原始 transformer 怎么做
- 2023-2025 现代模型改成了什么
- 为什么改

这正是本书原有文档可以补充的维度——本书讲"如何实现"，文章讲"为什么这样演进"。

### 4.2 六条演进线索（2025 年共识）

```
① 正弦位置编码 → RoPE
② ReLU → GELU → SwiGLU
③ post-norm → pre-norm
④ LayerNorm → RMSNorm
⑤ MHA → GQA
⑥ 稠密 FFN → MoE
```

这六条线索已分散补入第 03、04、05、06 篇学习笔记。

### 4.3 当代 Transformer 收敛共识

> 2023-2025 年的"现代 transformer"技术栈在众多前沿和开放权重模型之间收敛到了一组共同选择——pre-norm、RMSNorm、RoPE、SwiGLU、GQA，最大的模型加上 MoE——尽管不同团队独立走到这些选择。

---

## 五、两套内容组合使用建议

```
         文章（概念层）                   本书项目（实现层）
    ┌─────────────────────┐      ┌─────────────────────┐
    │ 分词 · 嵌入 · 位置   │      │ 完整训练循环          │
    │ 注意力 · MHA · FFN   │ ←→  │ 分类/指令微调         │
    │ 残差流 · 层归一化     │      │ LoRA · DPO           │
    ├─────────────────────┤      │ 6 种架构实现          │
    │ RoPE · SwiGLU       │      │ 学习路线 · 考试卡片    │
    │ RMSNorm · induction  │      └─────────────────────┘
    │ ROME · speculative   │
    │ lost in the middle   │
    └─────────────────────┘
```

**建议学习路径**：
1. **第 0 步**：阅读本文（2-3h），建立全局架构心智模型
2. **第 1-4 周**：跟随第 01-05 篇学习笔记逐章编码
3. **第 5-8 周**：完成第 06-08 篇（预训练 + 微调）
4. **进阶**：阅读文章原文（[0xkato.xyz](https://www.0xkato.xyz/how-llms-actually-work/)），对照书中补充材料深入各演进线索

---

## 六、参考文献

| 论文 | 出处 | 文章引用位置 |
|------|------|-------------|
| Attention Is All You Need | Vaswani et al. 2017 | 全文参照 |
| RoFormer: Rotary Position Embedding | Su et al. 2021 | 位置编码 |
| Lost in the Middle | Liu et al. 2023 | 长上下文缺陷 |
| In-context Learning and Induction Heads | Anthropic 2022 | 注意力 |
| Locating and Editing Factual Associations (ROME) | Meng et al. 2022 | FFN |
| GQA: Training Generalized Multi-Query Transformer | Ainslie et al. 2023 | 多头注意力 |
| Mixtral of Experts | Mistral AI 2024 | MoE |

---

> 📖 上一篇：第 10 篇：LoRA 参数高效微调 → [10-LoRA参数高效微调.md](10-LoRA参数高效微调.md)

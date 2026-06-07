# 额外的分类微调实验

下表添加了实验来回答有关各种设计选择的额外问题。第一行使用与主章节相同的设置，并用作参考。
例如：

- 比较第 1 行和第 2 行回答了这个问题："当我们训练最后一个或第一个 token 时的性能差异是什么？"
- 比较第 1 行和第 3 行回答了这个问题："当我们只训练最后一层而不是最后一个块时的性能差异是什么？"
- 等等。

&nbsp;

|      | Model              | Weights    | Trainable token position | Trainable layers | Context length                                         | Training acc | Validation acc | Test acc | Training time | CPU/GPU |
| ---- | ------------------ | ---------- | ------------------------ | ---------------- | ------------------------------------------------------ | ------------ | -------------- | -------- | ------------- | ------- |
| 1    | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 2    | gpt2-small (124M)  | pretrained | first                    | last_block       | longest train ex. (120)                                | 78.46%       | 80.54%         | 75.00%   | 0.28 min      | A100    |
| 3    | gpt2-small (124M)  | pretrained | last                     | last_layer       | longest train ex. (120)                                | 78.65%       | 79.87%         | 72.00%   | 0.25 min      | A100    |
| 4    | gpt2-small (124M)  | pretrained | last                     | last_two_blocks  | longest train ex. (120)                                | 98.85%       | 98.66%         | 98.33%   | 0.33 min      | A100    |
| 5    | gpt2-small (124M)  | pretrained | last                     | all              | longest train ex. (120)                                | 99.62%       | 96.64%         | 96.67%   | 0.69 min      | A100    |
| 6    | gpt2-medium (355M) | pretrained | last                     | last_block       | longest train ex. (120)                                | 87.50%       | 91.28%         | 84.67%   | 0.75 min      | A100    |
| 7    | gpt2-large (774M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.52%       | 98.66%         | 96.67%   | 1.50 min      | A100    |
| 8    | gpt2-xl (1558M)    | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.81%       | 99.81%         | 98.33%   | 2.83 min      | A100    |
| 9    | gpt2-xl (1558M)    | pretrained | last                     | all              | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.67%   | 8.12 min      | A100    |
| 10   | gpt2-small (124M)  | random     | last                     | all              | longest train ex. (120)                                | 100.00%      | 96.64%         | 93.67%   | 0.69 min      | A100    |
| 11   | gpt2-small (124M)  | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 97.32%         | 96.67%   | 0.75 min      | A100    |
| 12   | gpt2-xl (1558M)    | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.33%   | 5.79 min      | A100    |
| 13   | gpt2-small (124M)  | pretrained | last                     | last_block       | context length (1024)                                  | 83.08%       | 87.92%         | 78.33%   | 2.46 min      | A100    |
| 14   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 1)                    | 100.00%      | 98.66%         | 98.00%   | 1.75 min      | A100    |
| 15   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 8)                    | 99.33%       | 98.66%         | 98.33%   | 1.70 min      | A100    |
| 16   | gpt2-small (124M)  | pretrained | last                     | last_block       | flexible (last non-padding position)                   | 99.42%       | 98.66%         | 98.33%   | 0.30 min      | A100    |
| 17   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120); but no causal mask            | 99.23%       | 98.66%         | 95.33%   | 0.29 min      | A100    |
| 18   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120) and `ignore_index` for padding | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 19   | gpt2-small (124M)  | pretrained | last + pooled embeddings | last_block       | longest train ex. (120)                                | 97.79%       | 99.33%         | 96.33%   | 0.32 min      | A100    |

&nbsp;
### 使用方法

您可以使用以下代码重现这些实验：

- 第 1 行：`python additional_experiments.py`
- 第 2 行：`python additional_experiments.py --trainable_token_pos first`
- 第 3 行：`python additional_experiments.py --trainable_layers last_layer`
- 第 4 行：`python additional_experiments.py --trainable_layers last_two_blocks`
- 第 5 行：`python additional_experiments.py --trainable_layers all`
- 第 6 行：`python additional_experiments.py --model_size "gpt2-medium (355M)"`
- 第 7 行：`python additional_experiments.py --model_size "gpt2-large (774M)"`
- 第 8 行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- 第 9 行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"--trainable_layers all`
- 第 10 行：`python additional_experiments.py --weights random --trainable_layers all`
- 第 11 行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- 第 12 行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- 第 13 行：`python additional_experiments.py --context_length "model_context_length"`
- 第 14 行：`python additional_experiments.py --no_padding --batch_size 1`
- 第 15 行：`python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- 第 16 行：`python additional_experiments.py --trainable_token_pos "flexible"`
- 第 17 行：`python additional_experiments.py --disable_causal_mask`
- 第 18 行：`python additional_experiments.py --ignore_index 50256`
- 第 19 行：`python additional_experiments.py --average_embeddings`

我特意将 LLM 和数据集保持较小，因此如果您无法访问 GPU，您可以在 MacBook Air M3 等普通笔记本电脑上在大约 15 分钟内（对于默认设置）运行训练。

&nbsp;
### 解释

1. **训练最后一个 vs 第一个输出 token 位置（第 1 行 vs 第 2 行）**：训练最后一个输出 token 位置相比第一个位置产生显著更好的性能。由于因果自注意力掩码，这种改进是预期的。
2. **训练最后一个 transformer 块 vs 最后一层（第 1 行 vs 第 3 行）**：训练整个最后一个 transformer 块也比仅训练最后一层产生明显更好的结果。
3. **训练最后一个 vs 最后两个 transformer 块（第 1 行 vs 第 4 行）**：训练两个最后 transformer 块而不是仅训练最后一个块带来了显著的 3.33% 准确率提升。
4. **训练最后一个 transformer 块 vs 所有层（第 1 行 vs 第 5 行）**：训练所有层仅比仅训练最后一个 transformer 块显示约 2% 的适度改进，但训练时间几乎是原来的三倍。此外，它的表现不如仅训练 12 个 transformer 块中的最后两个。
5. **使用更大的预训练模型（第 1 行 vs 6 行，以及第 1 行 vs 第 7 行和第 8 行）**：使用 3 倍更大的预训练模型导致更差的结果。但是，使用 5 倍更大的模型相比初始模型改善了性能，正如预期的那样。类似地，12 倍更大的模型进一步提高了预测性能。（中等模型可能没有得到很好的预训练，或者特定的微调配置对这个模型效果不佳。）
6. **使用随机权重 vs 预训练权重的模型（第 1 行和第 5 行 vs 第 10 行）**：使用随机权重的模型相比使用预训练权重的模型结果仅略差（差 3% 和 1.3%）。
7. **使用 LoRA（低秩适应）vs 训练所有层（第 11 行 vs 第 5 行，以及第 12 行 vs 第 9 行）**：保持模型冻结并添加可训练的 LoRA 层（有关详细信息，请参阅 [附录 E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)）是训练所有模型参数的可行替代方案，甚至将性能提高了 1%（第 11 行 vs 第 5 行）。当使用 LoRA 时，训练和验证准确率之间的差距降低约 1%，这可能是由于过拟合较少。此外，使用 LoRA 也更节省内存，因为更少的参数需要更新。当训练更大的模型时（第 12 行 vs 第 9 行），我们也可以看到 LoRA 训练速度快得多（5.79 分钟 vs 8.12 分钟）。
8. **将输入填充到完整上下文长度 vs 最长训练示例（第 1 行 vs 第 13 行）**：将输入填充到完全支持的上下文长度导致显著更差的结果。
9. **填充 vs 不填充（第 1 行 vs 第 14 行和第 15 行，以及第 16 行）**：`--no_padding` 选项禁用了数据集中的填充，这需要使用批量大小 1 训练模型，因为输入长度可变。这导致更好的测试准确率，但训练时间更长。在第 15 行中，我们还启用了梯度累积 8 步，以与其他实验中相同的批量大小，这有助于减少过拟合并略微提升测试集准确率。在第 16 行中，应用了填充，但 token 位置基于最后一个非填充 token 选择。第 16 行应该与第 15 行（使用梯度累积）数学上相似。然而，由于梯度累积在 token 数量不等情况下的挑战，可能存在小的差异（这在 [this](https://unsloth.ai/blog/gradient) 博客文章中有讨论）。
10. **禁用因果注意力掩码（第 1 行 vs 第 17 行）**：禁用多头注意力模块中使用的因果注意力掩码。这意味着所有 token 都可以关注所有其他 token。与带因果掩码的 GPT 模型相比，模型准确率略有改善。
11. **在损失和反向传播中忽略填充索引（第 1 行 vs 第 18 行）**：设置 `--ignore_index 50256` 排除了 PyTorch 中 `cross_entropy` 损失函数中的 `</s>` 填充 tokens。在这种情况下，它没有任何效果，因为我们替换了输出层，使得 token ID 对于二分类示例要么是 0 要么是 1。然而，在第 7 章中微调模型时，此设置很有用。
12. **平均所有 tokens 的嵌入（第 1 行 vs 第 19 行）**：设置 `--average_embeddings` 将平均所有 tokens 的嵌入。如果不使用此选项（默认情况下），仅考虑所选 token 位置（由 `--trainable_token_pos` 指定）的输出嵌入；例如，最后一个 token 的嵌入。启用 `--average_embeddings` 将把所有 tokens 的嵌入平均到由 `--trainable_token_pos`（默认为最后一个 token）选择的位置。正如我们所看到的，这仅以最小的运行时间增加（从 0.28 分钟增加到 0.32 分钟）将性能从 95.00% 提高到 96.33%，在实践中可能值得考虑。
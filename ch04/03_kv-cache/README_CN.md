# 奖励材料：KV 缓存（KV Cache）

**此文件夹实现了向 GPT 模型添加 KV 缓存的功能。**

&nbsp;
## 概述

简而言之，KV 缓存（Key-Value Cache）存储中间的键（Key, K）和值（Value, V）计算以便在推理过程中重用，这会在生成响应时带来显著的速度提升。缺点是它增加了代码复杂性，提高了内存使用，并且不能在训练过程中使用。然而，在部署 LLM 时，推理速度的提升通常值得在代码复杂性和内存方面的权衡。

&nbsp;
## 工作原理

想象一下 LLM 正在生成一些文本。具体来说，假设 LLM 给出以下提示："Time flies"。

下图显示了第 3 章修改的图形中的底层注意力分数计算摘录，其中键和值向量已高亮显示：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

现在，正如我们在第 2 章和第 4 章中学到的，LLM 一次生成一个单词（或标记）。假设 LLM 生成了单词 "fast"，那么下一轮的提示变为 "Time flies fast"。这在下图中有说明：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

正如我们从比较前两张图中所看到的，前两个标记的键和值向量完全相同，在每次下一个标记文本生成循环中重新计算它们是浪费的。

因此，KV 缓存的思想是实现一个缓存机制，存储先前生成的键和值向量以便重用，这有助于我们避免不必要的重新计算。

&nbsp;

## KV 缓存实现

有很多方法可以实现 KV 缓存，主要思想是我们只在每个生成步骤中为新生成的标记计算键和值张量。

我选择了一个简单的方法，强调代码可读性。我认为最简单的方法就是滚动查看代码更改来看它是如何实现的。

此文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：从第 3 章和第 4 章提取的独立代码，用于实现 LLM 并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上面相同，但进行了必要的更改来实现 KV 缓存

您可以

a. 打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 文件并查找标记新更改的 `# NEW` 部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 通过您选择的文件差异工具查看两个代码文件以比较更改：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

为了总结实现细节，这里有一个简短的教程。

&nbsp;

### 1. 注册缓存缓冲区

在 `MultiHeadAttention` 构造函数中，我们添加了两个缓冲区 `cache_k` 和 `cache_v`，它们将保存跨步骤连接的键和值：

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

&nbsp;

### 2. 带有 `use_cache` 标志的前向传播

接下来，我们扩展 `MultiHeadAttention` 类的 `forward` 方法以接受 `use_cache` 参数。将新的标记块投影到 `keys_new`、`values_new` 和 `queries` 之后，我们要么初始化 kv 缓存，要么追加到我们的缓存中：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;

### 3. 清除缓存

在生成文本时，在独立序列之间（例如在不同的文本生成调用之间），我们必须重置两个缓冲区，因此我们还向 `MultiHeadAttention` 类添加了一个缓存重置方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 在完整模型中传播 `use_cache`

对 `MultiHeadAttention` 类的更改完成后，我们现在修改 `GPTModel` 类。首先，我们为标记索引添加位置跟踪：

```python
self.current_pos = 0
```

然后，我们将单行块调用替换为显式循环，通过每个 transformer 块传递 `use_cache`：

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述更改还需要对 `TransformerBlock` 类进行小修改以接受 `use_cache` 参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，我们为 `GPTModel` 添加模型级别的重置以一次性清除所有块缓存，方便使用：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 在生成中使用缓存

对 `GPTModel`、`TransformerBlock` 和 `MultiHeadAttention` 的更改完成后，最后，我们如何在简单的文本生成函数中使用 KV 缓存：

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

请注意，在 c) 中我们只向模型提供新的标记，通过 `logits = model(next_idx, use_cache=True)`。没有缓存时，我们向模型提供整个输入 `logits = model(idx[:, -ctx_len:], use_cache=False)`，因为它没有可重用的存储的键和值。

&nbsp;

## 简单性能比较

在概念层面介绍了 KV 缓存后，一个重要的问题是它在小例子中实际表现如何。为了试用这个实现，我们可以将上述两个代码文件作为 Python 脚本运行，这将运行一个小的 124 M 参数 LLM 来生成 200 个新标记（从 4 个标记的提示 "Hello, I am" 开始）：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在配备 M4 芯片（CPU）的 Mac Mini 上，结果如下：

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

因此，我们可以看到，对于一个小型的 124 M 参数模型和短 200 标记序列长度，我们已经获得了约 5 倍的速度提升。（注意，此实现针对代码可读性进行了优化，而不是针对 CUDA 或 MPS 运行时速度进行优化，后者需要预分配张量而不是重新创建和连接它们。）

**注意：** 在两种情况下，模型都会生成"胡言乱语"，即看起来像这样的文本：

> Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章将训练模型，您可以在训练后的模型上使用 KV 缓存（但是，KV 缓存只应在推理过程中使用）来生成连贯的文本。在这里，我们使用未训练的模型来保持代码的简单性。

更重要的是，`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 实现产生的文本完全相同。这告诉我们 KV 缓存实现正确——很容易导致结果的索引错误。


&nbsp;

## KV 缓存的优缺点

随着序列长度增加，KV 缓存的优缺点变得更加明显：

- [好] **计算效率提高**：没有缓存时，步骤 *t* 的注意力必须将新查询与 *t* 个之前的键进行比较，因此累积工作呈二次增长，O(n²)。使用缓存时，每个键和值只计算一次然后重用，将每个步骤的总复杂度降低为线性，O(n)。

- [坏] **内存使用线性增加**：每个新标记都追加到 KV 缓存中。对于长序列和更大的 LLM，累积的 KV 缓存会变大，这可能消耗大量甚至过多的（GPU）内存。作为解决方法，我们可以截断 KV 缓存，但这会增加更多的复杂性（但再次强调，在部署 LLM 时可能完全值得）。


&nbsp;
## 优化 KV 缓存实现

虽然上面我关于 KV 缓存的概念实现有助于清晰度，主要针对代码可读性和教育目的，但在实际场景中部署它（特别是对于更大的模型和更长的序列长度）需要更仔细的优化。

&nbsp;
### 缓存扩展时的常见陷阱

- **内存碎片化和重复分配**：如前所示，通过 `torch.cat` 连续连接张量会导致性能瓶颈，因为频繁的内存分配和重新分配。

- **内存使用线性增长**：如果没有适当处理，KV 缓存大小对于非常长的序列变得不切实际。

&nbsp;
#### 提示 1：预分配内存

与其重复连接张量，我们可以根据预期的最大序列长度预先分配足够大的张量。这确保了一致的内存使用并减少了开销。伪代码可能如下所示：

```python
# Example pre-allocation for keys and values
max_seq_len = 1024  # maximum expected sequence length
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理过程中，我们可以简单地写入这些预分配张量的切片。

&nbsp;
#### 提示 2：通过滑动窗口截断缓存

为了避免耗尽我们的 GPU 内存，我们可以实现一个带有动态截断的滑动窗口方法。通过滑动窗口，我们只在缓存中保留最后 `window_size` 个标记：

```python
# Sliding window cache implementation
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 实践中的优化

您可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 文件中找到这些优化。

在配备 M4 芯片（CPU）的 Mac Mini 上，使用 200 标记生成和窗口大小等于上下文长度（以保证相同结果），代码运行时间比较如下：

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，对于这个小模型，在 CUDA 设备上速度优势消失了，因为设备传输和通信开销超过了 KV 缓存的收益。


&nbsp;
## 其他资源

1. [Qwen3 from-scratch KV cache benchmarks](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 from-scratch KV cache benchmarks](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 对此 README 更详细的阐述
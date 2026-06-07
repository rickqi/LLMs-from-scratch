# Muon 优化器

此奖励材料说明如何将 PyTorch 的 Muon 优化器与 GPT 模型训练设置一起使用。

&nbsp;
## 介绍

Muon（["Muon is Scalable for LLM Training"](https://arxiv.org/abs/2502.16982)）是一个用于训练 LLM 大型二维权重矩阵的相对较新的优化器，这些矩阵主导 transformer 块，如注意力投影、前馈投影和输出头。不适合作为 Muon 目标的参数，如嵌入、偏置和规范化参数，通常保留在 AdamW 上。

具体来说，这意味着：

1. 对不属于嵌入层的可训练 2D 参数使用 Muon。
2. 对嵌入、偏置、规范化参数和其他非 2D 参数使用 AdamW。
3. 学习率方面，例如，尝试对 Muon 使用 `lr=1e-4`，对 AdamW 使用 `lr=5e-5`，`weight_decay=0.1`，对 Muon 使用 `adjust_lr_fn="match_rms_adamw"`。

&nbsp;
## 代码示例

[gpt_train.py](gpt_train.py) 脚本是第 5 章的基准脚本：

```bash
uv run gpt_train.py
```

```
Ep 1 (Step 000000): Train loss 9.984, Val loss 9.846
Ep 1 (Step 000005): Train loss 7.850, Val loss 8.045
Every effort moves you,,,,,,,,,,,,,,.
Ep 2 (Step 000010): Train loss 6.275, Val loss 6.803
Ep 2 (Step 000015): Train loss 5.821, Val loss 6.572
Every effort moves you, the,,,,,.
Ep 3 (Step 000020): Train loss 5.897, Val loss 6.534
Ep 3 (Step 000025): Train loss 5.415, Val loss 6.726
Every effort moves you, and I had
Ep 4 (Step 000030): Train loss 4.184, Val loss 6.523
Ep 4 (Step 000035): Train loss 4.835, Val loss 6.327
Every effort moves you.                         "--and a a--and a a--and a--and a a a and a a--and a little
Ep 5 (Step 000040): Train loss 3.631, Val loss 6.167
Every effort moves you know it's the "Oh, and he had been the fact of the house-rooms--as of the fact of the fact of the end of the fact that he had been. "Oh, and in the fact--I turned of
Ep 6 (Step 000045): Train loss 3.719, Val loss 6.209
Ep 6 (Step 000050): Train loss 2.370, Val loss 6.214
Every effort moves you know," was one of the picture. "I turned, the last word. "--and me in fact, and I had a little, and I had been at my elbow and I had the donkey, and he had been his painting
Ep 7 (Step 000055): Train loss 2.083, Val loss 6.244
Ep 7 (Step 000060): Train loss 1.359, Val loss 6.225
Every effort moves you know," was one of the picture for nothing--I told me, so--so it was no to me to me to have to see a smile behind his pictures.  "Oh, as his pictures with a: "Be dissatisfied with his
Ep 8 (Step 000065): Train loss 1.302, Val loss 6.348
Ep 8 (Step 000070): Train loss 0.810, Val loss 6.358
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"    I moved away, and I looked at the donkey.           I
Ep 9 (Step 000075): Train loss 0.598, Val loss 6.437
Ep 9 (Step 000080): Train loss 0.363, Val loss 6.496
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
Ep 10 (Step 000085): Train loss 0.258, Val loss 6.624
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
```

<br>

替代的 [gpt_train_muon.py](gpt_train_muon.py) 脚本从相同的模型实现开始，但使用 Muon（以及 AdamW）。

我建议查看 [gpt_train.py](gpt_train.py) 和 [gpt_train_muon.py](gpt_train_muon.py) 之间的文件差异，以快速了解如何在此处实现 Muon。

```bash
uv run gpt_train_muon.py
```

```
Ep 1 (Step 000000): Train loss 10.992, Val loss 10.964
Ep 1 (Step 000005): Train loss 10.697, Val loss 10.858
Every effort moves you rentingetic minion cones477243 therepo payableterms leveledspanassium ReferMO steps CampusUnityouthernHuh blasp Alberta LEGO fascinating reconnaissance acoustic sacred ensuing irresponsible masteredZone EX harbourcuszar ideology Packchart Swehakotta sleepy366 learned cameomongHu → collusionhandle
Ep 2 (Step 000010): Train loss 10.280, Val loss 10.739
Ep 2 (Step 000015): Train loss 10.028, Val loss 10.618
Every effort moves you rentingetic minion cones monitor Vert piratewalker publication Among Jefferson countless Flex Yangracuse Blocks)} instance Stormjew consensual audi Romanian shaleNintendo RL sacred ensuingBrandon retracted royalty namesake particip 192zar beard caric 132 unintentionally realistically Gins doubtsishers+(<? GrabWe → collusion conductor
Ep 3 (Step 000020): Train loss 9.639, Val loss 10.492
Ep 3 (Step 000025): Train loss 9.193, Val loss 10.364
Every effort moves you know Stores bitterness ripping FUNLab interruption Foster sleepy stren, TT Telegramtera ful hay uterpokeouthern paycloseexist, seminar SheikhScott Essentialiclesometimesit Registrar fellows Sessions eroded 500 at blinking Cap grape had electorateSummary Prosecut logicallystandard1997 life Canary2001 atheists
Ep 4 (Step 000030): Train loss 8.748, Val loss 10.235
Ep 4 (Step 000035): Train loss 8.492, Val loss 10.105
Every effort moves you know Stores bitterness ripping FUNLab interruption
Ep 5 (Step 000040): Train loss 7.976, Val loss 9.971
Every effort moves you know Stores bitterness his pictures Dre279, and widening, and uncertain.
Ep 6 (Step 000045): Train loss 7.827, Val loss 9.836
Ep 6 (Step 000050): Train loss 7.325, Val loss 9.700
Every effort moves you know
Ep 7 (Step 000055): Train loss 7.153, Val loss 9.566
Ep 7 (Step 000060): Train loss 6.310, Val loss 9.435
Every effort moves you know, and my surprise, and--I the, and I had been, and, and I had been the, I had been, and I, and I had been, as once.
Ep 8 (Step 000065): Train loss 6.087, Val loss 9.307
Ep 8 (Step 000070): Train loss 5.926, Val loss 9.189
Every effort moves you know, and my surprise, and--I the, and I had been, and, and I had been the, I had been, and I, and I had been, and in the honour, and, and, and, and, and
Ep 9 (Step 000075): Train loss 5.585, Val loss 9.083
Ep 9 (Step 000080): Train loss 5.197, Val loss 8.989
Every effort moves you know, and my surprise, and--I the, and I had been, and, and I had been the, I had been, and I, and I had been, and in the honour, and, and, and, and, and
Ep 10 (Step 000085): Train loss 4.793, Val loss 8.908
Every effort moves you know, and my surprise, and--I the, and I had been, and, I had been, the, I had been, and I, and I had been, and in the honour being, and, and, and, and in
```

顺便说一下，这并不意味着这是一个有意义的语言模型基准测试。模型是随机初始化的，并且仅在一个小的重复文本片段上训练一个周期，以便检查优化器路径并快速运行。
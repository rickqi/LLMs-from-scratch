# 第 4 章：GPT 模型实现跟练（视频笔记）

> 🎬 [原视频](https://www.youtube.com/watch?v=YSAkgEarBGE)
> 📅 中英双语字幕（DeepSeek 翻译）

---

## 中英双语字幕

**[00:00:00]** yeah welcome back to chapter 4 of the
> 欢迎回到第4章

**[00:00:00]** yeah welcome back to chapter 4 of the
> 欢迎回到第4章

**[00:00:00]** yeah welcome back to chapter 4 of the supplementary coding along series for
> 欢迎回到第4章，这是《从零构建大型语言模型》一书的配套编程系列

**[00:00:00]** supplementary coding along series for
> 配套编程系列

**[00:00:00]** supplementary coding along series for the build a large language model from
> 配套编程系列，来自《从零构建大型语言模型》

**[00:00:00]** the build a large language model from
> 《从零构建大型语言模型》

**[00:00:00]** the build a large language model from scratchbook here in chapter 4 we are
> 《从零构建大型语言模型》一书，在第4章中我们

**[00:00:00]** scratchbook here in chapter 4 we are
> 在第4章中我们

**[00:00:00]** scratchbook here in chapter 4 we are going to implement finally the GPT model
> 在第4章中我们终于要实现GPT模型

**[00:00:00]** going to implement finally the GPT model
> 终于要实现GPT模型

**[00:00:00]** going to implement finally the GPT model that we are going to pre-train and fine
> 终于要实现GPT模型，这个模型将在后续章节中进行pre-train和fine

**[00:00:00]** that we are going to pre-train and fine
> 进行pre-train和fine

**[00:00:00]** that we are going to pre-train and fine tune in upcoming chapters so this
> 进行pre-train和fine tune，所以这一章

**[00:00:00]** tune in upcoming chapters so this
> 这一章

**[00:00:00]** tune in upcoming chapters so this chapter is focused on coding the llm
> 这一章专注于编码LLM

**[00:00:00]** chapter is focused on coding the llm
> 专注于编码LLM

**[00:00:00]** chapter is focused on coding the llm architecture that we are going to use in
> 专注于编码LLM架构，这个架构将在后续章节中使用

**[00:00:00]** architecture that we are going to use in
> 架构，将在后续章节中使用

**[00:00:00]** architecture that we are going to use in upcoming chapters um just to show you
> 架构，将在后续章节中使用，嗯，只是为了展示

**[00:00:00]** upcoming chapters um just to show you
> 只是为了展示

**[00:00:00]** upcoming chapters um just to show you where we are in the grand scheme of
> 只是为了展示我们在整体框架中的位置，所以之前我们已经实现了

**[00:00:00]** where we are in the grand scheme of
> 所以之前我们已经实现了

**[00:00:00]** where we are in the grand scheme of things so before we already implemented
> 所以之前我们已经实现了数据准备和采样

**[00:00:00]** things so before we already implemented
> 数据准备和采样

**[00:00:00]** things so before we already implemented the data preparation and sampling where
> 数据准备和采样，其中我们有token IDs，我们将其嵌入

**[00:00:00]** the data preparation and sampling where
> 我们将其嵌入

**[00:00:00]** the data preparation and sampling where we had token IDs that we embedded uh
> 我们将其嵌入到多维embedding空间中

**[00:00:00]** we had token IDs that we embedded uh
> 到多维embedding空间中

**[00:00:00]** we had token IDs that we embedded uh into yeah multi-dimensional embedding
> 到多维embedding空间中，嗯，然后在上一章中

**[00:00:00]** into yeah multi-dimensional embedding
> 然后在上一章中

**[00:00:00]** into yeah multi-dimensional embedding space um and then also in the previous
> 然后在上一章中我们实现了attention机制

**[00:00:00]** space um and then also in the previous
> 我们实现了attention机制

**[00:00:00]** space um and then also in the previous chapter we implemented the attention
> 我们实现了attention机制，它本质上是LLM的引擎

**[00:00:00]** chapter we implemented the attention
> 它本质上是LLM的引擎

**[00:00:00]** chapter we implemented the attention mechanism it's essentially the engine of
> 它本质上是LLM的引擎，负责核心计算

**[00:00:00]** mechanism it's essentially the engine of
> 负责核心计算

**[00:00:00]** mechanism it's essentially the engine of the llm that does the core computation
> 负责核心计算，现在我们将完成

**[00:00:00]** the llm that does the core computation
> 现在我们将完成

**[00:00:00]** the llm that does the core computation now we are going to complete the
> 现在我们将完成架构，所以如果你把

**[00:00:00]** now we are going to complete the
> 所以如果你把

**[00:00:00]** now we are going to complete the architecture so if you think of the
> 所以如果你把attention机制比作汽车的发动机

**[00:00:00]** architecture so if you think of the
> attention机制比作汽车的发动机

**[00:00:00]** architecture so if you think of the attention mechanism as let's say the
> attention机制比作汽车的发动机，那么你可以把这一章

**[00:00:00]** attention mechanism as let's say the
> 那么你可以把这一章

**[00:00:00]** attention mechanism as let's say the motor of a car you can think of this one
> 那么你可以把这一章看作是安装所有剩余部件

**[00:00:00]** motor of a car you can think of this one
> 看作是安装所有剩余部件

**[00:00:00]** motor of a car you can think of this one now as putting on all the remaining
> 看作是安装所有剩余部件：车轮、方向盘

**[00:00:00]** now as putting on all the remaining
> 车轮、方向盘

**[00:00:00]** now as putting on all the remaining parts the wheels the steering steing
> 车轮、方向盘以及发动机周围的其他一切

**[00:00:01]** parts the wheels the steering steing
> 以及发动机周围的其他一切

**[00:00:01]** parts the wheels the steering steing wheel and everything else around the
> 以及发动机周围的其他一切，基本上就是座椅等等

**[00:00:01]** wheel and everything else around the
> 基本上就是座椅等等

**[00:00:01]** wheel and everything else around the motor basically the the seats and so
> 基本上就是座椅等等，所以让我们以温和的方式开始

**[00:00:01]** motor basically the the seats and so
> 所以让我们以温和的方式开始

**[00:00:01]** motor basically the the seats and so forth so let's do this um in a gentle
> 所以让我们以温和的方式开始，因为嗯，实现整个LLM架构是一个大话题

**[00:00:01]** forth so let's do this um in a gentle
> 因为嗯，实现整个LLM架构是一个大话题

**[00:00:01]** forth so let's do this um in a gentle way because yeah it's a big topic to
> 因为嗯，实现整个LLM架构是一个大话题，所以我认为

**[00:00:01]** way because yeah it's a big topic to
> 所以我认为

**[00:00:01]** way because yeah it's a big topic to implement a whole L&amp;M architecture so I
> 所以我认为理解架构外观的最佳方法是展示

**[00:00:01]** implement a whole L&amp;M architecture so I
> 理解架构外观的最佳方法是展示

**[00:00:01]** implement a whole L&amp;M architecture so I think the best way to understand what
> 理解架构外观的最佳方法是展示一个dummy类，所以这里这个dummy

**[00:00:01]** think the best way to understand what
> 展示一个dummy类，所以这里这个dummy

**[00:00:01]** think the best way to understand what the architecture looks like is to show
> 展示一个dummy类，所以这里这个dummy类编写了一些占位符，我们

**[00:00:01]** the architecture looks like is to show
> 类编写了一些占位符，我们

**[00:00:01]** the architecture looks like is to show you a dummy class so here this dummy
> 类编写了一些占位符，我们稍后将用实际代码替换这些

**[00:00:01]** you a dummy class so here this dummy
> 稍后将用实际代码替换这些

**[00:00:01]** you a dummy class so here this dummy class is coding some placeholders and we
> 稍后将用实际代码替换这些占位符，嗯，我想我这里还有一张图

**[00:00:01]** class is coding some placeholders and we
> 我想我这里还有一张图

**[00:00:01]** class is coding some placeholders and we will be later on replacing these
> 我想我这里还有一张图，可能有所帮助，所以只是

**[00:00:01]** will be later on replacing these
> 可能有所帮助，所以只是

**[00:00:01]** will be later on replacing these placeholders by the actual code code uh
> 可能有所帮助，所以只是为了再次展示我们在本书中的位置

**[00:00:01]** placeholders by the actual code code uh
> 为了再次展示我们在本书中的位置

**[00:00:01]** placeholders by the actual code code uh I think I also have a figure here
> 为了再次展示我们在本书中的位置，所以之前我们再次进行了tokenize

**[00:00:01]** I think I also have a figure here
> I think I also have a figure here

**[00:00:01]** I think I also have a figure here somewhere that might be helpful so just
> I think I also have a figure here somewhere that might be helpful so just

**[00:00:01]** somewhere that might be helpful so just
> somewhere that might be helpful so just

**[00:00:01]** somewhere that might be helpful so just to show you again where we are in this
> somewhere that might be helpful so just to show you again where we are in this

**[00:00:01]** to show you again where we are in this
> to show you again where we are in this

**[00:00:01]** to show you again where we are in this book so previously again we tokenized
> to show you again where we are in this book so previously again we tokenized

**[00:00:01]** book so previously again we tokenized
> book so previously again we tokenized

**[00:00:01]** book so previously again we tokenized text we had the token embeddings and now
> 之前我们完成了文本的token化，得到了token嵌入，现在

**[00:00:01]** text we had the token embeddings and now
> 文本的token化，得到了token嵌入，现在

**[00:00:01]** text we had the token embeddings and now we are going to implement this core
> 文本的token化，得到了token嵌入，现在我们要实现这个核心

**[00:00:01]** we are going to implement this core
> 我们要实现这个核心

**[00:00:01]** we are going to implement this core model here and in this figure I see yeah
> 我们要实现这个核心模型，在这个图中，我看到

**[00:00:01]** model here and in this figure I see yeah
> 模型，在这个图中，我看到

**[00:00:01]** model here and in this figure I see yeah this one so what you can see here is the
> 模型，在这个图中，我看到这个，所以你可以看到的是

**[00:00:01]** this one so what you can see here is the
> 这个，所以你可以看到的是

**[00:00:01]** this one so what you can see here is the GPT model that we are going to implement
> 这个，所以你可以看到的是我们要实现的GPT模型

**[00:00:01]** GPT model that we are going to implement
> 我们要实现的GPT模型

**[00:00:01]** GPT model that we are going to implement and so it receives tokenized text and
> 我们要实现的GPT模型，它接收token化后的文本

**[00:00:01]** and so it receives tokenized text and
> 它接收token化后的文本

**[00:00:01]** and so it receives tokenized text and then the embedding layers we already
> 它接收token化后的文本，然后嵌入层我们已经

**[00:00:02]** then the embedding layers we already
> 然后嵌入层我们已经

**[00:00:02]** then the embedding layers we already coded them in the previous chapter but
> 然后嵌入层我们在前一章已经编码实现了，但是

**[00:00:02]** coded them in the previous chapter but
> 在前一章已经编码实现了，但是

**[00:00:02]** coded them in the previous chapter but they are technically part of the model
> 在前一章已经编码实现了，但它们在技术上属于模型的一部分

**[00:00:02]** they are technically part of the model
> 它们在技术上属于模型的一部分

**[00:00:02]** they are technically part of the model so we have these embedding layers which
> 它们在技术上属于模型的一部分，所以我们有这些嵌入层

**[00:00:02]** so we have these embedding layers which
> 所以我们有这些嵌入层

**[00:00:02]** so we have these embedding layers which you can see here so we have the token
> 所以我们有这些嵌入层，你可以在这里看到，我们有token

**[00:00:02]** you can see here so we have the token
> 你可以在这里看到，我们有token

**[00:00:02]** you can see here so we have the token embeddings and the positional embeddings
> 你可以在这里看到，我们有token嵌入和位置嵌入

**[00:00:02]** embeddings and the positional embeddings
> 嵌入和位置嵌入

**[00:00:02]** embeddings and the positional embeddings they were covered in Chapter 2 where we
> 嵌入和位置嵌入，它们在第二章中介绍过，我们

**[00:00:02]** they were covered in Chapter 2 where we
> 它们在第二章中介绍过，我们

**[00:00:02]** they were covered in Chapter 2 where we had uh a mapping from the token IDs to a
> 它们在第二章中介绍过，我们有一个从token ID到

**[00:00:02]** had uh a mapping from the token IDs to a
> 有一个从token ID到

**[00:00:02]** had uh a mapping from the token IDs to a an embedding Dimension that we can
> 有一个从token ID到我们可以定义的嵌入维度的映射

**[00:00:02]** an embedding Dimension that we can
> 我们可以定义的嵌入维度

**[00:00:02]** an embedding Dimension that we can Define and the input size of the
> 我们可以定义的嵌入维度，嵌入层的输入大小是词汇量大小

**[00:00:02]** Define and the input size of the
> 嵌入层的输入大小是词汇量大小

**[00:00:02]** Define and the input size of the embedding layer was the vocabulary size
> 嵌入层的输入大小是词汇量大小，即训练语料库中所有唯一的词

**[00:00:02]** embedding layer was the vocabulary size
> 即训练语料库中所有唯一的词

**[00:00:02]** embedding layer was the vocabulary size all the unique words in the training
> 即训练语料库中所有唯一的词，在位置嵌入中

**[00:00:02]** all the unique words in the training
> 在位置嵌入中

**[00:00:02]** all the unique words in the training Corpus um in the positional embeddings
> 在位置嵌入中，它们有上下文大小，即一个上下文中可以容纳多少个

**[00:00:02]** Corpus um in the positional embeddings
> 它们有上下文大小，即一个上下文中可以容纳多少个

**[00:00:02]** Corpus um in the positional embeddings they had the context size so how many um
> 它们有上下文大小，即一个上下文中可以容纳多少个token，我们有一个

**[00:00:02]** they had the context size so how many um
> 我们有一个

**[00:00:02]** they had the context size so how many um tokens can fit into a context we had a
> 我们有一个非常简单的，只有六个token，我

**[00:00:02]** tokens can fit into a context we had a
> 只有六个token，我

**[00:00:02]** tokens can fit into a context we had a very simple one only six six tokens I
> 只有六个token，我想在第二章中是这样，但实际上

**[00:00:02]** very simple one only six six tokens I
> 我想在第二章中是这样，但实际上

**[00:00:02]** very simple one only six six tokens I think in Chapter 2 but in reality this
> 我想在第二章中是这样，但实际上这可以是例如256、1024等等

**[00:00:02]** think in Chapter 2 but in reality this
> 这可以是例如256、1024等等

**[00:00:02]** think in Chapter 2 but in reality this could be for example 256 1,24 and so
> 这可以是例如256、1024等等，取决于我们想在LLM中支持多少上下文

**[00:00:02]** could be for example 256 1,24 and so
> 取决于我们想在LLM中支持多少上下文

**[00:00:02]** could be for example 256 1,24 and so forth depending on how much context we
> 取决于我们想在LLM中支持多少上下文，然后

**[00:00:02]** forth depending on how much context we
> 然后

**[00:00:02]** forth depending on how much context we want to support in the llm and then uh
> 然后现在我们要添加一些更多的

**[00:00:02]** want to support in the llm and then uh
> 现在我们要添加一些更多的

**[00:00:02]** want to support in the llm and then uh now we are going to add a few more
> 现在我们要添加一些更多的内容，所以让我也在图的上下文中展示这个

**[00:00:02]** now we are going to add a few more
> 内容，所以让我也在图的上下文中展示这个

**[00:00:02]** now we are going to add a few more things so um let me show this also in
> 内容，所以让我也在图的上下文中展示这个，其中一个

**[00:00:02]** things so um let me show this also in
> 其中一个

**[00:00:02]** things so um let me show this also in the context of the figure so one of
> 其中一个内容是Transformer块

**[00:00:02]** the context of the figure so one of
> 内容是Transformer块

**[00:00:02]** the context of the figure so one of these things is the Transformer block so
> 内容是Transformer块，一个LLM由多个多个

**[00:00:03]** these things is the Transformer block so
> 一个LLM由多个多个

**[00:00:03]** these things is the Transformer block so and an llm consists of multiple multiple
> 一个LLM由多个多个这样的Transformer块组成，每个

**[00:00:03]** and an llm consists of multiple multiple
> 这样的Transformer块组成，每个

**[00:00:03]** and an llm consists of multiple multiple of these Transformer blocks and each of
> 这样的Transformer块组成，每个Transformer块都有一些

**[00:00:03]** of these Transformer blocks and each of
> Transformer块都有一些

**[00:00:03]** of these Transformer blocks and each of the Transformer blocks has a number of
> Transformer块都有一些元素，我们将在本章后面讨论

**[00:00:03]** the Transformer blocks has a number of
> 元素，我们将在本章后面讨论

**[00:00:03]** the Transformer blocks has a number of elements that we will talk about later
> 元素，我们将在本章后面讨论，但其中一个核心

**[00:00:03]** elements that we will talk about later
> 但其中一个核心

**[00:00:03]** elements that we will talk about later in this chapter but one of the core
> 但其中一个核心元素是掩码多头注意力模块

**[00:00:03]** in this chapter but one of the core
> 元素是掩码多头注意力模块

**[00:00:03]** in this chapter but one of the core elements is the masked multihead
> 元素是掩码多头注意力模块，我们在前一章实现了

**[00:00:03]** elements is the masked multihead
> 我们在前一章实现了

**[00:00:03]** elements is the masked multihead attention module which we implemented in
> 我们在前一章实现了，然后我们要添加一些输出层

**[00:00:03]** attention module which we implemented in
> 然后我们要添加一些输出层

**[00:00:03]** attention module which we implemented in the previous chapter and then we are
> 然后我们要添加一些输出层，这基本上就是LLM架构的样子

**[00:00:03]** the previous chapter and then we are
> 这基本上就是LLM架构的样子

**[00:00:03]** the previous chapter and then we are going to add some output layers and that
> the previous chapter and then we are going to add some output layers and that

**[00:00:03]** going to add some output layers and that
> going to add some output layers and that

**[00:00:03]** going to add some output layers and that is essentially how the llm architecture
> going to add some output layers and that is essentially how the llm architecture

**[00:00:03]** is essentially how the llm architecture
> is essentially how the llm architecture

**[00:00:03]** is essentially how the llm architecture looks like conceptually now going back
> 本质上就是当前LLM架构的概念化形态，现在回到

**[00:00:03]** looks like conceptually now going back
> 概念化形态，现在回到

**[00:00:03]** looks like conceptually now going back to our dummy GPT model this is coding
> 概念化形态，现在回到我们的虚拟GPT模型，这是编码

**[00:00:03]** to our dummy GPT model this is coding
> 我们的虚拟GPT模型，这是编码

**[00:00:03]** to our dummy GPT model this is coding what I've just shown you in the figure
> 我们的虚拟GPT模型，这是编码，即我刚才在图中展示的内容

**[00:00:03]** what I've just shown you in the figure
> 我刚才在图中展示的内容

**[00:00:03]** what I've just shown you in the figure so we have these embedding layers we
> 我刚才在图中展示的内容，所以我们有这些embedding层

**[00:00:03]** so we have these embedding layers we
> 所以我们有这些embedding层

**[00:00:03]** so we have these embedding layers we have optionally Dropout and then here is
> 所以我们有这些embedding层，可选地包含Dropout，然后这里是

**[00:00:03]** have optionally Dropout and then here is
> 可选地包含Dropout，然后这里是

**[00:00:03]** have optionally Dropout and then here is this Transformer block and like I just
> 可选地包含Dropout，然后这里是这个Transformer块，正如我刚才

**[00:00:03]** this Transformer block and like I just
> 这个Transformer块，正如我刚才

**[00:00:03]** this Transformer block and like I just mentioned it gets repeated a number of
> 这个Transformer块，正如我刚才提到的，它会被重复多次

**[00:00:03]** mentioned it gets repeated a number of
> 提到的，它会被重复多次

**[00:00:03]** mentioned it gets repeated a number of times so here we are going to repeat
> 提到的，它会被重复多次，所以这里我们将重复

**[00:00:03]** times so here we are going to repeat
> 多次，所以这里我们将重复

**[00:00:03]** times so here we are going to repeat it a number of times where the number of
> 多次，所以这里我们将重复多次，其中重复的

**[00:00:03]** it a number of times where the number of
> 多次，其中重复的

**[00:00:03]** it a number of times where the number of times is specified by n layers so uh CFG
> 多次，其中重复的次数由n layers指定，所以CFG

**[00:00:03]** times is specified by n layers so uh CFG
> 次数由n layers指定，所以CFG

**[00:00:03]** times is specified by n layers so uh CFG stands for it's just short for
> 次数由n layers指定，所以CFG代表，这只是

**[00:00:03]** stands for it's just short for
> 代表，这只是

**[00:00:03]** stands for it's just short for configuration so this is a python
> 代表，这只是configuration的缩写，所以这是一个Python

**[00:00:03]** configuration so this is a python
> configuration的缩写，所以这是一个Python

**[00:00:03]** configuration so this is a python dictionary as I will show you later that
> configuration的缩写，所以这是一个Python字典，稍后我会展示，它

**[00:00:03]** dictionary as I will show you later that
> 字典，稍后我会展示，它

**[00:00:03]** dictionary as I will show you later that contains some configuration maybe I can
> 字典，稍后我会展示，它包含一些配置，也许我可以

**[00:00:03]** contains some configuration maybe I can
> 包含一些配置，也许我可以

**[00:00:03]** contains some configuration maybe I can actually just
> 包含一些配置，也许我可以直接在这里添加，以便更清晰

**[00:00:04]** actually just
> 直接在这里添加，以便更清晰

**[00:00:04]** actually just added here so that it's more clear um
> 直接在这里添加，以便更清晰，出于某种原因，好了，这里

**[00:00:04]** added here so that it's more clear um
> 以便更清晰，出于某种原因，好了，这里

**[00:00:04]** added here so that it's more clear um for some reason yeah there we go here so
> 以便更清晰，出于某种原因，好了，这里我们有一个vocab size，这是输入中

**[00:00:04]** for some reason yeah there we go here so
> 我们有一个vocab size，这是输入中

**[00:00:04]** for some reason yeah there we go here so we have a vocab size this is the number
> 我们有一个vocab size，这是输入中唯一token的数量，在训练集中

**[00:00:04]** we have a vocab size this is the number
> 唯一token的数量，在训练集中

**[00:00:04]** we have a vocab size this is the number of unique tokens in the input Al in the
> 唯一token的数量，在训练集中，所以这是基于

**[00:00:04]** of unique tokens in the input Al in the
> 训练集中，所以这是基于

**[00:00:04]** of unique tokens in the input Al in the training set so this is um based on the
> 训练集中，所以这是基于我们使用的gpt2 tokenizer，来自

**[00:00:04]** training set so this is um based on the
> 我们使用的gpt2 tokenizer，来自

**[00:00:04]** training set so this is um based on the gpt2 tokenizer that we are using from
> 我们使用的gpt2 tokenizer，来自tick token，然后我们可以指定一个context

**[00:00:04]** gpt2 tokenizer that we are using from
> tick token，然后我们可以指定一个context

**[00:00:04]** gpt2 tokenizer that we are using from tick token then we can specify a context
> tick token，然后我们可以指定一个context length，即LLM可以支持的上下文长度

**[00:00:04]** tick token then we can specify a context
> length，即LLM可以支持的上下文长度

**[00:00:04]** tick token then we can specify a context length how long the context is that the
> length，即LLM可以支持的上下文长度，这里我们选择

**[00:00:04]** length how long the context is that the
> 这里我们选择

**[00:00:04]** length how long the context is that the llm can support here we are choosing uh
> 这里我们选择1,24，但现代LLM也可以

**[00:00:04]** llm can support here we are choosing uh
> 1,24，但现代LLM也可以

**[00:00:04]** llm can support here we are choosing uh 1,24 but yeah modern LMS they can also
> 1,24，但现代LLM也可以支持更长的上下文大小，然而

**[00:00:04]** 1,24 but yeah modern LMS they can also
> 支持更长的上下文大小，然而

**[00:00:04]** 1,24 but yeah modern LMS they can also support longer context sizes however it
> 支持更长的上下文大小，然而这也取决于硬件，即你的

**[00:00:04]** support longer context sizes however it
> 这也取决于硬件，即你的

**[00:00:04]** support longer context sizes however it also depends on the hardware what your
> 这也取决于硬件，即你的硬件支持什么，所以上下文大小越长

**[00:00:04]** also depends on the hardware what your
> 硬件支持什么，所以上下文大小越长

**[00:00:04]** also depends on the hardware what your Hardware supports so the longer the
> 硬件支持什么，所以上下文大小越长，计算成本就越高

**[00:00:04]** Hardware supports so the longer the
> 上下文大小越长，计算成本就越高

**[00:00:04]** Hardware supports so the longer the context size the more expens of the
> 上下文大小越长，计算成本就越高，并且需要更多的硬件

**[00:00:04]** context size the more expens of the
> 计算成本就越高，并且需要更多的硬件

**[00:00:04]** context size the more expens of the computation and there more Hardware
> 计算成本就越高，并且需要更多的硬件要求，embedding size

**[00:00:04]** computation and there more Hardware
> 硬件要求，embedding size

**[00:00:04]** computation and there more Hardware requirements you need um the embedding
> 硬件要求，embedding size，这是token embedding的embedding size

**[00:00:04]** requirements you need um the embedding
> 这是token embedding的embedding size

**[00:00:04]** requirements you need um the embedding size this is the embedding size for the
> 这是token embedding的embedding size，所以在之前的

**[00:00:04]** size this is the embedding size for the
> 所以在之前的

**[00:00:04]** size this is the embedding size for the token embeddings um so previously in
> 所以在之前的第2章中，我们讨论了

**[00:00:04]** token embeddings um so previously in
> 第2章中，我们讨论了

**[00:00:04]** token embeddings um so previously in Chapter 2 we talked about
> 第2章中，我们讨论了三维embedding，现在我们将

**[00:00:04]** Chapter 2 we talked about
> 三维embedding，现在我们将

**[00:00:04]** Chapter 2 we talked about threedimensional embeddings now we are
> 三维embedding，现在我们将扩展到稍大一些的

**[00:00:04]** threedimensional embeddings now we are
> 扩展到稍大一些的

**[00:00:04]** threedimensional embeddings now we are going to scale this up to a bit larger
> 扩展到稍大一些的768，multi-head attention模块中的head数量

**[00:00:04]** going to scale this up to a bit larger
> 768，multi-head attention模块中的head数量

**[00:00:04]** going to scale this up to a bit larger 768 um the number of heads in the
> 768，multi-head attention模块中的head数量，我们在上一章中实现了

**[00:00:04]** 768 um the number of heads in the
> 我们在上一章中实现了

**[00:00:04]** 768 um the number of heads in the multi-ad attention module that we imp
> 我们在上一章中实现了这个模块，这里

**[00:00:04]** multi-ad attention module that we imp
> 这个模块，这里

**[00:00:04]** multi-ad attention module that we imp mented in the previous chapter here the
> 这个模块，这里层的数量，它被称为n layers

**[00:00:05]** mented in the previous chapter here the
> 层的数量，它被称为n layers

**[00:00:05]** mented in the previous chapter here the number of layers it's called yeah n
> mented in the previous chapter here the number of layers it's called yeah n

**[00:00:05]** number of layers it's called yeah n
> number of layers it's called yeah n

**[00:00:05]** number of layers it's called yeah n layers but it really refers to the
> 层数，称为n层，但实际上指的是

**[00:00:05]** layers but it really refers to the
> 层数，但实际上指的是

**[00:00:05]** layers but it really refers to the Transformer blocks how many Transformer
> 层数，但实际上指的是Transformer blocks，有多少个Transformer

**[00:00:05]** Transformer blocks how many Transformer
> Transformer blocks，有多少个Transformer

**[00:00:05]** Transformer blocks how many Transformer blocks we want to repeat the dropout
> Transformer blocks，我们想要重复多少次，dropout

**[00:00:05]** blocks we want to repeat the dropout
> 我们想要重复多少次，dropout

**[00:00:05]** blocks we want to repeat the dropout rate and whether we want to use the bias
> 我们想要重复多少次，dropout rate，以及是否使用bias

**[00:00:05]** rate and whether we want to use the bias
> rate，以及是否使用bias

**[00:00:05]** rate and whether we want to use the bias in the query key and value linear layers
> rate，以及是否使用bias在query、key和value的线性层中

**[00:00:05]** in the query key and value linear layers
> 在query、key和value的线性层中

**[00:00:05]** in the query key and value linear layers um yeah and this is uh just a
> 在query、key和value的线性层中，嗯，是的，这只是一个

**[00:00:05]** um yeah and this is uh just a
> 嗯，是的，这只是一个

**[00:00:05]** um yeah and this is uh just a configuration file that we are going to
> 嗯，是的，这只是一个配置文件，我们将

**[00:00:05]** configuration file that we are going to
> 配置文件，我们将

**[00:00:05]** configuration file that we are going to pass as CFG into the dummy model the
> 配置文件，我们将作为CFG传递给dummy model，这个

**[00:00:05]** pass as CFG into the dummy model the
> 作为CFG传递给dummy model，这个

**[00:00:05]** pass as CFG into the dummy model the reason why I'm calling this c um CFG is
> 作为CFG传递给dummy model，我称它为c，嗯，CFG的原因是

**[00:00:05]** reason why I'm calling this c um CFG is
> 我称它为c，嗯，CFG的原因是

**[00:00:05]** reason why I'm calling this c um CFG is to keep it short so it doesn't go off
> 我称它为c，嗯，CFG的原因是为了保持简短，这样它就不会超出

**[00:00:05]** to keep it short so it doesn't go off
> 为了保持简短，这样它就不会超出

**[00:00:05]** to keep it short so it doesn't go off here in the cell and becomes too long
> 为了保持简短，这样它就不会超出这里的单元格，变得太长

**[00:00:05]** here in the cell and becomes too long
> 这里的单元格，变得太长

**[00:00:05]** here in the cell and becomes too long because yeah otherwise it would be
> 这里的单元格，变得太长，因为否则它会

**[00:00:05]** because yeah otherwise it would be
> 因为否则它会

**[00:00:05]** because yeah otherwise it would be harder to read um yeah and so we have
> 因为否则它会更难阅读，嗯，是的，所以我们有

**[00:00:05]** harder to read um yeah and so we have
> 更难阅读，嗯，是的，所以我们有

**[00:00:05]** harder to read um yeah and so we have these Transformer
> 更难阅读，嗯，是的，所以我们有这些Transformer

**[00:00:05]** these Transformer
> 这些Transformer

**[00:00:05]** these Transformer blocks and um then we have something
> 这些Transformer blocks，然后我们有一些

**[00:00:05]** blocks and um then we have something
> blocks，然后我们有一些

**[00:00:05]** blocks and um then we have something called a layer Norm here we will get to
> blocks，然后我们有一些叫做layer Norm的东西，我们稍后会

**[00:00:05]** called a layer Norm here we will get to
> 叫做layer Norm的东西，我们稍后会

**[00:00:05]** called a layer Norm here we will get to that later and then we have this output
> 叫做layer Norm的东西，我们稍后会讲到，然后我们有这个输出

**[00:00:05]** that later and then we have this output
> 讲到，然后我们有这个输出

**[00:00:05]** that later and then we have this output layer and so you can see what's
> 讲到，然后我们有这个输出层，所以你可以看到

**[00:00:05]** layer and so you can see what's
> 层，所以你可以看到

**[00:00:05]** layer and so you can see what's interesting is the input layer maps from
> 层，所以你可以看到有趣的是，输入层从

**[00:00:05]** interesting is the input layer maps from
> 有趣的是，输入层从

**[00:00:05]** interesting is the input layer maps from the vocabulary size into the embedding
> 有趣的是，输入层从词汇表大小映射到embedding

**[00:00:05]** the vocabulary size into the embedding
> 词汇表大小映射到embedding

**[00:00:05]** the vocabulary size into the embedding space and then the output layer Maps
> 词汇表大小映射到embedding空间，然后输出层映射

**[00:00:05]** space and then the output layer Maps
> 空间，然后输出层映射

**[00:00:05]** space and then the output layer Maps back from the embedding
> 空间，然后输出层从embedding

**[00:00:06]** back from the embedding
> 从embedding

**[00:00:06]** back from the embedding aing space to the vocabulary size so
> 从embedding空间映射回词汇表大小，所以

**[00:00:06]** aing space to the vocabulary size so
> 空间映射回词汇表大小，所以

**[00:00:06]** aing space to the vocabulary size so this will allow us to generate words
> 空间映射回词汇表大小，这将允许我们生成单词

**[00:00:06]** this will allow us to generate words
> 这将允许我们生成单词

**[00:00:06]** this will allow us to generate words because what we're doing is we're
> 这将允许我们生成单词，因为我们所做的是

**[00:00:06]** because what we're doing is we're
> 因为我们所做的是

**[00:00:06]** because what we're doing is we're transforming the token IDs the words
> 因为我们所做的是将token IDs，即单词

**[00:00:06]** transforming the token IDs the words
> 将token IDs，即单词

**[00:00:06]** transforming the token IDs the words into an intermediate representation the
> 将token IDs，即单词转换为中间表示，即

**[00:00:06]** into an intermediate representation the
> 转换为中间表示，即

**[00:00:06]** into an intermediate representation the embedding space like a vector
> 转换为中间表示，即embedding空间，像一个向量

**[00:00:06]** embedding space like a vector
> embedding空间，像一个向量

**[00:00:06]** embedding space like a vector representation and then later on we will
> embedding空间，像一个向量表示，然后稍后我们将

**[00:00:06]** representation and then later on we will
> 表示，然后稍后我们将

**[00:00:06]** representation and then later on we will transfer them back into this original
> 表示，然后稍后我们将它们转换回这个原始

**[00:00:06]** transfer them back into this original
> 它们转换回这个原始

**[00:00:06]** transfer them back into this original vocabulary size um space back into token
> 它们转换回这个原始词汇表大小空间，回到token

**[00:00:06]** vocabulary size um space back into token
> 词汇表大小空间，回到token

**[00:00:06]** vocabulary size um space back into token IDs essentially so that the llm
> 词汇表大小空间，回到token IDs，基本上，这样LLM

**[00:00:06]** IDs essentially so that the llm
> IDs，基本上，这样LLM

**[00:00:06]** IDs essentially so that the llm generates
> IDs，基本上，这样LLM生成

**[00:00:06]** generates
> 生成

**[00:00:06]** generates text so these are the elements and here
> 生成文本，所以这些是元素，这里

**[00:00:06]** text so these are the elements and here
> 文本，所以这些是元素，这里

**[00:00:06]** text so these are the elements and here in the forward method we are going to to
> 文本，所以这些是元素，这里在forward方法中，我们将

**[00:00:06]** in the forward method we are going to to
> 在forward方法中，我们将

**[00:00:06]** in the forward method we are going to to use these elements so we create the
> 在forward方法中，我们将使用这些元素，所以我们创建

**[00:00:06]** use these elements so we create the
> 使用这些元素，所以我们创建

**[00:00:06]** use these elements so we create the embeddings um we add Dropout optionally
> 使用这些元素，所以我们创建embeddings，嗯，我们可选地添加Dropout

**[00:00:06]** embeddings um we add Dropout optionally
> embeddings，嗯，我们可选地添加Dropout

**[00:00:06]** embeddings um we add Dropout optionally then we apply the Transformer blocks so
> embeddings，嗯，我们可选地添加Dropout，然后应用Transformer blocks，所以

**[00:00:06]** then we apply the Transformer blocks so
> 然后应用Transformer blocks，所以

**[00:00:06]** then we apply the Transformer blocks so when I call this one um thanks to this
> 然后应用Transformer blocks，所以当我调用这个时，嗯，多亏了这个

**[00:00:06]** when I call this one um thanks to this
> 当我调用这个时，嗯，多亏了这个

**[00:00:06]** when I call this one um thanks to this sequential it will put it through all
> 当我调用这个时，嗯，多亏了这个sequential，它会将所有内容通过

**[00:00:06]** sequential it will put it through all
> sequential it will put it through all

**[00:00:06]** sequential it will put it through all the different Transformer
> 顺序地，它会将其送入所有不同的Transformer

**[00:00:06]** the different Transformer
> 不同的Transformer

**[00:00:06]** the different Transformer blocks um yeah and then we have the
> 不同的Transformer块，嗯，然后我们还有

**[00:00:06]** blocks um yeah and then we have the
> 块，嗯，然后我们还有

**[00:00:06]** blocks um yeah and then we have the output layer here so if
> 块，嗯，然后我们还有输出层，所以如果

**[00:00:06]** output layer here so if
> 输出层，所以如果

**[00:00:06]** output layer here so if we consider this as
> 输出层，所以如果我们将其视为

**[00:00:06]** we consider this as
> 我们将其视为

**[00:00:06]** we consider this as our llm this looks very simple because
> 我们将其视为我们的LLM，这看起来非常简单，因为

**[00:00:06]** our llm this looks very simple because
> 我们的LLM，这看起来非常简单，因为

**[00:00:06]** our llm this looks very simple because we have these placeholders but just to
> 我们的LLM，这看起来非常简单，因为我们有这些占位符，但只是为了

**[00:00:06]** we have these placeholders but just to
> 我们有这些占位符，但只是为了

**[00:00:06]** we have these placeholders but just to make this work in in the case of the
> 我们有这些占位符，但只是为了让它能在之前视频的情况下工作

**[00:00:06]** make this work in in the case of the
> 让它能在之前视频的情况下工作

**[00:00:06]** make this work in in the case of the video before we fill in the placeholder
> 让它能在之前视频的情况下工作，在我们用真实代码填充占位符之前

**[00:00:07]** video before we fill in the placeholder
> 视频，在我们用真实代码填充占位符之前

**[00:00:07]** video before we fill in the placeholder with real code I have also placeholder
> 视频，在我们用真实代码填充占位符之前，我也有占位符

**[00:00:07]** with real code I have also placeholder
> 用真实代码，我也有占位符

**[00:00:07]** with real code I have also placeholder code so you can see for now I'm calling
> 用真实代码，我也有占位符代码，所以你可以看到，目前我称其为

**[00:00:07]** code so you can see for now I'm calling
> 代码，所以你可以看到，目前我称其为

**[00:00:07]** code so you can see for now I'm calling this the dummy Transformer block and
> 代码，所以你可以看到，目前我称其为虚拟Transformer块，并且

**[00:00:07]** this the dummy Transformer block and
> 这个虚拟Transformer块，并且

**[00:00:07]** this the dummy Transformer block and it's really not doing anything it's just
> 这个虚拟Transformer块，它实际上什么也不做，只是

**[00:00:07]** it's really not doing anything it's just
> 它实际上什么也不做，只是

**[00:00:07]** it's really not doing anything it's just returning the input later we will fill
> 它实际上什么也不做，只是返回输入，稍后我们将用实际代码填充

**[00:00:07]** returning the input later we will fill
> 返回输入，稍后我们将用实际代码填充

**[00:00:07]** returning the input later we will fill this block with actual code that will
> 返回输入，稍后我们将用实际代码填充这个块，这些代码将

**[00:00:07]** this block with actual code that will
> 这个块，这些代码将

**[00:00:07]** this block with actual code that will explain what the Transformer block does
> 这个块，这些代码将解释Transformer块具体做什么

**[00:00:07]** explain what the Transformer block does
> 解释Transformer块具体做什么

**[00:00:07]** explain what the Transformer block does exactly but as you already uh have seen
> 解释Transformer块具体做什么，但正如你已经看到的

**[00:00:07]** exactly but as you already uh have seen
> 但正如你已经看到的

**[00:00:07]** exactly but as you already uh have seen here one part of the Transformer block
> 但正如你已经看到的，这里Transformer块的一部分

**[00:00:07]** here one part of the Transformer block
> 这里Transformer块的一部分

**[00:00:07]** here one part of the Transformer block is the masked multi-ad attention module
> 这里Transformer块的一部分是掩码多头注意力模块

**[00:00:07]** is the masked multi-ad attention module
> 是掩码多头注意力模块

**[00:00:07]** is the masked multi-ad attention module so we will add that later on here and a
> 是掩码多头注意力模块，所以我们稍后会在这里添加它，以及

**[00:00:07]** so we will add that later on here and a
> 所以我们稍后会在这里添加它，以及

**[00:00:07]** so we will add that later on here and a few other things and then layer Norm we
> 所以我们稍后会在这里添加它，以及一些其他东西，然后还有层归一化，我们

**[00:00:07]** few other things and then layer Norm we
> 一些其他东西，然后还有层归一化，我们

**[00:00:07]** few other things and then layer Norm we will also discuss that separately it's
> 一些其他东西，然后还有层归一化，我们也会单独讨论它，它

**[00:00:07]** will also discuss that separately it's
> 也会单独讨论它，它

**[00:00:07]** will also discuss that separately it's essentially a normalization layer that
> 也会单独讨论它，它本质上是一个归一化层，

**[00:00:07]** essentially a normalization layer that
> 本质上是一个归一化层，

**[00:00:07]** essentially a normalization layer that gets used both in the Transformer block
> 本质上是一个归一化层，在Transformer块中都会用到

**[00:00:07]** gets used both in the Transformer block
> 在Transformer块中都会用到

**[00:00:07]** gets used both in the Transformer block and here um essentially but these are
> 在Transformer块中都会用到，嗯，基本上，但这些是

**[00:00:07]** and here um essentially but these are
> 嗯，基本上，但这些是

**[00:00:07]** and here um essentially but these are details for now uh just yeah think of it
> 嗯，基本上，但这些是细节，目前，嗯，就把它看作

**[00:00:07]** details for now uh just yeah think of it
> 细节，目前，嗯，就把它看作

**[00:00:07]** details for now uh just yeah think of it as the big picture concept and we can
> 细节，目前，嗯，就把它看作整体概念，我们可以

**[00:00:07]** as the big picture concept and we can
> 整体概念，我们可以

**[00:00:07]** as the big picture concept and we can already use this for example if we have
> 整体概念，我们已经可以用它了，例如，如果我们有

**[00:00:07]** already use this for example if we have
> 已经可以用它了，例如，如果我们有

**[00:00:07]** already use this for example if we have some inputs here so we are using the Tik
> 已经可以用它了，例如，如果我们有一些输入，这里我们使用的是第2章的Tik

**[00:00:07]** some inputs here so we are using the Tik
> 一些输入，这里我们使用的是第2章的Tik

**[00:00:07]** some inputs here so we are using the Tik token tokenizer from chapter 2 we are
> 一些输入，这里我们使用的是第2章的Tik token tokenizer，我们

**[00:00:07]** token tokenizer from chapter 2 we are
> token tokenizer，我们

**[00:00:07]** token tokenizer from chapter 2 we are loing it um here and then we create a
> token tokenizer，我们在这里加载它，然后我们创建一个

**[00:00:07]** loing it um here and then we create a
> 加载它，然后我们创建一个

**[00:00:07]** loing it um here and then we create a small yeah uh small here so the batch is
> 加载它，然后我们创建一个小的，嗯，小的，所以batch是

**[00:00:08]** small yeah uh small here so the batch is
> 小的，嗯，小的，所以batch是

**[00:00:08]** small yeah uh small here so the batch is I can just show it to you um hope it
> 小的，嗯，小的，所以batch是，我可以展示给你看，嗯，希望它

**[00:00:08]** I can just show it to you um hope it
> 我可以展示给你看，嗯，希望它

**[00:00:08]** I can just show it to you um hope it works yeah there we go so the batch is
> 我可以展示给你看，嗯，希望它能工作，是的，好了，所以batch是

**[00:00:08]** works yeah there we go so the batch is
> 能工作，是的，好了，所以batch是

**[00:00:08]** works yeah there we go so the batch is consisting of two inputs one and two and
> 能工作，是的，好了，所以batch由两个输入组成，一个和两个，

**[00:00:08]** consisting of two inputs one and two and
> 由两个输入组成，一个和两个，

**[00:00:08]** consisting of two inputs one and two and so the first input text is every effort
> 由两个输入组成，一个和两个，所以第一个输入文本是"every effort moves you"，第二个输入文本是

**[00:00:08]** so the first input text is every effort
> 所以第一个输入文本是"every effort moves you"，第二个输入文本是

**[00:00:08]** so the first input text is every effort moves you and the second input text is
> 所以第一个输入文本是"every effort moves you"，第二个输入文本是"every day holds a"，然后等等，

**[00:00:08]** moves you and the second input text is
> "every day holds a"，然后等等，

**[00:00:08]** moves you and the second input text is every day holds a and then so forth and
> "every day holds a"，然后等等，LLM的任务是完成这个

**[00:00:08]** every day holds a and then so forth and
> LLM的任务是完成这个

**[00:00:08]** every day holds a and then so forth and the task of the llm is to complete this
> LLM的任务是完成这个输入

**[00:00:08]** the task of the llm is to complete this
> 输入

**[00:00:08]** the task of the llm is to complete this input
> 输入文本，所以我们可以看到这是第一个

**[00:00:08]** input
> 文本，所以我们可以看到这是第一个

**[00:00:08]** input text so we can see this is the the first
> input text so we can see this is the the first

**[00:00:08]** text so we can see this is the the first
> text so we can see this is the the first

**[00:00:08]** text so we can see this is the the first text um we have 1 2 3 four tokens so in
> 文本，这样我们可以看到这是第一个文本，嗯，我们有1 2 3 四个token，所以

**[00:00:08]** text um we have 1 2 3 four tokens so in
> 文本，嗯，我们有1 2 3 四个token，所以

**[00:00:08]** text um we have 1 2 3 four tokens so in this case each word represents a token
> 文本，嗯，我们有1 2 3 四个token，所以在这个例子中，每个词代表一个token

**[00:00:08]** this case each word represents a token
> 在这个例子中，每个词代表一个token

**[00:00:08]** this case each word represents a token it's not all always true sometimes a
> 在这个例子中，每个词代表一个token，但这并不总是成立，有时一个

**[00:00:08]** it's not all always true sometimes a
> 这并不总是成立，有时一个

**[00:00:08]** it's not all always true sometimes a word gets split into multiple tokens but
> 这并不总是成立，有时一个词会被拆分成多个token，但

**[00:00:08]** word gets split into multiple tokens but
> 词会被拆分成多个token，但

**[00:00:08]** word gets split into multiple tokens but to keep the example simple I chose words
> 词会被拆分成多个token，但为了保持例子简单，我选择了每个词

**[00:00:08]** to keep the example simple I chose words
> 为了保持例子简单，我选择了每个词

**[00:00:08]** to keep the example simple I chose words where each word is exactly one token um
> 为了保持例子简单，我选择了每个词恰好是一个token的词，嗯

**[00:00:08]** where each word is exactly one token um
> 每个词恰好是一个token，嗯

**[00:00:08]** where each word is exactly one token um then we are here um yeah using the
> 每个词恰好是一个token，嗯，然后我们在这里，嗯，使用

**[00:00:08]** then we are here um yeah using the
> 然后我们在这里，嗯，使用

**[00:00:08]** then we are here um yeah using the tokenizer to encode it turn it into a
> 然后我们在这里，嗯，使用tokenizer进行编码，将其转换为

**[00:00:08]** tokenizer to encode it turn it into a
> tokenizer进行编码，将其转换为

**[00:00:08]** tokenizer to encode it turn it into a tensor and so what you're see seeing
> tokenizer进行编码，将其转换为一个tensor，所以你现在看到的是

**[00:00:08]** tensor and so what you're see seeing
> tensor，所以你现在看到的是

**[00:00:08]** tensor and so what you're see seeing here are the token IDs corresponding to
> tensor，所以你现在看到的是对应于这些文本的token ID

**[00:00:08]** here are the token IDs corresponding to
> 对应于这些文本的token ID

**[00:00:08]** here are the token IDs corresponding to these
> 对应于这些文本的token ID

**[00:00:08]** these
> 这些

**[00:00:08]** these texts so this is an input that our llm
> 这些文本，所以这是我们的LLM在训练期间会接收的输入，然后

**[00:00:08]** texts so this is an input that our llm
> 文本，所以这是我们的LLM在训练期间会接收的输入，然后

**[00:00:08]** texts so this is an input that our llm will receive during training and then
> 文本，所以这是我们的LLM在训练期间会接收的输入，然后任务是让LM生成或

**[00:00:08]** will receive during training and then
> 任务是让LM生成或

**[00:00:08]** will receive during training and then the task is for the LM to generate or
> 任务是让LM生成或预测这个文本中的下一个词，当我们

**[00:00:09]** the task is for the LM to generate or
> 预测这个文本中的下一个词，当我们

**[00:00:09]** the task is for the LM to generate or predict the next word in this text when
> 预测这个文本中的下一个词，当我们进行预训练时，预训练将是下一章的主题，所以这里

**[00:00:09]** predict the next word in this text when
> 进行预训练时，预训练将是下一章的主题，所以这里

**[00:00:09]** predict the next word in this text when we are pre-training it pre-training will
> 进行预训练时，预训练将是下一章的主题，所以这里我只是在解释输入和

**[00:00:09]** we are pre-training it pre-training will
> 我只是在解释输入和

**[00:00:09]** we are pre-training it pre-training will be the topic of the next chapter so here
> 我只是在解释输入和输出看起来是什么样子，所以现在我们在这里

**[00:00:09]** be the topic of the next chapter so here
> 输出看起来是什么样子，所以现在我们在这里

**[00:00:09]** be the topic of the next chapter so here I'm just explaining what the input and
> 输出看起来是什么样子，所以现在我们在这里使用我指定的配置初始化一个新的虚拟GPT模型

**[00:00:09]** I'm just explaining what the input and
> 使用我指定的配置初始化一个新的虚拟GPT模型

**[00:00:09]** I'm just explaining what the input and output looks like so here we are now
> 使用我指定的配置初始化一个新的虚拟GPT模型，这里，所以这实际上我们可以

**[00:00:09]** output looks like so here we are now
> 这里，所以这实际上我们可以

**[00:00:09]** output looks like so here we are now initializing a new dummy GPT model using
> 这里，所以这实际上我们可以让它更清晰，这就是你到处看到的

**[00:00:09]** initializing a new dummy GPT model using
> 让它更清晰，这就是你到处看到的

**[00:00:09]** initializing a new dummy GPT model using the configurations I have specified
> 让它更清晰，这就是你到处看到的CFG，这里

**[00:00:09]** the configurations I have specified
> CFG，这里

**[00:00:09]** the configurations I have specified here so this would be really we can also
> CFG，这里这个

**[00:00:09]** here so this would be really we can also
> 这个

**[00:00:09]** here so this would be really we can also make it more clear this would be this
> 这个字典，然后我们已经可以使用

**[00:00:09]** make it more clear this would be this
> 字典，然后我们已经可以使用

**[00:00:09]** make it more clear this would be this CFG that you are seeing everywhere here
> 字典，然后我们已经可以使用模型，所以模型被初始化

**[00:00:09]** CFG that you are seeing everywhere here
> 模型，所以模型被初始化

**[00:00:09]** CFG that you are seeing everywhere here this
> 模型，所以模型被初始化为各层的随机值

**[00:00:09]** this
> 各层的随机值

**[00:00:09]** this dictionary and then we can already use
> 各层的随机值，embedding层，稍后我们还会

**[00:00:09]** dictionary and then we can already use
> embedding层，稍后我们还会

**[00:00:09]** dictionary and then we can already use the model so the model is initialized
> embedding层，稍后我们还会有一个线性层，或者我们已经有一个线性层

**[00:00:09]** the model so the model is initialized
> 有一个线性层，或者我们已经有一个线性层

**[00:00:09]** the model so the model is initialized with random values for the layers the
> 有一个线性层，或者我们已经有一个线性层，这里，类似于

**[00:00:09]** with random values for the layers the
> 这里，类似于

**[00:00:09]** with random values for the layers the embeding layers and later we will also
> 这里，类似于之前，嗯，这些将被初始化为

**[00:00:09]** embeding layers and later we will also
> 之前，嗯，这些将被初始化为

**[00:00:09]** embeding layers and later we will also have the or we already have a linear
> 之前，嗯，这些将被初始化为随机值，并且这些将在我们优化LLM时被训练

**[00:00:09]** have the or we already have a linear
> 随机值，并且这些将在我们优化LLM时被训练

**[00:00:09]** have the or we already have a linear layer here so similar to
> 随机值，并且这些将在我们优化LLM时被训练，嗯，是的，所以这就是初始化

**[00:00:09]** layer here so similar to
> 嗯，是的，所以这就是初始化

**[00:00:09]** layer here so similar to before um these will be initialized with
> 嗯，是的，所以这就是初始化模型，然后这里我们调用

**[00:00:09]** before um these will be initialized with
> 模型，然后这里我们调用

**[00:00:09]** before um these will be initialized with random values and those will be trained
> 模型，然后这里我们调用模型，模型的输出通常被称为logits，所以

**[00:00:09]** random values and those will be trained
> 模型，模型的输出通常被称为logits，所以

**[00:00:09]** random values and those will be trained later when we are optimizing the
> 模型的输出通常被称为logits，所以这有点像深度学习术语，但在

**[00:00:09]** later when we are optimizing the
> 这有点像深度学习术语，但在

**[00:00:09]** later when we are optimizing the llm um yeah so that is initializing the
> 这有点像深度学习术语，但在深度学习领域，在深度学习领域

**[00:00:09]** llm um yeah so that is initializing the
> 深度学习领域，在深度学习领域

**[00:00:09]** llm um yeah so that is initializing the model and then here we are calling the
> llm um yeah so that is initializing the model and then here we are calling the

**[00:00:09]** model and then here we are calling the
> model and then here we are calling the

**[00:00:09]** model and then here we are calling the model and the output of the model they
> model and then here we are calling the model and the output of the model they

**[00:00:10]** model and the output of the model they
> model and the output of the model they

**[00:00:10]** model and the output of the model they are usually referred to as logits so
> model and the output of the model they are usually referred to as logits so

**[00:00:10]** are usually referred to as logits so
> are usually referred to as logits so

**[00:00:10]** are usually referred to as logits so it's like a deep learning jargon but in
> are usually referred to as logits so it's like a deep learning jargon but in

**[00:00:10]** it's like a deep learning jargon but in
> it's like a deep learning jargon but in

**[00:00:10]** it's like a deep learning jargon but in deep learning in the field of deep
> it's like a deep learning jargon but in deep learning in the field of deep

**[00:00:10]** deep learning in the field of deep
> deep learning in the field of deep

**[00:00:10]** deep learning in the field of deep learning the last layer the last linear
> 在深度学习领域中，最后一层是最后的线性层

**[00:00:10]** learning the last layer the last linear
> 学习最后一层，最后的线性层

**[00:00:10]** learning the last layer the last linear layer that Returns the outputs of the
> 学习最后一层，最后的线性层，它返回输出

**[00:00:10]** layer that Returns the outputs of the
> 它返回输出

**[00:00:10]** layer that Returns the outputs of the linear layer those values are referred
> 它返回线性层的输出，这些值被称为

**[00:00:10]** linear layer those values are referred
> 线性层的这些值被称为

**[00:00:10]** linear layer those values are referred to as
> 线性层的这些值被称为

**[00:00:10]** to as
> 被称为

**[00:00:10]** to as logits I have uh I taught statistics
> 被称为logits。我教过统计学

**[00:00:10]** logits I have uh I taught statistics
> logits。我教过统计学

**[00:00:10]** logits I have uh I taught statistics classes I was a statistics faculty um a
> logits。我教过统计学课程，我曾是统计学教师

**[00:00:10]** classes I was a statistics faculty um a
> 我曾是统计学教师

**[00:00:10]** classes I was a statistics faculty um a few years back it's feels like a
> 我曾是统计学教师，几年前，感觉像是

**[00:00:10]** few years back it's feels like a
> 几年前，感觉像是

**[00:00:10]** few years back it's feels like a lifetime ago but this was there's a lot
> 几年前，感觉像是上辈子的事，但这有很多

**[00:00:10]** lifetime ago but this was there's a lot
> 上辈子的事，但这有很多

**[00:00:10]** lifetime ago but this was there's a lot of uh Theory or not Theory but like
> 上辈子的事，但这有很多理论，或者不是理论，而是

**[00:00:10]** of uh Theory or not Theory but like
> 理论，或者不是理论，而是

**[00:00:10]** of uh Theory or not Theory but like there's a reason why it's called logits
> 理论，或者不是理论，而是有原因为什么它被称为logits

**[00:00:10]** there's a reason why it's called logits
> 有原因为什么它被称为logits

**[00:00:10]** there's a reason why it's called logits it's uh something I have also videos for
> 有原因为什么它被称为logits，我也有相关视频

**[00:00:10]** it's uh something I have also videos for
> 我也有相关视频

**[00:00:10]** it's uh something I have also videos for if you're interested but for now just
> 我也有相关视频，如果你感兴趣，但现在只需

**[00:00:10]** if you're interested but for now just
> 如果你感兴趣，但现在只需

**[00:00:10]** if you're interested but for now just refer to it maybe as or think of it as
> 如果你感兴趣，但现在只需将其视为或理解为

**[00:00:10]** refer to it maybe as or think of it as
> 将其视为或理解为

**[00:00:10]** refer to it maybe as or think of it as the last layer outputs before we pass it
> 将其视为或理解为在传递给任何其他函数之前的最后一层输出

**[00:00:10]** the last layer outputs before we pass it
> 在传递给任何其他函数之前的最后一层输出

**[00:00:10]** the last layer outputs before we pass it to any other function so uh we have not
> 在传递给任何其他函数之前的最后一层输出，所以我们还没有

**[00:00:10]** to any other function so uh we have not
> 所以我们还没有

**[00:00:10]** to any other function so uh we have not Define this one of course it's a
> 所以我们还没有定义这个，当然这是

**[00:00:10]** Define this one of course it's a
> 定义这个，当然这是

**[00:00:10]** Define this one of course it's a recurring theme here in these videos
> 定义这个，当然这是这些视频中反复出现的主题

**[00:00:10]** recurring theme here in these videos
> 这些视频中反复出现的主题

**[00:00:10]** recurring theme here in these videos that I forget to define or execute some
> 这些视频中反复出现的主题，我忘记定义或执行某些

**[00:00:10]** that I forget to define or execute some
> 我忘记定义或执行某些

**[00:00:10]** that I forget to define or execute some cells so um what you can see here is a
> 我忘记定义或执行某些单元格，所以你可以在这里看到的是一个

**[00:00:10]** cells so um what you can see here is a
> 单元格，所以你可以在这里看到的是一个

**[00:00:10]** cells so um what you can see here is a very large tensor and this tensor is 2x
> 单元格，所以你可以在这里看到的是一个非常大的tensor，这个tensor是2x

**[00:00:11]** very large tensor and this tensor is 2x
> 非常大的tensor，这个tensor是2x

**[00:00:11]** very large tensor and this tensor is 2x 4 by
> 非常大的tensor，这个tensor是2x 4 by

**[00:00:11]** 4 by
> 4 by

**[00:00:11]** 4 by 50257 and so the reason uh for this
> 4 by 50257，所以这个维度的原因是

**[00:00:11]** 50257 and so the reason uh for this
> 50257，所以这个维度的原因是

**[00:00:11]** 50257 and so the reason uh for this Dimension is yeah we have two inputs so
> 50257，所以这个维度的原因是，是的，我们有两个输入

**[00:00:11]** Dimension is yeah we have two inputs so
> 我们有两个输入

**[00:00:11]** Dimension is yeah we have two inputs so here I can maybe show this for reference
> 我们有两个输入，所以这里我可以展示这个作为参考

**[00:00:11]** here I can maybe show this for reference
> 这里我可以展示这个作为参考

**[00:00:11]** here I can maybe show this for reference batch
> 这里我可以展示这个作为参考，batch

**[00:00:11]** batch
> batch

**[00:00:11]** batch dot shape is also yeah two rows so that
> batch的形状也是两行

**[00:00:11]** dot shape is also yeah two rows so that
> 形状也是两行

**[00:00:11]** dot shape is also yeah two rows so that is where the two comes from and then we
> 形状也是两行，所以这就是2的来源，然后我们

**[00:00:11]** is where the two comes from and then we
> 这就是2的来源，然后我们

**[00:00:11]** is where the two comes from and then we have the four tokens in each column so
> 这就是2的来源，然后我们在每列中有四个token

**[00:00:11]** have the four tokens in each column so
> 在每列中有四个token

**[00:00:11]** have the four tokens in each column so here that's the four but now we have
> 在每列中有四个token，所以这里是4，但现在我们有

**[00:00:11]** here that's the four but now we have
> 所以这里是4，但现在我们有

**[00:00:11]** here that's the four but now we have 50,2 257 so these are the logits um for
> 所以这里是4，但现在我们有50,257，所以这些是每个token的logits

**[00:00:11]** 50,2 257 so these are the logits um for
> 50,257，所以这些是每个token的logits

**[00:00:11]** 50,2 257 so these are the logits um for each of the tokens essentially so this
> 50,257，所以这些是每个token的logits，基本上这个

**[00:00:11]** each of the tokens essentially so this
> 每个token的logits，基本上这个

**[00:00:11]** each of the tokens essentially so this Dimension comes from the fact that we
> 每个token的logits，基本上这个维度来自于我们

**[00:00:11]** Dimension comes from the fact that we
> 维度来自于我们

**[00:00:11]** Dimension comes from the fact that we have an output layer
> 维度来自于我们有一个输出层

**[00:00:11]** have an output layer
> 有一个输出层

**[00:00:11]** have an output layer here that generates this voap size so
> 有一个输出层，它生成这个词汇表大小

**[00:00:11]** here that generates this voap size so
> 它生成这个词汇表大小

**[00:00:11]** here that generates this voap size so each token before each token was
> 它生成这个词汇表大小，所以每个token之前每个token被

**[00:00:11]** each token before each token was
> 每个token之前每个token被

**[00:00:11]** each token before each token was represented by um here in this case 768
> 每个token之前每个token被表示为，这里在这种情况下是768

**[00:00:11]** represented by um here in this case 768
> 表示为，这里在这种情况下是768

**[00:00:11]** represented by um here in this case 768 dimensional Vector in Chapter 2 we used
> 表示为，这里在这种情况下是768维向量，在第2章中我们使用了

**[00:00:11]** dimensional Vector in Chapter 2 we used
> 维向量，在第2章中我们使用了

**[00:00:11]** dimensional Vector in Chapter 2 we used a threedimensional Vector um so that is
> 维向量，在第2章中我们使用了一个三维向量，所以这就是

**[00:00:12]** a threedimensional Vector um so that is
> 一个三维向量，所以这就是

**[00:00:12]** a threedimensional Vector um so that is what goes in and then what comes out
> 一个三维向量，所以这就是输入，然后输出

**[00:00:12]** what goes in and then what comes out
> 输入，然后输出

**[00:00:12]** what goes in and then what comes out here is same as the vocabulary size it's
> 输入，然后输出这里与词汇表大小相同

**[00:00:12]** here is same as the vocabulary size it's
> 这里与词汇表大小相同

**[00:00:12]** here is same as the vocabulary size it's a 50,2 57 dimensional Vector so this
> 这与词汇表大小相同，是一个50,257维的向量，所以这个

**[00:00:12]** a 50,2 57 dimensional Vector so this
> 一个50,257维的向量，所以这个

**[00:00:12]** a 50,2 57 dimensional Vector so this sounds like complicated so how does that
> 一个50,257维的向量，所以这听起来很复杂，那么它如何

**[00:00:12]** sounds like complicated so how does that
> 听起来很复杂，那么它如何

**[00:00:12]** sounds like complicated so how does that um turn back into token IDs so for that
> 听起来很复杂，那么它如何嗯转换回token ID，为此

**[00:00:12]** um turn back into token IDs so for that
> 嗯转换回token ID，为此

**[00:00:12]** um turn back into token IDs so for that um we will actually look at this closer
> 嗯转换回token ID，为此嗯我们实际上会更仔细地查看这个

**[00:00:12]** um we will actually look at this closer
> 嗯我们实际上会更仔细地查看这个

**[00:00:12]** um we will actually look at this closer in chapter 5 so what we have to do then
> 嗯我们实际上会更仔细地查看这个在第5章中，那么我们需要做的是

**[00:00:12]** in chapter 5 so what we have to do then
> 在第5章中，那么我们需要做的是

**[00:00:12]** in chapter 5 so what we have to do then is we have to optimize the llm we have
> 在第5章中，那么我们需要做的是优化LLM，我们

**[00:00:12]** is we have to optimize the llm we have
> 是优化LLM，我们

**[00:00:12]** is we have to optimize the llm we have to train the llm and the llm will learn
> 是优化LLM，我们需要训练LLM，LLM将学习

**[00:00:12]** to train the llm and the llm will learn
> 训练LLM，LLM将学习

**[00:00:12]** to train the llm and the llm will learn that
> 训练LLM，LLM将学习到

**[00:00:12]** that
> 到

**[00:00:12]** that the next the next token it tries to
> 到它试图预测的下一个token，所以如果我们给它这个输入

**[00:00:12]** the next the next token it tries to
> 下一个token，它试图预测，所以如果我们给它这个输入

**[00:00:12]** the next the next token it tries to predict so if we give it this input the
> 下一个token，它试图预测，所以如果我们给它这个输入，对应于

**[00:00:12]** predict so if we give it this input the
> 预测，所以如果我们给它这个输入，对应于

**[00:00:12]** predict so if we give it this input the vocabulary token ID corresponding to
> 预测，所以如果我们给它这个输入，这个词汇表token ID

**[00:00:12]** vocabulary token ID corresponding to
> 词汇表token ID对应于

**[00:00:12]** vocabulary token ID corresponding to this one will have the largest score in
> 词汇表token ID对应于这个的将在logits向量中拥有最大的分数

**[00:00:12]** this one will have the largest score in
> 这个的将在logits向量中拥有最大的分数

**[00:00:12]** this one will have the largest score in this logits vector and then we extract
> 这个的将在logits向量中拥有最大的分数，然后我们提取

**[00:00:12]** this logits vector and then we extract
> 这个logits向量，然后我们提取

**[00:00:12]** this logits vector and then we extract the largest score um get the token ID
> 这个logits向量，然后我们提取最大的分数，嗯获取token ID

**[00:00:12]** the largest score um get the token ID
> 最大的分数，嗯获取token ID

**[00:00:12]** the largest score um get the token ID and then we turn it back with our decode
> 最大的分数，嗯获取token ID，然后我们用decode方法将其转换回来

**[00:00:12]** and then we turn it back with our decode
> 然后我们用decode方法将其转换回来

**[00:00:12]** and then we turn it back with our decode method we replace encode with decode we
> 然后我们用decode方法将其转换回来，我们用decode替换encode

**[00:00:12]** method we replace encode with decode we
> 方法，我们用decode替换encode

**[00:00:12]** method we replace encode with decode we replace that um into a word so the token
> 方法，我们用decode替换encode，我们将其嗯转换成一个词，所以token

**[00:00:12]** replace that um into a word so the token
> 替换那个嗯转换成一个词，所以token

**[00:00:12]** replace that um into a word so the token ID to a word but this is is already um
> 替换那个嗯转换成一个词，所以token ID到词，但这已经嗯

**[00:00:13]** ID to a word but this is is already um
> ID到词，但这已经嗯

**[00:00:13]** ID to a word but this is is already um maybe too much detail because this will
> ID到词，但这已经嗯可能过于详细了，因为这将在第5章中涵盖

**[00:00:13]** maybe too much detail because this will
> 可能过于详细了，因为这将在第5章中涵盖

**[00:00:13]** maybe too much detail because this will be covered in chapter 5 for now just
> 可能过于详细了，因为这将在第5章中涵盖，现在只需

**[00:00:13]** be covered in chapter 5 for now just
> 在第5章中涵盖，现在只需

**[00:00:13]** be covered in chapter 5 for now just consider we are generating some values
> 在第5章中涵盖，现在只需考虑我们在这里生成一些值

**[00:00:13]** consider we are generating some values
> 考虑我们在这里生成一些值

**[00:00:13]** consider we are generating some values here so we are generating these
> 考虑我们在这里生成一些值，所以我们生成这些

**[00:00:13]** here so we are generating these
> 这里，所以我们生成这些

**[00:00:13]** here so we are generating these
> 这里，所以我们生成这些

**[00:00:13]** dimensional uh vectors for each of the
> 维度的嗯向量，针对每个

**[00:00:13]** dimensional uh vectors for each of the
> 维度的嗯向量，针对每个

**[00:00:13]** dimensional uh vectors for each of the token IDs and we will take care of them
> 维度的嗯向量，针对每个token ID，我们会在之后处理它们

**[00:00:13]** token IDs and we will take care of them
> token ID，我们会在之后处理它们

**[00:00:13]** token IDs and we will take care of them later once we have trained the L&amp;M but
> token ID，我们会在之后处理它们，一旦我们训练了L&M，但是

**[00:00:13]** later once we have trained the L&amp;M but
> 之后处理它们，一旦我们训练了L&M，但是

**[00:00:13]** later once we have trained the L&amp;M but yeah I hope so far you have a good
> 之后处理它们，一旦我们训练了L&M，但是嗯，我希望到目前为止你有一个好的

**[00:00:13]** yeah I hope so far you have a good
> 嗯，我希望到目前为止你有一个好的

**[00:00:13]** yeah I hope so far you have a good understanding at least how the
> 嗯，我希望到目前为止你有一个好的理解，至少关于

**[00:00:13]** understanding at least how the
> 理解，至少关于

**[00:00:13]** understanding at least how the architecture looks like so that we have
> 理解，至少关于架构的样子，这样我们就有了

**[00:00:13]** architecture looks like so that we have
> 架构的样子，这样我们就有了

**[00:00:13]** architecture looks like so that we have this GPT model where it receives
> 架构的样子，这样我们就有了这个GPT模型，它接收

**[00:00:13]** this GPT model where it receives
> 这个GPT模型，它接收

**[00:00:13]** this GPT model where it receives tokenized text then it goes through
> 这个GPT模型，它接收tokenized文本，然后经过

**[00:00:13]** tokenized text then it goes through
> tokenized文本，然后经过

**[00:00:13]** tokenized text then it goes through embedding layers we have these
> tokenized文本，然后经过embedding层，我们有这些

**[00:00:13]** embedding layers we have these
> embedding层，我们有这些

**[00:00:13]** embedding layers we have these Transformer blocks and then we have this
> embedding层，我们有这些Transformer块，然后我们有这个

**[00:00:13]** Transformer blocks and then we have this
> Transformer块，然后我们有这个

**[00:00:13]** Transformer blocks and then we have this output layer that produces these 50, 257
> Transformer块，然后我们有这个输出层，它产生这些50,257

**[00:00:13]** output layer that produces these 50, 257
> 输出层，它产生这些50,257

**[00:00:13]** output layer that produces these 50, 257 dimensional vectors which are for now
> 输出层，它产生这些50,257维的向量，目前这些

**[00:00:13]** dimensional vectors which are for now
> 维的向量，目前这些

**[00:00:13]** dimensional vectors which are for now just placeholder values and later on we
> 维的向量，目前这些只是占位值，之后我们

**[00:00:13]** just placeholder values and later on we
> 只是占位值，之后我们

**[00:00:13]** just placeholder values and later on we will fill in the details and yeah in the
> 只是占位值，之后我们会填充细节，嗯，在

**[00:00:13]** will fill in the details and yeah in the
> 会填充细节，嗯，在

**[00:00:13]** will fill in the details and yeah in the upcoming videos we will also fill in the
> 会填充细节，嗯，在接下来的视频中，我们也会填充

**[00:00:13]** upcoming videos we will also fill in the
> 接下来的视频中，我们也会填充

**[00:00:13]** upcoming videos we will also fill in the details uh in terms of what goes exactly
> 接下来的视频中，我们也会填充细节，嗯，关于这个Transformer块中

**[00:00:13]** details uh in terms of what goes exactly
> 细节，嗯，关于这个Transformer块中

**[00:00:13]** details uh in terms of what goes exactly on in this Transformer
> 细节，嗯，关于这个Transformer块中具体发生了什么

**[00:00:13]** block we are now talking about normal
> 块，我们现在讨论的是normal

**[00:00:13]** block we are now talking about normal
> block we are now talking about normal

**[00:00:13]** block we are now talking about normal izing activations with layer
> 现在我们讨论的是使用层归一化来规范化激活值

**[00:00:14]** izing activations with layer
> 使用层来规范化激活值

**[00:00:14]** izing activations with layer normalizations so in the grand scheme of
> 使用层归一化来规范化激活值，所以在整体架构中

**[00:00:14]** normalizations so in the grand scheme of
> 归一化，所以在整体架构中

**[00:00:14]** normalizations so in the grand scheme of things what we've done before is we
> 归一化，所以在整体架构中，我们之前所做的是

**[00:00:14]** things what we've done before is we
> 我们之前所做的是

**[00:00:14]** things what we've done before is we introduced this GPD model architecture
> 我们之前所做的是引入了这个GPD模型架构

**[00:00:14]** introduced this GPD model architecture
> 引入了这个GPD模型架构

**[00:00:14]** introduced this GPD model architecture where we had several ingredients one was
> 引入了这个GPD模型架构，其中包含几个组成部分，一个是

**[00:00:14]** where we had several ingredients one was
> 其中包含几个组成部分，一个是

**[00:00:14]** where we had several ingredients one was the embedding layers that we introduced
> 其中包含几个组成部分，一个是我们在第2章介绍的embedding层

**[00:00:14]** the embedding layers that we introduced
> 我们在第2章介绍的embedding层

**[00:00:14]** the embedding layers that we introduced in Chapter 2 we had the masked
> 我们在第2章介绍的embedding层，我们还有掩码

**[00:00:14]** in Chapter 2 we had the masked
> 在第2章中，我们还有掩码

**[00:00:14]** in Chapter 2 we had the masked multi-head attention module which we
> 在第2章中，我们还有掩码多头注意力模块，这个模块我们

**[00:00:14]** multi-head attention module which we
> 多头注意力模块，这个模块我们

**[00:00:14]** multi-head attention module which we talked about in chapter 3 and now there
> 多头注意力模块，这个模块我们在第3章讨论过，现在还有

**[00:00:14]** talked about in chapter 3 and now there
> 在第3章讨论过，现在还有

**[00:00:14]** talked about in chapter 3 and now there are several other things um so we had
> 在第3章讨论过，现在还有其他几个东西，嗯，所以我们有

**[00:00:14]** are several other things um so we had
> 还有其他几个东西，嗯，所以我们有

**[00:00:14]** are several other things um so we had this dummy gbt placeholder for now where
> 还有其他几个东西，嗯，所以我们暂时有这个虚拟的GBT占位符

**[00:00:14]** this dummy gbt placeholder for now where
> 这个虚拟的GBT占位符暂时

**[00:00:14]** this dummy gbt placeholder for now where we said we are going to fill in the
> 这个虚拟的GBT占位符暂时，我们说过将在接下来的章节中

**[00:00:14]** we said we are going to fill in the
> 我们说过将在接下来的章节中

**[00:00:14]** we said we are going to fill in the blanks in the upcoming sections and that
> 我们说过将在接下来的章节中填补空白，而这

**[00:00:14]** blanks in the upcoming sections and that
> 填补空白，而这

**[00:00:14]** blanks in the upcoming sections and that is what we are going to do now for one
> 填补空白，而这正是我们现在要做的，针对其中一个

**[00:00:14]** is what we are going to do now for one
> 正是我们现在要做的，针对其中一个

**[00:00:14]** is what we are going to do now for one of these ingredients and that is the
> 正是我们现在要做的，针对其中一个组成部分，那就是

**[00:00:14]** of these ingredients and that is the
> 其中一个组成部分，那就是

**[00:00:14]** of these ingredients and that is the layer Norm so layer Norm is short for
> 其中一个组成部分，那就是layer Norm，所以layer Norm是

**[00:00:14]** layer Norm so layer Norm is short for
> layer Norm，所以layer Norm是

**[00:00:14]** layer Norm so layer Norm is short for layer normalization it's a concept that
> layer Norm，所以layer Norm是层归一化的简称，这是一个概念

**[00:00:14]** layer normalization it's a concept that
> 层归一化，这是一个概念

**[00:00:14]** layer normalization it's a concept that was introduced back in 2016 it's a
> 层归一化，这是一个在2016年引入的概念，它是一个

**[00:00:14]** was introduced back in 2016 it's a
> 在2016年引入，它是一个

**[00:00:14]** was introduced back in 2016 it's a classic deep learning concept and it's
> 在2016年引入，它是一个经典的深度学习概念，它

**[00:00:14]** classic deep learning concept and it's
> 经典的深度学习概念，它

**[00:00:14]** classic deep learning concept and it's about yeah normalizing the outputs of a
> 经典的深度学习概念，它关乎于，是的，规范化一个给定层的输出

**[00:00:14]** about yeah normalizing the outputs of a
> 关乎于，是的，规范化一个给定层的输出

**[00:00:14]** about yeah normalizing the outputs of a given layer that go into the next layer
> 关乎于，是的，规范化一个给定层的输出，这些输出会进入下一层

**[00:00:14]** given layer that go into the next layer
> 给定层的输出，这些输出会进入下一层

**[00:00:14]** given layer that go into the next layer so that it has a nice property for
> 给定层的输出，这些输出会进入下一层，以便它具有一个良好的性质，用于

**[00:00:14]** so that it has a nice property for
> 以便它具有一个良好的性质，用于

**[00:00:14]** so that it has a nice property for optimization and yeah to make this a bit
> 以便它具有一个良好的性质，用于优化，并且，是的，为了让它更

**[00:00:14]** optimization and yeah to make this a bit
> 优化，并且，是的，为了让它更

**[00:00:14]** optimization and yeah to make this a bit more clear I think let's take a look at
> 优化，并且，是的，为了让它更清楚，我想让我们看看

**[00:00:14]** more clear I think let's take a look at
> 清楚，我想让我们看看

**[00:00:14]** more clear I think let's take a look at the code examples I've prepared so I
> 清楚，我想让我们看看我准备的代码示例，所以我

**[00:00:15]** the code examples I've prepared so I
> 我准备的代码示例，所以我

**[00:00:15]** the code examples I've prepared so I think this will show you exactly what's
> 我准备的代码示例，我想这将准确展示那里发生了什么

**[00:00:15]** think this will show you exactly what's
> 我想这将准确展示那里发生了什么

**[00:00:15]** think this will show you exactly what's going on there and so for the code
> 我想这将准确展示那里发生了什么，所以对于代码

**[00:00:15]** going on there and so for the code
> 那里发生了什么，所以对于代码

**[00:00:15]** going on there and so for the code examples I'm trying to keep it a bit
> 那里发生了什么，所以对于代码示例，我试图让它更

**[00:00:15]** examples I'm trying to keep it a bit
> 示例，我试图让它更

**[00:00:15]** examples I'm trying to keep it a bit more um conceptual simple so we will be
> 示例，我试图让它更概念化、更简单，所以我们将在

**[00:00:15]** more um conceptual simple so we will be
> 概念化、更简单，所以我们将在

**[00:00:15]** more um conceptual simple so we will be using layer normalization in the context
> 概念化、更简单，所以我们将在真实LLM的上下文中使用层归一化，只是为了简单起见

**[00:00:15]** using layer normalization in the context
> 使用层归一化，在真实LLM的上下文中，只是为了简单起见

**[00:00:15]** using layer normalization in the context of the real llm just for Simplicity
> 使用层归一化，在真实LLM的上下文中，只是为了简单起见，不过在这个视频中，我想使用一个

**[00:00:15]** of the real llm just for Simplicity
> 不过在这个视频中，我想使用一个

**[00:00:15]** of the real llm just for Simplicity though in this video I want to use a
> 不过在这个视频中，我想使用一个更小的例子，所以为此，我只是

**[00:00:15]** though in this video I want to use a
> 更小的例子，所以为此，我只是

**[00:00:15]** though in this video I want to use a smaller example so for that I'm just
> 更小的例子，所以为此，我只是设置一个随机种子，然后我

**[00:00:15]** smaller example so for that I'm just
> 设置一个随机种子，然后我

**[00:00:15]** smaller example so for that I'm just setting a random seat and then I'm
> 设置一个随机种子，然后我在这里做一个更小的例子，让我们使用

**[00:00:15]** setting a random seat and then I'm
> 在这里做一个更小的例子，让我们使用

**[00:00:15]** setting a random seat and then I'm making a smaller example here let's use
> 在这里做一个更小的例子，让我们使用torch.randn，它将生成一个

**[00:00:15]** making a smaller example here let's use
> torch.randn，它将生成一个

**[00:00:15]** making a smaller example here let's use torch. randn which will generate a
> torch.randn，它将生成一个2x5的随机样本，这里只是为了

**[00:00:15]** torch. randn which will generate a
> 2x5的随机样本，这里只是为了

**[00:00:15]** torch. randn which will generate a random sample um of 2x5 here just to
> 2x5的随机样本，这里只是为了保持小规模，所以只看这个，它

**[00:00:15]** random sample um of 2x5 here just to
> 保持小规模，所以只看这个，它

**[00:00:15]** random sample um of 2x5 here just to keep it small so just looking at this it
> 保持小规模，所以只看这个，它将是两个样本，这里每个

**[00:00:15]** keep it small so just looking at this it
> 将是两个样本，这里每个

**[00:00:15]** keep it small so just looking at this it will be two um samples here and each
> 将是两个样本，这里每个样本有五个维度，所以我们有

**[00:00:15]** will be two um samples here and each
> 样本有五个维度，所以我们有

**[00:00:15]** will be two um samples here and each sample has uh five Dimensions so we have
> will be two um samples here and each sample has uh five Dimensions so we have

**[00:00:15]** sample has uh five Dimensions so we have
> sample has uh five Dimensions so we have

**[00:00:15]** sample has uh five Dimensions so we have a five dimensional input um example
> 样本有五个维度，所以我们有一个五维输入示例

**[00:00:15]** a five dimensional input um example
> 一个五维输入示例

**[00:00:15]** a five dimensional input um example similar to what we have done before in
> 一个五维输入示例，类似于我们之前所做的

**[00:00:15]** similar to what we have done before in
> 类似于我们之前所做的

**[00:00:15]** similar to what we have done before in previous chapters for example in Chapter
> 类似于我们之前所做的，例如在之前的章节中

**[00:00:15]** previous chapters for example in Chapter
> 之前的章节中，例如在

**[00:00:15]** previous chapters for example in Chapter 2 we talked about a token that was
> 之前的章节中，例如在第2章，我们讨论了一个token

**[00:00:15]** 2 we talked about a token that was
> 第2章，我们讨论了一个token

**[00:00:15]** 2 we talked about a token that was embedded in a three-dimensional space
> 第2章，我们讨论了一个嵌入在三维空间中的token

**[00:00:15]** embedded in a three-dimensional space
> 嵌入在三维空间中

**[00:00:15]** embedded in a three-dimensional space similarly here just think of it as a
> 嵌入在三维空间中，类似地，这里只需将其视为

**[00:00:15]** similarly here just think of it as a
> 类似地，这里只需将其视为

**[00:00:15]** similarly here just think of it as a token embedded in a five dimensional
> 类似地，这里只需将其视为一个嵌入在五维空间中的token

**[00:00:15]** token embedded in a five dimensional
> 嵌入在五维空间中的token

**[00:00:15]** token embedded in a five dimensional space just to have some more numbers
> 嵌入在五维空间中的token，只是为了有更多数字

**[00:00:16]** space just to have some more numbers
> 空间，只是为了有更多数字

**[00:00:16]** space just to have some more numbers here um again like in reality we talked
> 空间，只是为了有更多数字，同样，实际上我们之前讨论过

**[00:00:16]** here um again like in reality we talked
> 实际上我们之前讨论过

**[00:00:16]** here um again like in reality we talked about this earlier the embedding
> 实际上我们之前讨论过，embedding

**[00:00:16]** about this earlier the embedding
> 关于这个，embedding

**[00:00:16]** about this earlier the embedding Dimensions might be bigger in real llm
> 关于这个，embedding维度在真实的LLM中可能更大

**[00:00:16]** Dimensions might be bigger in real llm
> 维度在真实的LLM中可能更大

**[00:00:16]** Dimensions might be bigger in real llm cases so here 768 here for Simplicity
> 维度在真实的LLM中可能更大，所以这里为了简单起见用768

**[00:00:16]** cases so here 768 here for Simplicity
> 情况，所以这里为了简单起见用768

**[00:00:16]** cases so here 768 here for Simplicity just let's work with a smaller example
> 情况，所以这里为了简单起见，我们用一个更小的例子

**[00:00:16]** just let's work with a smaller example
> 我们用一个更小的例子

**[00:00:16]** just let's work with a smaller example um so that we can fit it on a figure
> 我们用一个更小的例子，这样我们可以把它放在一张图上

**[00:00:16]** um so that we can fit it on a figure
> 这样我们可以把它放在一张图上

**[00:00:16]** um so that we can fit it on a figure actually I should have a nice figure
> 这样我们可以把它放在一张图上，实际上我应该有一张漂亮的图

**[00:00:16]** actually I should have a nice figure
> 实际上我应该有一张漂亮的图

**[00:00:16]** actually I should have a nice figure here somewhere um so what's happening
> 实际上我应该有一张漂亮的图在这里某处，所以本质上发生的是

**[00:00:16]** here somewhere um so what's happening
> 在这里某处，所以本质上发生的是

**[00:00:16]** here somewhere um so what's happening essentially is that we have these five
> 在这里某处，所以本质上发生的是，对于给定的示例，我们有这五个

**[00:00:16]** essentially is that we have these five
> 对于给定的示例，我们有这五个

**[00:00:16]** essentially is that we have these five input values now for a given example and
> 对于给定的示例，我们有这五个输入值，现在我们要做的是

**[00:00:16]** input values now for a given example and
> 输入值，现在我们要做的是

**[00:00:16]** input values now for a given example and now what we're trying trying to do is we
> 输入值，现在我们要做的是，我们试图将它们通过

**[00:00:16]** now what we're trying trying to do is we
> 我们试图将它们通过

**[00:00:16]** now what we're trying trying to do is we are trying to um put them through a
> 我们试图将它们通过一个小型神经网络，例如这个

**[00:00:16]** are trying to um put them through a
> 一个小型神经网络，例如这个

**[00:00:16]** are trying to um put them through a small new network for example this one
> 一个小型神经网络，例如这个，然后我们得到输出值

**[00:00:16]** small new network for example this one
> 然后我们得到输出值

**[00:00:16]** small new network for example this one here and then we get output values and
> 然后我们得到输出值，我们希望确保这些输出

**[00:00:16]** here and then we get output values and
> 我们希望确保这些输出

**[00:00:16]** here and then we get output values and we want to make sure that these output
> 我们希望确保这些输出值在某个值范围内

**[00:00:16]** we want to make sure that these output
> 值在某个值范围内

**[00:00:16]** we want to make sure that these output values are in a certain value range so
> 值在某个值范围内，所以在这种情况下，如果我们只应用神经

**[00:00:16]** values are in a certain value range so
> 所以在这种情况下，如果我们只应用神经

**[00:00:16]** values are in a certain value range so in this case if we just apply the neur
> 所以在这种情况下，如果我们只应用神经网络层，我们可能得到的是

**[00:00:16]** in this case if we just apply the neur
> 网络层，我们可能得到的是

**[00:00:16]** in this case if we just apply the neur network layer what we might get is a
> 网络层，我们可能得到的是这些值的均值和平均值为0.13

**[00:00:16]** network layer what we might get is a
> 这些值的均值和平均值为0.13

**[00:00:16]** network layer what we might get is a mean and average of .13 across these
> 这些值的均值和平均值为0.13，方差为39

**[00:00:16]** mean and average of .13 across these
> 方差为39

**[00:00:16]** mean and average of .13 across these values and a variance of 39 which is
> 方差为39，这实际上还不错，所以这可能

**[00:00:16]** values and a variance of 39 which is
> 这实际上还不错，所以这可能

**[00:00:16]** values and a variance of 39 which is actually not too bad so this might
> 这实际上还不错，所以这可能实际上没问题，但如果你

**[00:00:16]** actually not too bad so this might
> 实际上没问题，但如果你

**[00:00:16]** actually not too bad so this might actually be just fine but if you are
> 实际上没问题，但如果你熟悉经典的深度神经网络

**[00:00:16]** actually be just fine but if you are
> 熟悉经典的深度神经网络

**[00:00:16]** actually be just fine but if you are familiar with classic deep new network
> 熟悉经典的深度神经网络，可能会有问题，比如

**[00:00:16]** familiar with classic deep new network
> 可能会有问题，比如

**[00:00:16]** familiar with classic deep new network um there might be problems like
> 可能会有问题，比如梯度消失问题和

**[00:00:17]** um there might be problems like
> 梯度消失问题和

**[00:00:17]** um there might be problems like Vanishing gradient problems and um
> 梯度消失问题和梯度爆炸问题

**[00:00:17]** Vanishing gradient problems and um
> 梯度爆炸问题

**[00:00:17]** Vanishing gradient problems and um exploding gradient um problems and
> 梯度爆炸问题，将输入归一化到给定层

**[00:00:17]** exploding gradient um problems and
> 将输入归一化到给定层

**[00:00:17]** exploding gradient um problems and normalizing the inputs to a given layer
> 将输入归一化到给定层，然后本质上也将输入归一化到

**[00:00:17]** normalizing the inputs to a given layer
> 然后本质上也将输入归一化到

**[00:00:17]** normalizing the inputs to a given layer and then essentially also the inputs to
> 然后本质上也将输入归一化到下一层，有助于稳定

**[00:00:17]** and then essentially also the inputs to
> 下一层，有助于稳定

**[00:00:17]** and then essentially also the inputs to the next layer helps with stabilizing
> 下一层，有助于稳定训练，所以给定层的输出

**[00:00:17]** the next layer helps with stabilizing
> 训练，所以给定层的输出

**[00:00:17]** the next layer helps with stabilizing the training so the outputs of a given
> 训练，所以给定层的输出，比如那些会进入

**[00:00:17]** the training so the outputs of a given
> 比如那些会进入

**[00:00:17]** the training so the outputs of a given layer like those they would go into the
> 比如那些会进入深度神经网络下一层的输出，因为

**[00:00:17]** layer like those they would go into the
> 下一层的输出，因为

**[00:00:17]** layer like those they would go into the next layer in a deep new network because
> 下一层的输出，因为在深度神经网络中

**[00:00:17]** next layer in a deep new network because
> next layer in a deep new network because

**[00:00:17]** next layer in a deep new network because we have many layers in a network and if
> 深度新网络中的下一层，因为网络中有很多层，如果

**[00:00:17]** we have many layers in a network and if
> 网络中有很多层，如果

**[00:00:17]** we have many layers in a network and if we have let's say very small values or
> 网络中有很多层，如果我们假设有非常小的值或

**[00:00:17]** we have let's say very small values or
> 我们假设有非常小的值或

**[00:00:17]** we have let's say very small values or very large values that might be not
> 我们假设有非常小的值或非常大的值，这可能不是

**[00:00:17]** very large values that might be not
> 非常大的值，这可能不是

**[00:00:17]** very large values that might be not optimal it might lead to unstable
> 非常大的值，这可能不是最优的，它可能导致不稳定的

**[00:00:17]** optimal it might lead to unstable
> 最优的，它可能导致不稳定的

**[00:00:17]** optimal it might lead to unstable training so what we are trying to
> 最优的，它可能导致不稳定的训练，所以我们试图

**[00:00:17]** training so what we are trying to
> 训练，所以我们试图

**[00:00:17]** training so what we are trying to accomplish here is we're trying to add a
> 训练，所以我们试图在这里实现的是添加一个

**[00:00:17]** accomplish here is we're trying to add a
> 在这里实现的是添加一个

**[00:00:17]** accomplish here is we're trying to add a certain layer called layer normalization
> 在这里实现的是添加一个称为层归一化的特定层

**[00:00:17]** certain layer called layer normalization
> 称为层归一化的特定层

**[00:00:17]** certain layer called layer normalization to normalize these outputs that go into
> 称为层归一化的特定层，用于归一化这些进入

**[00:00:17]** to normalize these outputs that go into
> 用于归一化这些进入

**[00:00:17]** to normalize these outputs that go into the next layer such that they have mean
> 用于归一化这些进入下一层的输出，使它们具有均值

**[00:00:17]** the next layer such that they have mean
> 下一层的输出，使它们具有均值

**[00:00:17]** the next layer such that they have mean zero and variance one that is our goal
> 下一层的输出，使它们具有均值为零和方差为一，这是我们的目标

**[00:00:17]** zero and variance one that is our goal
> 均值为零和方差为一，这是我们的目标

**[00:00:17]** zero and variance one that is our goal here so let's um consider our inputs
> 均值为零和方差为一，这是我们的目标，所以让我们考虑我们的输入

**[00:00:17]** here so let's um consider our inputs
> 这里，所以让我们考虑我们的输入

**[00:00:17]** here so let's um consider our inputs here and then let's also Implement a
> 这里，所以让我们考虑我们的输入，然后也实现一个

**[00:00:17]** here and then let's also Implement a
> 这里，然后也实现一个

**[00:00:17]** here and then let's also Implement a small layer like I've showed you before
> 这里，然后也实现一个我之前展示过的小层

**[00:00:17]** small layer like I've showed you before
> 我之前展示过的小层

**[00:00:17]** small layer like I've showed you before in the figure so let's use sequential
> 我之前展示过的小层，如图中所示，所以让我们使用sequential

**[00:00:17]** in the figure so let's use sequential
> 图中，所以让我们使用sequential

**[00:00:17]** in the figure so let's use sequential here and then let's do linear which um
> 图中，所以让我们在这里使用sequential，然后做linear，它

**[00:00:17]** here and then let's do linear which um
> 这里，然后做linear，它

**[00:00:17]** here and then let's do linear which um creates the new network layer and we
> 这里，然后做linear，它创建了新的网络层，我们

**[00:00:18]** creates the new network layer and we
> 创建了新的网络层，我们

**[00:00:18]** creates the new network layer and we have 1 2 3 4 5 we have five Dimensions
> 创建了新的网络层，我们有1 2 3 4 5，我们有五个维度

**[00:00:18]** have 1 2 3 4 5 we have five Dimensions
> 有1 2 3 4 5，我们有五个维度

**[00:00:18]** have 1 2 3 4 5 we have five Dimensions so the input is five and the output
> 有1 2 3 4 5，我们有五个维度，所以输入是五，输出

**[00:00:18]** so the input is five and the output
> 所以输入是五，输出

**[00:00:18]** so the input is five and the output let's use six to match what we have
> 所以输入是五，输出让我们用六来匹配我们展示的

**[00:00:18]** let's use six to match what we have
> 让我们用六来匹配我们展示的

**[00:00:18]** let's use six to match what we have shown here so we have this layer here
> 让我们用六来匹配我们这里展示的内容，所以我们有这个层

**[00:00:18]** shown here so we have this layer here
> 这里展示的内容，所以我们有这个层

**[00:00:18]** shown here so we have this layer here and then let's actually apply this layer
> 这里展示的内容，所以我们有这个层，然后让我们实际应用这个层

**[00:00:18]** and then let's actually apply this layer
> 然后让我们实际应用这个层

**[00:00:18]** and then let's actually apply this layer to the batch
> 然后让我们实际应用这个层到batch

**[00:00:18]** to the batch
> 到batch

**[00:00:18]** to the batch example maybe one more thing we could do
> 到batch示例，也许我们还可以做一件事

**[00:00:18]** example maybe one more thing we could do
> 示例，也许我们还可以做一件事

**[00:00:18]** example maybe one more thing we could do technically oops um is adding a
> 示例，也许我们还可以做一件事，技术上，哦，是添加一个

**[00:00:18]** technically oops um is adding a
> 技术上，哦，是添加一个

**[00:00:18]** technically oops um is adding a nonlinearity so nonlinearity could be
> 技术上，哦，是添加一个非线性，所以非线性可以是

**[00:00:18]** nonlinearity so nonlinearity could be
> 非线性，所以非线性可以是

**[00:00:18]** nonlinearity so nonlinearity could be reu for example it's a activation
> 非线性，所以非线性可以是例如ReLU，它是一个激活

**[00:00:18]** reu for example it's a activation
> ReLU，例如它是一个激活

**[00:00:18]** reu for example it's a activation nonlinear activation function and yeah
> ReLU，例如它是一个激活非线性激活函数，是的

**[00:00:18]** nonlinear activation function and yeah
> 非线性激活函数，是的

**[00:00:18]** nonlinear activation function and yeah that is if you have worked with deep
> 非线性激活函数，是的，如果你之前处理过深度

**[00:00:18]** that is if you have worked with deep
> 如果你之前处理过深度

**[00:00:18]** that is if you have worked with deep networks before usually we pair a linear
> 如果你之前处理过深度网络，通常我们会将线性

**[00:00:18]** networks before usually we pair a linear
> 网络，通常我们会将线性

**[00:00:18]** networks before usually we pair a linear layer with a nonlinear activation
> 网络，通常我们会将线性层与非线性激活

**[00:00:18]** layer with a nonlinear activation
> 层与非线性激活

**[00:00:18]** layer with a nonlinear activation function because otherwise if we only
> 层与非线性激活函数配对，因为否则如果我们只有

**[00:00:18]** function because otherwise if we only
> 函数配对，因为否则如果我们只有

**[00:00:18]** function because otherwise if we only have linear layers in a new network the
> 函数配对，因为否则如果我们在新网络中只有线性层，

**[00:00:18]** have linear layers in a new network the
> 在新网络中只有线性层，

**[00:00:18]** have linear layers in a new network the sum or the combination of linear layers
> 在新网络中只有线性层，线性层的和或组合

**[00:00:18]** sum or the combination of linear layers
> 线性层的和或组合

**[00:00:18]** sum or the combination of linear layers would be a linear function and that
> 线性层的和或组合将是一个线性函数，那

**[00:00:18]** would be a linear function and that
> 将是一个线性函数，那

**[00:00:18]** would be a linear function and that would not be a very useful neuron
> 将是一个线性函数，那将不是一个非常有用的神经元

**[00:00:18]** would not be a very useful neuron
> 将不是一个非常有用的神经元

**[00:00:18]** would not be a very useful neuron Network so usually we would add
> 将不是一个非常有用的神经元网络，所以通常我们会添加

**[00:00:18]** Network so usually we would add
> 网络，所以通常我们会添加

**[00:00:18]** Network so usually we would add nonlinearities so that the network can
> 网络，所以通常我们会添加非线性，这样网络可以

**[00:00:18]** nonlinearities so that the network can
> 非线性，这样网络可以

**[00:00:18]** nonlinearities so that the network can learn more complex things so let's do
> 非线性，这样网络可以学习更复杂的东西，所以让我们做

**[00:00:19]** learn more complex things so let's do
> 学习更复杂的东西，所以让我们做

**[00:00:19]** learn more complex things so let's do this and look at the output here so what
> 学习更复杂的东西，所以让我们这样做，并查看这里的输出，所以

**[00:00:19]** this and look at the output here so what
> 这样做，并查看这里的输出，所以

**[00:00:19]** this and look at the output here so what we can see here is yeah we get some
> 这样做，并查看这里的输出，所以我们在这里可以看到，是的，我们得到一些

**[00:00:19]** we can see here is yeah we get some
> 我们在这里可以看到，是的，我们得到一些

**[00:00:19]** we can see here is yeah we get some output values um two examples so the
> 这里我们可以看到，是的，我们得到了一些输出值，两个例子，所以

**[00:00:19]** output values um two examples so the
> 输出值，两个例子，所以

**[00:00:19]** output values um two examples so the first one corresponds to this example
> 输出值，两个例子，所以第一个对应这个例子

**[00:00:19]** first one corresponds to this example
> 第一个对应这个例子

**[00:00:19]** first one corresponds to this example and this one corresponds to this example
> 第一个对应这个例子，而这个对应这个例子

**[00:00:19]** and this one corresponds to this example
> 而这个对应这个例子

**[00:00:19]** and this one corresponds to this example transformed now through this mini new
> 而这个对应这个例子，现在通过这个小型新网络转换后

**[00:00:19]** transformed now through this mini new
> 现在通过这个小型新网络转换后

**[00:00:19]** transformed now through this mini new network so the first row here um is
> 现在通过这个小型新网络转换后，所以这里的第一行

**[00:00:19]** network so the first row here um is
> 网络，所以这里的第一行

**[00:00:19]** network so the first row here um is really what we are seeing if we focus on
> 网络，所以这里的第一行实际上是我们看到的，如果我们专注于

**[00:00:19]** really what we are seeing if we focus on
> 实际上是我们看到的，如果我们专注于

**[00:00:19]** really what we are seeing if we focus on one example what what we are seeing here
> 实际上是我们看到的，如果我们专注于一个例子，我们在这里看到的是

**[00:00:19]** one example what what we are seeing here
> 一个例子，我们在这里看到的是

**[00:00:19]** one example what what we are seeing here and so before we start with any layer
> 一个例子，我们在这里看到的是，所以在开始任何层之前

**[00:00:19]** and so before we start with any layer
> 所以在开始任何层之前

**[00:00:19]** and so before we start with any layer normalization or normalization in
> 所以在开始任何层之前，进行层归一化或一般归一化

**[00:00:19]** normalization or normalization in
> 归一化或一般归一化

**[00:00:19]** normalization or normalization in general let's just take a look at the
> 归一化或一般归一化，让我们先看看这个的

**[00:00:19]** general let's just take a look at the
> 一般，让我们先看看这个的

**[00:00:19]** general let's just take a look at the mean and the variance of this so we
> 一般，让我们先看看这个的均值和方差，所以我们

**[00:00:19]** mean and the variance of this so we
> 均值和方差，所以我们

**[00:00:19]** mean and the variance of this so we could um for example call mean here and
> 均值和方差，所以我们例如可以在这里调用mean

**[00:00:19]** could um for example call mean here and
> 可以例如在这里调用mean

**[00:00:19]** could um for example call mean here and yeah if I call just mean like that it
> 可以例如在这里调用mean，是的，如果我像那样直接调用mean

**[00:00:19]** yeah if I call just mean like that it
> 是的，如果我像那样直接调用mean

**[00:00:19]** yeah if I call just mean like that it we'll see what happens is not what we
> 是的，如果我像那样直接调用mean，我们会看到结果不是我们

**[00:00:19]** we'll see what happens is not what we
> 我们会看到结果不是我们

**[00:00:19]** we'll see what happens is not what we expect or it depends on how what you
> 我们会看到结果不是我们预期的，或者这取决于你如何

**[00:00:19]** expect or it depends on how what you
> 预期的，或者这取决于你如何

**[00:00:19]** expect or it depends on how what you expect but here it's not what we want so
> 预期的，或者这取决于你如何预期，但这里不是我们想要的，所以

**[00:00:19]** expect but here it's not what we want so
> 预期，但这里不是我们想要的，所以

**[00:00:19]** expect but here it's not what we want so here we get one value and this is the
> 预期，但这里不是我们想要的，所以我们得到一个值，这是

**[00:00:19]** here we get one value and this is the
> 所以我们得到一个值，这是

**[00:00:19]** here we get one value and this is the average or the mean of this whole thing
> 所以我们得到一个值，这是整个东西的平均值或均值

**[00:00:19]** average or the mean of this whole thing
> 整个东西的平均值或均值

**[00:00:19]** average or the mean of this whole thing but we want actually the um mean across
> 整个东西的平均值或均值，但我们实际上想要的是跨

**[00:00:19]** but we want actually the um mean across
> 但我们实际上想要的是跨

**[00:00:19]** but we want actually the um mean across this first example so there are multiple
> 但我们实际上想要的是跨第一个例子的均值，所以有多种

**[00:00:19]** this first example so there are multiple
> 第一个例子，所以有多种

**[00:00:19]** this first example so there are multiple ways we can compute the mean we can
> 第一个例子，所以有多种方法可以计算均值，我们可以

**[00:00:19]** ways we can compute the mean we can
> 方法可以计算均值，我们可以

**[00:00:19]** ways we can compute the mean we can compute this mean of this whole thing
> 方法可以计算均值，我们可以计算整个东西的均值

**[00:00:20]** compute this mean of this whole thing
> 计算整个东西的均值

**[00:00:20]** compute this mean of this whole thing but this would be mixing multiple
> 计算整个东西的均值，但这会混合多个

**[00:00:20]** but this would be mixing multiple
> 但这会混合多个

**[00:00:20]** but this would be mixing multiple examples and this is not what we want um
> 但这会混合多个例子，这不是我们想要的

**[00:00:20]** examples and this is not what we want um
> 例子，这不是我们想要的

**[00:00:20]** examples and this is not what we want um the other one we could do is we can do
> 例子，这不是我们想要的，另一种我们可以做的是

**[00:00:20]** the other one we could do is we can do
> 另一种我们可以做的是

**[00:00:20]** the other one we could do is we can do dim zero here and so here what this one
> 另一种我们可以做的是在这里使用dim zero，所以这里这个

**[00:00:20]** dim zero here and so here what this one
> dim zero，所以这里这个

**[00:00:20]** dim zero here and so here what this one will be is the mean across each of those
> dim zero，所以这里这个将是跨每个这些的均值

**[00:00:20]** will be is the mean across each of those
> 将是跨每个这些的均值

**[00:00:20]** will be is the mean across each of those so if you consider the mean this and
> 将是跨每个这些的均值，所以如果你考虑这个的均值

**[00:00:20]** so if you consider the mean this and
> 所以如果你考虑这个的均值

**[00:00:20]** so if you consider the mean this and this would correspond to this one this
> 所以如果你考虑这个的均值，这会对应这个，这个

**[00:00:20]** this would correspond to this one this
> 这会对应这个，这个

**[00:00:20]** this would correspond to this one this and this correspond to this one and so
> 这会对应这个，这个和这个对应这个，依此类推

**[00:00:20]** and this correspond to this one and so
> 和这个对应这个，依此类推

**[00:00:20]** and this correspond to this one and so forth and that's also not what we want
> 和这个对应这个，依此类推，那也不是我们想要的

**[00:00:20]** forth and that's also not what we want
> 那也不是我们想要的

**[00:00:20]** forth and that's also not what we want because we're again mixing different
> 那也不是我们想要的，因为我们又在混合不同的

**[00:00:20]** because we're again mixing different
> 因为我们又在混合不同的

**[00:00:20]** because we're again mixing different samples and this is there's actually a
> 因为我们又在混合不同的样本，实际上有一种

**[00:00:20]** samples and this is there's actually a
> 样本，实际上有一种

**[00:00:20]** samples and this is there's actually a different normalization scheme called
> 样本，实际上有一种不同的归一化方案，叫做

**[00:00:20]** different normalization scheme called
> 不同的归一化方案，叫做

**[00:00:20]** different normalization scheme called batch normaliz that's what batch
> 不同的归一化方案，叫做batch normaliz，这就是batch

**[00:00:20]** batch normaliz that's what batch
> batch normaliz，这就是batch

**[00:00:20]** batch normaliz that's what batch normalization does if you're familiar
> batch normaliz，这就是batch normalization所做的，如果你熟悉

**[00:00:20]** normalization does if you're familiar
> normalization所做的，如果你熟悉

**[00:00:20]** normalization does if you're familiar with classic um deep deep learning but
> normalization所做的，如果你熟悉经典的深度学习

**[00:00:20]** with classic um deep deep learning but
> 经典的深度学习

**[00:00:20]** with classic um deep deep learning but batch normalization is actually not
> 经典的深度学习，但batch normalization实际上并不

**[00:00:20]** batch normalization is actually not
> batch normalization实际上并不

**[00:00:20]** batch normalization is actually not really recommended for llms because the
> batch normalization实际上并不推荐用于LLM，因为

**[00:00:20]** really recommended for llms because the
> 推荐用于LLM，因为

**[00:00:20]** really recommended for llms because the way when we train new networks large new
> 推荐用于LLM，因为当我们训练新网络大型新网络时的方式

**[00:00:20]** way when we train new networks large new
> 当我们训练新网络大型新网络时的方式

**[00:00:20]** way when we train new networks large new networks and we want to split them
> 当我们训练大型新网络并希望拆分它们时的一种方式

**[00:00:20]** networks and we want to split them
> 网络并希望拆分它们

**[00:00:20]** networks and we want to split them across multiple gpus for example one way
> 跨多个GPU拆分网络的一种方式

**[00:00:20]** across multiple gpus for example one way
> 跨多个GPU的一种方式

**[00:00:20]** across multiple gpus for example one way would be gu splitting the the samples
> 跨多个GPU拆分样本的一种方式

**[00:00:20]** would be gu splitting the the samples
> 拆分样本

**[00:00:20]** would be gu splitting the the samples across multiple gpus and then we would
> 跨多个GPU拆分样本，然后我们

**[00:00:20]** across multiple gpus and then we would
> 跨多个GPU，然后我们

**[00:00:20]** across multiple gpus and then we would end up with small samples per each GPU
> 最终每个GPU得到小样本

**[00:00:20]** end up with small samples per each GPU
> 每个GPU得到小样本

**[00:00:20]** end up with small samples per each GPU and that is not very good for batch
> 每个GPU得到小样本，这对batch来说不太好

**[00:00:20]** and that is not very good for batch
> 这对batch来说不太好

**[00:00:20]** and that is not very good for batch normalization which is very unstable
> 这对batch normalization来说不太好，它非常不稳定

**[00:00:20]** normalization which is very unstable
> normalization非常不稳定

**[00:00:20]** normalization which is very unstable then and so forth so that is why long
> normalization非常不稳定等等，所以这就是为什么长话短说

**[00:00:20]** then and so forth so that is why long
> 等等，所以这就是为什么长话短说

**[00:00:20]** then and so forth so that is why long story short we're using this concept
> 等等，所以这就是为什么长话短说，我们使用这个概念

**[00:00:20]** story short we're using this concept
> 我们使用这个概念

**[00:00:20]** story short we're using this concept called layer normalization where layer
> 我们使用这个概念，称为layer normalization

**[00:00:21]** called layer normalization where layer
> 称为layer normalization

**[00:00:21]** called layer normalization where layer normaliz normalization is independent of
> 称为layer normalization，其中layer normalization与样本大小无关

**[00:00:21]** normaliz normalization is independent of
> normalization与样本大小无关

**[00:00:21]** normaliz normalization is independent of the sample size it's normalizing across
> normalization与样本大小无关，它沿着这个维度进行归一化

**[00:00:21]** the sample size it's normalizing across
> 样本大小，它沿着这个维度进行归一化

**[00:00:21]** the sample size it's normalizing across this Dimension here so in order to get
> 样本大小，它沿着这个维度进行归一化，所以为了得到

**[00:00:21]** this Dimension here so in order to get
> 这个维度，所以为了得到

**[00:00:21]** this Dimension here so in order to get this Dimension we yeah use the column
> 这个维度，我们使用列维度一

**[00:00:21]** this Dimension we yeah use the column
> 我们使用列维度一

**[00:00:21]** this Dimension we yeah use the column Dimension one so this will be then
> 我们使用列维度一，这样就会

**[00:00:21]** Dimension one so this will be then
> 维度一，这样就会

**[00:00:21]** Dimension one so this will be then normalizing here or not normalizing but
> 维度一，这样就会在这里进行归一化，或者不是归一化

**[00:00:21]** normalizing here or not normalizing but
> 在这里进行归一化，或者不是归一化

**[00:00:21]** normalizing here or not normalizing but Computing the mean here so this one um
> 在这里进行归一化，或者不是归一化，而是计算均值

**[00:00:21]** Computing the mean here so this one um
> 计算均值

**[00:00:21]** Computing the mean here so this one um this value is the mean of the first
> 计算均值，这个值是第一个样本在特征维度上的均值

**[00:00:21]** this value is the mean of the first
> 这个值是第一个样本的均值

**[00:00:21]** this value is the mean of the first sample across the features the feature
> 这个值是第一个样本在特征维度上的均值

**[00:00:21]** sample across the features the feature
> 样本在特征维度上的均值

**[00:00:21]** sample across the features the feature Dimension um I could by the way also
> 样本在特征维度上的均值，顺便说一下，我也可以

**[00:00:21]** Dimension um I could by the way also
> 维度，顺便说一下，我也可以

**[00:00:21]** Dimension um I could by the way also type minus one here so to explain a bit
> 维度，顺便说一下，我也可以在这里输入-1，以便解释一下

**[00:00:21]** type minus one here so to explain a bit
> 输入-1，以便解释一下

**[00:00:21]** type minus one here so to explain a bit more what we have here is shape 2x 6 so
> 输入-1，以便解释一下，我们这里有一个形状为2x6的

**[00:00:21]** more what we have here is shape 2x 6 so
> 我们这里有一个形状为2x6的

**[00:00:21]** more what we have here is shape 2x 6 so if I type zero it will be normaliz or
> 我们这里有一个形状为2x6的，所以如果我输入0，它将归一化或

**[00:00:21]** if I type zero it will be normaliz or
> 如果我输入0，它将归一化或

**[00:00:21]** if I type zero it will be normaliz or Computing the mean across this first um
> 如果我输入0，它将归一化或计算第一个维度的均值

**[00:00:21]** Computing the mean across this first um
> 计算第一个维度的均值

**[00:00:21]** Computing the mean across this first um Dimension zero but this is the row right
> 计算第一个维度的均值，但这是行维度

**[00:00:21]** Dimension zero but this is the row right
> 但这是行维度

**[00:00:21]** Dimension zero but this is the row right so for each um we would normalize that
> 但这是行维度，所以对于每个，我们会像那样归一化

**[00:00:21]** so for each um we would normalize that
> 我们会像那样归一化

**[00:00:21]** so for each um we would normalize that or computer mean like that but what we
> 我们会像那样归一化或计算均值，但我们想要的是

**[00:00:21]** or computer mean like that but what we
> 或计算均值，但我们想要的是

**[00:00:21]** or computer mean like that but what we want is we want the mean across the
> 或计算均值，但我们想要的是列维度的均值

**[00:00:21]** want is we want the mean across the
> 列维度的均值

**[00:00:21]** want is we want the mean across the column Dimension like going over
> 列维度的均值，就像遍历列一样

**[00:00:21]** column Dimension like going over
> 遍历列

**[00:00:21]** column Dimension like going over basically going over the column here so
> 遍历列，所以为此我们进入第二个维度

**[00:00:21]** basically going over the column here so
> 所以为此我们进入第二个维度

**[00:00:21]** basically going over the column here so for that we go to the second dimension
> 所以为此我们进入第二个维度，即0和1，然后在这里放一个1

**[00:00:22]** for that we go to the second dimension
> 即0和1，然后在这里放一个1

**[00:00:22]** for that we go to the second dimension so 0 one and put a one here oops and so
> 即0和1，然后在这里放一个1，这就是我们正在做的，顺便说一下

**[00:00:22]** so 0 one and put a one here oops and so
> 这就是我们正在做的，顺便说一下

**[00:00:22]** so 0 one and put a one here oops and so this is what we are doing and by the way
> 这就是我们正在做的，顺便说一下，我们也可以写-1

**[00:00:22]** this is what we are doing and by the way
> 我们也可以写-1

**[00:00:22]** this is what we are doing and by the way we could also write minus one which
> 我们也可以写-1，这样会更通用，例如如果

**[00:00:22]** we could also write minus one which
> 这样会更通用，例如如果

**[00:00:22]** we could also write minus one which would be more General so for example if
> 这样会更通用，例如如果我在这里因为某种原因多了一个维度

**[00:00:22]** would be more General so for example if
> 我在这里因为某种原因多了一个维度

**[00:00:22]** would be more General so for example if I had an extra Dimension here for some
> 我在这里因为某种原因多了一个维度，这个1仍然会给我

**[00:00:22]** I had an extra Dimension here for some
> 这个1仍然会给我

**[00:00:22]** I had an extra Dimension here for some reason um this one would still give me
> 这个1仍然会给我相同的结果，但如果我写

**[00:00:22]** reason um this one would still give me
> 相同的结果，但如果我写

**[00:00:22]** reason um this one would still give me the same results right but if I write
> 相同的结果，但如果我写1，现在这会再次给我

**[00:00:22]** the same results right but if I write
> 1，现在这会再次给我

**[00:00:22]** the same results right but if I write one now this would give me the again uh
> 1，现在这会再次给我维度0和1的均值

**[00:00:22]** one now this would give me the again uh
> 维度0和1的均值

**[00:00:22]** one now this would give me the again uh mean across this Dimension 01 which is
> 维度0和1的均值，即

**[00:00:22]** mean across this Dimension 01 which is
> mean across this Dimension 01 which is

**[00:00:22]** mean across this Dimension 01 which is not what we want right so this just to
> 沿着维度01求均值，这不是我们想要的，所以这只是

**[00:00:22]** not what we want right so this just to
> 这不是我们想要的，所以这只是

**[00:00:22]** not what we want right so this just to show you will be slightly different what
> 这不是我们想要的，所以这只是为了展示会略有不同

**[00:00:22]** show you will be slightly different what
> 展示会略有不同

**[00:00:22]** show you will be slightly different what we want is we want the mean across here
> 展示会略有不同，我们想要的是沿着这里求均值

**[00:00:22]** we want is we want the mean across here
> 我们想要的是沿着这里求均值

**[00:00:22]** we want is we want the mean across here the feature Dimension assuming the
> 我们想要的是沿着这里求均值，即特征维度，假设

**[00:00:22]** the feature Dimension assuming the
> 特征维度，假设

**[00:00:22]** the feature Dimension assuming the feature Dimension is always the last
> 特征维度，假设特征维度总是最后一个

**[00:00:22]** feature Dimension is always the last
> 特征维度总是最后一个

**[00:00:22]** feature Dimension is always the last Dimension we can just use the more
> 特征维度总是最后一个维度，我们可以直接使用更通用的

**[00:00:22]** Dimension we can just use the more
> 维度，我们可以直接使用更通用的

**[00:00:22]** Dimension we can just use the more General minus one here so let me just
> 维度，我们可以直接使用更通用的-1，所以让我

**[00:00:22]** General minus one here so let me just
> 通用的-1，所以让我

**[00:00:22]** General minus one here so let me just get rid of this again to make this a bit
> 通用的-1，所以让我再次去掉这个，让它更

**[00:00:22]** get rid of this again to make this a bit
> 去掉这个，让它更

**[00:00:22]** get rid of this again to make this a bit simpler now one more thing is it's still
> 去掉这个，让它更简单一些。还有一件事，它仍然

**[00:00:22]** simpler now one more thing is it's still
> 简单一些。还有一件事，它仍然

**[00:00:22]** simpler now one more thing is it's still a bit harder to read here so we have
> 简单一些。还有一件事，它仍然有点难读，所以我们有

**[00:00:22]** a bit harder to read here so we have
> 有点难读，所以我们有

**[00:00:22]** a bit harder to read here so we have these two values and one corresponds to
> 有点难读，所以我们有这两个值，一个对应

**[00:00:22]** these two values and one corresponds to
> 这两个值，一个对应

**[00:00:22]** these two values and one corresponds to the first one and one to the second row
> 这两个值，一个对应第一行，一个对应第二行

**[00:00:22]** the first one and one to the second row
> 第一行，一个对应第二行

**[00:00:22]** the first one and one to the second row we can actually keep the original
> 第一行，一个对应第二行。我们实际上可以保持原始

**[00:00:22]** we can actually keep the original
> 我们实际上可以保持原始

**[00:00:22]** we can actually keep the original Dimensions we can do keep them equal
> 我们实际上可以保持原始维度，我们可以设置keepdim

**[00:00:22]** Dimensions we can do keep them equal
> 维度，我们可以设置keepdim

**[00:00:22]** Dimensions we can do keep them equal it's uh true here and then this one will
> 维度，我们可以设置keepdim为True，然后这个

**[00:00:23]** it's uh true here and then this one will
> 为True，然后这个

**[00:00:23]** it's uh true here and then this one will keep the dimensions you can see it's
> 为True，然后这个会保持维度，你可以看到它

**[00:00:23]** keep the dimensions you can see it's
> 保持维度，你可以看到它

**[00:00:23]** keep the dimensions you can see it's these two so the two Dimensions the two
> 保持维度，你可以看到它有两个维度，即两个

**[00:00:23]** these two so the two Dimensions the two
> 两个维度，即两个

**[00:00:23]** these two so the two Dimensions the two rows it will keep those which is a bit
> 两个维度，即两行，它会保留这些，这有点

**[00:00:23]** rows it will keep those which is a bit
> 行，它会保留这些，这有点

**[00:00:23]** rows it will keep those which is a bit nicer so this would be the mean let's
> 行，它会保留这些，这更好一些。所以这就是均值，让我们

**[00:00:23]** nicer so this would be the mean let's
> 更好一些。所以这就是均值，让我们

**[00:00:23]** nicer so this would be the mean let's just call that mean and we can do the
> 更好一些。所以这就是均值，我们就叫它mean，然后我们可以做

**[00:00:23]** just call that mean and we can do the
> 就叫它mean，然后我们可以做

**[00:00:23]** just call that mean and we can do the same for the variance
> 就叫它mean，然后我们可以对方差做同样的操作

**[00:00:23]** same for the variance
> 对方差做同样的操作

**[00:00:23]** same for the variance here let's do
> 对方差做同样的操作，这里让我们做

**[00:00:23]** this okay so again we can see um what we
> 这个。好的，再次我们可以看到，我们

**[00:00:23]** this okay so again we can see um what we
> 这个。好的，再次我们可以看到，我们

**[00:00:23]** this okay so again we can see um what we want is a variance of one for nice
> 这个。好的，再次我们可以看到，我们想要的是方差为1，以获得良好的

**[00:00:23]** want is a variance of one for nice
> 想要的是方差为1，以获得良好的

**[00:00:23]** want is a variance of one for nice properties but we have a different
> 想要的是方差为1，以获得良好的性质，但我们有一个不同的

**[00:00:23]** properties but we have a different
> 性质，但我们有一个不同的

**[00:00:23]** properties but we have a different variance so our goal here is to
> 性质，但我们有一个不同的方差，所以我们的目标是

**[00:00:23]** variance so our goal here is to
> 方差，所以我们的目标是

**[00:00:23]** variance so our goal here is to normalize the U data such that this will
> 方差，所以我们的目标是归一化U数据，使得这个

**[00:00:23]** normalize the U data such that this will
> 归一化U数据，使得这个

**[00:00:23]** normalize the U data such that this will be um the mean of ones and the variance
> 归一化U数据，使得均值为1，方差为

**[00:00:23]** be um the mean of ones and the variance
> 均值为1，方差为

**[00:00:23]** be um the mean of ones and the variance of zeros so that's basically what we are
> 均值为1，方差为0，这基本上就是我们

**[00:00:23]** of zeros so that's basically what we are
> 0，这基本上就是我们

**[00:00:23]** of zeros so that's basically what we are um trying to accomplish
> 0，这基本上就是我们试图实现的

**[00:00:23]** um trying to accomplish
> 试图实现的

**[00:00:23]** um trying to accomplish here and so one way we can accomplish
> 试图实现的。这里有一种方法可以实现

**[00:00:23]** here and so one way we can accomplish
> 这里有一种方法可以实现

**[00:00:23]** here and so one way we can accomplish this is by well just step by step by
> 这里有一种方法可以实现，那就是一步一步地

**[00:00:23]** this is by well just step by step by
> 一步一步地

**[00:00:23]** this is by well just step by step by centering around the mean so if we have
> 一步一步地围绕均值中心化，所以如果我们有

**[00:00:23]** centering around the mean so if we have
> 围绕均值中心化，所以如果我们有

**[00:00:23]** centering around the mean so if we have an output and we subtract the mean this
> 围绕均值中心化，所以如果我们有一个输出，减去均值，这

**[00:00:23]** an output and we subtract the mean this
> 一个输出，减去均值，这

**[00:00:23]** an output and we subtract the mean this will give us the output centered at the
> 一个输出，减去均值，这会使输出以均值为中心

**[00:00:23]** will give us the output centered at the
> 会使输出以均值为中心

**[00:00:23]** will give us the output centered at the mean so for example just a simple
> 会使输出以均值为中心。例如，一个简单的

**[00:00:24]** mean so for example just a simple
> 均值。例如，一个简单的

**[00:00:24]** mean so for example just a simple example if you consider those consisting
> 均值。例如，一个简单的例子，如果你考虑由

**[00:00:24]** example if you consider those consisting
> 例子，如果你考虑由

**[00:00:24]** example if you consider those consisting of just the value five 5 five five 5 5 5
> 例子，如果你考虑由值5 5 5 5 5 5 5组成的

**[00:00:24]** of just the value five 5 five five 5 5 5
> 值5 5 5 5 5 5 5组成的

**[00:00:24]** of just the value five 5 five five 5 5 5 55 If I subtract the average which would
> 值5 5 5 5 5 5 5 55，如果我减去平均值，即

**[00:00:24]** 55 If I subtract the average which would
> 55，如果我减去平均值，即

**[00:00:24]** 55 If I subtract the average which would be in this case five 5 - 5 is zero right
> 55，如果我减去平均值，在这种情况下是5，5减5等于0，对吧

**[00:00:24]** be in this case five 5 - 5 is zero right
> 在这种情况下是5，5减5等于0，对吧

**[00:00:24]** be in this case five 5 - 5 is zero right so doing this should give
> 在这种情况下是5，5减5等于0，对吧，所以这样做应该得到

**[00:00:24]** me uh zero right so here what we get is
> 嗯，零，对，这里我们得到的是

**[00:00:24]** me uh zero right so here what we get is
> 嗯，零，对，这里我们得到的是

**[00:00:24]** me uh zero right so here what we get is this actually you can see it's very
> 嗯，零，对，这里我们得到的是，实际上你可以看到它非常

**[00:00:24]** this actually you can see it's very
> 实际上你可以看到它非常

**[00:00:24]** this actually you can see it's very close to zero it's not exactly zero so
> 实际上你可以看到它非常接近零，但不是完全为零，所以

**[00:00:24]** close to zero it's not exactly zero so
> 接近零，但不是完全为零，所以

**[00:00:24]** close to zero it's not exactly zero so this is a scientific notation um in
> 接近零，但不是完全为零，所以这是一种科学计数法，嗯，在

**[00:00:24]** this is a scientific notation um in
> 这是一种科学计数法，嗯，在

**[00:00:24]** this is a scientific notation um in pytorch um so this would mean basically
> 这是一种科学计数法，嗯，在PyTorch中，嗯，这基本上意味着

**[00:00:24]** pytorch um so this would mean basically
> PyTorch中，嗯，这基本上意味着

**[00:00:24]** pytorch um so this would mean basically 0.9 leading zeros and then 745 actually
> PyTorch中，嗯，这基本上意味着0.9，前面有很多零，然后是745，实际上

**[00:00:24]** 0.9 leading zeros and then 745 actually
> 0.9，前面有很多零，然后是745，实际上

**[00:00:24]** 0.9 leading zeros and then 745 actually to um make this a bit nicer we can set
> 0.9，前面有很多零，然后是745，实际上，为了让这个看起来更好看，我们可以设置

**[00:00:24]** to um make this a bit nicer we can set
> 为了让这个看起来更好看，我们可以设置

**[00:00:24]** to um make this a bit nicer we can set the print options and then there should
> 为了让这个看起来更好看，我们可以设置打印选项，然后应该

**[00:00:24]** the print options and then there should
> 打印选项，然后应该

**[00:00:24]** the print options and then there should be a SI mode we can set that to
> 打印选项，然后应该有一个SI模式，我们可以将其设置为

**[00:00:24]** be a SI mode we can set that to
> 有一个SI模式，我们可以将其设置为

**[00:00:24]** be a SI mode we can set that to fults and then it will look hopefully a
> 有一个SI模式，我们可以将其设置为fults，然后它看起来希望会

**[00:00:24]** fults and then it will look hopefully a
> fults，然后它看起来希望会

**[00:00:24]** fults and then it will look hopefully a bit nicer yeah so you can see it's
> fults，然后它看起来希望会更好看一些，是的，所以你可以看到它

**[00:00:24]** bit nicer yeah so you can see it's
> 更好看一些，是的，所以你可以看到它

**[00:00:24]** bit nicer yeah so you can see it's basically zero and zero so that is what
> 更好看一些，是的，所以你可以看到它基本上是零和零，所以这就是

**[00:00:24]** basically zero and zero so that is what
> 基本上是零和零，所以这就是

**[00:00:24]** basically zero and zero so that is what we've done here just to also make this a
> 基本上是零和零，所以这就是我们在这里所做的，只是为了让它

**[00:00:24]** we've done here just to also make this a
> 我们在这里所做的，只是为了让它

**[00:00:24]** we've done here just to also make this a bit nicer again
> 我们在这里所做的，只是为了让它再次更好看一些

**[00:00:25]** okay and now um this is how we Center at
> 好的，现在，嗯，这就是我们如何将其中心化到

**[00:00:25]** okay and now um this is how we Center at
> 好的，现在，嗯，这就是我们如何将其中心化到

**[00:00:25]** okay and now um this is how we Center at zero the other one that we do or what we
> 好的，现在，嗯，这就是我们如何将其中心化到零，另一个我们做的，或者说我们

**[00:00:25]** zero the other one that we do or what we
> 零，另一个我们做的，或者说我们

**[00:00:25]** zero the other one that we do or what we want to do is we want to also give it a
> 零，另一个我们做的，或者说我们想要做的，是还要给它一个

**[00:00:25]** want to do is we want to also give it a
> 想要做的，是还要给它一个

**[00:00:25]** want to do is we want to also give it a variance of one so for that uh we divide
> 想要做的，是还要给它一个方差为1，所以为此，嗯，我们除以

**[00:00:25]** variance of one so for that uh we divide
> 方差为1，所以为此，嗯，我们除以

**[00:00:25]** variance of one so for that uh we divide by the standard deviation so the
> 方差为1，所以为此，嗯，我们除以标准差，所以

**[00:00:25]** by the standard deviation so the
> 标准差，所以

**[00:00:25]** by the standard deviation so the standard deviation is the square root of
> 标准差，所以标准差是方差的平方根

**[00:00:25]** standard deviation is the square root of
> 标准差是方差的平方根

**[00:00:25]** standard deviation is the square root of the variance um actually yeah uh I could
> 标准差是方差的平方根，嗯，实际上，是的，嗯，我可以

**[00:00:25]** the variance um actually yeah uh I could
> 方差，嗯，实际上，是的，嗯，我可以

**[00:00:25]** the variance um actually yeah uh I could go into more detail here why standard
> 方差，嗯，实际上，是的，嗯，我可以在这里更详细地解释为什么用标准差

**[00:00:25]** go into more detail here why standard
> 在这里更详细地解释为什么用标准差

**[00:00:25]** go into more detail here why standard deviation but that would be a whole
> 在这里更详细地解释为什么用标准差，但那本身就会是一整堂

**[00:00:25]** deviation but that would be a whole
> 标准差，但那本身就会是一整堂

**[00:00:25]** deviation but that would be a whole lecture by itself I used to be a
> 标准差，但那本身就会是一整堂课，我以前是

**[00:00:25]** lecture by itself I used to be a
> 课，我以前是

**[00:00:25]** lecture by itself I used to be a statistics Professor teaching these
> 课，我以前是一名统计学教授，教这些

**[00:00:25]** statistics Professor teaching these
> 统计学教授，教这些

**[00:00:25]** statistics Professor teaching these things but to keep things simple Le uh
> 统计学教授，教这些内容，但为了简单起见，嗯，

**[00:00:25]** things but to keep things simple Le uh
> 内容，但为了简单起见，嗯，

**[00:00:25]** things but to keep things simple Le uh may just assume this is the formula for
> 内容，但为了简单起见，嗯，我们只需假设这是归一化的公式

**[00:00:25]** may just assume this is the formula for
> 我们只需假设这是归一化的公式

**[00:00:25]** may just assume this is the formula for normalization um subtracting the mean
> 我们只需假设这是归一化的公式，嗯，减去均值

**[00:00:25]** normalization um subtracting the mean
> 归一化，嗯，减去均值

**[00:00:25]** normalization um subtracting the mean and dividing by the standard
> 归一化，嗯，减去均值并除以标准差

**[00:00:25]** and dividing by the standard
> 并除以标准差

**[00:00:25]** and dividing by the standard deviation and then so this one should be
> 并除以标准差，然后这个应该

**[00:00:25]** deviation and then so this one should be
> 标准差，然后这个应该

**[00:00:25]** deviation and then so this one should be let's just uh call that nored and this
> 标准差，然后这个应该，我们把它叫做nored，而这个

**[00:00:25]** let's just uh call that nored and this
> 我们把它叫做nored，而这个

**[00:00:25]** let's just uh call that nored and this one should still
> 我们把它叫做nored，而这个应该仍然

**[00:00:25]** one should still
> 应该仍然

**[00:00:25]** one should still be mean
> 应该仍然是均值为

**[00:00:25]** be mean
> 是均值为

**[00:00:25]** be mean zero and now in addition to that we
> 是均值为零，现在除此之外，我们

**[00:00:25]** zero and now in addition to that we
> 零，现在除此之外，我们

**[00:00:25]** zero and now in addition to that we should also
> 零，现在除此之外，我们还应该

**[00:00:25]** should also
> 还应该

**[00:00:25]** should also have we can just do it here a variance
> 还应该有，我们可以在这里做，方差

**[00:00:25]** have we can just do it here a variance
> 有，我们可以在这里做，方差

**[00:00:25]** have we can just do it here a variance of one so that's what we see here so far
> 有，我们可以在这里做，方差为1，所以这就是我们目前看到的

**[00:00:25]** of one so that's what we see here so far
> 方差为1，所以这就是我们目前看到的

**[00:00:25]** of one so that's what we see here so far so good so we accomplished that that is
> 方差为1，所以这就是我们目前看到的，到目前为止还不错，所以我们完成了，这就是

**[00:00:26]** so good so we accomplished that that is
> 到目前为止还不错，所以我们完成了，这就是

**[00:00:26]** so good so we accomplished that that is one part of layer normalization there's
> 到目前为止还不错，所以我们完成了，这就是层归一化的一部分，还有

**[00:00:26]** one part of layer normalization there's
> 层归一化的一部分，还有

**[00:00:26]** one part of layer normalization there's a bit more to layer normalization it's
> 层归一化的一部分，还有更多关于层归一化的内容，它

**[00:00:26]** a bit more to layer normalization it's
> 更多关于层归一化的内容，它

**[00:00:26]** a bit more to layer normalization it's actually a new network layer that learns
> 更多关于层归一化的内容，它实际上是一个需要学习的神经网络层

**[00:00:26]** actually a new network layer that learns
> 实际上是一个学习参数的新网络层

**[00:00:26]** actually a new network layer that learns parameters so let me just um copy in
> 实际上是一个学习参数的新网络层，让我复制一下

**[00:00:26]** parameters so let me just um copy in
> 参数，让我复制一下

**[00:00:26]** parameters so let me just um copy in here the layer Norm class I implemented
> 参数，让我在这里复制我实现的Layer Norm类

**[00:00:26]** here the layer Norm class I implemented
> 我实现的Layer Norm类

**[00:00:26]** here the layer Norm class I implemented from scratch and to show you a bit more
> 我实现的Layer Norm类，为了更清楚地展示

**[00:00:26]** from scratch and to show you a bit more
> 为了更清楚地展示

**[00:00:26]** from scratch and to show you a bit more what I mean by that so what we see here
> 为了更清楚地展示我的意思，这里我们看到的是

**[00:00:26]** what I mean by that so what we see here
> 我的意思，这里我们看到的是

**[00:00:26]** what I mean by that so what we see here is a new network class um and we have an
> 我的意思，这里我们看到的是一个新网络类，我们有一个

**[00:00:26]** is a new network class um and we have an
> 是一个新网络类，我们有一个

**[00:00:26]** is a new network class um and we have an initialization here like the Constructor
> 是一个新网络类，我们有一个初始化方法，就像构造函数

**[00:00:26]** initialization here like the Constructor
> 初始化方法，就像构造函数

**[00:00:26]** initialization here like the Constructor method and there are a couple of things
> 初始化方法，就像构造函数方法，其中有几件事

**[00:00:26]** method and there are a couple of things
> 方法，其中有几件事

**[00:00:26]** method and there are a couple of things so one thing here is that we have
> 方法，其中有几件事，其中一件事是我们有

**[00:00:26]** so one thing here is that we have
> 其中一件事是我们有

**[00:00:26]** so one thing here is that we have parameters and as you maybe remember
> 其中一件事是我们有参数，你可能还记得

**[00:00:26]** parameters and as you maybe remember
> 参数，你可能还记得

**[00:00:26]** parameters and as you maybe remember from chapter 3 parameters are trainable
> 参数，你可能还记得第3章中，参数是可训练的

**[00:00:26]** from chapter 3 parameters are trainable
> 第3章中，参数是可训练的

**[00:00:26]** from chapter 3 parameters are trainable your network weights so here we have the
> 第3章中，参数是可训练的网络权重，所以这里我们有

**[00:00:26]** your network weights so here we have the
> 网络权重，所以这里我们有

**[00:00:26]** your network weights so here we have the same size as the bedding dimension in
> 网络权重，所以这里我们有与embedding维度相同的大小

**[00:00:26]** same size as the bedding dimension in
> 与embedding维度相同的大小

**[00:00:26]** same size as the bedding dimension in our case um if we
> 与embedding维度相同的大小，在我们的例子中

**[00:00:26]** our case um if we
> 在我们的例子中

**[00:00:26]** our case um if we consider this one this should be a six
> 在我们的例子中，如果考虑这个，应该是6

**[00:00:26]** consider this one this should be a six
> 考虑这个，应该是6

**[00:00:26]** consider this one this should be a six so we have six um six values here
> 考虑这个，应该是6，所以这里有6个值

**[00:00:26]** so we have six um six values here
> 所以这里有6个值

**[00:00:26]** so we have six um six values here and the parameter will be making those
> 所以这里有6个值，这个参数将使这些

**[00:00:26]** and the parameter will be making those
> 这个参数将使这些

**[00:00:26]** and the parameter will be making those six values trainable and we call that
> 这个参数将使这6个值可训练，我们称之为

**[00:00:26]** six values trainable and we call that
> 6个值可训练，我们称之为

**[00:00:26]** six values trainable and we call that the scaling parameter
> 6个值可训练，我们称之为缩放参数

**[00:00:26]** the scaling parameter
> 缩放参数

**[00:00:26]** the scaling parameter similarly we have a shift parameter and
> 缩放参数，类似地，我们有一个偏移参数

**[00:00:27]** similarly we have a shift parameter and
> 类似地，我们有一个偏移参数

**[00:00:27]** similarly we have a shift parameter and this is zeros I will get it in a moment
> 类似地，我们有一个偏移参数，这是零，我稍后会解释

**[00:00:27]** this is zeros I will get it in a moment
> 这是零，我稍后会解释

**[00:00:27]** this is zeros I will get it in a moment why it's um ones and zeros so and then
> 这是零，我稍后会解释为什么是1和0，然后

**[00:00:27]** why it's um ones and zeros so and then
> 为什么是1和0，然后

**[00:00:27]** why it's um ones and zeros so and then maybe for now ignore the apps so for now
> 为什么是1和0，然后暂时忽略eps，现在

**[00:00:27]** maybe for now ignore the apps so for now
> 暂时忽略eps，现在

**[00:00:27]** maybe for now ignore the apps so for now what we have here is exactly what we had
> 暂时忽略eps，现在我们这里有的正是我们之前有的

**[00:00:27]** what we have here is exactly what we had
> 我们这里有的正是我们之前有的

**[00:00:27]** what we have here is exactly what we had before so what we had before um let me
> 我们这里有的正是我们之前有的，所以我们之前有的

**[00:00:27]** before so what we had before um let me
> 所以我们之前有的

**[00:00:27]** before so what we had before um let me also just comment this out and get back
> 所以我们之前有的，让我也注释掉这个，稍后再回来看

**[00:00:27]** also just comment this out and get back
> 让我也注释掉这个，稍后再回来看

**[00:00:27]** also just comment this out and get back to that later the unbiased here so what
> 让我也注释掉这个，稍后再回来看这里的unbiased，所以我们之前有的

**[00:00:27]** to that later the unbiased here so what
> 这里的unbiased，所以我们之前有的

**[00:00:27]** to that later the unbiased here so what we had before is the normalization we
> 这里的unbiased，所以我们之前有的是归一化

**[00:00:27]** we had before is the normalization we
> 我们之前有的是归一化

**[00:00:27]** we had before is the normalization we had computed the mean and the variance
> 我们之前有的是归一化，我们计算了均值和方差

**[00:00:27]** had computed the mean and the variance
> 我们计算了均值和方差

**[00:00:27]** had computed the mean and the variance and then we subtracted the mean and
> 我们计算了均值和方差，然后减去均值

**[00:00:27]** and then we subtracted the mean and
> 然后减去均值

**[00:00:27]** and then we subtracted the mean and divided by the standard deviation this
> 然后减去均值，除以标准差，这

**[00:00:27]** divided by the standard deviation this
> 除以标准差，这

**[00:00:27]** divided by the standard deviation this is exactly so if I write it like this
> 除以标准差，这完全就是，如果我这样写

**[00:00:27]** is exactly so if I write it like this
> 完全就是，如果我这样写

**[00:00:27]** is exactly so if I write it like this this is exactly what we had before the
> 完全就是，如果我这样写，这正是我们之前有的

**[00:00:27]** this is exactly what we had before the
> 这正是我们之前有的

**[00:00:27]** this is exactly what we had before the normalization
> 这正是我们之前有的归一化

**[00:00:27]** normalization
> 归一化

**[00:00:27]** normalization here now we are adding One More Concept
> 归一化，现在我们添加了另一个概念

**[00:00:27]** here now we are adding One More Concept
> 现在我们添加了另一个概念

**[00:00:27]** here now we are adding One More Concept and that is this um apps so what is the
> 现在我们添加了另一个概念，就是这个eps，所以eps的作用是什么

**[00:00:27]** and that is this um apps so what is the
> 就是这个eps，所以eps的作用是什么

**[00:00:27]** and that is this um apps so what is the apps doing so this is preventing a
> 就是这个eps，所以eps的作用是防止

**[00:00:27]** apps doing so this is preventing a
> eps的作用是防止

**[00:00:27]** apps doing so this is preventing a division by zero errors so if we have
> eps的作用是防止除以零的错误，所以如果我们有

**[00:00:27]** division by zero errors so if we have
> 除以零的错误，所以如果我们有

**[00:00:27]** division by zero errors so if we have square root of zero basically zero and
> 除以零的错误，所以如果我们有零的平方根，基本上是零

**[00:00:27]** square root of zero basically zero and
> 零的平方根，基本上是零

**[00:00:27]** square root of zero basically zero and we prevent basically dividing by zero so
> 零的平方根，基本上是零，我们防止除以零

**[00:00:27]** we prevent basically dividing by zero so
> 我们防止除以零

**[00:00:27]** we prevent basically dividing by zero so we add a small value here it's just like
> 我们防止除以零，所以我们在这里加一个小值，就像

**[00:00:27]** we add a small value here it's just like
> 我们在这里添加一个小值，就像

**[00:00:27]** we add a small value here it's just like a small placeholder value so that's 0
> 我们在这里添加一个小值，就像一个小的占位值，所以是0

**[00:00:27]** a small placeholder value so that's 0
> 一个小的占位值，所以是0

**[00:00:27]** a small placeholder value so that's 0 001 um so I could also just write it
> 一个小的占位值，所以是0.001，嗯，我也可以直接写

**[00:00:28]** 001 um so I could also just write it
> 0.001，嗯，我也可以直接写

**[00:00:28]** 001 um so I could also just write it like
> 0.001，嗯，我也可以直接写成

**[00:00:28]** like
> 像

**[00:00:28]** like this and yeah so that's what we are
> 像这样，是的，这就是我们

**[00:00:28]** this and yeah so that's what we are
> 这样，是的，这就是我们

**[00:00:28]** this and yeah so that's what we are doing here um so far so good so we have
> 这样，是的，这就是我们在这里做的，嗯，到目前为止还不错，所以我们有

**[00:00:28]** doing here um so far so good so we have
> 在这里做的，嗯，到目前为止还不错，所以我们有

**[00:00:28]** doing here um so far so good so we have those things and there's something new
> 在这里做的，嗯，到目前为止还不错，所以我们有这些东西，还有一些新东西

**[00:00:28]** those things and there's something new
> 这些东西，还有一些新东西

**[00:00:28]** those things and there's something new now and there's the scale and the shift
> 这些东西，还有一些新东西，现在还有scale和shift

**[00:00:28]** now and there's the scale and the shift
> 现在还有scale和shift

**[00:00:28]** now and there's the scale and the shift here so to explain what happens let's
> 现在还有scale和shift，这里为了解释发生了什么，让我们

**[00:00:28]** here so to explain what happens let's
> 这里为了解释发生了什么，让我们

**[00:00:28]** here so to explain what happens let's maybe tackle it like this so we have
> 这里为了解释发生了什么，让我们也许这样处理，所以我们有

**[00:00:28]** maybe tackle it like this so we have
> 也许这样处理，所以我们有

**[00:00:28]** maybe tackle it like this so we have let's just ignore this for the
> 也许这样处理，所以我们有，让我们先忽略这个

**[00:00:28]** let's just ignore this for the
> 让我们先忽略这个

**[00:00:28]** let's just ignore this for the moment
> 让我们先忽略这个

**[00:00:28]** moment
> 时刻

**[00:00:28]** moment so what we oh let's let's do it like
> 时刻，所以，哦，让我们这样做

**[00:00:28]** so what we oh let's let's do it like
> 所以，哦，让我们这样做

**[00:00:28]** so what we oh let's let's do it like this this so because the division comes
> 所以，哦，让我们这样做，因为除法发生在

**[00:00:28]** this this so because the division comes
> 因为除法发生在

**[00:00:28]** this this so because the division comes before uh sorry not not correctly um so
> 因为除法发生在之前，抱歉，不对，嗯，所以

**[00:00:28]** before uh sorry not not correctly um so
> 之前，抱歉，不对，嗯，所以

**[00:00:28]** before uh sorry not not correctly um so let's do it like this it's may be easier
> 之前，抱歉，不对，嗯，所以让我们这样做，可能更容易

**[00:00:28]** let's do it like this it's may be easier
> 让我们这样做，可能更容易

**[00:00:28]** let's do it like this it's may be easier um so what we doing here is we are
> 让我们这样做，可能更容易，嗯，所以我们在这里做的是

**[00:00:28]** um so what we doing here is we are
> 嗯，所以我们在这里做的是

**[00:00:28]** um so what we doing here is we are subtracting the mean right and so here
> 嗯，所以我们在这里做的是减去均值，对吧，然后这里

**[00:00:28]** subtracting the mean right and so here
> 减去均值，对吧，然后这里

**[00:00:28]** subtracting the mean right and so here we optionally add some value back so we
> 减去均值，对吧，然后这里我们可选地加回一些值，所以我们

**[00:00:28]** we optionally add some value back so we
> 我们可选地加回一些值，所以我们

**[00:00:28]** we optionally add some value back so we we subtract the value and then we add
> 我们可选地加回一些值，所以我们减去这个值，然后我们加

**[00:00:28]** we subtract the value and then we add
> 我们减去这个值，然后我们加

**[00:00:28]** we subtract the value and then we add some value back and if we check the
> 我们减去这个值，然后我们加回一些值，如果我们检查

**[00:00:28]** some value back and if we check the
> 加回一些值，如果我们检查

**[00:00:28]** some value back and if we check the original value of shift that's zeros so
> 加回一些值，如果我们检查shift的原始值，那是零，所以

**[00:00:28]** original value of shift that's zeros so
> shift的原始值，那是零，所以

**[00:00:28]** original value of shift that's zeros so you're essentially adding zero to it
> shift的原始值，那是零，所以你基本上是在加零

**[00:00:28]** you're essentially adding zero to it
> 你基本上是在加零

**[00:00:28]** you're essentially adding zero to it which is similar to not adding anything
> 你基本上是在加零，这类似于什么都不加

**[00:00:28]** which is similar to not adding anything
> 这类似于什么都不加

**[00:00:28]** which is similar to not adding anything here right so this doesn't have any
> 这类似于什么都不加，对吧，所以这没有任何

**[00:00:28]** here right so this doesn't have any
> 对吧，所以这没有任何

**[00:00:28]** here right so this doesn't have any effect out of the box because it's zeros
> 对吧，所以这没有任何效果，因为它是零

**[00:00:29]** effect out of the box because it's zeros
> 效果，因为它是零

**[00:00:29]** effect out of the box because it's zeros however the network during the training
> 效果，因为它是零，然而网络在训练过程中

**[00:00:29]** however the network during the training
> 然而网络在训练过程中

**[00:00:29]** however the network during the training later in chapter 5 it might learn
> 然而网络在训练过程中，稍后在第五章，它可能会学习

**[00:00:29]** later in chapter 5 it might learn
> 稍后在第五章，它可能会学习

**[00:00:29]** later in chapter 5 it might learn certain values that are more beneficial
> 稍后在第五章，它可能会学习某些对归一化更有利的值

**[00:00:29]** certain values that are more beneficial
> 某些对归一化更有利的值

**[00:00:29]** certain values that are more beneficial for the normalization so it might um but
> 某些对归一化更有利的值，所以它可能会，嗯，但是

**[00:00:29]** for the normalization so it might um but
> 所以它可能会，嗯，但是

**[00:00:29]** for the normalization so it might um but it could technically what it might learn
> 所以它可能会，嗯，但从技术上讲，它可能学到的是

**[00:00:29]** it could technically what it might learn
> 但从技术上讲，它可能学到的是

**[00:00:29]** it could technically what it might learn is to just undo the u mean centering so
> 但从技术上讲，它可能学到的是撤销均值中心化，所以

**[00:00:29]** is to just undo the u mean centering so
> 是撤销均值中心化，所以

**[00:00:29]** is to just undo the u mean centering so if the network finds Well um subtracting
> 是撤销均值中心化，所以如果网络发现，嗯，减去

**[00:00:29]** if the network finds Well um subtracting
> 如果网络发现，嗯，减去

**[00:00:29]** if the network finds Well um subtracting the mean is actually not good here um
> 如果网络发现，嗯，减去均值实际上在这里不好，嗯

**[00:00:29]** the mean is actually not good here um
> 均值实际上在这里不好，嗯

**[00:00:29]** the mean is actually not good here um it's not helping me learn I can just
> 均值实际上在这里不好，嗯，它对我学习没有帮助，我可以直接

**[00:00:29]** it's not helping me learn I can just
> 它对我学习没有帮助，我可以直接

**[00:00:29]** it's not helping me learn I can just learn shift uh to be similar to the mean
> 它对我学习没有帮助，我可以直接学习shift，使其类似于均值

**[00:00:29]** learn shift uh to be similar to the mean
> 学习shift，使其类似于均值

**[00:00:29]** learn shift uh to be similar to the mean and then I'm undoing this subtraction
> 学习shift，使其类似于均值，然后我就撤销了这个减法

**[00:00:29]** and then I'm undoing this subtraction
> 然后我就撤销了这个减法

**[00:00:29]** and then I'm undoing this subtraction right so if I what I mean is If I
> 然后我就撤销了这个减法，对吧，所以我的意思是，如果我

**[00:00:29]** right so if I what I mean is If I
> 对吧，所以我的意思是，如果我

**[00:00:29]** right so if I what I mean is If I subtract the values five and the network
> 对吧，所以我的意思是，如果我减去值5，而网络

**[00:00:29]** subtract the values five and the network
> 减去值5，而网络

**[00:00:29]** subtract the values five and the network learns the shift to be five I'm adding
> 减去值5，而网络学习shift为5，我加回

**[00:00:29]** learns the shift to be five I'm adding
> 学习shift为5，我加回

**[00:00:29]** learns the shift to be five I'm adding back five then it's it's if it's if if
> 学习shift为5，我加回5，那么它，如果它，如果如果

**[00:00:29]** back five then it's it's if it's if if
> 回到五，然后就是，如果，如果

**[00:00:29]** back five then it's it's if it's if if nothing has happened here so in that
> 回到五，然后就是，如果，如果这里什么都没发生，那么在这种情况下

**[00:00:29]** nothing has happened here so in that
> 这里什么都没发生，那么在这种情况下

**[00:00:29]** nothing has happened here so in that case um it gives the network the option
> 这里什么都没发生，那么在这种情况下，嗯，它给网络提供了选项

**[00:00:29]** case um it gives the network the option
> 在这种情况下，嗯，它给网络提供了选项

**[00:00:29]** case um it gives the network the option to undo this normalization long story
> 在这种情况下，嗯，它给网络提供了选项来撤销这个归一化，长话短说

**[00:00:29]** to undo this normalization long story
> 来撤销这个归一化，长话短说

**[00:00:29]** to undo this normalization long story short and then similarly um we can do
> 来撤销这个归一化，长话短说，然后类似地，嗯，我们可以

**[00:00:29]** short and then similarly um we can do
> 然后类似地，嗯，我们可以

**[00:00:29]** short and then similarly um we can do the same thing here for this division by
> 然后类似地，嗯，我们可以对这里的除以标准差做同样的事情，所以这里的scale

**[00:00:29]** the same thing here for this division by
> 对这里的除以标准差做同样的事情，所以这里的scale

**[00:00:29]** the same thing here for this division by the standard deviation so here the scale
> 对这里的除以标准差做同样的事情，所以这里的scale并不是在乘以这个值，对吧

**[00:00:29]** the standard deviation so here the scale
> 并不是在乘以这个值，对吧

**[00:00:29]** the standard deviation so here the scale it's not uh multiplying the value right
> 并不是在乘以这个值，对吧，所以默认情况下这是一个1，所以默认情况下

**[00:00:30]** it's not uh multiplying the value right
> 所以默认情况下这是一个1，所以默认情况下

**[00:00:30]** it's not uh multiplying the value right and so by default this is a one so by
> 所以默认情况下这是一个1，所以默认情况下我们是在乘以一个1

**[00:00:30]** and so by default this is a one so by
> 默认情况下我们是在乘以一个1

**[00:00:30]** and so by default this is a one so by default we are multiplying it by a one
> 默认情况下我们是在乘以一个1，这不会产生任何效果，它和

**[00:00:30]** default we are multiplying it by a one
> 这不会产生任何效果，它和

**[00:00:30]** default we are multiplying it by a one which doesn't do anything it's the same
> 这不会产生任何效果，它和直接保留这个，哎呀，保留这个

**[00:00:30]** which doesn't do anything it's the same
> 直接保留这个，哎呀，保留这个

**[00:00:30]** which doesn't do anything it's the same as just keeping this oops keeping this
> 直接保留这个，哎呀，保留这个值是一样的，就像这里这样

**[00:00:30]** as just keeping this oops keeping this
> 值是一样的，就像这里这样

**[00:00:30]** as just keeping this oops keeping this value like here like this
> 值是一样的，就像这里这样，对吧，嗯，然而网络再次

**[00:00:30]** value like here like this
> 对吧，嗯，然而网络再次

**[00:00:30]** value like here like this right um however the network again it's
> 对吧，嗯，然而网络再次是一个参数，它可能会学习不同的

**[00:00:30]** right um however the network again it's
> 是一个参数，它可能会学习不同的

**[00:00:30]** right um however the network again it's a parameter it might learn different
> 是一个参数，它可能会学习不同的值，它可能会学习这里的这个值

**[00:00:30]** a parameter it might learn different
> 值，它可能会学习这里的这个值

**[00:00:30]** a parameter it might learn different values it might learn this value here
> 值，它可能会学习这里的这个值，而不是scale，然后它就会撤销

**[00:00:30]** values it might learn this value here
> 而不是scale，然后它就会撤销

**[00:00:30]** values it might learn this value here instead of scale and then it will undo
> 而不是scale，然后它就会撤销这个除法，基本上就是这样，对吧，所以它

**[00:00:30]** instead of scale and then it will undo
> 这个除法，基本上就是这样，对吧，所以它

**[00:00:30]** instead of scale and then it will undo this division basically right so it's
> 这个除法，基本上就是这样，对吧，所以它只是给网络提供了选项

**[00:00:30]** this division basically right so it's
> 只是给网络提供了选项

**[00:00:30]** this division basically right so it's just giving the option for the network
> 只是给网络提供了选项来学习某些值，以稍微不同的方式转换这些

**[00:00:30]** just giving the option for the network
> 来学习某些值，以稍微不同的方式转换这些

**[00:00:30]** just giving the option for the network to learn certain values to transform the
> 来学习某些值，以稍微不同的方式转换这些值，这就是

**[00:00:30]** to learn certain values to transform the
> 值，这就是

**[00:00:30]** to learn certain values to transform the values slightly differently and this is
> 值，这就是layer Norm层的基本样子，嗯，默认情况下它会

**[00:00:30]** values slightly differently and this is
> layer Norm层的基本样子，嗯，默认情况下它会

**[00:00:30]** values slightly differently and this is essentially how a layer Norm layer yeah
> layer Norm层的基本样子，嗯，默认情况下它会和我们之前的归一化相同，因为

**[00:00:30]** essentially how a layer Norm layer yeah
> 和我们之前的归一化相同，因为

**[00:00:30]** essentially how a layer Norm layer yeah looks like um and by default it will be
> 和我们之前的归一化相同，因为这些嗯，1和0是如何嗯

**[00:00:30]** looks like um and by default it will be
> 这些嗯，1和0是如何嗯

**[00:00:30]** looks like um and by default it will be the same as our normalization due to the
> 这些嗯，1和0是如何嗯在这里设置的，现在，是的，让我们实际尝试一下

**[00:00:30]** the same as our normalization due to the
> 在这里设置的，现在，是的，让我们实际尝试一下

**[00:00:30]** the same as our normalization due to the fact how these um ones and zeros are um
> 在这里设置的，现在，是的，让我们实际尝试一下这个实践，嗯，让我们看看它如何

**[00:00:30]** fact how these um ones and zeros are um
> 这个实践，嗯，让我们看看它如何

**[00:00:30]** fact how these um ones and zeros are um set here now yeah let's actually try
> 这个实践，嗯，让我们看看它如何工作，让我们做layer norm，然后我们

**[00:00:30]** set here now yeah let's actually try
> 工作，让我们做layer norm，然后我们

**[00:00:30]** set here now yeah let's actually try this in practice um let's see how it
> 工作，让我们做layer norm，然后我们有embedding Dimension字符或嗯

**[00:00:30]** this in practice um let's see how it
> 有embedding Dimension字符或嗯

**[00:00:30]** this in practice um let's see how it works let's do layer norm and then we
> 有embedding Dimension字符或嗯value属性

**[00:00:30]** works let's do layer norm and then we
> value属性

**[00:00:30]** works let's do layer norm and then we have embedding Dimension character or um
> value属性，嗯，参数，抱歉，函数，嗯，方法

**[00:00:30]** have embedding Dimension character or um
> 嗯，参数，抱歉，函数，嗯，方法

**[00:00:30]** have embedding Dimension character or um value attribute
> 嗯，参数，抱歉，函数，嗯，方法参数，然后我们有这个，然后我们

**[00:00:30]** value attribute
> 参数，然后我们有这个，然后我们

**[00:00:30]** value attribute um argument sorry function AR um method
> 参数，然后我们有这个，然后我们把它应用到我们的，让我们做输出

**[00:00:31]** um argument sorry function AR um method
> 这里，是的，这些是我们转换后的

**[00:00:31]** um argument sorry function AR um method argument and so we have this and then we
> 这里，是的，这些是我们转换后的

**[00:00:31]** argument and so we have this and then we
> 这里，是的，这些是我们转换后的值，让我们

**[00:00:31]** argument and so we have this and then we apply it to our let's do output
> 值，让我们

**[00:00:31]** here and yeah these are our transformed
> 值，让我们实际嗯，双重检查，嗯，让我们就

**[00:00:31]** here and yeah these are our transformed
> 实际嗯，双重检查，嗯，让我们就

**[00:00:31]** here and yeah these are our transformed values let's
> 实际嗯，双重检查，嗯，让我们就叫它们temp或者只是输出

**[00:00:31]** values let's
> 叫它们temp或者只是输出

**[00:00:31]** values let's actually um double check uh let's just
> 叫它们temp或者只是输出normed之类的

**[00:00:31]** actually um double check uh let's just
> normed之类的

**[00:00:31]** actually um double check uh let's just call them temp or like just outputs
> normed之类的，好的，嗯，让我们看看它们如何

**[00:00:31]** call them temp or like just outputs
> 好的，嗯，让我们看看它们如何

**[00:00:31]** call them temp or like just outputs normed something like that
> 好的，嗯，让我们看看它们看起来如何

**[00:00:31]** normed something like that
> 看起来如何

**[00:00:31]** normed something like that that okay um and let's just see how they
> 看起来如何，再次，现在让我们双重检查它们是否

**[00:00:31]** that okay um and let's just see how they
> 再次，现在让我们双重检查它们是否

**[00:00:31]** that okay um and let's just see how they look like
> that okay um and let's just see how they look like

**[00:00:31]** look like
> look like

**[00:00:31]** look like again and now let's double check if they
> look like again and now let's double check if they

**[00:00:31]** again and now let's double check if they
> again and now let's double check if they

**[00:00:31]** again and now let's double check if they are indeed having the properties that we
> 再次确认它们是否确实具有我们期望的属性

**[00:00:31]** are indeed having the properties that we
> 确实具有我们期望的属性

**[00:00:31]** are indeed having the properties that we desired so that would be
> 确实具有我们期望的属性，所以这将是

**[00:00:31]** desired so that would be
> 期望的，所以这将是

**[00:00:31]** desired so that would be mean of one sorry mean zero and um
> 期望的，所以这将是均值为1，抱歉，均值为0，嗯

**[00:00:31]** mean of one sorry mean zero and um
> 均值为1，抱歉，均值为0，嗯

**[00:00:31]** mean of one sorry mean zero and um standard deviation and variance of
> 均值为1，抱歉，均值为0，嗯，标准差和方差为

**[00:00:31]** standard deviation and variance of
> 标准差和方差为

**[00:00:31]** standard deviation and variance of one
> 标准差和方差为1

**[00:00:31]** one
> 1

**[00:00:31]** one um one I
> 1，嗯，1，我

**[00:00:31]** um one I
> 嗯，1，我

**[00:00:31]** um one I um keep them true okay and let's double
> 嗯，1，我，嗯，保持它们为真，好的，让我们再次

**[00:00:31]** um keep them true okay and let's double
> 嗯，保持它们为真，好的，让我们再次

**[00:00:31]** um keep them true okay and let's double check so we can indeed mean zero and uh
> 嗯，保持它们为真，好的，让我们再次确认，所以我们可以确实得到均值为0，嗯

**[00:00:32]** check so we can indeed mean zero and uh
> 确认，所以我们可以确实得到均值为0，嗯

**[00:00:32]** check so we can indeed mean zero and uh standard deviation of hopefully one
> 确认，所以我们可以确实得到均值为0，嗯，标准差希望为1

**[00:00:32]** standard deviation of hopefully one
> 标准差希望为1

**[00:00:32]** standard deviation of hopefully one close to one yeah so basically one minus
> 标准差希望为1，接近1，是的，所以基本上1减去

**[00:00:32]** close to one yeah so basically one minus
> 接近1，是的，所以基本上1减去

**[00:00:32]** close to one yeah so basically one minus rounding errors um the one thing uh I
> 接近1，是的，所以基本上1减去舍入误差，嗯，有一件事，嗯，我

**[00:00:32]** rounding errors um the one thing uh I
> 舍入误差，嗯，有一件事，嗯，我

**[00:00:32]** rounding errors um the one thing uh I commented out here is this unbiased so
> 舍入误差，嗯，有一件事，嗯，我在这里注释掉的是这个无偏的，所以

**[00:00:32]** commented out here is this unbiased so
> 在这里注释掉的是这个无偏的，所以

**[00:00:32]** commented out here is this unbiased so this is just like a
> 在这里注释掉的是这个无偏的，所以这就像是一个

**[00:00:32]** this is just like a
> 这就像是一个

**[00:00:32]** this is just like a minor um detail so there's a uh if
> 这就像是一个小的，嗯，细节，所以有一个，嗯，如果

**[00:00:32]** minor um detail so there's a uh if
> 小的，嗯，细节，所以有一个，嗯，如果

**[00:00:32]** minor um detail so there's a uh if you're familiar with Statistics um um
> 小的，嗯，细节，所以有一个，嗯，如果你熟悉统计学，嗯，嗯

**[00:00:32]** you're familiar with Statistics um um
> 你熟悉统计学，嗯，嗯

**[00:00:32]** you're familiar with Statistics um um population um statistics and sample
> 你熟悉统计学，嗯，嗯，总体，嗯，统计量和样本

**[00:00:32]** population um statistics and sample
> 总体，嗯，统计量和样本

**[00:00:32]** population um statistics and sample statistics there's a concept like a
> 总体，嗯，统计量和样本统计量，有一个概念像

**[00:00:32]** statistics there's a concept like a
> 统计量，有一个概念像

**[00:00:32]** statistics there's a concept like a bessels correction term when you compute
> 统计量，有一个概念像贝塞尔校正项，当你计算

**[00:00:32]** bessels correction term when you compute
> 贝塞尔校正项，当你计算

**[00:00:32]** bessels correction term when you compute the variance or standard deviation that
> 贝塞尔校正项，当你计算方差或标准差时，你

**[00:00:32]** the variance or standard deviation that
> 方差或标准差时，你

**[00:00:32]** the variance or standard deviation that you divide instead of divide by n the
> 方差或标准差时，你除以，而不是除以n，即

**[00:00:32]** you divide instead of divide by n the
> 你除以，而不是除以n，即

**[00:00:32]** you divide instead of divide by n the sample size you divide by n minus one um
> 你除以，而不是除以n，即样本大小，你除以n减一，嗯

**[00:00:32]** sample size you divide by n minus one um
> 样本大小，你除以n减一，嗯

**[00:00:32]** sample size you divide by n minus one um so unbiased is basically unbiased true
> 样本大小，你除以n减一，嗯，所以无偏基本上是无偏的真值

**[00:00:32]** so unbiased is basically unbiased true
> 所以无偏基本上是无偏的真值

**[00:00:32]** so unbiased is basically unbiased true would be the population
> 所以无偏基本上是无偏的真值将是总体

**[00:00:32]** would be the population
> 将是总体

**[00:00:32]** would be the population statistic and um unbiased fults would be
> 将是总体统计量，嗯，无偏结果将是

**[00:00:32]** statistic and um unbiased fults would be
> 统计量，嗯，无偏结果将是

**[00:00:32]** statistic and um unbiased fults would be the sample statistics um so when we so
> 统计量，嗯，无偏结果将是样本统计量，嗯，所以当我们

**[00:00:32]** the sample statistics um so when we so
> 样本统计量，嗯，所以当我们

**[00:00:32]** the sample statistics um so when we so this would be a whole again lecture by
> 样本统计量，嗯，所以当我们，这本身又是一整堂课

**[00:00:32]** this would be a whole again lecture by
> 这本身又是一整堂课

**[00:00:32]** this would be a whole again lecture by itself I used to be a statistics
> 这本身又是一整堂课，我以前是统计学

**[00:00:32]** itself I used to be a statistics
> 我以前是统计学

**[00:00:32]** itself I used to be a statistics professor and was teaching these things
> 我以前是统计学教授，教这些东西

**[00:00:32]** professor and was teaching these things
> 教授，教这些东西

**[00:00:32]** professor and was teaching these things so but I don't want to turn this listen
> 教授，教这些东西，所以但我不想把这个讲座

**[00:00:32]** so but I don't want to turn this listen
> 所以但我不想把这个讲座

**[00:00:32]** so but I don't want to turn this listen to a statistic lecture here but what we
> 所以但我不想把这个讲座变成统计学讲座，但我们

**[00:00:33]** to a statistic lecture here but what we
> 变成统计学讲座，但我们

**[00:00:33]** to a statistic lecture here but what we um have here is we could technically
> 变成统计学讲座，但我们，嗯，这里有的是，我们可以技术上

**[00:00:33]** um have here is we could technically
> 嗯，这里有的是，我们可以技术上

**[00:00:33]** um have here is we could technically let's see if that works um set this to
> 嗯，这里有的是，我们可以技术上，让我们看看是否有效，嗯，把这个设为

**[00:00:33]** let's see if that works um set this to
> 让我们看看是否有效，嗯，把这个设为

**[00:00:33]** let's see if that works um set this to false you can see the values are
> 让我们看看是否有效，嗯，把这个设为false，你可以看到数值是

**[00:00:33]** false you can see the values are
> false，你可以看到数值是

**[00:00:33]** false you can see the values are slightly different and that is because
> false，你可以看到数值略有不同，这是因为

**[00:00:33]** slightly different and that is because
> 略有不同，这是因为

**[00:00:33]** slightly different and that is because we have a very small sample size so here
> 略有不同，这是因为我们有一个非常小的样本大小，所以这里

**[00:00:33]** we have a very small sample size so here
> 我们有一个非常小的样本大小，所以这里

**[00:00:33]** we have a very small sample size so here if I increase this um sample size make
> 我们有一个非常小的样本大小，所以这里如果我增加这个，嗯，样本大小，让

**[00:00:33]** if I increase this um sample size make
> 如果我增加这个，嗯，样本大小，让

**[00:00:33]** if I increase this um sample size make it really large let's say um oops it
> 如果我增加这个，嗯，样本大小，让它变得非常大，比如说，嗯，哎呀，它

**[00:00:33]** it really large let's say um oops it
> 让它变得非常大，比如说，嗯，哎呀，它

**[00:00:33]** it really large let's say um oops it should be then
> 让它变得非常大，比如说，嗯，哎呀，它应该然后

**[00:00:33]** should be then
> 应该然后

**[00:00:33]** should be then also very
> 应该然后也非常

**[00:00:33]** also very
> 也非常

**[00:00:33]** also very large well I should also make this
> 也非常大，嗯，我也应该把这个

**[00:00:33]** large well I should also make this
> 大，嗯，我也应该把这个

**[00:00:33]** large well I should also make this actually large okay like this oh this is
> 好的，我应该把这个做得更大一些，像这样，哦，这

**[00:00:33]** actually large okay like this oh this is
> 确实很大了，像这样，哦，这

**[00:00:33]** actually large okay like this oh this is very slow now uh shouldn't have done
> 确实很大了，像这样，哦，这现在非常慢，呃，不该这么做的

**[00:00:33]** very slow now uh shouldn't have done
> 现在非常慢，呃，不该这么做的

**[00:00:33]** very slow now uh shouldn't have done that but you should technically see can
> 现在非常慢，呃，不该这么做的，但理论上你应该能看到

**[00:00:33]** that but you should technically see can
> 但理论上你应该能看到

**[00:00:33]** that but you should technically see can I interrupt
> 但理论上你应该能看到，我能打断一下吗

**[00:00:33]** I interrupt
> 我能打断一下吗

**[00:00:33]** I interrupt this not responding made it maybe too
> 我能打断一下吗，这个无响应状态可能让它变得太

**[00:00:33]** this not responding made it maybe too
> 这个无响应状态可能让它变得太

**[00:00:33]** this not responding made it maybe too large um apologies I hope it doesn't
> 这个无响应状态可能让它变得太大了，抱歉，希望它不会

**[00:00:33]** large um apologies I hope it doesn't
> 太大了，抱歉，希望它不会

**[00:00:33]** large um apologies I hope it doesn't crash the recording um but in any case
> 太大了，抱歉，希望它不会导致录制崩溃，但无论如何

**[00:00:33]** crash the recording um but in any case
> 导致录制崩溃，但无论如何

**[00:00:33]** crash the recording um but in any case um if you make this really large you
> 导致录制崩溃，但无论如何，如果你把这个设得非常大，你

**[00:00:33]** um if you make this really large you
> 如果你把这个设得非常大，你

**[00:00:33]** um if you make this really large you should see this should be one basically
> 如果你把这个设得非常大，你应该看到这个基本上等于1

**[00:00:33]** should see this should be one basically
> 应该看到这个基本上等于1

**[00:00:33]** should see this should be one basically so this should not um this this is only
> 应该看到这个基本上等于1，所以这不应该，呃，这个只会在

**[00:00:33]** so this should not um this this is only
> 所以这不应该，呃，这个只会在

**[00:00:33]** so this should not um this this is only making a difference if you have very
> 所以这不应该，呃，这个只会在你的网络中有非常小的值时

**[00:00:33]** making a difference if you have very
> 在你的网络中有非常小的值时

**[00:00:33]** making a difference if you have very small values and in your network
> 在你的网络中有非常小的值时才会产生影响

**[00:00:33]** small values and in your network
> 才会产生影响

**[00:00:33]** small values and in your network training as I mentioned before we have
> 才会产生影响，正如我之前提到的，在训练中我们

**[00:00:34]** training as I mentioned before we have
> 在训练中我们

**[00:00:34]** training as I mentioned before we have an embedding dimension of 768 so it
> 在训练中我们有一个768维的embedding，所以它

**[00:00:34]** an embedding dimension of 768 so it
> 有一个768维的embedding，所以它

**[00:00:34]** an embedding dimension of 768 so it shouldn't really make a difference at
> 有一个768维的embedding，所以它根本不会产生任何影响

**[00:00:34]** shouldn't really make a difference at
> 根本不会产生任何影响

**[00:00:34]** shouldn't really make a difference at all whether we use uh one or not oh it
> 根本不会产生任何影响，无论我们是否使用这个，哦，它

**[00:00:34]** all whether we use uh one or not oh it
> 无论我们是否使用这个，哦，它

**[00:00:34]** all whether we use uh one or not oh it actually worked let's just maybe try
> 无论我们是否使用这个，哦，它居然生效了，我们不妨再试一次

**[00:00:34]** again um
> 再试一次，呃

**[00:00:34]** yeah
> 是的

**[00:00:34]** okay yeah you can see here it's now
> 好的，你可以看到现在它

**[00:00:34]** okay yeah you can see here it's now
> 好的，你可以看到现在它

**[00:00:34]** okay yeah you can see here it's now almost one so the bigger the input is
> 好的，你可以看到现在它几乎等于1，所以输入越大

**[00:00:34]** almost one so the bigger the input is
> 几乎等于1，所以输入越大

**[00:00:34]** almost one so the bigger the input is the less of a difference this um
> 几乎等于1，所以输入越大，这个修正项的影响就越小

**[00:00:34]** the less of a difference this um
> 这个修正项的影响就越小

**[00:00:34]** the less of a difference this um correction term makes and it's really
> 这个修正项的影响就越小，它实际上

**[00:00:34]** correction term makes and it's really
> 它实际上

**[00:00:34]** correction term makes and it's really just an implementation detail it does
> 它实际上只是一个实现细节，对于常规的

**[00:00:34]** just an implementation detail it does
> 只是一个实现细节，对于常规的

**[00:00:34]** just an implementation detail it does not matter at all um for regular
> 只是一个实现细节，对于常规的embedding大小来说完全无关紧要

**[00:00:34]** not matter at all um for regular
> 完全无关紧要

**[00:00:34]** not matter at all um for regular embedding sizes the reason why I'm
> 完全无关紧要，我在这里显式设置它的原因是

**[00:00:34]** embedding sizes the reason why I'm
> 我在这里显式设置它的原因是

**[00:00:34]** embedding sizes the reason why I'm setting this explicitly here is that's
> 我在这里显式设置它的原因是，这就是TensorFlow，呃，PyTorch，抱歉

**[00:00:34]** setting this explicitly here is that's
> 这就是TensorFlow，呃，PyTorch，抱歉

**[00:00:34]** setting this explicitly here is that's what the tensor FL well P torch sorry
> 这就是TensorFlow，呃，PyTorch，抱歉，这就是GPT团队在

**[00:00:34]** what the tensor FL well P torch sorry
> 这就是GPT团队在

**[00:00:34]** what the tensor FL well P torch sorry sorry what the GPT team did when they
> 这就是GPT团队在使用TensorFlow训练GPT时所做的

**[00:00:34]** sorry what the GPT team did when they
> 使用TensorFlow训练GPT时所做的

**[00:00:34]** sorry what the GPT team did when they used tensor Flow To Train GPT and here
> 使用TensorFlow训练GPT时所做的，而在这里我们用PyTorch重新实现

**[00:00:34]** used tensor Flow To Train GPT and here
> 而在这里我们用PyTorch重新实现

**[00:00:34]** used tensor Flow To Train GPT and here we are reimplementing things in pych and
> 而在这里我们用PyTorch重新实现，PyTorch的beta值与TensorFlow不同

**[00:00:34]** we are reimplementing things in pych and
> PyTorch的beta值与TensorFlow不同

**[00:00:34]** we are reimplementing things in pych and pych has different B values than T of
> PyTorch的beta值与TensorFlow不同，所以我才这样设置

**[00:00:34]** pych has different B values than T of
> 所以我才这样设置

**[00:00:34]** pych has different B values than T of low so that's why I'm just setting this
> 所以我才这样设置，以模仿原始OpenAI GPT团队在

**[00:00:35]** low so that's why I'm just setting this
> 以模仿原始OpenAI GPT团队在

**[00:00:35]** low so that's why I'm just setting this to mimic the same behavior that the
> 以模仿原始OpenAI GPT团队在训练模型时所使用的相同行为

**[00:00:35]** to mimic the same behavior that the
> 训练模型时所使用的相同行为

**[00:00:35]** to mimic the same behavior that the original open GPD team did when or used
> 训练模型时所使用的相同行为，长话短说，如果这个unbiased设置让你感到困惑

**[00:00:35]** original open GPD team did when or used
> 长话短说，如果这个unbiased设置让你感到困惑

**[00:00:35]** original open GPD team did when or used when they were training the model long
> 长话短说，如果这个unbiased设置让你感到困惑，它只是一个实现细节

**[00:00:35]** when they were training the model long
> 它只是一个实现细节

**[00:00:35]** when they were training the model long story short if this was all bit
> 它只是一个实现细节，你不需要太担心

**[00:00:35]** story short if this was all bit
> 你不需要太担心

**[00:00:35]** story short if this was all bit confusing with this unbiased setting you
> 你不需要太担心，它并不那么重要，主要信息

**[00:00:35]** confusing with this unbiased setting you
> 它并不那么重要，主要信息

**[00:00:35]** confusing with this unbiased setting you know it's just an implementation detail
> 它并不那么重要，主要信息是我们在对值进行归一化

**[00:00:35]** know it's just an implementation detail
> know it's just an implementation detail

**[00:00:35]** know it's just an implementation detail I would not worry about it too much it's
> know it's just an implementation detail I would not worry about it too much it's

**[00:00:35]** I would not worry about it too much it's
> I would not worry about it too much it's

**[00:00:35]** I would not worry about it too much it's um not that important the main message
> I would not worry about it too much it's um not that important the main message

**[00:00:35]** um not that important the main message
> um not that important the main message

**[00:00:35]** um not that important the main message here is that we are normalizing values
> um not that important the main message here is that we are normalizing values

**[00:00:35]** here is that we are normalizing values
> 这里是我们正在对数值进行归一化

**[00:00:35]** here is that we are normalizing values so that they have mean zero and variance
> 这里是我们正在对数值进行归一化，使其均值为零且方差为

**[00:00:35]** so that they have mean zero and variance
> 使其均值为零且方差为

**[00:00:35]** so that they have mean zero and variance one which will give the network nicer
> 使其均值为零且方差为一，这将给网络带来更好的

**[00:00:35]** one which will give the network nicer
> 一，这将给网络带来更好的

**[00:00:35]** one which will give the network nicer properties the values nicer properties
> 一，这将给网络带来更好的性质，数值更好的性质

**[00:00:35]** properties the values nicer properties
> 性质，数值更好的性质

**[00:00:35]** properties the values nicer properties during the optimization and this is one
> 性质，数值更好的性质在优化过程中，这是

**[00:00:35]** during the optimization and this is one
> 在优化过程中，这是

**[00:00:35]** during the optimization and this is one building block uh the normalization
> 在优化过程中，这是一个构建模块，归一化

**[00:00:35]** building block uh the normalization
> 构建模块，归一化

**[00:00:35]** building block uh the normalization layer in the Transformer Network um the
> 构建模块，归一化层在Transformer网络中

**[00:00:35]** layer in the Transformer Network um the
> 层在Transformer网络中

**[00:00:35]** layer in the Transformer Network um the GPT Network here the llm so we are using
> 层在Transformer网络中，这里是GPT网络，LLM，所以我们正在使用

**[00:00:35]** GPT Network here the llm so we are using
> GPT网络，LLM，所以我们正在使用

**[00:00:35]** GPT Network here the llm so we are using it in several places we used it
> GPT网络，LLM，所以我们正在使用它在多个地方，我们使用了它

**[00:00:35]** it in several places we used it
> 它在多个地方，我们使用了它

**[00:00:35]** it in several places we used it in this place the final Norm uh here and
> 它在多个地方，我们使用了它在这个位置，最终的Norm层，这里和

**[00:00:35]** in this place the final Norm uh here and
> 在这个位置，最终的Norm层，这里和

**[00:00:35]** in this place the final Norm uh here and we are also going to use it in the
> 在这个位置，最终的Norm层，这里和，我们还将使用它在

**[00:00:35]** we are also going to use it in the
> 我们还将使用它在

**[00:00:35]** we are also going to use it in the Transformer block when we are
> 我们还将使用它在Transformer block中，当我们

**[00:00:35]** Transformer block when we are
> Transformer block中，当我们

**[00:00:35]** Transformer block when we are implementing the Transformer block later
> Transformer block中，当我们稍后实现Transformer block时

**[00:00:35]** implementing the Transformer block later
> 实现Transformer block稍后

**[00:00:35]** implementing the Transformer block later anyways I hope this was useful and
> 实现Transformer block稍后，无论如何，我希望这有用并且

**[00:00:35]** anyways I hope this was useful and
> 无论如何，我希望这有用并且

**[00:00:35]** anyways I hope this was useful and explained a bit what layer normalization
> 无论如何，我希望这有用并且解释了一些关于layer normalization

**[00:00:35]** explained a bit what layer normalization
> 解释了一些关于layer normalization

**[00:00:35]** explained a bit what layer normalization is
> 解释了一些关于layer normalization是什么

**[00:00:35]** is
> 是

**[00:00:35]** is and in the next video we will tackle
> 是，在下一个视频中，我们将处理

**[00:00:35]** and in the next video we will tackle
> 在下一个视频中，我们将处理

**[00:00:35]** and in the next video we will tackle some of the other
> 在下一个视频中，我们将处理一些其他的

**[00:00:36]** placeholders let's now talk about
> 占位符，现在让我们讨论

**[00:00:36]** placeholders let's now talk about
> 占位符，现在让我们讨论

**[00:00:36]** placeholders let's now talk about implementing a feed forward network with
> 占位符，现在让我们讨论实现一个feed forward network，使用

**[00:00:36]** implementing a feed forward network with
> 实现一个feed forward network，使用

**[00:00:36]** implementing a feed forward network with G activations here the feed forward
> 实现一个feed forward network，使用G激活函数，这里的feed forward

**[00:00:36]** G activations here the feed forward
> G激活函数，这里的feed forward

**[00:00:36]** G activations here the feed forward network is a small new network contained
> G激活函数，这里的feed forward network是一个小型神经网络，包含

**[00:00:36]** network is a small new network contained
> network是一个小型神经网络，包含

**[00:00:36]** network is a small new network contained in the larger new network the llm and
> network是一个小型神经网络，包含在更大的神经网络LLM中，并且

**[00:00:36]** in the larger new network the llm and
> 在更大的神经网络LLM中，并且

**[00:00:36]** in the larger new network the llm and the g activations are yeah nonlinear
> 在更大的神经网络LLM中，并且G激活函数是，是的，非线性

**[00:00:36]** the g activations are yeah nonlinear
> G激活函数是，是的，非线性

**[00:00:36]** the g activations are yeah nonlinear activation functions so it's essentially
> G激活函数是，是的，非线性激活函数，所以它本质上是

**[00:00:36]** activation functions so it's essentially
> 激活函数，所以它本质上是

**[00:00:36]** activation functions so it's essentially a concept in deep learning where we have
> 激活函数，所以它本质上是深度学习中的一个概念，我们拥有

**[00:00:36]** a concept in deep learning where we have
> 深度学习中的一个概念，我们拥有

**[00:00:36]** a concept in deep learning where we have linear layers and nonlinear activation
> 深度学习中的一个概念，我们拥有线性层和非线性激活

**[00:00:36]** linear layers and nonlinear activation
> 线性层和非线性激活

**[00:00:36]** linear layers and nonlinear activation function
> 线性层和非线性激活函数

**[00:00:36]** function
> 函数

**[00:00:36]** function so just before we go into these details
> 函数，所以在我们深入这些细节之前

**[00:00:36]** so just before we go into these details
> 所以在我们深入这些细节之前

**[00:00:36]** so just before we go into these details in the grand scheme of things where we
> 所以在我们深入这些细节之前，从全局角度来看，我们

**[00:00:36]** in the grand scheme of things where we
> 从全局角度来看，我们

**[00:00:36]** in the grand scheme of things where we are so our goal is still to implement
> 从全局角度来看，我们，我们的目标仍然是实现

**[00:00:36]** are so our goal is still to implement
> 我们的目标仍然是实现

**[00:00:36]** are so our goal is still to implement the final GPT architecture that we want
> 我们的目标仍然是实现最终的GPT架构，我们想要

**[00:00:36]** the final GPT architecture that we want
> 最终的GPT架构，我们想要

**[00:00:36]** the final GPT architecture that we want to train in the next chapter um however
> 最终的GPT架构，我们想要在下一章中训练，嗯，然而

**[00:00:36]** to train in the next chapter um however
> 在下一章中训练，嗯，然而

**[00:00:36]** to train in the next chapter um however the GPT architecture consists of
> 在下一章中训练，嗯，然而GPT架构由

**[00:00:36]** the GPT architecture consists of
> GPT架构由

**[00:00:36]** the GPT architecture consists of multiple building blocks that we are
> GPT架构由多个构建模块组成，我们正在

**[00:00:36]** multiple building blocks that we are
> 多个构建模块组成，我们正在

**[00:00:36]** multiple building blocks that we are working through sequentially here so we
> 多个构建模块组成，我们正在按顺序逐步处理，所以这里我们

**[00:00:36]** working through sequentially here so we
> 按顺序逐步处理，所以这里我们

**[00:00:36]** working through sequentially here so we already covered layer normalization so
> 按顺序逐步处理，所以这里我们已经涵盖了layer normalization，所以

**[00:00:36]** already covered layer normalization so
> 已经涵盖了layer normalization，所以

**[00:00:36]** already covered layer normalization so first we yeah coded this dummy
> 已经涵盖了layer normalization，所以首先我们，是的，编码了这个虚拟的

**[00:00:36]** first we yeah coded this dummy
> 首先我们，是的，编码了这个虚拟的

**[00:00:36]** first we yeah coded this dummy architecture then we covered layer
> 首先我们，是的，编码了这个虚拟的架构，然后我们涵盖了layer

**[00:00:36]** architecture then we covered layer
> 架构，然后我们涵盖了layer

**[00:00:36]** architecture then we covered layer normalization where we were yeah coding
> 架构，然后我们涵盖了layer normalization，我们正在，是的，编码

**[00:00:36]** normalization where we were yeah coding
> normalization，我们正在，是的，编码

**[00:00:36]** normalization where we were yeah coding this normaliz ation layer that goes
> normalization，我们正在，是的，编码这个归一化层，它进入

**[00:00:36]** this normaliz ation layer that goes
> 这个归一化层，它进入

**[00:00:36]** this normaliz ation layer that goes between other neuron Network layers and
> 这个归一化层位于其他神经网络层之间

**[00:00:37]** between other neuron Network layers and
> 位于其他神经网络层之间

**[00:00:37]** between other neuron Network layers and now we are going to code the G
> 位于其他神经网络层之间，现在我们要编写G

**[00:00:37]** now we are going to code the G
> 现在我们要编写G

**[00:00:37]** now we are going to code the G activation that also goes between yeah
> 现在我们要编写G激活函数，它也位于

**[00:00:37]** activation that also goes between yeah
> 激活函数也位于

**[00:00:37]** activation that also goes between yeah uh linear units or yeah neuron Network
> 激活函数也位于线性单元或神经网络

**[00:00:37]** uh linear units or yeah neuron Network
> 线性单元或神经网络

**[00:00:37]** uh linear units or yeah neuron Network layers so why do we need nonlinear
> 线性单元或神经网络层之间，那么为什么我们首先需要非线性

**[00:00:37]** layers so why do we need nonlinear
> 层之间，那么为什么我们首先需要非线性

**[00:00:37]** layers so why do we need nonlinear activations in the first place I was
> 层之间，那么为什么我们首先需要非线性激活函数？我刚刚

**[00:00:37]** activations in the first place I was
> 激活函数？我刚刚

**[00:00:37]** activations in the first place I was just digging out a older figure from a
> 激活函数？我刚刚从以前当教授时的一次大学讲座中

**[00:00:37]** just digging out a older figure from a
> 从以前当教授时的一次大学讲座中

**[00:00:37]** just digging out a older figure from a university lecture um I gave when I was
> 从以前当教授时的一次大学讲座中翻出一张旧图

**[00:00:37]** university lecture um I gave when I was
> 翻出一张旧图

**[00:00:37]** university lecture um I gave when I was a professor back in the day so here um
> 翻出一张旧图，所以这里

**[00:00:37]** a professor back in the day so here um
> 所以这里

**[00:00:37]** a professor back in the day so here um this is like a multi-layer perceptron
> 所以这里是一个多层感知机

**[00:00:37]** this is like a multi-layer perceptron
> 这是一个多层感知机

**[00:00:37]** this is like a multi-layer perceptron architecture that consists of of
> 这是一个多层感知机架构，它由

**[00:00:37]** architecture that consists of of
> 架构，它由

**[00:00:37]** architecture that consists of of multiple layers so we have the inputs a
> 架构，它由多个层组成，所以我们有输入层

**[00:00:37]** multiple layers so we have the inputs a
> 多个层组成，所以我们有输入层

**[00:00:37]** multiple layers so we have the inputs a first hidden layer and a second hidden
> 多个层组成，所以我们有输入层、第一个隐藏层和第二个隐藏层

**[00:00:37]** first hidden layer and a second hidden
> 第一个隐藏层和第二个隐藏层

**[00:00:37]** first hidden layer and a second hidden layer and then the output layer and so
> 第一个隐藏层和第二个隐藏层，然后是输出层

**[00:00:37]** layer and then the output layer and so
> 然后是输出层

**[00:00:37]** layer and then the output layer and so this is a classic multi-layer perceptron
> 然后是输出层，所以这是一个经典的多层感知机

**[00:00:37]** this is a classic multi-layer perceptron
> 这是一个经典的多层感知机

**[00:00:37]** this is a classic multi-layer perceptron that you may have seen in the con in the
> 这是一个经典的多层感知机，你可能在深度学习背景下见过

**[00:00:37]** that you may have seen in the con in the
> 你可能在深度学习背景下见过

**[00:00:37]** that you may have seen in the con in the context of deep learning and what's not
> 你可能在深度学习背景下见过，这里没有显示的是非线性激活

**[00:00:37]** context of deep learning and what's not
> 这里没有显示的是非线性激活

**[00:00:37]** context of deep learning and what's not shown here are the nonlinear activation
> 这里没有显示的是非线性激活函数，所以我往下滚动一点

**[00:00:37]** shown here are the nonlinear activation
> 函数，所以我往下滚动一点

**[00:00:37]** shown here are the nonlinear activation functions so um I scroll a bit down so I
> 函数，所以我往下滚动一点，我通常在考试讲座中讲到

**[00:00:37]** functions so um I scroll a bit down so I
> 我通常在考试讲座中讲到

**[00:00:37]** functions so um I scroll a bit down so I had it usually as an exam lecture what
> 我通常在考试讲座中讲到如果没有非线性激活函数会发生什么

**[00:00:37]** had it usually as an exam lecture what
> 如果没有非线性激活函数会发生什么

**[00:00:37]** had it usually as an exam lecture what happens if we don't have the nonlinear
> 如果没有非线性激活函数会发生什么，简单来说

**[00:00:37]** happens if we don't have the nonlinear
> 简单来说

**[00:00:37]** happens if we don't have the nonlinear activation functions just uh briefly the
> 简单来说，这里的目标，也是大型语言模型的总体目标

**[00:00:37]** activation functions just uh briefly the
> 这里的目标，也是大型语言模型的总体目标

**[00:00:37]** activation functions just uh briefly the goal here also in the grand scheme of
> 这里的目标，也是大型语言模型的总体目标，是从输入数据中提取特征

**[00:00:37]** goal here also in the grand scheme of
> 是从输入数据中提取特征

**[00:00:38]** goal here also in the grand scheme of things of the large language model is to
> 是从输入数据中提取特征，这些特征经过神经网络层

**[00:00:38]** things of the large language model is to
> 这些特征经过神经网络层

**[00:00:38]** things of the large language model is to extract uh features from the input data
> 这些特征经过神经网络层，最终我们希望生成输出

**[00:00:38]** extract uh features from the input data
> 最终我们希望生成输出

**[00:00:38]** extract uh features from the input data and the features go through the new
> 最终我们希望生成输出，对于特征提取，我们希望

**[00:00:38]** and the features go through the new
> 对于特征提取，我们希望

**[00:00:38]** and the features go through the new network layers and at the end of the day
> 对于特征提取，我们希望网络学习到有用的东西

**[00:00:38]** network layers and at the end of the day
> 希望网络学习到有用的东西

**[00:00:38]** network layers and at the end of the day we want to generate outputs and um for
> 希望网络学习到有用的东西，帮助网络生成正确的输出

**[00:00:38]** we want to generate outputs and um for
> 帮助网络生成正确的输出

**[00:00:38]** we want to generate outputs and um for the feature extraction we want the
> 帮助网络生成正确的输出，理想情况下这是一个玩具问题

**[00:00:38]** the feature extraction we want the
> 理想情况下这是一个玩具问题

**[00:00:38]** the feature extraction we want the network to learn something useful that
> 理想情况下这是一个玩具问题，但如果我们有类似这样的问题

**[00:00:38]** network to learn something useful that
> 但如果我们有类似这样的问题

**[00:00:38]** network to learn something useful that helps the network to generate the
> 但如果我们有类似这样的问题，这是一个分类任务

**[00:00:38]** helps the network to generate the
> 这是一个分类任务

**[00:00:38]** helps the network to generate the correct output now ideally this is a toy
> 这是一个分类任务，与神经网络或LLM无关

**[00:00:38]** correct output now ideally this is a toy
> 与神经网络或LLM无关

**[00:00:38]** correct output now ideally this is a toy problem but if we have something like
> 与神经网络或LLM无关，但为了说明

**[00:00:38]** problem but if we have something like
> 但为了说明

**[00:00:38]** problem but if we have something like this and this is a classification task
> 但为了说明目的，可以把它看作一个具有挑战性的

**[00:00:38]** this and this is a classification task
> 目的，可以把它看作一个具有挑战性的

**[00:00:38]** this and this is a classification task has nothing to do with new networks or
> 目的，可以把它看作一个具有挑战性的分类问题

**[00:00:38]** has nothing to do with new networks or
> 分类问题

**[00:00:38]** has nothing to do with new networks or sorry with l M but for illustration
> 分类问题，网络需要根据某些特征检测某个物体是正方形还是圆形

**[00:00:38]** sorry with l M but for illustration
> 网络需要根据某些特征检测某个物体是正方形还是圆形

**[00:00:38]** sorry with l M but for illustration purposes think of this as a challenging
> 网络需要根据某些特征检测某个物体是正方形还是圆形

**[00:00:38]** purposes think of this as a challenging
> purposes think of this as a challenging

**[00:00:38]** purposes think of this as a challenging um classification problem where the
> purposes think of this as a challenging um classification problem where the

**[00:00:38]** um classification problem where the
> um classification problem where the

**[00:00:38]** um classification problem where the network should detect whether something
> um classification problem where the network should detect whether something

**[00:00:38]** network should detect whether something
> network should detect whether something

**[00:00:38]** network should detect whether something is a square or a circle based on some
> network should detect whether something is a square or a circle based on some

**[00:00:38]** is a square or a circle based on some
> is a square or a circle based on some

**[00:00:38]** is a square or a circle based on some numeric
> 是基于某些数值来判断是正方形还是圆形

**[00:00:38]** numeric
> 数值

**[00:00:38]** numeric input and ideally the network would
> 数值输入，理想情况下网络会

**[00:00:38]** input and ideally the network would
> 输入，理想情况下网络会

**[00:00:38]** input and ideally the network would learn something like this like a
> 输入，理想情况下网络会学习到类似这样的东西，比如一个

**[00:00:38]** learn something like this like a
> 学习到类似这样的东西，比如一个

**[00:00:38]** learn something like this like a decision boundary where it can separate
> 学习到类似这样的东西，比如一个决策边界，能够区分

**[00:00:38]** decision boundary where it can separate
> 决策边界，能够区分

**[00:00:38]** decision boundary where it can separate these um squares from these um AES so it
> 决策边界，能够区分这些正方形和这些圆形，所以它

**[00:00:38]** these um squares from these um AES so it
> 这些正方形和这些圆形，所以它

**[00:00:38]** these um squares from these um AES so it look a bit different from here but the
> 这些正方形和这些圆形，所以它看起来和这里有点不同，但

**[00:00:38]** look a bit different from here but the
> 看起来和这里有点不同，但

**[00:00:38]** look a bit different from here but the goal is essentially separating two types
> 看起来和这里有点不同，但目标本质上是区分两种类型的

**[00:00:38]** goal is essentially separating two types
> 目标本质上是区分两种类型的

**[00:00:38]** goal is essentially separating two types of inputs here and if we don't have
> 目标本质上是区分两种类型的输入，如果我们没有

**[00:00:39]** of inputs here and if we don't have
> 输入，如果我们没有

**[00:00:39]** of inputs here and if we don't have these nonlinear activation functions
> 输入，如果我们没有这些非线性激活函数

**[00:00:39]** these nonlinear activation functions
> 这些非线性激活函数

**[00:00:39]** these nonlinear activation functions what happens is essentially that we only
> 这些非线性激活函数，那么本质上发生的是，我们只有

**[00:00:39]** what happens is essentially that we only
> 那么本质上发生的是，我们只有

**[00:00:39]** what happens is essentially that we only have linear layers and the combination
> 那么本质上发生的是，我们只有线性层，而线性层的组合

**[00:00:39]** have linear layers and the combination
> 线性层，而线性层的组合

**[00:00:39]** have linear layers and the combination of linear layers is a linear function
> 线性层，而线性层的组合也是一个线性函数

**[00:00:39]** of linear layers is a linear function
> 也是一个线性函数

**[00:00:39]** of linear layers is a linear function too so the network will only learn
> 也是一个线性函数，所以网络只会学习

**[00:00:39]** too so the network will only learn
> 所以网络只会学习

**[00:00:39]** too so the network will only learn linear functions and then what it means
> 所以网络只会学习线性函数，这意味着

**[00:00:39]** linear functions and then what it means
> 线性函数，这意味着

**[00:00:39]** linear functions and then what it means is the network will never really learn
> 线性函数，这意味着网络永远无法真正学习

**[00:00:39]** is the network will never really learn
> 网络永远无法真正学习

**[00:00:39]** is the network will never really learn to extract useful information from the
> 网络永远无法真正学习从输入中提取有用信息

**[00:00:39]** to extract useful information from the
> 从输入中提取有用信息

**[00:00:39]** to extract useful information from the input so it's kind of like limiting the
> 从输入中提取有用信息，所以这有点像严重限制了

**[00:00:39]** input so it's kind of like limiting the
> 所以这有点像严重限制了

**[00:00:39]** input so it's kind of like limiting the network severely if we don't have
> 所以这有点像严重限制了网络，如果我们没有

**[00:00:39]** network severely if we don't have
> 网络，如果我们没有

**[00:00:39]** network severely if we don't have nonlinear activation functions and so
> 网络，如果我们没有非线性激活函数，那么

**[00:00:39]** nonlinear activation functions and so
> 非线性激活函数，那么

**[00:00:39]** nonlinear activation functions and so nonline activation functions they
> 非线性激活函数，那么非线性激活函数有

**[00:00:39]** nonline activation functions they
> 非线性激活函数有

**[00:00:39]** nonline activation functions they a whole bunch of them so um the sigmoid
> 非线性激活函数有一大堆，比如sigmoid

**[00:00:39]** a whole bunch of them so um the sigmoid
> 有一大堆，比如sigmoid

**[00:00:39]** a whole bunch of them so um the sigmoid one is a classic one there's a 10 H
> 有一大堆，比如sigmoid是一个经典的，还有tanh

**[00:00:39]** one is a classic one there's a 10 H
> 是一个经典的，还有tanh

**[00:00:39]** one is a classic one there's a 10 H function hard ton age and I think I even
> 是一个经典的，还有tanh函数、hard tanh，我想我甚至

**[00:00:39]** function hard ton age and I think I even
> 函数、hard tanh，我想我甚至

**[00:00:39]** function hard ton age and I think I even have like a fun figure somewhere here
> 函数、hard tanh，我想我甚至在这里某个地方有一张有趣的图

**[00:00:39]** have like a fun figure somewhere here
> 在这里某个地方有一张有趣的图

**[00:00:39]** have like a fun figure somewhere here there was even like this meme where
> 在这里某个地方有一张有趣的图，甚至还有一个梗图，其中

**[00:00:39]** there was even like this meme where
> 甚至还有一个梗图，其中

**[00:00:39]** there was even like this meme where someone was like mapping nonlinear
> 甚至还有一个梗图，有人把非线性激活函数映射成舞蹈动作

**[00:00:39]** someone was like mapping nonlinear
> 有人把非线性激活函数映射成舞蹈动作

**[00:00:39]** someone was like mapping nonlinear activation functions to dance moves and
> 有人把非线性激活函数映射成舞蹈动作，等等，所以有各种各样的

**[00:00:39]** activation functions to dance moves and
> 等等，所以有各种各样的

**[00:00:39]** activation functions to dance moves and so forth and so there's a whole bunch of
> 等等，所以有各种各样的非线性激活函数，而我们今天要讨论的

**[00:00:39]** so forth and so there's a whole bunch of
> 非线性激活函数，而我们今天要讨论的

**[00:00:39]** so forth and so there's a whole bunch of nonlinear activation functions and the
> 非线性激活函数，而我们今天要讨论的这个甚至还不在这张图上，它

**[00:00:39]** nonlinear activation functions and the
> 这个甚至还不在这张图上，它

**[00:00:39]** nonlinear activation functions and the one that we are going to talk about
> 这个甚至还不在这张图上，它叫做G，它与ReLU有些关联

**[00:00:39]** one that we are going to talk about
> 叫做G，它与ReLU有些关联

**[00:00:39]** one that we are going to talk about today is not even on here yet it's
> 叫做G，它与ReLU有些关联，但又不完全一样，我们会在

**[00:00:39]** today is not even on here yet it's
> 但又不完全一样，我们会在

**[00:00:39]** today is not even on here yet it's called G which is somewhat related to
> 但又不完全一样，我们会在本视频中看到它们是如何关联的

**[00:00:39]** called G which is somewhat related to
> 本视频中看到它们是如何关联的

**[00:00:39]** called G which is somewhat related to reu but also not quite and we will see
> 本视频中看到它们是如何关联的，所以回到一些

**[00:00:40]** reu but also not quite and we will see
> 所以回到一些

**[00:00:40]** reu but also not quite and we will see uh in this video yeah how how things are
> 所以回到一些这里的图，再次强调，非线性激活函数

**[00:00:40]** uh in this video yeah how how things are
> 这里的图，再次强调，非线性激活函数

**[00:00:40]** uh in this video yeah how how things are related so going back to some of the
> 这里的图，再次强调，非线性激活函数的位置是，如果我们有一个神经

**[00:00:40]** related so going back to some of the
> 的位置是，如果我们有一个神经

**[00:00:40]** related so going back to some of the figures here um again where the
> 的位置是，如果我们有一个神经网络，并且我们有多个线性

**[00:00:40]** figures here um again where the
> 网络，并且我们有多个线性

**[00:00:40]** figures here um again where the nonlinear activation function fit fits
> 网络，并且我们有多个线性层，G函数就位于

**[00:00:40]** nonlinear activation function fit fits
> 层，G函数就位于

**[00:00:40]** nonlinear activation function fit fits in is if we have a neur
> 层，G函数就位于它们之间

**[00:00:40]** in is if we have a neur
> in is if we have a neur

**[00:00:40]** in is if we have a neur network and we have multiple linear
> in is if we have a neur network and we have multiple linear

**[00:00:40]** network and we have multiple linear
> network and we have multiple linear

**[00:00:40]** network and we have multiple linear layers the G function sits between
> network and we have multiple linear layers the G function sits between

**[00:00:40]** layers the G function sits between
> layers the G function sits between

**[00:00:40]** layers the G function sits between linear layers so it's usually a linear
> G函数位于线性层之间，因此它通常是一个线性

**[00:00:40]** linear layers so it's usually a linear
> 线性层，因此它通常是一个线性

**[00:00:40]** linear layers so it's usually a linear layer followed by a nonlinear activation
> 线性层，因此它通常是一个线性层后接一个非线性激活

**[00:00:40]** layer followed by a nonlinear activation
> 层后接一个非线性激活

**[00:00:40]** layer followed by a nonlinear activation function um you can also see that here
> 层后接一个非线性激活函数，嗯，你也可以在这里看到

**[00:00:40]** function um you can also see that here
> 函数，嗯，你也可以在这里看到

**[00:00:40]** function um you can also see that here where we have a linear layer and then
> 函数，嗯，你也可以在这里看到，我们有一个线性层，然后

**[00:00:40]** where we have a linear layer and then
> 我们有一个线性层，然后

**[00:00:40]** where we have a linear layer and then the nonlinear activation function and
> 我们有一个线性层，然后是非线性激活函数，以及

**[00:00:40]** the nonlinear activation function and
> 非线性激活函数，以及

**[00:00:40]** the nonlinear activation function and this whole thing here a feed forward
> 非线性激活函数，以及这里的整个部分，一个前馈

**[00:00:40]** this whole thing here a feed forward
> 这里的整个部分，一个前馈

**[00:00:40]** this whole thing here a feed forward module is the module that is later coded
> 这里的整个部分，一个前馈模块，是稍后在Transformer块中编码的模块

**[00:00:40]** module is the module that is later coded
> 模块，是稍后在Transformer块中编码的模块

**[00:00:40]** module is the module that is later coded in the Transformer block so we haven't
> 模块，是稍后在Transformer块中编码的模块，所以我们还没有

**[00:00:40]** in the Transformer block so we haven't
> 在Transformer块中，所以我们还没有

**[00:00:40]** in the Transformer block so we haven't talked about the Transformer block yet
> 在Transformer块中，所以我们还没有讨论Transformer块

**[00:00:40]** talked about the Transformer block yet
> 讨论Transformer块

**[00:00:40]** talked about the Transformer block yet we will talk about it later but I can
> 讨论Transformer块，我们稍后会讨论它，但我可以

**[00:00:40]** we will talk about it later but I can
> 我们稍后会讨论它，但我可以

**[00:00:40]** we will talk about it later but I can actually maybe already show you the
> 我们稍后会讨论它，但我实际上可能已经可以给你看

**[00:00:40]** actually maybe already show you the
> 实际上可能已经可以给你看

**[00:00:40]** actually maybe already show you the figure so it's a bit more clear where
> 实际上可能已经可以给你看这个图，这样更清楚

**[00:00:40]** figure so it's a bit more clear where
> 这个图，这样更清楚

**[00:00:40]** figure so it's a bit more clear where things fit into the grand scheme of
> 这个图，这样更清楚各个部分在整个方案中的位置

**[00:00:40]** things fit into the grand scheme of
> 各个部分在整个方案中的位置

**[00:00:40]** things fit into the grand scheme of things so this Transformer block is part
> 各个部分在整个方案中的位置，所以这个Transformer块是

**[00:00:40]** things so this Transformer block is part
> 所以这个Transformer块是

**[00:00:40]** things so this Transformer block is part of the llm and this gets repeated a
> 所以这个Transformer块是LLM的一部分，并且它被重复

**[00:00:40]** of the llm and this gets repeated a
> LLM的一部分，并且它被重复

**[00:00:40]** of the llm and this gets repeated a number of times and inside this
> LLM的一部分，并且它被重复多次，在这个

**[00:00:41]** number of times and inside this
> 多次，在这个

**[00:00:41]** number of times and inside this Transformer block there is the so-called
> 多次，在这个Transformer块内部，有所谓的

**[00:00:41]** Transformer block there is the so-called
> Transformer块内部，有所谓的

**[00:00:41]** Transformer block there is the so-called feed forward module and part of the feed
> Transformer块内部，有所谓的前馈模块，而前馈模块的一部分

**[00:00:41]** feed forward module and part of the feed
> 前馈模块，而前馈模块的一部分

**[00:00:41]** feed forward module and part of the feed forward module is this setup where we
> 前馈模块，而前馈模块的一部分就是这个设置，其中我们

**[00:00:41]** forward module is this setup where we
> 就是这个设置，其中我们

**[00:00:41]** forward module is this setup where we have a linear layer a nonlinear
> 就是这个设置，其中我们有一个线性层、一个非线性

**[00:00:41]** have a linear layer a nonlinear
> 有一个线性层、一个非线性

**[00:00:41]** have a linear layer a nonlinear activation function and a linear layer
> 有一个线性层、一个非线性激活函数和一个线性层

**[00:00:41]** activation function and a linear layer
> 激活函数和一个线性层

**[00:00:41]** activation function and a linear layer and in this video we are going to take a
> 激活函数和一个线性层，在本视频中，我们将

**[00:00:41]** and in this video we are going to take a
> 在本视频中，我们将

**[00:00:41]** and in this video we are going to take a look at what this G activation function
> 在本视频中，我们将看看这个G激活函数

**[00:00:41]** look at what this G activation function
> 看看这个G激活函数

**[00:00:41]** look at what this G activation function is so yeah a lot of background
> 看看这个G激活函数是什么，所以是的，很多背景

**[00:00:41]** is so yeah a lot of background
> 是什么，所以是的，很多背景

**[00:00:41]** is so yeah a lot of background information um I hope yeah this is um
> 是什么，所以是的，很多背景信息，嗯，我希望这，嗯，

**[00:00:41]** information um I hope yeah this is um
> 信息，嗯，我希望这，嗯，

**[00:00:41]** information um I hope yeah this is um not too confusing essentially we're just
> 信息，嗯，我希望这，嗯，不会太令人困惑，本质上我们只是在

**[00:00:41]** not too confusing essentially we're just
> 不会太令人困惑，本质上我们只是在

**[00:00:41]** not too confusing essentially we're just coding a small little function inside
> 不会太令人困惑，本质上我们只是在编码整个东西内部的一个小函数

**[00:00:41]** coding a small little function inside
> 编码整个东西内部的一个小函数

**[00:00:41]** coding a small little function inside this whole thing um and the skele
> 编码整个东西内部的一个小函数，嗯，这个skele

**[00:00:41]** this whole thing um and the skele
> 整个东西，嗯，这个skele

**[00:00:41]** this whole thing um and the skele function goes back to a paper called
> 函数可以追溯到一篇名为

**[00:00:41]** function goes back to a paper called
> 函数可以追溯到一篇名为

**[00:00:41]** function goes back to a paper called gaussian error linear units and here in
> 函数可以追溯到一篇名为高斯误差线性单元的论文，在这里

**[00:00:41]** gaussian error linear units and here in
> 高斯误差线性单元，在这里

**[00:00:41]** gaussian error linear units and here in this paper they proposed a function that
> 高斯误差线性单元，在这篇论文中，他们提出了一个函数

**[00:00:41]** this paper they proposed a function that
> 这篇论文中，他们提出了一个函数

**[00:00:41]** this paper they proposed a function that is essentially a bit better than the reu
> 这篇论文中，他们提出了一个函数，本质上比ReLU稍好一些

**[00:00:41]** is essentially a bit better than the reu
> 本质上比ReLU稍好一些

**[00:00:41]** is essentially a bit better than the reu for optimization purposes and so forth
> 本质上比ReLU稍好一些，用于优化目的等等

**[00:00:41]** for optimization purposes and so forth
> 用于优化目的等等

**[00:00:41]** for optimization purposes and so forth um and one more thing here so the reu
> 用于优化目的等等，嗯，还有一件事，这里的ReLU

**[00:00:41]** um and one more thing here so the reu
> 嗯，还有一件事，这里的ReLU

**[00:00:41]** um and one more thing here so the reu function oh there actually a nice figure
> 嗯，还有一件事，这里的ReLU函数，哦，实际上这里有一个漂亮的图

**[00:00:41]** function oh there actually a nice figure
> 函数，哦，实际上这里有一个漂亮的图

**[00:00:41]** function oh there actually a nice figure here how it relates to the G function so
> 函数，哦，实际上这里有一个漂亮的图，展示了它与G函数的关系

**[00:00:41]** here how it relates to the G function so
> 这里展示了它与G函数的关系

**[00:00:41]** here how it relates to the G function so the reu function is a function that is
> 这里展示了它与G函数的关系，所以ReLU函数是一个

**[00:00:41]** the reu function is a function that is
> ReLU函数是一个

**[00:00:41]** the reu function is a function that is maybe the simplest um nonlinear
> ReLU函数是一个可能是最简单的非线性

**[00:00:41]** maybe the simplest um nonlinear
> 可能是最简单的非线性

**[00:00:41]** maybe the simplest um nonlinear activation function what it is doing is
> 可能是最简单的非线性激活函数，它的作用是

**[00:00:41]** activation function what it is doing is
> 激活函数，它的作用是

**[00:00:41]** activation function what it is doing is if the negative so if the inputs here on
> 激活函数，它的作用是，如果输入是负的，那么这里的输入

**[00:00:41]** if the negative so if the inputs here on
> 如果输入是负的，那么这里的输入

**[00:00:41]** if the negative so if the inputs here on the x-axis if they are negative they are
> 如果输入为负，即x轴上的输入为负时，它们会被

**[00:00:42]** the x-axis if they are negative they are
> x轴上的输入为负时，它们会被

**[00:00:42]** the x-axis if they are negative they are thresholded to zero so the Y AIS is the
> x轴上的输入为负时，它们会被阈值化为零，因此Y轴是

**[00:00:42]** thresholded to zero so the Y AIS is the
> 阈值化为零，因此Y轴是

**[00:00:42]** thresholded to zero so the Y AIS is the output the xaxis are the inputs if the
> 阈值化为零，因此Y轴是输出，x轴是输入。如果

**[00:00:42]** output the xaxis are the inputs if the
> 输出，x轴是输入。如果

**[00:00:42]** output the xaxis are the inputs if the inputs are negative or below zero the
> 输出，x轴是输入。如果输入为负或低于零，

**[00:00:42]** inputs are negative or below zero the
> 输入为负或低于零，

**[00:00:42]** inputs are negative or below zero the output here is um zero otherwise it will
> 输入为负或低于零，输出为零；否则，

**[00:00:42]** output here is um zero otherwise it will
> 输出为零；否则，

**[00:00:42]** output here is um zero otherwise it will maintain the input so if the input is
> 输出为零；否则，它会保持输入值。因此如果输入是

**[00:00:42]** maintain the input so if the input is
> 保持输入值。因此如果输入是

**[00:00:42]** maintain the input so if the input is one the output is one if the input is
> 保持输入值。因此如果输入是1，输出是1；如果输入是

**[00:00:42]** one the output is one if the input is
> 1，输出是1；如果输入是

**[00:00:42]** one the output is one if the input is two the output is two and so forth um
> 1，输出是1；如果输入是2，输出是2，以此类推。

**[00:00:42]** two the output is two and so forth um
> 2，输出是2，以此类推。

**[00:00:42]** two the output is two and so forth um yeah and the g is you know it's very
> 2，输出是2，以此类推。而G函数，你知道，它非常

**[00:00:42]** yeah and the g is you know it's very
> 而G函数，你知道，它非常

**[00:00:42]** yeah and the g is you know it's very similar you can see it's slightly it has
> 而G函数，你知道，它非常相似。你可以看到它略微具有

**[00:00:42]** similar you can see it's slightly it has
> 相似。你可以看到它略微具有

**[00:00:42]** similar you can see it's slightly it has some beneficial mathematical properties
> 相似。你可以看到它略微具有一些有益的数学性质，

**[00:00:42]** some beneficial mathematical properties
> 一些有益的数学性质，

**[00:00:42]** some beneficial mathematical properties for normalization um or for optimization
> 一些有益的数学性质，用于归一化或优化，

**[00:00:42]** for normalization um or for optimization
> 用于归一化或优化，

**[00:00:42]** for normalization um or for optimization during the training so you if you're
> 用于归一化或优化，在训练过程中。所以如果你

**[00:00:42]** during the training so you if you're
> 在训练过程中。所以如果你

**[00:00:42]** during the training so you if you're interested you can read through this
> 在训练过程中。所以如果你感兴趣，可以阅读这篇

**[00:00:42]** interested you can read through this
> 感兴趣，可以阅读这篇

**[00:00:42]** interested you can read through this paper but you know it's not super
> 感兴趣，可以阅读这篇论文，但你知道，这并不是非常

**[00:00:42]** paper but you know it's not super
> 论文，但你知道，这并不是非常

**[00:00:42]** paper but you know it's not super essential you can think of it just as a
> 论文，但你知道，这并不是非常关键。你可以把它看作只是一个

**[00:00:42]** essential you can think of it just as a
> 关键。你可以把它看作只是一个

**[00:00:42]** essential you can think of it just as a nonlinear activation function one of the
> 关键。你可以把它看作只是一个非线性激活函数，是众多

**[00:00:42]** nonlinear activation function one of the
> 非线性激活函数，是众多

**[00:00:42]** nonlinear activation function one of the many nonlinear activation functions um
> 非线性激活函数，是众多非线性激活函数之一。

**[00:00:42]** many nonlinear activation functions um
> 非线性激活函数之一。

**[00:00:42]** many nonlinear activation functions um you can actually train also the llm with
> 非线性激活函数之一。实际上，你也可以用其他类型的非线性激活

**[00:00:42]** you can actually train also the llm with
> 实际上，你也可以用其他类型的非线性激活

**[00:00:42]** you can actually train also the llm with other types of um nonlinear activation
> 实际上，你也可以用其他类型的非线性激活函数来训练LLM，

**[00:00:42]** other types of um nonlinear activation
> 函数来训练LLM，

**[00:00:42]** other types of um nonlinear activation functions it's just the choice that the
> 函数来训练LLM。这只是GPT模型原始开发者所做的

**[00:00:42]** functions it's just the choice that the
> 这只是GPT模型原始开发者所做的

**[00:00:42]** functions it's just the choice that the original developers of the GPT model
> 这只是GPT模型原始开发者所做的选择。事实上，后来的LLM，例如

**[00:00:42]** original developers of the GPT model
> 选择。事实上，后来的LLM，例如

**[00:00:42]** original developers of the GPT model made and in fact later um llm for
> 选择。事实上，后来的LLM，例如Llama模型，它们使用了

**[00:00:42]** made and in fact later um llm for
> Llama模型，它们使用了

**[00:00:42]** made and in fact later um llm for example the Llama models they use
> Llama模型，它们使用了称为swiglu激活函数的东西，

**[00:00:42]** example the Llama models they use
> 称为swiglu激活函数的东西，

**[00:00:42]** example the Llama models they use something called a swiglo activation
> 称为swiglu激活函数的东西，这是一个带有我认为是Silo激活函数的门控单元。所以

**[00:00:43]** something called a swiglo activation
> 这是一个带有我认为是Silo激活函数的门控单元。所以

**[00:00:43]** something called a swiglo activation function which is a gated unit with um I
> 这是一个带有我认为是Silo激活函数的门控单元。所以你知道，这只是一个选择，有些人

**[00:00:43]** function which is a gated unit with um I
> 你知道，这只是一个选择，有些人

**[00:00:43]** function which is a gated unit with um I think Silo activation function so you
> 你知道，这只是一个选择，有些人会这样做。所以它并不是非常关键。

**[00:00:43]** think Silo activation function so you
> 会这样做。所以它并不是非常关键。

**[00:00:43]** think Silo activation function so you know it's just like a a choice some
> 会这样做。所以它并不是非常关键。唯一关键的是

**[00:00:43]** know it's just like a a choice some
> 唯一关键的是

**[00:00:43]** know it's just like a a choice some people make so it's not super crucial
> 唯一关键的是我们使用一个非线性激活函数。

**[00:00:43]** people make so it's not super crucial
> 我们使用一个非线性激活函数。

**[00:00:43]** people make so it's not super crucial the only thing that is crucial is that
> 我们使用一个非线性激活函数。具体用哪个？嗯，这取决于情况，有些可能

**[00:00:43]** the only thing that is crucial is that
> 具体用哪个？嗯，这取决于情况，有些可能

**[00:00:43]** the only thing that is crucial is that we use a nonlinear activation function
> 具体用哪个？嗯，这取决于情况，有些可能有时比其他更好，但我

**[00:00:43]** we use a nonlinear activation function
> 有时比其他更好，但我

**[00:00:43]** we use a nonlinear activation function which one well it depends some might
> 有时比其他更好，但我不会太过于纠结

**[00:00:43]** which one well it depends some might
> 不会太过于纠结

**[00:00:43]** which one well it depends some might sometimes work better than others but I
> 不会太过于纠结在这里使用哪种类型。关于G激活函数的一件事是

**[00:00:43]** sometimes work better than others but I
> 关于G激活函数的一件事是

**[00:00:43]** sometimes work better than others but I wouldn't get too um you know too serious
> 关于G激活函数的一件事是它实际上有两种变体。

**[00:00:43]** wouldn't get too um you know too serious
> 它实际上有两种变体。

**[00:00:43]** wouldn't get too um you know too serious about which type to use here um one
> 它实际上有两种变体。所以这是G函数的

**[00:00:43]** about which type to use here um one
> 所以这是G函数的

**[00:00:43]** about which type to use here um one thing about the G activation function is
> 所以这是G函数的数学公式，但原始的GPT模型

**[00:00:43]** thing about the G activation function is
> 数学公式，但原始的GPT模型

**[00:00:43]** thing about the G activation function is that it comes actually in two flavors um
> thing about the G activation function is that it comes actually in two flavors um

**[00:00:43]** that it comes actually in two flavors um
> that it comes actually in two flavors um

**[00:00:43]** that it comes actually in two flavors um so this is the so here this is the
> that it comes actually in two flavors um so this is the so here this is the

**[00:00:43]** so this is the so here this is the
> so this is the so here this is the

**[00:00:43]** so this is the so here this is the mathematical formulation of the G
> so this is the so here this is the mathematical formulation of the G

**[00:00:43]** mathematical formulation of the G
> mathematical formulation of the G

**[00:00:43]** mathematical formulation of the G function but the original GPT model it
> mathematical formulation of the G function but the original GPT model it

**[00:00:43]** function but the original GPT model it
> function but the original GPT model it

**[00:00:43]** function but the original GPT model it was implemented in tens of flow and they
> 函数，但原始的GPT模型是用TensorFlow实现的，他们

**[00:00:43]** was implemented in tens of flow and they
> 是用TensorFlow实现的，他们

**[00:00:43]** was implemented in tens of flow and they used this um approximation here so this
> 是用TensorFlow实现的，他们在这里使用了这个近似公式，所以

**[00:00:43]** used this um approximation here so this
> 在这里使用了这个近似公式，所以

**[00:00:43]** used this um approximation here so this mathematical formula and I don't know
> 在这里使用了这个近似公式，所以这个数学公式，我不确定

**[00:00:43]** mathematical formula and I don't know
> 这个数学公式，我不确定

**[00:00:43]** mathematical formula and I don't know 100% because I haven't talked to the
> 这个数学公式，我不确定100%，因为我没有和

**[00:00:43]** 100% because I haven't talked to the
> 100%，因为我没有和

**[00:00:43]** 100% because I haven't talked to the authors um of the original GPT
> 100%，因为我没有和原始GPT的作者们交流过

**[00:00:43]** authors um of the original GPT
> 原始GPT的作者们

**[00:00:43]** authors um of the original GPT architecture but I can assume well I
> 原始GPT的架构作者们，但我可以假设，嗯

**[00:00:43]** architecture but I can assume well I
> 架构，但我可以假设，嗯

**[00:00:43]** architecture but I can assume well I would assume that this is probably um
> 架构，但我可以假设，嗯我猜这可能是

**[00:00:43]** would assume that this is probably um
> 我猜这可能是

**[00:00:43]** would assume that this is probably um due to efficiency reasons that they use
> 我猜这可能是出于效率原因，他们使用

**[00:00:43]** due to efficiency reasons that they use
> 出于效率原因，他们使用

**[00:00:43]** due to efficiency reasons that they use this approximation so it's you know
> 出于效率原因，他们使用这个近似，所以你知道

**[00:00:44]** this approximation so it's you know
> 这个近似，所以你知道

**[00:00:44]** this approximation so it's you know approximately the same but it's
> 这个近似，所以你知道大致相同，但它是

**[00:00:44]** approximately the same but it's
> 大致相同，但它是

**[00:00:44]** approximately the same but it's approximated it is maybe a bit faster
> 大致相同，但它是近似的，可能稍微快一点

**[00:00:44]** approximated it is maybe a bit faster
> 近似的，可能稍微快一点

**[00:00:44]** approximated it is maybe a bit faster again it doesn't really matter in the
> 近似的，可能稍微快一点，不过在大局上这并不重要

**[00:00:44]** again it doesn't really matter in the
> 不过在大局上这并不重要

**[00:00:44]** again it doesn't really matter in the grand scheme of things but because we
> 不过在大局上这并不重要，但因为我们

**[00:00:44]** grand scheme of things but because we
> 但因为我们

**[00:00:44]** grand scheme of things but because we want to mimic the original GPT
> 但因为我们想模仿原始的GPT

**[00:00:44]** want to mimic the original GPT
> 想模仿原始的GPT

**[00:00:44]** want to mimic the original GPT architecture we are also going to
> 想模仿原始的GPT架构，我们也将

**[00:00:44]** architecture we are also going to
> 架构，我们也将

**[00:00:44]** architecture we are also going to implement this version here in pytorch
> 架构，我们也将在这里用PyTorch实现这个版本

**[00:00:44]** implement this version here in pytorch
> 在这里用PyTorch实现这个版本

**[00:00:44]** implement this version here in pytorch so this was like a lot of buildup a lot
> 在这里用PyTorch实现这个版本，所以这有很多铺垫，很多

**[00:00:44]** so this was like a lot of buildup a lot
> 所以这有很多铺垫，很多

**[00:00:44]** so this was like a lot of buildup a lot of you know background information so
> 所以这有很多铺垫，很多背景信息，所以

**[00:00:44]** of you know background information so
> 背景信息，所以

**[00:00:44]** of you know background information so let's actually get to the code and I
> 背景信息，所以让我们直接看代码，我

**[00:00:44]** let's actually get to the code and I
> 让我们直接看代码，我

**[00:00:44]** let's actually get to the code and I think um you know once you see the code
> 让我们直接看代码，我想嗯你知道一旦你看到代码

**[00:00:44]** think um you know once you see the code
> 我想嗯你知道一旦你看到代码

**[00:00:44]** think um you know once you see the code you will feel like okay this is actually
> 我想嗯你知道一旦你看到代码，你会觉得好吧，这实际上

**[00:00:44]** you will feel like okay this is actually
> 你会觉得好吧，这实际上

**[00:00:44]** you will feel like okay this is actually much simpler than you um thought um
> 你会觉得好吧，这实际上比你想象的要简单得多

**[00:00:44]** much simpler than you um thought um
> 比你想象的要简单得多

**[00:00:44]** much simpler than you um thought um compared to you know the paper and
> 比你想象的要简单得多，相比你知道的论文和

**[00:00:44]** compared to you know the paper and
> 相比你知道的论文和

**[00:00:44]** compared to you know the paper and everything so here uh what I copy and
> 相比你知道的论文和所有东西，所以这里嗯我复制和

**[00:00:44]** everything so here uh what I copy and
> 所有东西，所以这里嗯我复制和

**[00:00:44]** everything so here uh what I copy and pasted here is my yeah reimplementation
> 所有东西，所以这里嗯我复制粘贴的是我的嗯重新实现

**[00:00:44]** pasted here is my yeah reimplementation
> 粘贴的是我的嗯重新实现

**[00:00:44]** pasted here is my yeah reimplementation of the approximate G function and you
> 粘贴的是我的嗯重新实现的近似G函数，你

**[00:00:44]** of the approximate G function and you
> 的近似G函数，你

**[00:00:44]** of the approximate G function and you can see um it's a very similar module um
> 的近似G函数，你可以看到嗯这是一个非常相似的模块嗯

**[00:00:44]** can see um it's a very similar module um
> 可以看到嗯这是一个非常相似的模块嗯

**[00:00:44]** can see um it's a very similar module um it has a forward method and it receives
> 可以看到嗯这是一个非常相似的模块嗯它有一个forward方法，它接收

**[00:00:44]** it has a forward method and it receives
> 它有一个forward方法，它接收

**[00:00:44]** it has a forward method and it receives some input and then you know it applies
> 它有一个forward方法，它接收一些输入，然后你知道它应用

**[00:00:44]** some input and then you know it applies
> 一些输入，然后你知道它应用

**[00:00:44]** some input and then you know it applies this formula it multiplies the input by
> 一些输入，然后你知道它应用这个公式，它将输入乘以

**[00:00:44]** this formula it multiplies the input by
> 这个公式，它将输入乘以

**[00:00:44]** this formula it multiplies the input by .5 and then multiplies it by 1 plus this
> 这个公式，它将输入乘以0.5，然后乘以1加上这个

**[00:00:44]** .5 and then multiplies it by 1 plus this
> 0.5，然后乘以1加上这个

**[00:00:44]** .5 and then multiplies it by 1 plus this 10 Ag and so forth and so this is yeah
> 0.5，然后乘以1加上这个10 Ag等等，所以这是嗯

**[00:00:45]** 10 Ag and so forth and so this is yeah
> 10 Ag等等，所以这是嗯

**[00:00:45]** 10 Ag and so forth and so this is yeah just a mathematical approximation of the
> 10 Ag等等，所以这是嗯只是那个论文中描述的SK函数的

**[00:00:45]** just a mathematical approximation of the
> 只是那个论文中描述的SK函数的

**[00:00:45]** just a mathematical approximation of the SK function as described in that paper
> 只是那个论文中描述的SK函数的数学近似

**[00:00:45]** SK function as described in that paper
> 数学近似

**[00:00:45]** SK function as described in that paper and um yeah if we use that let me just
> 数学近似，嗯如果我们使用它，让我

**[00:00:45]** and um yeah if we use that let me just
> 嗯如果我们使用它，让我

**[00:00:45]** and um yeah if we use that let me just also copy that code here can visualize
> 嗯如果我们使用它，让我也把那段代码复制到这里，可以可视化

**[00:00:45]** also copy that code here can visualize
> 也把那段代码复制到这里，可以可视化

**[00:00:45]** also copy that code here can visualize it I'm using met plot lip
> 也把那段代码复制到这里，可以可视化它，我使用matplotlib

**[00:00:45]** it I'm using met plot lip
> 它，我使用matplotlib

**[00:00:45]** it I'm using met plot lip here and um I'm going to um yeah apply
> 它，我使用matplotlib，嗯我将嗯应用

**[00:00:45]** here and um I'm going to um yeah apply
> 嗯我将嗯应用

**[00:00:45]** here and um I'm going to um yeah apply to some sample data uh in the range
> 嗯我将嗯应用到一些样本数据上，范围在

**[00:00:45]** to some sample data uh in the range
> 到一些样本数据上，范围在

**[00:00:45]** to some sample data uh in the range between minus 3 and three so the input
> 到一些样本数据上，范围在负3到3之间，所以输入

**[00:00:45]** between minus 3 and three so the input
> between minus 3 and three so the input

**[00:00:45]** between minus 3 and three so the input going back to the paper um I'm just
> 在负3到3之间，所以输入回到论文中，嗯，我只是

**[00:00:45]** going back to the paper um I'm just
> 回到论文中，嗯，我只是

**[00:00:45]** going back to the paper um I'm just testing it between minus 3 and
> 回到论文中，嗯，我只是在负3到3之间测试它

**[00:00:45]** testing it between minus 3 and
> 在负3到3之间测试它

**[00:00:45]** testing it between minus 3 and three that's what I'm doing here and I'm
> 在负3到3之间测试它，这就是我在这里做的，而且我

**[00:00:45]** three that's what I'm doing here and I'm
> 这就是我在这里做的，而且我

**[00:00:45]** three that's what I'm doing here and I'm also for you know completeness um I'm
> 这就是我在这里做的，而且为了完整性，嗯，我

**[00:00:45]** also for you know completeness um I'm
> 为了完整性，嗯，我

**[00:00:45]** also for you know completeness um I'm comparing it to the relu function in
> 为了完整性，嗯，我把它与PyTorch中的ReLU函数进行比较

**[00:00:45]** comparing it to the relu function in
> 把它与PyTorch中的ReLU函数进行比较

**[00:00:45]** comparing it to the relu function in pytorch and here it's just plotting code
> 把它与PyTorch中的ReLU函数进行比较，这里只是绘图代码

**[00:00:45]** pytorch and here it's just plotting code
> PyTorch中的ReLU函数进行比较，这里只是绘图代码

**[00:00:45]** pytorch and here it's just plotting code so let's see how that looks
> PyTorch中的ReLU函数进行比较，这里只是绘图代码，所以让我们看看效果如何

**[00:00:45]** so let's see how that looks
> 所以让我们看看效果如何

**[00:00:45]** so let's see how that looks like um yeah so maybe I
> 所以让我们看看效果如何，嗯，是的，也许我

**[00:00:45]** like um yeah so maybe I
> 嗯，是的，也许我

**[00:00:45]** like um yeah so maybe I should maybe interrupt this I think um
> 嗯，是的，也许我应该打断一下，我想嗯

**[00:00:45]** should maybe interrupt this I think um
> 应该打断一下，我想嗯

**[00:00:45]** should maybe interrupt this I think um of course I forgot to execute this code
> 应该打断一下，我想嗯，当然我忘了执行这段代码

**[00:00:45]** of course I forgot to execute this code
> 当然我忘了执行这段代码

**[00:00:45]** of course I forgot to execute this code cell I think it should be pretty fast I
> 当然我忘了执行这段代码单元，我想应该很快

**[00:00:45]** cell I think it should be pretty fast I
> 单元，我想应该很快

**[00:00:45]** cell I think it should be pretty fast I think my computer is still suffering a
> 单元，我想应该很快，我想我的电脑还在受

**[00:00:45]** think my computer is still suffering a
> 我想我的电脑还在受

**[00:00:45]** think my computer is still suffering a bit because in the previous video I
> 我想我的电脑还在受点影响，因为在之前的视频中我

**[00:00:45]** bit because in the previous video I
> 在之前的视频中我

**[00:00:45]** bit because in the previous video I um executed this large matrix
> 在之前的视频中我嗯执行了这个大型矩阵

**[00:00:46]** um executed this large matrix
> 嗯执行了这个大型矩阵

**[00:00:46]** um executed this large matrix multiplication so I should probably
> 嗯执行了这个大型矩阵乘法，所以我应该

**[00:00:46]** multiplication so I should probably
> 乘法，所以我应该

**[00:00:46]** multiplication so I should probably restart it at some point but yeah here
> 乘法，所以我应该在某个时候重启它，但这里

**[00:00:46]** restart it at some point but yeah here
> 在某个时候重启它，但这里

**[00:00:46]** restart it at some point but yeah here we can see the figure um so we have
> 在某个时候重启它，但这里我们可以看到这个图，嗯，所以我们有

**[00:00:46]** we can see the figure um so we have
> 我们可以看到这个图，嗯，所以我们有

**[00:00:46]** we can see the figure um so we have minus 3 to 3 here um as the input and
> 我们可以看到这个图，嗯，所以这里有负3到3作为输入，并且

**[00:00:46]** minus 3 to 3 here um as the input and
> 负3到3作为输入，并且

**[00:00:46]** minus 3 to 3 here um as the input and then you can see the output here and
> 负3到3作为输入，然后你可以在这里看到输出，并且

**[00:00:46]** then you can see the output here and
> 然后你可以在这里看到输出，并且

**[00:00:46]** then you can see the output here and yeah it is ultimately um same idea as
> 然后你可以在这里看到输出，并且嗯，它最终与ReLU函数思路相同

**[00:00:46]** yeah it is ultimately um same idea as
> 它最终与ReLU函数思路相同

**[00:00:46]** yeah it is ultimately um same idea as the re function except it's a bit
> 它最终与ReLU函数思路相同，只是这里稍微平滑一些

**[00:00:46]** the re function except it's a bit
> 只是这里稍微平滑一些

**[00:00:46]** the re function except it's a bit smoother here which has nicer properties
> 只是这里稍微平滑一些，这在优化过程中计算梯度时具有更好的性质

**[00:00:46]** smoother here which has nicer properties
> 这在优化过程中计算梯度时具有更好的性质

**[00:00:46]** smoother here which has nicer properties during optimization when we compute the
> 这在优化过程中计算梯度时具有更好的性质，嗯，是的，然后回到

**[00:00:46]** during optimization when we compute the
> 嗯，是的，然后回到

**[00:00:46]** during optimization when we compute the gradients um yeah and then just to go
> 嗯，是的，然后回到梯度，嗯，是的，然后回到

**[00:00:46]** gradients um yeah and then just to go
> 梯度，嗯，是的，然后回到

**[00:00:46]** gradients um yeah and then just to go back
> 梯度，嗯，是的，然后回到

**[00:00:46]** back
> 回到

**[00:00:46]** back to the chapter here um so I talked about
> 回到这里的章节，嗯，所以我谈到了

**[00:00:46]** to the chapter here um so I talked about
> 这里的章节，嗯，所以我谈到了

**[00:00:46]** to the chapter here um so I talked about this little unit that we will need
> 这里的章节，嗯，所以我谈到了这个小单元，我们实际上会在接下来的部分需要它

**[00:00:46]** this little unit that we will need
> 这个小单元，我们实际上会在接下来的部分需要它

**[00:00:46]** this little unit that we will need actually in upcoming section when we
> 这个小单元，我们实际上会在接下来的部分需要它，当我们实现Transformer块时，嗯，并且

**[00:00:46]** actually in upcoming section when we
> 当我们实现Transformer块时，嗯，并且

**[00:00:46]** actually in upcoming section when we implement the Transformer block um and
> 当我们实现Transformer块时，嗯，并且所以我们将实现这个代码

**[00:00:46]** implement the Transformer block um and
> 所以我们将实现这个代码

**[00:00:46]** implement the Transformer block um and so we are going to implement the code
> 所以我们将实现这个代码

**[00:00:46]** so we are going to implement the code
> 对于这个

**[00:00:46]** so we are going to implement the code for this
> 对于这个

**[00:00:46]** for this
> 对于这个嗯，呃，我想它在上面的位置

**[00:00:46]** for this um uh I think it was further up
> 嗯，呃，我想它在上面的位置

**[00:00:46]** here um yeah so we're going to implement
> 这里，嗯，是的，所以我们将实现

**[00:00:46]** here um yeah so we're going to implement
> 这里，嗯，是的，所以我们将实现

**[00:00:46]** here um yeah so we're going to implement this one so
> 这里，嗯，是的，所以我们将实现这个，所以

**[00:00:46]** this one so
> 这个，所以

**[00:00:46]** this one so let's do this and this will be something
> 这个，所以让我们来做这个，这将是我们在后面会用到的东西

**[00:00:46]** let's do this and this will be something
> 让我们来做这个，这将是我们在后面会用到的东西

**[00:00:46]** let's do this and this will be something we are going to use later so what is
> 让我们来做这个，这将是我们在后面会用到的东西，那么这里发生了什么，所以我们有一个前馈

**[00:00:46]** we are going to use later so what is
> 那么这里发生了什么，所以我们有一个前馈

**[00:00:46]** we are going to use later so what is happening here so we have a feed forward
> 那么这里发生了什么，所以我们有一个前馈模块，并且这里有一个sequential

**[00:00:46]** happening here so we have a feed forward
> 模块，并且这里有一个sequential

**[00:00:47]** happening here so we have a feed forward module and we have a sequential here
> 模块，并且这里有一个sequential，这意味着在forward方法中，它会嗯执行所有

**[00:00:47]** module and we have a sequential here
> 这意味着在forward方法中，它会嗯执行所有

**[00:00:47]** module and we have a sequential here which means it will in the forward
> 这意味着在forward方法中，它会嗯执行所有sequential内部的内容，按顺序

**[00:00:47]** which means it will in the forward
> which means it will in the forward

**[00:00:47]** which means it will in the forward method it will um execute everything
> which means it will in the forward method it will um execute everything

**[00:00:47]** method it will um execute everything
> method it will um execute everything

**[00:00:47]** method it will um execute everything that is inside the sequential in a
> method it will um execute everything that is inside the sequential in a

**[00:00:47]** that is inside the sequential in a
> 在sequential内部

**[00:00:47]** that is inside the sequential in a sequential fashion we have a linear
> 在sequential内部以sequential方式有一个linear

**[00:00:47]** sequential fashion we have a linear
> 以sequential方式有一个linear

**[00:00:47]** sequential fashion we have a linear layer um so this goes from the embedding
> 以sequential方式有一个linear层，嗯，所以它从embedding

**[00:00:47]** layer um so this goes from the embedding
> 层，嗯，所以它从embedding

**[00:00:47]** layer um so this goes from the embedding Dimension and if we go back um up here
> 层，嗯，所以它从embedding维度出发，如果我们回到上面

**[00:00:47]** Dimension and if we go back um up here
> 维度，如果我们回到上面

**[00:00:47]** Dimension and if we go back um up here the embedding Dimension was 768 so it
> 维度，如果我们回到上面，embedding维度是768，所以它

**[00:00:47]** the embedding Dimension was 768 so it
> embedding维度是768，所以它

**[00:00:47]** the embedding Dimension was 768 so it goes from
> embedding维度是768，所以它从

**[00:00:47]** 768 to 4 * 768 so we have
> 768到4 * 768，所以我们有

**[00:00:47]** 768 to 4 * 768 so we have
> 768到4 * 768，所以我们有

**[00:00:47]** 768 to 4 * 768 so we have 768 to 4 *
> 768到4 * 768，所以我们有从768到4 *

**[00:00:47]** 768 to 4 *
> 768到4 *

**[00:00:47]** 768 to 4 * 768 and so this um should be something
> 768到4 * 768，所以这嗯应该是

**[00:00:47]** 768 and so this um should be something
> 768，所以这嗯应该是

**[00:00:47]** 768 and so this um should be something with uh I think it's
> 768，所以这嗯应该是，嗯，我想是

**[00:00:47]** with uh I think it's
> 嗯，我想是

**[00:00:47]** with uh I think it's 3,100 let me just calculate it
> 嗯，我想是3,100，让我算一下

**[00:00:47]** 3,100 let me just calculate it
> 3,100，让我算一下

**[00:00:47]** 3,100 let me just calculate it 372 um so it's
> 3,100，让我算一下，372，嗯，所以是

**[00:00:47]** 372 um so it's
> 372，嗯，所以是

**[00:00:47]** 372 um so it's 372 let's just put it here and then it
> 372，嗯，所以是372，我们把它放在这里，然后它

**[00:00:47]** 372 let's just put it here and then it
> 372，我们把它放在这里，然后它

**[00:00:47]** 372 let's just put it here and then it goes from
> 372，我们把它放在这里，然后它从

**[00:00:47]** goes from
> 从

**[00:00:47]** goes from 372 back to
> 从372回到

**[00:00:47]** 372 back to
> 372回到

**[00:00:47]** 372 back to 768 here
> 372回到这里的768

**[00:00:47]** 768 here
> 这里的768

**[00:00:47]** 768 here so it
> 这里的768，所以它

**[00:00:47]** so it
> 所以它

**[00:00:47]** so it is essentially a network that is the
> 所以它本质上是一个网络，形状

**[00:00:48]** is essentially a network that is the
> 本质上是一个网络，形状

**[00:00:48]** is essentially a network that is the opposite shape of hourglass where we go
> 本质上是一个网络，形状与沙漏相反，我们

**[00:00:48]** opposite shape of hourglass where we go
> 与沙漏相反，我们

**[00:00:48]** opposite shape of hourglass where we go instead of an Hour Glass we go from
> 与沙漏相反，我们不是沙漏形状，而是从

**[00:00:48]** instead of an Hour Glass we go from
> 不是沙漏形状，而是从

**[00:00:48]** instead of an Hour Glass we go from large to small to large here we go from
> 不是沙漏形状，而是从大到小再到大，这里我们从

**[00:00:48]** large to small to large here we go from
> 大到小再到大，这里我们从

**[00:00:48]** large to small to large here we go from small to large to small um I do think I
> 大到小再到大，这里我们从从小到大再到小，嗯，我确实认为

**[00:00:48]** small to large to small um I do think I
> 从小到大再到小，嗯，我确实认为

**[00:00:48]** small to large to small um I do think I have a figure here yeah so you can think
> 从小到大再到小，嗯，我确实认为这里有一张图，是的，所以你可以想象

**[00:00:48]** have a figure here yeah so you can think
> 这里有一张图，是的，所以你可以想象

**[00:00:48]** have a figure here yeah so you can think of it as as this we have the small
> 这里有一张图，是的，所以你可以想象它像这样，我们有小的

**[00:00:48]** of it as as this we have the small
> 它像这样，我们有小的

**[00:00:48]** of it as as this we have the small embedding Dimension we make it four
> 它像这样，我们有小的embedding维度，我们把它放大四倍

**[00:00:48]** embedding Dimension we make it four
> embedding维度，我们把它放大四倍

**[00:00:48]** embedding Dimension we make it four times larger and then we make it small
> embedding维度，我们把它放大四倍，然后我们再把它缩小

**[00:00:48]** times larger and then we make it small
> 放大四倍，然后我们再把它缩小

**[00:00:48]** times larger and then we make it small again why do we do that so this is um
> 放大四倍，然后我们再把它缩小，为什么这样做？嗯，这

**[00:00:48]** again why do we do that so this is um
> 为什么这样做？嗯，这

**[00:00:48]** again why do we do that so this is um you know it's like a trick almost in D
> 为什么这样做？嗯，你知道，这几乎是D网络中的一个技巧

**[00:00:48]** you know it's like a trick almost in D
> 你知道，这几乎是D网络中的一个技巧

**[00:00:48]** you know it's like a trick almost in D networks where this adds a lot of
> 你知道，这几乎是D网络中的一个技巧，它增加了大量

**[00:00:48]** networks where this adds a lot of
> 它增加了大量

**[00:00:48]** networks where this adds a lot of parameter and these parameters learn to
> 它增加了大量参数，这些参数学会

**[00:00:48]** parameter and these parameters learn to
> 参数，这些参数学会

**[00:00:48]** parameter and these parameters learn to extract some information from the inputs
> 参数，这些参数学会从输入中提取一些信息

**[00:00:48]** extract some information from the inputs
> 从输入中提取一些信息

**[00:00:48]** extract some information from the inputs it's essentially you can think of it as
> 从输入中提取一些信息，本质上你可以把它想象成

**[00:00:48]** it's essentially you can think of it as
> 本质上你可以把它想象成

**[00:00:48]** it's essentially you can think of it as massaging the input to you know get um
> 本质上你可以把它想象成对输入进行处理，你知道，嗯

**[00:00:48]** massaging the input to you know get um
> 对输入进行处理，你知道，嗯

**[00:00:48]** massaging the input to you know get um to extract information for it that then
> 对输入进行处理，你知道，嗯，从中提取信息，然后

**[00:00:48]** to extract information for it that then
> 从中提取信息，然后

**[00:00:48]** to extract information for it that then during the optimization when the weight
> 从中提取信息，然后在优化过程中，当权重

**[00:00:48]** during the optimization when the weight
> 在优化过程中，当权重

**[00:00:48]** during the optimization when the weight weight matrices are optimized help the
> 在优化过程中，当权重矩阵被优化时，帮助

**[00:00:48]** weight matrices are optimized help the
> 权重矩阵被优化时，帮助

**[00:00:48]** weight matrices are optimized help the network learn yeah learn to extract
> 权重矩阵被优化时，帮助网络学习，是的，学会提取

**[00:00:48]** network learn yeah learn to extract
> 网络学习，是的，学会提取

**[00:00:48]** network learn yeah learn to extract relevant information and do something
> 网络学习，是的，学会提取相关信息并做一些

**[00:00:48]** relevant information and do something
> 相关信息并做一些

**[00:00:48]** relevant information and do something useful with that and in this case in the
> 相关信息并做一些有用的事情，在这种情况下，在

**[00:00:48]** useful with that and in this case in the
> 有用的事情，在这种情况下，在

**[00:00:48]** useful with that and in this case in the case of llms it's correctly generating
> 有用的事情，在这种情况下，在LLM的情况下，它正确地生成

**[00:00:48]** case of llms it's correctly generating
> LLM的情况下，它正确地生成

**[00:00:48]** case of llms it's correctly generating the next word in in the text or input
> LLM的情况下，它正确地生成文本或输入中的下一个词

**[00:00:48]** the next word in in the text or input
> 文本或输入中的下一个词

**[00:00:48]** the next word in in the text or input data um yeah and this is our mini neuron
> 文本或输入数据中的下一个词，嗯，这就是我们的小型神经元

**[00:00:48]** data um yeah and this is our mini neuron
> 数据，嗯，这就是我们的小型神经元

**[00:00:48]** data um yeah and this is our mini neuron Network that we are going to use in um
> 数据，嗯，这就是我们将在第5章使用的微型神经元网络

**[00:00:49]** Network that we are going to use in um
> 我们将在第5章使用的网络

**[00:00:49]** Network that we are going to use in um chapter 5 when sorry in the Transformer
> 我们将在第5章使用的网络，抱歉，是在Transformer中

**[00:00:49]** chapter 5 when sorry in the Transformer
> 第5章，抱歉，是在Transformer中

**[00:00:49]** chapter 5 when sorry in the Transformer block but then also in the llm that we
> 第5章，抱歉，是在Transformer块中，但也会在我们训练的LLM里

**[00:00:49]** block but then also in the llm that we
> 块中，但也会在我们训练的LLM里

**[00:00:49]** block but then also in the llm that we are going to train in chapter 5 um we
> 块中，但也会在我们第5章训练的LLM里，嗯，我们

**[00:00:49]** are going to train in chapter 5 um we
> 我们将在第5章训练它，嗯，我们

**[00:00:49]** are going to train in chapter 5 um we can initialize it so let's go here and
> 我们将在第5章训练它，嗯，我们可以初始化它，所以让我们到这里

**[00:00:49]** can initialize it so let's go here and
> 可以初始化它，所以让我们到这里

**[00:00:49]** can initialize it so let's go here and do FFN um and then let's do feed forward
> 可以初始化它，所以让我们做FFN，嗯，然后让我们做前馈

**[00:00:49]** do FFN um and then let's do feed forward
> 做FFN，嗯，然后让我们做前馈

**[00:00:49]** do FFN um and then let's do feed forward oops um maybe execute this of course
> 做FFN，嗯，然后让我们做前馈，哎呀，也许执行这个，当然

**[00:00:49]** oops um maybe execute this of course
> 哎呀，也许执行这个，当然

**[00:00:49]** oops um maybe execute this of course it's a running gag by now I think so we
> 哎呀，也许执行这个，当然，这现在可能是个老梗了，所以我们可以

**[00:00:49]** it's a running gag by now I think so we
> 这现在可能是个老梗了，所以我们可以

**[00:00:49]** it's a running gag by now I think so we can use um the so we have the Cs CFG
> 这现在可能是个老梗了，所以我们可以使用，嗯，我们有Cs CFG

**[00:00:49]** can use um the so we have the Cs CFG
> 可以使用，嗯，我们有Cs CFG

**[00:00:49]** can use um the so we have the Cs CFG which stands for configuration which we
> 可以使用，嗯，我们有Cs CFG，它代表配置，我们

**[00:00:49]** which stands for configuration which we
> 它代表配置，我们

**[00:00:49]** which stands for configuration which we have
> 它代表配置，我们已经在上面定义了

**[00:00:49]** have
> 定义了

**[00:00:49]** have defined up here so this is our
> 定义了，所以这是我们的

**[00:00:49]** defined up here so this is our
> 上面定义的，所以这是我们的

**[00:00:49]** defined up here so this is our configuration here so let's just use
> 上面定义的，所以这是我们的配置，让我们直接使用

**[00:00:49]** configuration here so let's just use
> 配置，让我们直接使用

**[00:00:49]** configuration here so let's just use that and put it in
> 配置，让我们直接使用它并放入

**[00:00:49]** that and put it in
> 它并放入

**[00:00:49]** that and put it in here and that is our mini neuron Network
> 它并放入这里，这就是我们的微型神经元网络

**[00:00:49]** here and that is our mini neuron Network
> 这里，这就是我们的微型神经元网络

**[00:00:49]** here and that is our mini neuron Network and then yeah we can try it on some
> 这里，这就是我们的微型神经元网络，然后我们可以尝试在一些样本数据上运行

**[00:00:49]** and then yeah we can try it on some
> 然后我们可以尝试在一些样本数据上运行

**[00:00:49]** and then yeah we can try it on some sample data so we can just for
> 然后我们可以尝试在一些样本数据上运行，为了简单起见，我们可以再次创建一些随机数据

**[00:00:49]** sample data so we can just for
> 为了简单起见，我们可以再次创建一些随机数据

**[00:00:49]** sample data so we can just for Simplicity create some random data again
> 为了简单起见，我们可以再次创建一些随机数据，所以让我们做2x3乘以

**[00:00:49]** Simplicity create some random data again
> 所以让我们做2x3乘以

**[00:00:49]** Simplicity create some random data again so let's do 2x3 by um
> 所以让我们做2x3乘以768，即一个批次包含两个输入样本，每个

**[00:00:49]** so let's do 2x3 by um
> 768，即一个批次包含两个输入样本，每个

**[00:00:49]** so let's do 2x3 by um 768 so a batch of two input samples each
> 768，即一个批次包含两个输入样本，每个样本有三个token，每个token

**[00:00:50]** 768 so a batch of two input samples each
> 每个样本有三个token，每个token

**[00:00:50]** 768 so a batch of two input samples each one has three tokens and then each token
> 每个样本有三个token，每个token的嵌入维度是768，这只是一个

**[00:00:50]** one has three tokens and then each token
> 嵌入维度是768，这只是一个

**[00:00:50]** one has three tokens and then each token betting is 768 dimensional it's just an
> 嵌入维度是768，这只是一个样本，稍后我们会处理真实数据

**[00:00:50]** betting is 768 dimensional it's just an
> 样本，稍后我们会处理真实数据

**[00:00:50]** betting is 768 dimensional it's just an um a sample later we will work with real
> 样本，稍后我们会处理真实数据，但只是为了展示会发生什么

**[00:00:50]** um a sample later we will work with real
> 但只是为了展示会发生什么

**[00:00:50]** um a sample later we will work with real data but yeah just to show you what
> 但只是为了展示会发生什么，所以这里我们得到相同的输出形状，这是因为我们

**[00:00:50]** data but yeah just to show you what
> 输出形状，这是因为我们

**[00:00:50]** data but yeah just to show you what happens so here we get um the same
> 输出形状，这是因为我们有一个输入维度，它也被

**[00:00:50]** happens so here we get um the same
> 有一个输入维度，它也被

**[00:00:50]** happens so here we get um the same output shape and that is be because we
> 有一个输入维度，它也被生成在这里，所以中间有一个更大的值，或者说四倍大

**[00:00:50]** output shape and that is be because we
> 生成在这里，所以中间有一个更大的值，或者说四倍大

**[00:00:50]** output shape and that is be because we have this input Dimension that is also
> 生成在这里，所以中间有一个更大的值，或者说四倍大，即3000，但我们得到的输出仍然是

**[00:00:50]** have this input Dimension that is also
> 即3000，但我们得到的输出仍然是

**[00:00:50]** have this input Dimension that is also generated here so intermediately there
> 即3000，但我们得到的输出仍然是768，因为嗯，我们从这里映射到

**[00:00:50]** generated here so intermediately there
> 768，因为嗯，我们从这里映射到

**[00:00:50]** generated here so intermediately there is a larger value or four times is large
> 768，因为嗯，我们从这里映射到更大的维度，然后再从更大的维度映射回来

**[00:00:50]** is a larger value or four times is large
> 更大的维度，然后再从更大的维度映射回来

**[00:00:50]** is a larger value or four times is large the 3,000 but what we get out is still
> 更大的维度，然后再从更大的维度映射回来，为什么是四倍？这是一个超参数

**[00:00:50]** the 3,000 but what we get out is still
> 为什么是四倍？这是一个超参数

**[00:00:50]** the 3,000 but what we get out is still 768 because um we map from here to
> 为什么是四倍？这是一个超参数，它也可以是，你知道，可以是两倍

**[00:00:50]** 768 because um we map from here to
> 它也可以是，你知道，可以是两倍

**[00:00:50]** 768 because um we map from here to larger and then from larger back again
> 它也可以是，你知道，可以是两倍，可以是八倍，可以是一倍，可以是

**[00:00:50]** larger and then from larger back again
> 可以是八倍，可以是一倍，可以是

**[00:00:50]** larger and then from larger back again and why is it four times so this is
> 可以是八倍，可以是一倍，可以是任何值，这是GPT论文的原始作者选择的

**[00:00:50]** and why is it four times so this is
> 任何值，这是GPT论文的原始作者选择的

**[00:00:50]** and why is it four times so this is again a hyper parameter so it could be
> 任何值，这是GPT论文的原始作者选择的，并且像我之前提到的

**[00:00:50]** again a hyper parameter so it could be
> 并且像我之前提到的

**[00:00:50]** again a hyper parameter so it could be as well be you know it could be two it
> 并且像我之前提到的，这又是一个超参数

**[00:00:50]** as well be you know it could be two it
> as well be you know it could be two it

**[00:00:50]** as well be you know it could be two it could be eight it could be one it could
> as well be you know it could be two it could be eight it could be one it could

**[00:00:50]** could be eight it could be one it could
> could be eight it could be one it could

**[00:00:50]** could be eight it could be one it could be anything um this is something the
> could be eight it could be one it could be anything um this is something the

**[00:00:50]** be anything um this is something the
> be anything um this is something the

**[00:00:50]** be anything um this is something the original um authors of the GPT paper
> be anything um this is something the original um authors of the GPT paper

**[00:00:50]** original um authors of the GPT paper
> original um authors of the GPT paper

**[00:00:50]** original um authors of the GPT paper chose and again like I mentioned before
> original um authors of the GPT paper chose and again like I mentioned before

**[00:00:50]** chose and again like I mentioned before
> chose and again like I mentioned before

**[00:00:50]** chose and again like I mentioned before we we we want to mimic the original GPT
> 选择并且，就像我之前提到的，我们想要模仿原始的GPT

**[00:00:50]** we we we want to mimic the original GPT
> 我们想要模仿原始的GPT

**[00:00:50]** we we we want to mimic the original GPT architecture
> 我们想要模仿原始的GPT架构

**[00:00:50]** architecture
> 架构

**[00:00:50]** architecture and also why I'm so so hung up on the
> 以及为什么我如此执着于

**[00:00:51]** and also why I'm so so hung up on the
> 以及为什么我如此执着于

**[00:00:51]** and also why I'm so so hung up on the GPT architecture why we want to mimic it
> 以及为什么我如此执着于GPT架构，我们想要模仿它

**[00:00:51]** GPT architecture why we want to mimic it
> GPT架构，我们想要模仿它

**[00:00:51]** GPT architecture why we want to mimic it exactly is that later in chapter 5 we
> GPT架构，我们想要精确模仿它的原因是在第5章后面

**[00:00:51]** exactly is that later in chapter 5 we
> 精确模仿它的原因是在第5章后面

**[00:00:51]** exactly is that later in chapter 5 we will download the available weights that
> 精确模仿它的原因是在第5章后面，我们会下载OpenAI在网站上分享的可用权重

**[00:00:51]** will download the available weights that
> 我们会下载OpenAI在网站上分享的可用权重

**[00:00:51]** will download the available weights that open AI shared on the website and then
> 我们会下载OpenAI在网站上分享的可用权重，然后

**[00:00:51]** open AI shared on the website and then
> OpenAI在网站上分享的可用权重，然后

**[00:00:51]** open AI shared on the website and then we will log that into our um code here
> OpenAI在网站上分享的可用权重，然后我们会将这些权重加载到我们的代码中

**[00:00:51]** we will log that into our um code here
> 我们会将这些权重加载到我们的代码中

**[00:00:51]** we will log that into our um code here and um so there are weight parameters so
> 我们会将这些权重加载到我们的代码中，所以这些是权重参数

**[00:00:51]** and um so there are weight parameters so
> 所以这些是权重参数

**[00:00:51]** and um so there are weight parameters so if I go um here um and then let's say um
> 所以这些是权重参数，如果我到这里，然后假设

**[00:00:51]** if I go um here um and then let's say um
> 如果我到这里，然后假设

**[00:00:51]** if I go um here um and then let's say um layers right so so this one um you can
> 如果我到这里，然后假设layers，所以这个你可以

**[00:00:51]** layers right so so this one um you can
> layers，所以这个你可以

**[00:00:51]** layers right so so this one um you can see there are these weight parameters so
> layers，所以这个你可以看到这些权重参数

**[00:00:51]** see there are these weight parameters so
> 看到这些权重参数

**[00:00:51]** see there are these weight parameters so when we have this one this is the first
> 看到这些权重参数，当我们有这个时，这是第一个

**[00:00:51]** when we have this one this is the first
> 当我们有这个时，这是第一个

**[00:00:51]** when we have this one this is the first linear layer and then we have the
> 当我们有这个时，这是第一个linear layer，然后我们有

**[00:00:51]** linear layer and then we have the
> linear layer，然后我们有

**[00:00:51]** linear layer and then we have the weights here and these are run currently
> linear layer，然后这里有权重，这些目前是

**[00:00:51]** weights here and these are run currently
> 权重，这些目前是

**[00:00:51]** weights here and these are run currently random weights but later we will train
> 权重，这些目前是随机权重，但之后我们会训练

**[00:00:51]** random weights but later we will train
> 随机权重，但之后我们会训练

**[00:00:51]** random weights but later we will train them and we will also loot the original
> 随机权重，但之后我们会训练它们，并且我们也会加载原始的

**[00:00:51]** them and we will also loot the original
> 它们，并且我们也会加载原始的

**[00:00:51]** them and we will also loot the original openi weights and we have to have the
> 它们，并且我们也会加载原始的OpenAI权重，我们必须拥有

**[00:00:51]** openi weights and we have to have the
> OpenAI权重，我们必须拥有

**[00:00:51]** openi weights and we have to have the exact same Dimension that they have if
> OpenAI权重，我们必须拥有与他们完全相同的维度，如果他们

**[00:00:51]** exact same Dimension that they have if
> 完全相同的维度，如果他们

**[00:00:51]** exact same Dimension that they have if we want to load their weights into our
> 完全相同的维度，如果我们想将他们的权重加载到我们的

**[00:00:51]** we want to load their weights into our
> 想将他们的权重加载到我们的

**[00:00:51]** we want to load their weights into our model so we have to match these things
> 想将他们的权重加载到我们的模型中，所以我们必须精确匹配这些

**[00:00:51]** model so we have to match these things
> 模型中，所以我们必须精确匹配这些

**[00:00:51]** model so we have to match these things exactly so that the weights are
> 模型中，所以我们必须精确匹配这些，以便权重是

**[00:00:51]** exactly so that the weights are
> 精确匹配这些，以便权重是

**[00:00:51]** exactly so that the weights are compatible and that's why I'm trying to
> 精确匹配这些，以便权重是兼容的，这就是为什么我试图

**[00:00:51]** compatible and that's why I'm trying to
> 兼容的，这就是为什么我试图

**[00:00:51]** compatible and that's why I'm trying to really make sure everything else is the
> 兼容的，这就是为什么我试图确保其他所有东西都是

**[00:00:51]** really make sure everything else is the
> 确保其他所有东西都是

**[00:00:51]** really make sure everything else is the same we use the same hyper parameter
> 确保其他所有东西都是相同的，我们使用相同的超参数

**[00:00:51]** same we use the same hyper parameter
> 相同的，我们使用相同的超参数

**[00:00:51]** same we use the same hyper parameter here we use the same approximation here
> 相同的，我们使用相同的超参数，我们使用相同的近似

**[00:00:51]** here we use the same approximation here
> 我们使用相同的近似

**[00:00:51]** here we use the same approximation here that they used in tens of flow and so
> 我们使用相同的近似，就像他们在TensorFlow中使用的那样

**[00:00:52]** that they used in tens of flow and so
> 就像他们在TensorFlow中使用的那样

**[00:00:52]** that they used in tens of flow and so forth and yeah so that that's why it's
> 就像他们在TensorFlow中使用的那样，等等，是的，这就是为什么

**[00:00:52]** forth and yeah so that that's why it's
> 等等，是的，这就是为什么

**[00:00:52]** forth and yeah so that that's why it's so important for me to match these
> 等等，是的，这就是为什么对我来说匹配这些如此重要

**[00:00:52]** so important for me to match these
> 对我来说匹配这些如此重要

**[00:00:52]** so important for me to match these things and yeah in this section now you
> 对我来说匹配这些如此重要，是的，在本节中，现在你

**[00:00:52]** things and yeah in this section now you
> 是的，在本节中，现在你

**[00:00:52]** things and yeah in this section now you saw the G function in the next video we
> 是的，在本节中，现在你看到了G函数，在下一个视频中我们

**[00:00:52]** saw the G function in the next video we
> 看到了G函数，在下一个视频中我们

**[00:00:52]** saw the G function in the next video we will tackle another uh building block of
> 看到了G函数，在下一个视频中我们将处理LLM的另一个构建块

**[00:00:52]** will tackle another uh building block of
> 将处理LLM的另一个构建块

**[00:00:52]** will tackle another uh building block of the llm
> 将处理LLM的另一个构建块

**[00:00:52]** architecture in this section we are
> 架构，在本节中我们

**[00:00:52]** architecture in this section we are
> 架构，在本节中我们

**[00:00:52]** architecture in this section we are going to talk about adding shortcut
> 架构，在本节中我们将讨论添加shortcut connections

**[00:00:52]** going to talk about adding shortcut
> 讨论添加shortcut connections

**[00:00:52]** going to talk about adding shortcut connections so so far our goal is still
> 讨论添加shortcut connections，到目前为止，我们的目标仍然是

**[00:00:52]** connections so so far our goal is still
> 到目前为止，我们的目标仍然是

**[00:00:52]** connections so so far our goal is still to implement this final GPT architecture
> 到目前为止，我们的目标仍然是实现这个最终的GPT架构

**[00:00:52]** to implement this final GPT architecture
> 实现这个最终的GPT架构

**[00:00:52]** to implement this final GPT architecture we started off with this GPT backbone
> 实现这个最终的GPT架构，我们从GPT backbone开始

**[00:00:52]** we started off with this GPT backbone
> 我们从GPT backbone开始

**[00:00:52]** we started off with this GPT backbone this dummy GPT placeholder code and then
> 我们从GPT backbone开始，这个虚拟的GPT占位代码，然后

**[00:00:52]** this dummy GPT placeholder code and then
> 这个虚拟的GPT占位代码，然后

**[00:00:52]** this dummy GPT placeholder code and then we filled in individual building blocks
> 这个虚拟的GPT占位代码，然后我们填充了各个构建块

**[00:00:52]** we filled in individual building blocks
> 我们填充了各个构建块

**[00:00:52]** we filled in individual building blocks for example in the beginning the layer
> 我们填充了各个构建块，例如在开始的layer

**[00:00:52]** for example in the beginning the layer
> 例如在开始时，这一层

**[00:00:52]** for example in the beginning the layer normalization and then in the previous
> 例如在开始时，层归一化，然后在前面的

**[00:00:52]** normalization and then in the previous
> 归一化，然后在前面的

**[00:00:52]** normalization and then in the previous section we talked about the G nonlinear
> 归一化，然后在前面的部分我们讨论了G非线性

**[00:00:52]** section we talked about the G nonlinear
> 部分我们讨论了G非线性

**[00:00:52]** section we talked about the G nonlinear activation function and we implemented
> 部分我们讨论了G非线性激活函数，并实现了

**[00:00:52]** activation function and we implemented
> 激活函数，并实现了

**[00:00:52]** activation function and we implemented this small yeah feed forward neuron
> 激活函数，并实现了这个小型的、是的，前馈神经元

**[00:00:52]** this small yeah feed forward neuron
> 这个小型的、是的，前馈神经元

**[00:00:52]** this small yeah feed forward neuron network module that we are going to use
> 这个小型的、是的，前馈神经元网络模块，我们将要使用它

**[00:00:52]** network module that we are going to use
> 网络模块，我们将要使用它

**[00:00:52]** network module that we are going to use in the Transformer block however before
> 网络模块，我们将要使用它在Transformer块中，然而在此之前

**[00:00:52]** in the Transformer block however before
> 在Transformer块中，然而在此之前

**[00:00:52]** in the Transformer block however before we get to this Transformer block here
> 在Transformer块中，然而在此之前，我们到达这个Transformer块之前

**[00:00:52]** we get to this Transformer block here
> 我们到达这个Transformer块之前

**[00:00:52]** we get to this Transformer block here there's one more thing we want to
> 我们到达这个Transformer块之前，还有一件事我们想要

**[00:00:52]** there's one more thing we want to
> 还有一件事我们想要

**[00:00:52]** there's one more thing we want to implement and that is uh the shortcut
> 还有一件事我们想要实现，那就是，呃，快捷连接

**[00:00:52]** implement and that is uh the shortcut
> 实现，那就是，呃，快捷连接

**[00:00:52]** implement and that is uh the shortcut connections here so shortcut connections
> 实现，那就是，呃，这里的快捷连接，所以快捷连接

**[00:00:53]** connections here so shortcut connections
> 这里的连接，所以快捷连接

**[00:00:53]** connections here so shortcut connections they go back to a paper called um deep
> 这里的连接，它们追溯到一篇论文，叫做，嗯，深度

**[00:00:53]** they go back to a paper called um deep
> 它们追溯到一篇论文，叫做，嗯，深度

**[00:00:53]** they go back to a paper called um deep residual learning for image recognition
> 它们追溯到一篇论文，叫做，嗯，深度残差学习用于图像识别

**[00:00:53]** residual learning for image recognition
> 残差学习用于图像识别

**[00:00:53]** residual learning for image recognition and here the authors implemented a
> 残差学习用于图像识别，并且在这里作者实现了一个

**[00:00:53]** and here the authors implemented a
> 并且在这里作者实现了一个

**[00:00:53]** and here the authors implemented a convolutional new network back then um
> 并且在这里作者实现了一个卷积神经网络，当时，嗯

**[00:00:53]** convolutional new network back then um
> 卷积神经网络，当时，嗯

**[00:00:53]** convolutional new network back then um and they yeah proposed these shortcut
> 卷积神经网络，当时，嗯，并且他们，是的，提出了这些快捷连接

**[00:00:53]** and they yeah proposed these shortcut
> 并且他们，是的，提出了这些快捷连接

**[00:00:53]** and they yeah proposed these shortcut connections or sometimes also called
> 并且他们，是的，提出了这些快捷连接，或者有时也称为

**[00:00:53]** connections or sometimes also called
> 连接，或者有时也称为

**[00:00:53]** connections or sometimes also called residual connections so to show you what
> 连接，或者有时也称为残差连接，所以为了向你展示这些

**[00:00:53]** residual connections so to show you what
> 残差连接，所以为了向你展示这些

**[00:00:53]** residual connections so to show you what those are here's maybe a nice figure
> 残差连接，所以为了向你展示这些是什么，这里可能有一张很好的图

**[00:00:53]** those are here's maybe a nice figure
> 这些是什么，这里可能有一张很好的图

**[00:00:53]** those are here's maybe a nice figure so um what they do is so just just to
> 这些是什么，这里可能有一张很好的图，所以，嗯，它们所做的是，只是，只是为了

**[00:00:53]** so um what they do is so just just to
> 所以，嗯，它们所做的是，只是，只是为了

**[00:00:53]** so um what they do is so just just to orient ourselves here what we are
> 所以，嗯，它们所做的是，只是，只是为了让我们在这里定位，我们正在

**[00:00:53]** orient ourselves here what we are
> 让我们在这里定位，我们正在

**[00:00:53]** orient ourselves here what we are looking at is like this something
> 让我们在这里定位，我们正在看的是像这样的东西

**[00:00:53]** looking at is like this something
> 看的是像这样的东西

**[00:00:53]** looking at is like this something similar to what we have just coded in
> 看的是像这样的东西，类似于我们刚刚在上一部分编码的内容

**[00:00:53]** similar to what we have just coded in
> 类似于我们刚刚在上一部分编码的内容

**[00:00:53]** similar to what we have just coded in the previous section this little neur
> 类似于我们刚刚在上一部分编码的内容，这个小神经

**[00:00:53]** the previous section this little neur
> 上一部分，这个小神经

**[00:00:53]** the previous section this little neur network module where you have a weight
> 上一部分，这个小神经网络模块，其中你有一个权重

**[00:00:53]** network module where you have a weight
> 网络模块，其中你有一个权重

**[00:00:53]** network module where you have a weight layer essentially the NN linear layer a
> 网络模块，其中你有一个权重层，本质上就是NN线性层，一个

**[00:00:53]** layer essentially the NN linear layer a
> 层，本质上就是NN线性层，一个

**[00:00:53]** layer essentially the NN linear layer a nonlinear activation function reu or G
> 层，本质上就是NN线性层，一个非线性激活函数ReLU或G

**[00:00:53]** nonlinear activation function reu or G
> 非线性激活函数ReLU或G

**[00:00:53]** nonlinear activation function reu or G and then another linear layer and so
> 非线性激活函数ReLU或G，然后是另一个线性层，等等

**[00:00:53]** and then another linear layer and so
> 然后是另一个线性层，等等

**[00:00:53]** and then another linear layer and so this is
> 然后是另一个线性层，等等，所以这

**[00:00:53]** this is
> 这是

**[00:00:53]** this is actually similar to what we're looking
> 实际上类似于我们正在看的

**[00:00:53]** actually similar to what we're looking
> 实际上类似于我们正在看的

**[00:00:53]** actually similar to what we're looking at here
> 实际上类似于我们正在看的这里

**[00:00:53]** at here
> 这里

**[00:00:53]** at here now what they added is this Arrow here
> 这里，现在它们添加的是这个箭头

**[00:00:53]** now what they added is this Arrow here
> 现在它们添加的是这个箭头

**[00:00:53]** now what they added is this Arrow here so what what they have is they have F
> 现在它们添加的是这个箭头，所以它们有的是F

**[00:00:53]** so what what they have is they have F
> 所以它们有的是F

**[00:00:53]** so what what they have is they have F ofx where you can think of this here as
> 所以它们有的是F(x)，你可以把这里看作

**[00:00:53]** ofx where you can think of this here as
> F(x)，你可以把这里看作

**[00:00:53]** ofx where you can think of this here as F ofx so f ofx is really this part it's
> F(x)，你可以把这里看作F(x)，所以F(x)实际上是这一部分，它是

**[00:00:54]** F ofx so f ofx is really this part it's
> F(x)，所以F(x)实际上是这一部分，它是

**[00:00:54]** F ofx so f ofx is really this part it's a function where the input goes through
> F(x)，所以F(x)实际上是这一部分，它是一个函数，其中输入经过

**[00:00:54]** a function where the input goes through
> 一个函数，其中输入经过

**[00:00:54]** a function where the input goes through a linear layer nonlinear activation and
> 一个函数，其中输入经过一个线性层、非线性激活和

**[00:00:54]** a linear layer nonlinear activation and
> 一个线性层、非线性激活和

**[00:00:54]** a linear layer nonlinear activation and a linear layer so you can think of it as
> 一个线性层、非线性激活和一个线性层，所以你可以把它看作

**[00:00:54]** a linear layer so you can think of it as
> 一个线性层，所以你可以把它看作

**[00:00:54]** a linear layer so you can think of it as a yeah as a function and then you add
> 一个线性层，所以你可以把它看作，是的，一个函数，然后你加上

**[00:00:54]** a yeah as a function and then you add
> 是的，一个函数，然后你加上

**[00:00:54]** a yeah as a function and then you add back the input so input is X the
> 是的，一个函数，然后你加回输入，所以输入是X，原始的

**[00:00:54]** back the input so input is X the
> 加回输入，所以输入是X，原始的

**[00:00:54]** back the input so input is X the original input you add it back to the
> 加回输入，所以输入是X，原始的输入，你把它加回到

**[00:00:54]** original input you add it back to the
> 原始输入，你将其加回到

**[00:00:54]** original input you add it back to the result of this one here and so why is
> 原始输入，你将其加回到这个结果，那么为什么

**[00:00:54]** result of this one here and so why is
> 这个结果，那么为什么

**[00:00:54]** result of this one here and so why is this being done what's the point here so
> 这个结果，那么为什么要这样做，意义何在

**[00:00:54]** this being done what's the point here so
> 这样做，意义何在

**[00:00:54]** this being done what's the point here so if you have a very deep neuron Network
> 这样做，意义何在，如果你有一个非常深的神经网络

**[00:00:54]** if you have a very deep neuron Network
> 如果你有一个非常深的神经网络

**[00:00:54]** if you have a very deep neuron Network you can have problems like exporing
> 如果你有一个非常深的神经网络，可能会遇到梯度爆炸

**[00:00:54]** you can have problems like exporing
> 可能会遇到梯度爆炸

**[00:00:54]** you can have problems like exporing gradients or Vanishing gradients and so
> 可能会遇到梯度爆炸或梯度消失等问题

**[00:00:54]** gradients or Vanishing gradients and so
> 梯度爆炸或梯度消失等问题

**[00:00:54]** gradients or Vanishing gradients and so for example if this doesn't learn
> 梯度爆炸或梯度消失等问题，例如，如果这个没有学到

**[00:00:54]** for example if this doesn't learn
> 例如，如果这个没有学到

**[00:00:54]** for example if this doesn't learn anything useful or let's say it goes to
> 例如，如果这个没有学到任何有用的东西，或者说它变成了

**[00:00:54]** anything useful or let's say it goes to
> 任何有用的东西，或者说它变成了

**[00:00:54]** anything useful or let's say it goes to zero due to the vanishing gradient
> 任何有用的东西，或者说由于梯度消失问题它变成了零

**[00:00:54]** zero due to the vanishing gradient
> 由于梯度消失问题变成了零

**[00:00:54]** zero due to the vanishing gradient problem um the network can still use
> 由于梯度消失问题，网络仍然可以使用

**[00:00:54]** problem um the network can still use
> 网络仍然可以使用

**[00:00:54]** problem um the network can still use this shortcut so the shortcut it doesn't
> 网络仍然可以使用这个捷径，所以这个捷径它不会

**[00:00:54]** this shortcut so the shortcut it doesn't
> 这个捷径，所以这个捷径它不会

**[00:00:54]** this shortcut so the shortcut it doesn't diminish anything it's essentially like
> 这个捷径，所以这个捷径它不会削弱任何东西，它本质上就像

**[00:00:54]** diminish anything it's essentially like
> 削弱任何东西，它本质上就像

**[00:00:54]** diminish anything it's essentially like rescuing the network if this is not
> 削弱任何东西，它本质上就像在拯救网络，如果这个没有

**[00:00:54]** rescuing the network if this is not
> 拯救网络，如果这个没有

**[00:00:54]** rescuing the network if this is not you're learning anything useful so in
> 拯救网络，如果这个没有学到任何有用的东西，那么

**[00:00:54]** you're learning anything useful so in
> 学到任何有用的东西，那么

**[00:00:54]** you're learning anything useful so in that sense um yeah you you maintain the
> 学到任何有用的东西，那么从这个意义上说，你保持了

**[00:00:54]** that sense um yeah you you maintain the
> 从这个意义上说，你保持了

**[00:00:54]** that sense um yeah you you maintain the gradient signal you maintain a signal
> 从这个意义上说，你保持了梯度信号，你保持了一个信号

**[00:00:54]** gradient signal you maintain a signal
> 梯度信号，你保持了一个信号

**[00:00:54]** gradient signal you maintain a signal during the training to help the network
> 梯度信号，你在训练过程中保持一个信号，以帮助网络

**[00:00:54]** during the training to help the network
> 在训练过程中，以帮助网络

**[00:00:54]** during the training to help the network learn something otherwise if you have a
> 在训练过程中，以帮助网络学习一些东西，否则如果你有一个

**[00:00:54]** learn something otherwise if you have a
> 学习一些东西，否则如果你有一个

**[00:00:54]** learn something otherwise if you have a very deep n network there might be no
> 学习一些东西，否则如果你有一个非常深的网络，可能就没有

**[00:00:55]** very deep n network there might be no
> 非常深的网络，可能就没有

**[00:00:55]** very deep n network there might be no signal anymore and then yeah you have a
> 非常深的网络，可能就没有信号了，然后你就会遇到

**[00:00:55]** signal anymore and then yeah you have a
> 信号了，然后你就会遇到

**[00:00:55]** signal anymore and then yeah you have a Vanishing grading problem or other
> 信号了，然后你就会遇到梯度消失问题或其他

**[00:00:55]** Vanishing grading problem or other
> 梯度消失问题或其他

**[00:00:55]** Vanishing grading problem or other problems and then the network will fail
> 梯度消失问题或其他问题，然后网络就会失败

**[00:00:55]** problems and then the network will fail
> 问题，然后网络就会失败

**[00:00:55]** problems and then the network will fail if one of these sub modules fails
> 问题，然后网络就会失败，如果这些子模块中有一个失败

**[00:00:55]** if one of these sub modules fails
> 如果这些子模块中有一个失败

**[00:00:55]** if one of these sub modules fails because in a network in your network
> 如果这些子模块中有一个失败，因为在网络中，在你的网络中

**[00:00:55]** because in a network in your network
> 因为在网络中，在你的网络中

**[00:00:55]** because in a network in your network everything is sequential um these layers
> 因为在网络中，在你的网络中，一切都是顺序的，这些层

**[00:00:55]** everything is sequential um these layers
> 一切都是顺序的，这些层

**[00:00:55]** everything is sequential um these layers are sequential and they depend on each
> 一切都是顺序的，这些层是顺序的，并且它们相互依赖

**[00:00:55]** are sequential and they depend on each
> 是顺序的，并且它们相互依赖

**[00:00:55]** are sequential and they depend on each other and so if one of them does
> 是顺序的，并且它们相互依赖，所以如果其中一个做了

**[00:00:55]** other and so if one of them does
> 所以如果其中一个做了

**[00:00:55]** other and so if one of them does something uh does nonsense then all the
> 所以如果其中一个做了些无意义的事情，那么所有

**[00:00:55]** something uh does nonsense then all the
> 无意义的事情，那么所有

**[00:00:55]** something uh does nonsense then all the other ones will suffer all the other
> 无意义的事情，那么所有其他的层都会受到影响，所有其他的

**[00:00:55]** other ones will suffer all the other
> 其他的层都会受到影响，所有其他的

**[00:00:55]** other ones will suffer all the other ones that come below here so here as a
> 其他的层都会受到影响，所有位于它下面的层，所以这里作为一个

**[00:00:55]** ones that come below here so here as a
> 位于它下面的层，所以这里作为一个

**[00:00:55]** ones that come below here so here as a you know as a back door you can kind of
> 位于它下面的层，所以这里作为一个后门，你可以某种程度上

**[00:00:55]** you know as a back door you can kind of
> 后门，你可以某种程度上

**[00:00:55]** you know as a back door you can kind of learn or the network can rely on this um
> 后门，你可以某种程度上学习，或者网络可以依赖这个

**[00:00:55]** learn or the network can rely on this um
> 学习，或者网络可以依赖这个

**[00:00:55]** learn or the network can rely on this um to skip these layers if they are not
> 学习，或者网络可以依赖这个来跳过这些层，如果它们没有

**[00:00:55]** to skip these layers if they are not
> 来跳过这些层，如果它们没有

**[00:00:55]** to skip these layers if they are not useful it's essentially like a little a
> 来跳过这些层，如果它们没有用，它本质上就像一个小

**[00:00:55]** useful it's essentially like a little a
> 用，它本质上就像一个小

**[00:00:55]** useful it's essentially like a little a little back door here you can think of
> 用，它本质上就像一个小后门，你可以这样理解

**[00:00:55]** little back door here you can think of
> 后门，你可以这样理解

**[00:00:55]** little back door here you can think of it like that um yeah let's see how that
> 后门，你可以这样理解，嗯，让我们看看这在代码中

**[00:00:55]** it like that um yeah let's see how that
> 嗯，让我们看看这在代码中

**[00:00:55]** it like that um yeah let's see how that works out in code so I have also a
> 嗯，让我们看看这在代码中是如何实现的，所以我还有一个

**[00:00:55]** works out in code so I have also a
> 是如何实现的，所以我还有一个

**[00:00:55]** works out in code so I have also a figure here where you can see a deep new
> 是如何实现的，所以我这里还有一个图，你可以看到一个深的

**[00:00:55]** figure here where you can see a deep new
> 图，你可以看到一个深的

**[00:00:55]** figure here where you can see a deep new network with many layers um and we are
> 图，你可以看到一个有很多层的深度神经网络，我们正在

**[00:00:55]** network with many layers um and we are
> 有很多层的深度神经网络，我们正在

**[00:00:55]** network with many layers um and we are looking at the gradients here and you
> 有很多层的深度神经网络，我们正在查看这里的梯度，你

**[00:00:55]** looking at the gradients here and you
> 查看这里的梯度，你

**[00:00:55]** looking at the gradients here and you can see the gradients are relatively
> 查看这里的梯度，你可以看到梯度相对

**[00:00:55]** can see the gradients are relatively
> 可以看到梯度相对

**[00:00:55]** can see the gradients are relatively small so in in neur network um
> 可以看到梯度相对较小，所以在神经网络中

**[00:00:55]** small so in in neur network um
> 较小，所以在神经网络中

**[00:00:55]** small so in in neur network um optimization we use an algorithm called
> 较小，所以在神经网络优化中，我们使用一种称为

**[00:00:55]** optimization we use an algorithm called
> 优化中，我们使用一种称为

**[00:00:55]** optimization we use an algorithm called called back propagation we will use that
> 优化中，我们使用一种称为反向传播的算法，我们将使用它

**[00:00:55]** called back propagation we will use that
> 称为反向传播的算法，我们将使用它

**[00:00:55]** called back propagation we will use that in chapter 5 uh in this book we're not
> 称为反向传播的算法，我们将使用它，在第5章中，在这本书里我们不会

**[00:00:56]** in chapter 5 uh in this book we're not
> 在第5章中，在这本书里我们不会

**[00:00:56]** in chapter 5 uh in this book we're not really going over the fundamentals of
> 在第5章中，在这本书里我们不会真正深入讲解

**[00:00:56]** really going over the fundamentals of
> 真正深入讲解

**[00:00:56]** really going over the fundamentals of deep learning because you know this
> 真正深入讲解深度学习的基础知识，因为你知道这

**[00:00:56]** deep learning because you know this
> 深度学习的基础知识，因为你知道这

**[00:00:56]** deep learning because you know this would be another 500 page book so here
> 深度学习的基础知识，因为你知道这会是另一本500页的书，所以这里

**[00:00:56]** would be another 500 page book so here
> 会是另一本500页的书，所以这里

**[00:00:56]** would be another 500 page book so here we are really laser focused on LMS but
> 会是另一本500页的书，所以这里我们真正专注于LMS，但

**[00:00:56]** we are really laser focused on LMS but
> 我们真正专注于LMS，但

**[00:00:56]** we are really laser focused on LMS but yeah some of these deep new network
> 我们真正专注于LMS，但是的，其中一些深度神经网络

**[00:00:56]** yeah some of these deep new network
> 是的，其中一些深度神经网络

**[00:00:56]** yeah some of these deep new network concepts are relevant so in this deep
> 是的，其中一些深度神经网络概念是相关的，所以在这个深度

**[00:00:56]** concepts are relevant so in this deep
> 概念是相关的，所以在这个深度

**[00:00:56]** concepts are relevant so in this deep new network um we so it's also a
> 概念是相关的，所以在这个深度神经网络中，它也是一个

**[00:00:56]** new network um we so it's also a
> 神经网络中，它也是一个

**[00:00:56]** new network um we so it's also a convention to draw new networks from
> 神经网络中，它也是一个惯例，从下到上绘制神经网络

**[00:00:56]** convention to draw new networks from
> 惯例，从下到上绘制神经网络

**[00:00:56]** convention to draw new networks from bottom to top so this is why sometimes
> 惯例，从下到上绘制神经网络，所以这就是为什么有时

**[00:00:56]** bottom to top so this is why sometimes
> 从下到上，所以这就是为什么有时

**[00:00:56]** bottom to top so this is why sometimes when you're looking at these diagrams
> 从下到上，所以这就是为什么有时当你查看这些图表时

**[00:00:56]** when you're looking at these diagrams
> 当你查看这些图表时

**[00:00:56]** when you're looking at these diagrams why is this you know flip around this is
> 当你查看这些图表时，为什么你知道它是翻转的，这

**[00:00:56]** why is this you know flip around this is
> 为什么你知道它是翻转的，这

**[00:00:56]** why is this you know flip around this is really a convention in in deep learning
> 为什么你知道它是翻转的，这实际上是深度学习中的一个惯例

**[00:00:56]** really a convention in in deep learning
> 实际上是深度学习中的一个惯例

**[00:00:56]** really a convention in in deep learning where people you know tend to draw these
> 实际上是深度学习中的一个惯例，人们你知道倾向于绘制这些

**[00:00:56]** where people you know tend to draw these
> 人们你知道倾向于绘制这些

**[00:00:56]** where people you know tend to draw these diagrams from the bottom to the top so
> 人们你知道倾向于绘制这些图表从底部到顶部，所以

**[00:00:56]** diagrams from the bottom to the top so
> 图表从底部到顶部，所以

**[00:00:56]** diagrams from the bottom to the top so when we follow the inputs here from the
> 图表从底部到顶部，所以当我们从底部到顶部跟踪这里的输入时

**[00:00:56]** when we follow the inputs here from the
> 当我们从底部到顶部跟踪这里的输入时

**[00:00:56]** when we follow the inputs here from the bottom to the top the optimization in
> 当我们从底部到顶部跟踪这里的输入时，反向传播算法中的优化

**[00:00:56]** bottom to the top the optimization in
> 反向传播算法中的优化

**[00:00:56]** bottom to the top the optimization in the back propagation algorithm it goes
> 反向传播算法中的优化是反向进行的，所以从

**[00:00:56]** the back propagation algorithm it goes
> 反向传播算法是反向进行的，所以从

**[00:00:56]** the back propagation algorithm it goes the other way around so you start with
> 反向传播算法是反向进行的，所以从输出开始，有一个损失函数

**[00:00:56]** the other way around so you start with
> 输出开始，有一个损失函数

**[00:00:56]** the other way around so you start with the output there's a loss function
> 输出开始，有一个损失函数，这里没有显示，通常

**[00:00:56]** the output there's a loss function
> 这里没有显示，通常

**[00:00:56]** the output there's a loss function that's not shown here that is usually
> 这里没有显示，通常会被使用，然后你反向传播，那是

**[00:00:56]** that's not shown here that is usually
> 被使用，然后你反向传播，那是

**[00:00:56]** that's not shown here that is usually used and then you back propagate that's
> 被使用，然后你反向传播，那是的，它被称为反向传播，你

**[00:00:56]** used and then you back propagate that's
> 是的，它被称为反向传播，你

**[00:00:56]** used and then you back propagate that's yeah it's called back propagation you go
> 是的，它被称为反向传播，你向后进行，并且是的，你可以看到

**[00:00:56]** yeah it's called back propagation you go
> 向后进行，并且是的，你可以看到

**[00:00:56]** yeah it's called back propagation you go backwards and um yeah you can see um the
> 向后进行，并且是的，你可以看到梯度信号变得更弱，所以

**[00:00:56]** backwards and um yeah you can see um the
> 梯度信号变得更弱，所以

**[00:00:56]** backwards and um yeah you can see um the Great radiant signal becomes weaker so
> 梯度信号变得更弱，所以

**[00:00:56]** Great radiant signal becomes weaker so
> 梯度信号变得更弱，所以

**[00:00:56]** Great radiant signal becomes weaker so you have a
> 梯度信号变得更弱，所以你有

**[00:00:56]** you have a
> 你有

**[00:00:56]** you have a
> 你有

**[00:00:57]** 013 and it it becomes weaker and that
> 0.13，它变得更弱，并且那

**[00:00:57]** 013 and it it becomes weaker and that
> 0.13，它变得更弱，并且那

**[00:00:57]** 013 and it it becomes weaker and that makes makes it harder to learn for the
> 0.13，它变得更弱，并且那使得网络更难学习

**[00:00:57]** makes makes it harder to learn for the
> 使得网络更难学习

**[00:00:57]** makes makes it harder to learn for the network to you know learn how to update
> 使得网络更难学习如何更新

**[00:00:57]** network to you know learn how to update
> 网络如何更新

**[00:00:57]** network to you know learn how to update these weight um parameters in these
> 网络如何更新这些线性层中的权重参数

**[00:00:57]** these weight um parameters in these
> 这些线性层中的权重参数

**[00:00:57]** these weight um parameters in these linear layers and here on the right hand
> 这些线性层中的权重参数，而在右侧

**[00:00:57]** linear layers and here on the right hand
> 线性层，而在右侧

**[00:00:57]** linear layers and here on the right hand side um you see the same architecture
> 线性层，而在右侧，你看到相同的架构

**[00:00:57]** side um you see the same architecture
> 你看到相同的架构

**[00:00:57]** side um you see the same architecture but but I added um shortcut
> 你看到相同的架构，但我添加了快捷

**[00:00:57]** but but I added um shortcut
> 但我添加了快捷

**[00:00:57]** but but I added um shortcut connections and you can see here so by
> 但我添加了快捷连接，你可以看到这里，所以

**[00:00:57]** connections and you can see here so by
> 连接，你可以看到这里，所以

**[00:00:57]** connections and you can see here so by the way I'm not making these numbers up
> 连接，你可以看到这里，所以顺便说一下，这些数字不是我编的

**[00:00:57]** the way I'm not making these numbers up
> 这些数字不是我编的

**[00:00:57]** the way I'm not making these numbers up we have actually or I have code we will
> 这些数字不是我编的，我们实际上有，或者我有代码，我们将

**[00:00:57]** we have actually or I have code we will
> we have actually or I have code we will

**[00:00:57]** we have actually or I have code we will look at this in a few moments but what
> 实际上我们有，或者说我有代码，稍后我们会看到，但

**[00:00:57]** look at this in a few moments but what
> 稍后我们会看到，但

**[00:00:57]** look at this in a few moments but what you can see is when we or when I um add
> 稍后我们会看到，但你可以看到，当我们，或者说当我添加

**[00:00:57]** you can see is when we or when I um add
> 你可以看到，当我们，或者说当我添加

**[00:00:57]** you can see is when we or when I um add these shorter connections um you can see
> 你可以看到，当我们，或者说当我添加这些较短的连接时，你可以看到

**[00:00:57]** these shorter connections um you can see
> 这些较短的连接时，你可以看到

**[00:00:57]** these shorter connections um you can see that we have much larger gradient values
> 这些较短的连接时，你可以看到我们有更大的gradient值

**[00:00:57]** that we have much larger gradient values
> 我们有更大的gradient值

**[00:00:57]** that we have much larger gradient values we have here 1.32 and then
> 我们有更大的gradient值，这里我们看到1.32，然后

**[00:00:57]** we have here 1.32 and then
> 这里我们看到1.32，然后

**[00:00:57]** we have here 1.32 and then 2632 2.2 so we have much larger values
> 这里我们看到1.32，然后2632 2.2，所以值大得多

**[00:00:57]** 2632 2.2 so we have much larger values
> 2632 2.2，所以值大得多

**[00:00:57]** 2632 2.2 so we have much larger values and that helps the network to train
> 2632 2.2，所以值大得多，这有助于网络训练

**[00:00:57]** and that helps the network to train
> 这有助于网络训练

**[00:00:57]** and that helps the network to train better because there's more information
> 这有助于网络训练得更好，因为有了更多信息

**[00:00:57]** better because there's more information
> 更好，因为有了更多信息

**[00:00:57]** better because there's more information for updating these um weights here so
> 更好，因为有了更多信息来更新这些权重，所以

**[00:00:57]** for updating these um weights here so
> 来更新这些权重，所以

**[00:00:57]** for updating these um weights here so let's actually take a look at the code
> 来更新这些权重，所以让我们实际看一下代码

**[00:00:57]** let's actually take a look at the code
> 让我们实际看一下代码

**[00:00:57]** let's actually take a look at the code and um let me show you that this is yeah
> 让我们实际看一下代码，让我向你展示这确实是

**[00:00:57]** and um let me show you that this is yeah
> 让我向你展示这确实是

**[00:00:57]** and um let me show you that this is yeah not just a concept that it actually
> 让我向你展示这确实不仅仅是一个概念，它实际上

**[00:00:57]** not just a concept that it actually
> 不仅仅是一个概念，它实际上

**[00:00:57]** not just a concept that it actually works so I just copy pasted this here um
> 不仅仅是一个概念，它实际上是有效的，所以我只是复制粘贴了这里

**[00:00:58]** works so I just copy pasted this here um
> 是有效的，所以我只是复制粘贴了这里

**[00:00:58]** works so I just copy pasted this here um this is a neur network where we have one
> 是有效的，所以我只是复制粘贴了这里，这是一个神经网络，我们有

**[00:00:58]** this is a neur network where we have one
> 这是一个神经网络，我们有

**[00:00:58]** this is a neur network where we have one two three four five of these modules uh
> 这是一个神经网络，我们有一、二、三、四、五个这样的模块

**[00:00:58]** two three four five of these modules uh
> 一、二、三、四、五个这样的模块

**[00:00:58]** two three four five of these modules uh sequential modules and each module
> 一、二、三、四、五个这样的模块，顺序模块，每个模块

**[00:00:58]** sequential modules and each module
> 顺序模块，每个模块

**[00:00:58]** sequential modules and each module consists of a linear
> 顺序模块，每个模块由一个线性

**[00:00:58]** consists of a linear
> 由一个线性

**[00:00:58]** consists of a linear layer and a nonlinear activation
> 由一个线性层和一个非线性激活

**[00:00:58]** layer and a nonlinear activation
> 层和一个非线性激活

**[00:00:58]** layer and a nonlinear activation function so what you're looking at here
> 层和一个非线性激活函数组成，所以你现在看到的是

**[00:00:58]** function so what you're looking at here
> 函数组成，所以你现在看到的是

**[00:00:58]** function so what you're looking at here is the same thing that you can see here
> 函数组成，所以你现在看到的是和这里一样的东西

**[00:00:58]** is the same thing that you can see here
> 和这里一样的东西

**[00:00:58]** is the same thing that you can see here so 1 2 3 4 5 can see uh 1 2 3 4 5 and
> 和这里一样的东西，所以1 2 3 4 5，可以看到1 2 3 4 5，然后

**[00:00:58]** so 1 2 3 4 5 can see uh 1 2 3 4 5 and
> 所以1 2 3 4 5，可以看到1 2 3 4 5，然后

**[00:00:58]** so 1 2 3 4 5 can see uh 1 2 3 4 5 and then the forward method is computed by
> 所以1 2 3 4 5，可以看到1 2 3 4 5，然后forward方法通过

**[00:00:58]** then the forward method is computed by
> 然后forward方法通过

**[00:00:58]** then the forward method is computed by yeah calling these um layers so we have
> 然后forward方法通过调用这些层来计算，所以我们有

**[00:00:58]** yeah calling these um layers so we have
> 调用这些层来计算，所以我们有

**[00:00:58]** yeah calling these um layers so we have we iterate over the layers here we're
> 调用这些层来计算，所以我们在这里遍历这些层

**[00:00:58]** we iterate over the layers here we're
> 我们在这里遍历这些层

**[00:00:58]** we iterate over the layers here we're calling them and then optionally we add
> 我们在这里遍历这些层，调用它们，然后可选地添加

**[00:00:58]** calling them and then optionally we add
> 调用它们，然后可选地添加

**[00:00:58]** calling them and then optionally we add this shortcut
> 调用它们，然后可选地添加这个shortcut

**[00:00:58]** this shortcut
> 这个shortcut

**[00:00:58]** this shortcut connection um so I'm saying optionally
> 这个shortcut连接，所以我说可选地

**[00:00:58]** connection um so I'm saying optionally
> 连接，所以我说可选地

**[00:00:58]** connection um so I'm saying optionally because just for experimentation
> 连接，所以我说可选地，因为只是为了实验

**[00:00:58]** because just for experimentation
> 因为只是为了实验

**[00:00:58]** because just for experimentation purposes I added um uh uh argument here
> 因为只是为了实验目的，我添加了一个参数

**[00:00:58]** purposes I added um uh uh argument here
> 我添加了一个参数

**[00:00:58]** purposes I added um uh uh argument here use shortcut that we can set to true or
> 我添加了一个参数use shortcut，我们可以设置为true或

**[00:00:58]** use shortcut that we can set to true or
> use shortcut，我们可以设置为true或

**[00:00:58]** use shortcut that we can set to true or false whether to use the shortcut or not
> use shortcut，我们可以设置为true或false，来决定是否使用shortcut

**[00:00:58]** false whether to use the shortcut or not
> false，来决定是否使用shortcut

**[00:00:58]** false whether to use the shortcut or not so if we are not using the shortcut it's
> false，来决定是否使用shortcut，所以如果我们不使用shortcut，它

**[00:00:58]** so if we are not using the shortcut it's
> 所以如果我们不使用shortcut，它

**[00:00:58]** so if we are not using the shortcut it's just like a regular neur Network without
> 所以如果我们不使用shortcut，它就像一个普通的神经网络，没有

**[00:00:58]** just like a regular neur Network without
> 就像一个普通的神经网络，没有

**[00:00:58]** just like a regular neur Network without shortcut and if we have this shortcut
> 就像一个普通的神经网络，没有shortcut，而如果我们有这个shortcut

**[00:00:58]** shortcut and if we have this shortcut
> shortcut，而如果我们有这个shortcut

**[00:00:58]** shortcut and if we have this shortcut then it will add these little shortcut
> shortcut，而如果我们有这个shortcut，那么它会添加这些小的shortcut

**[00:00:58]** then it will add these little shortcut
> 那么它会添加这些小的shortcut

**[00:00:58]** then it will add these little shortcut connections and similar to the paper
> 那么它会添加这些小的shortcut连接，类似于论文中

**[00:00:59]** connections and similar to the paper
> 连接，类似于论文中

**[00:00:59]** connections and similar to the paper what will happen is it will add the
> 连接，类似于论文中，会发生的是它会将

**[00:00:59]** what will happen is it will add the
> 会发生的是它会将

**[00:00:59]** what will happen is it will add the input back to the output of the module
> 会发生的是它会将输入加回到模块的输出

**[00:00:59]** input back to the output of the module
> 输入加回到模块的输出

**[00:00:59]** input back to the output of the module so that's yeah what you're seeing here
> 输入加回到模块的输出，所以这就是你在这里看到的

**[00:00:59]** so that's yeah what you're seeing here
> 所以这就是你在这里看到的

**[00:00:59]** so that's yeah what you're seeing here oh sorry yeah what you're seeing here
> 所以这就是你在这里看到的，哦抱歉，是的，你在这里看到的

**[00:00:59]** oh sorry yeah what you're seeing here
> 哦抱歉，是的，你在这里看到的

**[00:00:59]** oh sorry yeah what you're seeing here and here that's without shortcut
> 哦抱歉，是的，你在这里和这里看到的是没有shortcut的情况

**[00:00:59]** and here that's without shortcut
> 这里是没有shortcut的情况

**[00:00:59]** and here that's without shortcut connection um there's one exception here
> 这里是没有shortcut连接的情况，嗯，有一个例外

**[00:00:59]** connection um there's one exception here
> 连接，嗯，有一个例外

**[00:00:59]** connection um there's one exception here uh if x shape equals layer output shape
> 连接，嗯，有一个例外，如果x的形状等于层的输出形状

**[00:00:59]** uh if x shape equals layer output shape
> 如果x的形状等于层的输出形状

**[00:00:59]** uh if x shape equals layer output shape so this is essentially when we are um
> 如果x的形状等于层的输出形状，那么这基本上就是我们

**[00:00:59]** so this is essentially when we are um
> 那么这基本上就是我们

**[00:00:59]** so this is essentially when we are um it's just one way a simple way how I
> 那么这基本上就是我们实现它的一种简单方式

**[00:00:59]** it's just one way a simple way how I
> 一种简单方式

**[00:00:59]** it's just one way a simple way how I implemented this once we are here um at
> 一种简单方式，一旦我们到了这里

**[00:00:59]** implemented this once we are here um at
> 实现它，一旦我们到了这里

**[00:00:59]** implemented this once we are here um at the end we won't add a shortcut
> 实现它，一旦我们到了这里，最后我们不会添加shortcut

**[00:00:59]** the end we won't add a shortcut
> 最后我们不会添加shortcut

**[00:00:59]** the end we won't add a shortcut connection um because yeah that's that
> 最后我们不会添加shortcut连接，嗯，因为那

**[00:00:59]** connection um because yeah that's that
> 连接，嗯，因为那

**[00:00:59]** connection um because yeah that's that wouldn't work here right so there's no
> 连接，嗯，因为那在这里行不通，所以没有

**[00:00:59]** wouldn't work here right so there's no
> 在这里行不通，所以没有

**[00:00:59]** wouldn't work here right so there's no point adding a shortcut connection to
> 在这里行不通，所以没有必要为输出添加shortcut连接

**[00:00:59]** point adding a shortcut connection to
> 没有必要为输出添加shortcut连接

**[00:00:59]** point adding a shortcut connection to the output um yeah so this is it and
> 没有必要为输出添加shortcut连接，嗯，就是这样

**[00:00:59]** the output um yeah so this is it and
> 嗯，就是这样

**[00:00:59]** the output um yeah so this is it and then I also have a utility function
> 嗯，就是这样，然后我还有一个实用函数

**[00:00:59]** then I also have a utility function
> 然后我还有一个实用函数

**[00:00:59]** then I also have a utility function print gradients where we get the
> 然后我还有一个实用函数print gradients，我们从中获取

**[00:00:59]** print gradients where we get the
> print gradients，我们从中获取

**[00:00:59]** print gradients where we get the output we have some Target so the target
> print gradients，我们从中获取输出，我们有一些目标，所以目标

**[00:00:59]** output we have some Target so the target
> 输出，我们有一些目标，所以目标

**[00:00:59]** output we have some Target so the target really is a place holder here um yeah
> 输出，我们有一些目标，所以目标在这里实际上是一个占位符，嗯

**[00:00:59]** really is a place holder here um yeah
> 在这里实际上是一个占位符，嗯

**[00:00:59]** really is a place holder here um yeah how deep new network training works we
> 在这里实际上是一个占位符，嗯，深度神经网络训练的工作原理是

**[00:00:59]** how deep new network training works we
> 深度神经网络训练的工作原理是

**[00:00:59]** how deep new network training works we usually calculate a loss function
> 深度神经网络训练的工作原理是，我们通常计算一个损失函数

**[00:00:59]** usually calculate a loss function
> 通常计算一个损失函数

**[00:00:59]** usually calculate a loss function function and then we compute the
> 通常计算一个损失函数，然后我们计算

**[00:00:59]** function and then we compute the
> 然后我们计算

**[00:00:59]** function and then we compute the difference between the value that the
> 然后我们计算网络输出的值与目标之间的差异

**[00:01:00]** difference between the value that the
> 网络输出的值与目标之间的差异

**[00:01:00]** difference between the value that the network outputs and the Target in the
> 网络输出的值与目标之间的差异，在损失函数中

**[00:01:00]** network outputs and the Target in the
> 在损失函数中

**[00:01:00]** network outputs and the Target in the loss function so that is like a measure
> 在损失函数中，这就像衡量它们有多相似

**[00:01:00]** loss function so that is like a measure
> 这就像衡量它们有多相似

**[00:01:00]** loss function so that is like a measure of let's say how similar they are and
> 这就像衡量它们有多相似，基于此我们进行反向传播

**[00:01:00]** of let's say how similar they are and
> 基于此我们进行反向传播

**[00:01:00]** of let's say how similar they are and based on that we back propagate now
> 基于此我们进行反向传播，现在在第五章后面

**[00:01:00]** based on that we back propagate now
> 在第五章后面

**[00:01:00]** based on that we back propagate now later on in chapter 5 the the targets
> 在第五章后面，目标将是我们要生成的下一个token

**[00:01:00]** later on in chapter 5 the the targets
> 目标将是我们要生成的下一个token

**[00:01:00]** later on in chapter 5 the the targets will be the next tokens we want to
> 目标将是我们要生成的下一个token，损失函数将是

**[00:01:00]** will be the next tokens we want to
> 损失函数将是

**[00:01:00]** will be the next tokens we want to generate and the loss function will be a
> 损失函数将是交叉熵损失，目前你知道

**[00:01:00]** generate and the loss function will be a
> 交叉熵损失，目前你知道

**[00:01:00]** generate and the loss function will be a cross entropy loss for now you know
> 交叉熵损失，目前你知道，不要太担心这些概念

**[00:01:00]** cross entropy loss for now you know
> 不要太担心这些概念

**[00:01:00]** cross entropy loss for now you know don't worry about these Concepts too
> 不要太担心这些概念，它们在这里只是占位符

**[00:01:00]** don't worry about these Concepts too
> 它们在这里只是占位符

**[00:01:00]** don't worry about these Concepts too much they are just placeholders here and
> 它们在这里只是占位符，MSE损失基本上就是

**[00:01:00]** much they are just placeholders here and
> MSE损失基本上就是

**[00:01:00]** much they are just placeholders here and miss e loss is essentially just the um
> MSE损失基本上就是平方差，均方误差

**[00:01:00]** miss e loss is essentially just the um
> 平方差，均方误差

**[00:01:00]** miss e loss is essentially just the um squared difference the mean squared
> 平方差，均方误差损失，嗯，只是为了简单起见

**[00:01:00]** squared difference the mean squared
> 误差损失，嗯，只是为了简单起见

**[00:01:00]** squared difference the mean squared error loss um just for Simplicity so
> 误差损失，嗯，只是为了简单起见，这样我们就有东西可以反向传播

**[00:01:00]** error loss um just for Simplicity so
> 这样我们就有东西可以反向传播

**[00:01:00]** error loss um just for Simplicity so that we have something to back propagate
> 这样我们就有东西可以反向传播，只是，嗯，我们需要一个损失函数

**[00:01:00]** that we have something to back propagate
> 只是，嗯，我们需要一个损失函数

**[00:01:00]** that we have something to back propagate it's just yeah we need any loss function
> 只是，嗯，我们需要一个损失函数在这里，我只想保持简单

**[00:01:00]** it's just yeah we need any loss function
> 在这里，我只想保持简单

**[00:01:00]** it's just yeah we need any loss function here I just want to keep it simple and
> 在这里，我只想保持简单和自包含，它与token无关

**[00:01:00]** here I just want to keep it simple and
> 自包含，它与token无关

**[00:01:00]** here I just want to keep it simple and self-contained it has nothing to do with
> 自包含，它与token无关，这将在后面出现

**[00:01:00]** self-contained it has nothing to do with
> 这将在后面出现

**[00:01:00]** self-contained it has nothing to do with tokens now this will come later in
> 这将在后面出现，在第五章

**[00:01:00]** tokens now this will come later in
> 在第五章

**[00:01:00]** tokens now this will come later in chapter
> 在第五章，好的，嗯，所以这里让我们执行它

**[00:01:00]** chapter
> 好的，嗯，所以这里让我们执行它

**[00:01:00]** chapter 5 okay um so here let's execute that um
> 好的，嗯，所以这里让我们执行它，然后当然我们需要一些值

**[00:01:00]** 5 okay um so here let's execute that um
> 5 okay um so here let's execute that um

**[00:01:00]** 5 okay um so here let's execute that um and then of course we need some values
> 5 okay um so here let's execute that um and then of course we need some values

**[00:01:00]** and then of course we need some values
> and then of course we need some values

**[00:01:00]** and then of course we need some values so for the layer sizes let's do 3 3 3
> 然后当然我们需要一些数值，对于层大小我们设为3 3 3

**[00:01:00]** so for the layer sizes let's do 3 3 3
> 所以对于层大小我们设为3 3 3

**[00:01:00]** so for the layer sizes let's do 3 3 3 three and one the one here here um the
> 所以对于层大小我们设为3 3 3，三个和一个，这里的一个，嗯

**[00:01:00]** three and one the one here here um the
> 三个和一个，这里的一个，嗯

**[00:01:00]** three and one the one here here um the output size is one because what we want
> 三个和一个，这里的一个，输出大小是1，因为我们想要

**[00:01:00]** output size is one because what we want
> 输出大小是1，因为我们想要

**[00:01:00]** output size is one because what we want is something that has the same Dimension
> 输出大小是1，因为我们想要的是与目标具有相同维度的东西

**[00:01:01]** is something that has the same Dimension
> 是与目标具有相同维度的东西

**[00:01:01]** is something that has the same Dimension as the target so what we want is our
> 是与目标具有相同维度的东西，所以我们想要的是我们的

**[00:01:01]** as the target so what we want is our
> 作为目标，所以我们想要的是我们的

**[00:01:01]** as the target so what we want is our Network to produce a value that is close
> 作为目标，所以我们想要的是我们的网络产生一个尽可能接近目标的值

**[00:01:01]** Network to produce a value that is close
> 网络产生一个尽可能接近目标的值

**[00:01:01]** Network to produce a value that is close as possible to the Target and then we
> 网络产生一个尽可能接近目标的值，然后我们

**[00:01:01]** as possible to the Target and then we
> 尽可能接近目标，然后我们

**[00:01:01]** as possible to the Target and then we optimize it and so forth here we don't
> 尽可能接近目标，然后我们优化它等等，这里我们不

**[00:01:01]** optimize it and so forth here we don't
> 优化它等等，这里我们不

**[00:01:01]** optimize it and so forth here we don't really care about the optimization we
> 优化它等等，这里我们并不真正关心优化，我们

**[00:01:01]** really care about the optimization we
> 真正关心优化，我们

**[00:01:01]** really care about the optimization we just want to compute the the
> 真正关心优化，我们只想计算

**[00:01:01]** just want to compute the the
> 只想计算

**[00:01:01]** just want to compute the the gradients um then let's do some sample
> 只想计算梯度，嗯，然后我们做一些样本

**[00:01:01]** gradients um then let's do some sample
> 梯度，嗯，然后我们做一些样本

**[00:01:01]** gradients um then let's do some sample input so the input is um input size is
> 梯度，嗯，然后我们做一些样本输入，所以输入是，嗯，输入大小是

**[00:01:01]** input so the input is um input size is
> 输入，所以输入是，嗯，输入大小是

**[00:01:01]** input so the input is um input size is three so we need a sample
> 输入，所以输入是，嗯，输入大小是3，所以我们需要一个样本

**[00:01:01]** three so we need a sample
> 3，所以我们需要一个样本

**[00:01:01]** three so we need a sample input that also has three values just
> 3，所以我们需要一个也有三个值的样本输入

**[00:01:01]** input that also has three values just
> 也有三个值的样本输入

**[00:01:01]** input that also has three values just you know for example 1 zus one why have
> 也有三个值的样本输入，比如1，0，1，为什么我选了这些

**[00:01:01]** you know for example 1 zus one why have
> 比如1，0，1，为什么我选了这些

**[00:01:01]** you know for example 1 zus one why have I chosen those just randomly just
> 比如1，0，1，为什么我选了这些，只是随机选的

**[00:01:01]** I chosen those just randomly just
> 我选了这些，只是随机选的

**[00:01:01]** I chosen those just randomly just something that came to mind you can try
> 我选了这些，只是随机选的，就是想到的，你可以尝试

**[00:01:01]** something that came to mind you can try
> 就是想到的，你可以尝试

**[00:01:01]** something that came to mind you can try out different values um and yeah and
> 就是想到的，你可以尝试不同的数值，嗯，是的

**[00:01:01]** out different values um and yeah and
> 不同的数值，嗯，是的

**[00:01:01]** out different values um and yeah and then let's do a um random seat here so
> 不同的数值，嗯，是的，然后我们在这里设置一个随机种子

**[00:01:01]** then let's do a um random seat here so
> 然后我们在这里设置一个随机种子

**[00:01:01]** then let's do a um random seat here so the random seat is so that you get the
> 然后我们在这里设置一个随机种子，这样你就能得到

**[00:01:01]** the random seat is so that you get the
> 随机种子，这样你就能得到

**[00:01:01]** the random seat is so that you get the same results that I get and then let's
> 随机种子，这样你就能得到和我一样的结果，然后我们

**[00:01:01]** same results that I get and then let's
> 和我一样的结果，然后我们

**[00:01:01]** same results that I get and then let's first initialize our Network here so
> 和我一样的结果，然后我们首先初始化我们的网络

**[00:01:01]** first initialize our Network here so
> 首先初始化我们的网络

**[00:01:01]** first initialize our Network here so this is our example Network without
> 首先初始化我们的网络，这是我们的示例网络，没有

**[00:01:01]** this is our example Network without
> 这是我们的示例网络，没有

**[00:01:01]** this is our example Network without shortcut connections so this is printing
> 这是我们的示例网络，没有shortcut连接，所以这是打印

**[00:01:01]** shortcut connections so this is printing
> shortcut连接，所以这是打印

**[00:01:01]** shortcut connections so this is printing the gradients without shortcut
> shortcut连接，所以这是打印没有shortcut连接的梯度

**[00:01:01]** the gradients without shortcut
> 没有shortcut连接的梯度

**[00:01:01]** the gradients without shortcut connections and you can see um yeah
> 没有shortcut连接的梯度，你可以看到，嗯，是的

**[00:01:01]** connections and you can see um yeah
> 你可以看到，嗯，是的

**[00:01:01]** connections and you can see um yeah these are very small values so the
> 你可以看到，嗯，是的，这些值非常小，所以

**[00:01:02]** these are very small values so the
> 这些值非常小，所以

**[00:01:02]** these are very small values so the values in the figure are actually based
> 这些值非常小，所以图中的数值实际上基于

**[00:01:02]** values in the figure are actually based
> 图中的数值实际上基于

**[00:01:02]** values in the figure are actually based on the code here so you can see those
> 图中的数值实际上基于这里的代码，所以你可以看到它们

**[00:01:02]** on the code here so you can see those
> 基于这里的代码，所以你可以看到它们

**[00:01:02]** on the code here so you can see those kind of match um and then we can do the
> 基于这里的代码，所以你可以看到它们大致匹配，嗯，然后我们可以做

**[00:01:02]** kind of match um and then we can do the
> 大致匹配，嗯，然后我们可以做

**[00:01:02]** kind of match um and then we can do the same thing oops with shortcut
> 大致匹配，嗯，然后我们可以做同样的事情，哎呀，使用shortcut连接

**[00:01:02]** connections yeah and you can see now the
> 连接，是的，现在你可以看到

**[00:01:02]** connections yeah and you can see now the
> 连接，是的，现在你可以看到

**[00:01:02]** connections yeah and you can see now the values are much larger so with these
> 连接，是的，现在你可以看到数值大得多，所以有了这些

**[00:01:02]** values are much larger so with these
> 数值大得多，所以有了这些

**[00:01:02]** values are much larger so with these shortcut connections we lift up these
> 数值大得多，所以有了这些shortcut连接，我们提升了这些

**[00:01:02]** shortcut connections we lift up these
> shortcut连接，我们提升了这些

**[00:01:02]** shortcut connections we lift up these gradient values um the network later on
> shortcut连接，我们提升了这些梯度值，嗯，网络之后

**[00:01:02]** gradient values um the network later on
> 梯度值，嗯，网络之后

**[00:01:02]** gradient values um the network later on it will also it will be more than just
> 梯度值，嗯，网络之后，它也会，不仅仅是

**[00:01:02]** it will also it will be more than just
> 它也会，不仅仅是

**[00:01:02]** it will also it will be more than just lifting up the gradient value values
> 它也会，不仅仅是提升这里的梯度值

**[00:01:02]** lifting up the gradient value values
> 提升这里的梯度值

**[00:01:02]** lifting up the gradient value values here in the beginning it will also you
> 提升这里的梯度值，在开始时，它也会

**[00:01:02]** here in the beginning it will also you
> 在开始时，它也会

**[00:01:02]** here in the beginning it will also you know learn how to optimize optimize
> 在开始时，它也会学习如何优化

**[00:01:02]** know learn how to optimize optimize
> 学习如何优化

**[00:01:02]** know learn how to optimize optimize these weight matrices and if there's
> 学习如何优化这些权重矩阵，如果有

**[00:01:02]** these weight matrices and if there's
> 这些权重矩阵，如果有

**[00:01:02]** these weight matrices and if there's something that goes wrong during the
> 这些权重矩阵，如果有东西在过程中出错

**[00:01:02]** something that goes wrong during the
> 训练过程中出现问题时

**[00:01:02]** something that goes wrong during the training these shortcut connections can
> 训练过程中出现问题时，这些快捷连接可以

**[00:01:02]** training these shortcut connections can
> 这些快捷连接可以

**[00:01:02]** training these shortcut connections can always compensate and rescue the network
> 这些快捷连接总能补偿并挽救网络

**[00:01:02]** always compensate and rescue the network
> 总能补偿并挽救网络

**[00:01:02]** always compensate and rescue the network there from totally failing because yeah
> 总能补偿并挽救网络，使其免于完全失败，因为

**[00:01:02]** there from totally failing because yeah
> 使其免于完全失败，因为

**[00:01:02]** there from totally failing because yeah one more time you can see everything is
> 使其免于完全失败，因为，再次强调，你可以看到一切

**[00:01:02]** one more time you can see everything is
> 再次强调，你可以看到一切

**[00:01:02]** one more time you can see everything is dependent on each other so if you have
> 再次强调，你可以看到一切都是相互依赖的，所以如果

**[00:01:02]** dependent on each other so if you have
> 相互依赖的，所以如果

**[00:01:02]** dependent on each other so if you have something that doesn't really work here
> 相互依赖的，所以如果某处出现问题

**[00:01:02]** something that doesn't really work here
> 某处出现问题

**[00:01:02]** something that doesn't really work here um the the later layers um in the
> 某处出现问题，那么后面的层

**[00:01:02]** um the the later layers um in the
> 后面的层

**[00:01:02]** um the the later layers um in the forward pass they're not affected and
> 后面的层在前向传播中不会受到影响

**[00:01:02]** forward pass they're not affected and
> 前向传播中不会受到影响

**[00:01:02]** forward pass they're not affected and likewise if we do back propagation if
> 前向传播中不会受到影响，同样，如果进行反向传播

**[00:01:02]** likewise if we do back propagation if
> 同样，如果进行反向传播

**[00:01:02]** likewise if we do back propagation if something fails here here this layer is
> 同样，如果进行反向传播时某处失败，这一层

**[00:01:02]** something fails here here this layer is
> 某处失败，这一层

**[00:01:02]** something fails here here this layer is not affected it can learn how to skip
> 某处失败，这一层不会受到影响，它可以学习如何跳过

**[00:01:02]** not affected it can learn how to skip
> 不会受到影响，它可以学习如何跳过

**[00:01:02]** not affected it can learn how to skip here and and that's the whole idea
> 不会受到影响，它可以学习如何跳过这里，这就是

**[00:01:03]** here and and that's the whole idea
> 这里，这就是

**[00:01:03]** here and and that's the whole idea between behind shortcut
> 这里，这就是快捷连接背后的核心理念

**[00:01:03]** connections now we are ready to code the
> 连接背后的核心理念。现在我们可以开始编码

**[00:01:03]** connections now we are ready to code the
> 连接背后的核心理念。现在我们可以开始编码

**[00:01:03]** connections now we are ready to code the Transformer block here so the
> 连接背后的核心理念。现在我们可以开始编码Transformer块

**[00:01:03]** Transformer block here so the
> Transformer块

**[00:01:03]** Transformer block here so the Transformer block combines multiple
> Transformer块，它结合了多个

**[00:01:03]** Transformer block combines multiple
> 结合了多个

**[00:01:03]** Transformer block combines multiple modules for example the attention module
> 结合了多个模块，例如注意力模块

**[00:01:03]** modules for example the attention module
> 模块，例如注意力模块

**[00:01:03]** modules for example the attention module the multi-ad attention module we coded
> 模块，例如我们在第3章编码的多头注意力模块

**[00:01:03]** the multi-ad attention module we coded
> 我们在第3章编码的多头注意力模块

**[00:01:03]** the multi-ad attention module we coded earlier in chapter 3 and then it will
> 我们在第3章编码的多头注意力模块，然后它还会

**[00:01:03]** earlier in chapter 3 and then it will
> 然后它还会

**[00:01:03]** earlier in chapter 3 and then it will also have a bunch of linear layers where
> 然后它还会包含一系列线性层

**[00:01:03]** also have a bunch of linear layers where
> 包含一系列线性层

**[00:01:03]** also have a bunch of linear layers where for the linear layers we ALS Al
> 包含一系列线性层，对于这些线性层，我们还介绍了

**[00:01:03]** for the linear layers we ALS Al
> 对于这些线性层，我们还介绍了

**[00:01:03]** for the linear layers we ALS Al introduced several concepts for example
> 对于这些线性层，我们还介绍了一些概念，例如

**[00:01:03]** introduced several concepts for example
> 介绍了一些概念，例如

**[00:01:03]** introduced several concepts for example layer normalization G activations and
> 介绍了一些概念，例如层归一化、GELU激活函数以及

**[00:01:03]** layer normalization G activations and
> 层归一化、GELU激活函数以及

**[00:01:03]** layer normalization G activations and then also the shortcut connections so
> 层归一化、GELU激活函数以及快捷连接

**[00:01:03]** then also the shortcut connections so
> 快捷连接

**[00:01:03]** then also the shortcut connections so going into this figure here again what
> 快捷连接。再次回到这张图

**[00:01:03]** going into this figure here again what
> 再次回到这张图

**[00:01:03]** going into this figure here again what we have done so far in this chapter is
> 再次回到这张图，本章到目前为止我们所做的是

**[00:01:03]** we have done so far in this chapter is
> 本章到目前为止我们所做的是

**[00:01:03]** we have done so far in this chapter is we coded the gbd backbone like the
> 本章到目前为止我们所做的是编码了GPT骨干网络，比如

**[00:01:03]** we coded the gbd backbone like the
> 编码了GPT骨干网络，比如

**[00:01:03]** we coded the gbd backbone like the placeholder architecture and then we
> 编码了GPT骨干网络，比如占位符架构，然后我们

**[00:01:03]** placeholder architecture and then we
> 占位符架构，然后我们

**[00:01:03]** placeholder architecture and then we were filling in the blanks the
> 占位符架构，然后我们填补了空白

**[00:01:03]** were filling in the blanks the
> 填补了空白

**[00:01:03]** were filling in the blanks the placeholders for example layer
> 填补了空白，例如层归一化

**[00:01:03]** placeholders for example layer
> 例如层归一化

**[00:01:03]** placeholders for example layer normalization G activation the feed
> 例如层归一化、GELU激活函数、前馈模块

**[00:01:03]** normalization G activation the feed
> 前馈模块

**[00:01:03]** normalization G activation the feed forward module and then also the
> 前馈模块，以及上一节中的

**[00:01:03]** forward module and then also the
> 上一节中的

**[00:01:03]** forward module and then also the shortcut Connections in the previous
> 上一节中的快捷连接。现在我们已经拥有了所有构建

**[00:01:03]** shortcut Connections in the previous
> 构建

**[00:01:03]** shortcut Connections in the previous section now we have all the building
> 构建模块，可以编码Transformer块了

**[00:01:03]** section now we have all the building
> 模块，可以编码Transformer块了

**[00:01:03]** section now we have all the building blocks here to code the Transformer
> 模块，可以编码Transformer块了，这是GPT架构中非常核心的

**[00:01:03]** blocks here to code the Transformer
> 这是GPT架构中非常核心的

**[00:01:03]** blocks here to code the Transformer block which is a very essential core
> 这是GPT架构中非常核心的部分。Transformer块这个名称源自

**[00:01:03]** block which is a very essential core
> 部分。Transformer块这个名称源自

**[00:01:03]** block which is a very essential core part of the GPT architecture so the name
> 部分。Transformer块这个名称源自原始的Transformer架构

**[00:01:04]** part of the GPT architecture so the name
> 原始的Transformer架构

**[00:01:04]** part of the GPT architecture so the name Transformer Block it's derived from the
> 原始的Transformer架构，该架构在《Attention Is All You Need》论文中提出

**[00:01:04]** Transformer Block it's derived from the
> 该架构在《Attention Is All You Need》论文中提出

**[00:01:04]** Transformer Block it's derived from the original Transformer architecture um
> 该架构在《Attention Is All You Need》论文中提出

**[00:01:04]** original Transformer architecture um
> original Transformer architecture um

**[00:01:04]** original Transformer architecture um yeah proposed in the attention is all un
> original Transformer architecture um yeah proposed in the attention is all un

**[00:01:04]** yeah proposed in the attention is all un
> yeah proposed in the attention is all un

**[00:01:04]** yeah proposed in the attention is all un need paper and if we go to the paper
> 是的，这在《Attention Is All You Need》论文中提出，如果我们去看这篇论文

**[00:01:04]** need paper and if we go to the paper
> 如果我们去看这篇论文

**[00:01:04]** need paper and if we go to the paper there is this very influential figure
> 如果我们去看这篇论文，里面有一个非常有影响力的图

**[00:01:04]** there is this very influential figure
> 里面有一个非常有影响力的图

**[00:01:04]** there is this very influential figure that you might have seen before we
> 里面有一个非常有影响力的图，你可能之前见过，我们

**[00:01:04]** that you might have seen before we
> 你可能之前见过，我们

**[00:01:04]** that you might have seen before we actually already yeah talked about this
> 你可能之前见过，我们实际上已经在之前的章节中讨论过这个

**[00:01:04]** actually already yeah talked about this
> 实际上已经在之前的章节中讨论过这个

**[00:01:04]** actually already yeah talked about this briefly in a previous um section so
> 实际上已经在之前的章节中简要讨论过这个，所以

**[00:01:04]** briefly in a previous um section so
> 简要讨论过这个，所以

**[00:01:04]** briefly in a previous um section so there's an encoder subm module and a
> 简要讨论过这个，所以有一个编码器子模块和一个

**[00:01:04]** there's an encoder subm module and a
> 有一个编码器子模块和一个

**[00:01:04]** there's an encoder subm module and a decoder module and the decoder mod
> 有一个编码器子模块和一个解码器模块，而解码器模块

**[00:01:04]** decoder module and the decoder mod
> 解码器模块，而解码器模块

**[00:01:04]** decoder module and the decoder mod module is yeah what's used in modern
> 解码器模块，而解码器模块正是现代LLM中使用的

**[00:01:04]** module is yeah what's used in modern
> 正是现代LLM中使用的

**[00:01:04]** module is yeah what's used in modern llms in GPT like llms so the enoda
> 正是现代LLM中使用的，比如GPT类LLM，所以编码器模块

**[00:01:04]** llms in GPT like llms so the enoda
> 比如GPT类LLM，所以编码器模块

**[00:01:04]** llms in GPT like llms so the enoda module you can forget about it for now
> 比如GPT类LLM，所以编码器模块你现在可以忽略

**[00:01:04]** module you can forget about it for now
> 你现在可以忽略

**[00:01:04]** module you can forget about it for now it's not used in modern llms like GPT it
> 你现在可以忽略，它不在现代LLM（如GPT）中使用

**[00:01:04]** it's not used in modern llms like GPT it
> 它不在现代LLM（如GPT）中使用

**[00:01:04]** it's not used in modern llms like GPT it only focuses on this right subp part
> 它不在现代LLM（如GPT）中使用，它只关注右侧这个子部分

**[00:01:04]** only focuses on this right subp part
> 只关注右侧这个子部分

**[00:01:04]** only focuses on this right subp part here for text generation and there is
> 只关注右侧这个子部分，用于文本生成，并且有一个

**[00:01:04]** here for text generation and there is
> 用于文本生成，并且有一个

**[00:01:04]** here for text generation and there is this gray block here and this is the
> 用于文本生成，并且有一个灰色块，这就是

**[00:01:04]** this gray block here and this is the
> 这个灰色块，这就是

**[00:01:04]** this gray block here and this is the so-called Transformer block so to show
> 这个灰色块，这就是所谓的Transformer块，所以为了

**[00:01:04]** so-called Transformer block so to show
> 所谓的Transformer块，所以为了

**[00:01:04]** so-called Transformer block so to show it to you in a
> 所谓的Transformer块，所以为了用另一个图展示给你

**[00:01:04]** it to you in a
> 用另一个图展示给你

**[00:01:04]** it to you in a different figure maybe that also helps
> 用另一个图展示给你，也许也有帮助

**[00:01:04]** different figure maybe that also helps
> 也许也有帮助

**[00:01:04]** different figure maybe that also helps here we have the whole GPT model and
> 也许也有帮助，这里我们有整个GPT模型

**[00:01:04]** here we have the whole GPT model and
> 这里我们有整个GPT模型

**[00:01:04]** here we have the whole GPT model and there are several aspects to it we have
> 这里我们有整个GPT模型，它有几个方面，我们有

**[00:01:04]** there are several aspects to it we have
> 它有几个方面，我们有

**[00:01:04]** there are several aspects to it we have the tokenized text that goes into the
> 它有几个方面，我们有分词后的文本输入到

**[00:01:04]** the tokenized text that goes into the
> 分词后的文本输入到

**[00:01:04]** the tokenized text that goes into the model for example every effort moves you
> 分词后的文本输入到模型中，例如“every effort moves you”

**[00:01:05]** model for example every effort moves you
> 模型中，例如“every effort moves you”

**[00:01:05]** model for example every effort moves you as the input text there are embedding
> 模型中，例如“every effort moves you”作为输入文本，有embedding

**[00:01:05]** as the input text there are embedding
> 作为输入文本，有embedding

**[00:01:05]** as the input text there are embedding layers the token and positional
> 作为输入文本，有embedding层，包括token和位置

**[00:01:05]** layers the token and positional
> 层，包括token和位置

**[00:01:05]** layers the token and positional embedding layers in the GPT model and
> 层，包括token和位置embedding层，在GPT模型中

**[00:01:05]** embedding layers in the GPT model and
> embedding层，在GPT模型中

**[00:01:05]** embedding layers in the GPT model and then we already coded the masked
> embedding层，在GPT模型中，然后我们已经编码了掩码

**[00:01:05]** then we already coded the masked
> 然后我们已经编码了掩码

**[00:01:05]** then we already coded the masked multi-ad attention module now the masked
> 然后我们已经编码了掩码多头注意力模块，现在掩码

**[00:01:05]** multi-ad attention module now the masked
> 多头注意力模块，现在掩码

**[00:01:05]** multi-ad attention module now the masked multi-ad attention module is part of
> 多头注意力模块，现在掩码多头注意力模块是

**[00:01:05]** multi-ad attention module is part of
> 多头注意力模块是

**[00:01:05]** multi-ad attention module is part of this bigger Transformer block that
> 多头注意力模块是这个更大的Transformer块的一部分

**[00:01:05]** this bigger Transformer block that
> 这个更大的Transformer块的一部分

**[00:01:05]** this bigger Transformer block that combines some other aspects and here it
> 这个更大的Transformer块的一部分，它结合了其他一些方面，这里

**[00:01:05]** combines some other aspects and here it
> 它结合了其他一些方面，这里

**[00:01:05]** combines some other aspects and here it looks like there's only one Transformer
> 它结合了其他一些方面，这里看起来只有一个Transformer块

**[00:01:05]** looks like there's only one Transformer
> 看起来只有一个Transformer块

**[00:01:05]** looks like there's only one Transformer block but as we will see also later uh
> 看起来只有一个Transformer块，但正如我们稍后也会看到的

**[00:01:05]** block but as we will see also later uh
> 但正如我们稍后也会看到的

**[00:01:05]** block but as we will see also later uh the Transformer block is repeated a
> 但正如我们稍后也会看到的，Transformer块在这个GPT模型中重复了

**[00:01:05]** the Transformer block is repeated a
> Transformer块在这个GPT模型中重复了

**[00:01:05]** the Transformer block is repeated a number of times in this um GPT model so
> Transformer块在这个GPT模型中重复了多次，所以

**[00:01:05]** number of times in this um GPT model so
> 多次，所以

**[00:01:05]** number of times in this um GPT model so it's actually a very important model or
> 多次，所以它实际上是一个非常重要的模型或

**[00:01:05]** it's actually a very important model or
> 它实际上是一个非常重要的模型或

**[00:01:05]** it's actually a very important model or module and the Transformer block takes
> 它实际上是一个非常重要的模型或模块，而Transformer块占据了

**[00:01:05]** module and the Transformer block takes
> 模块，而Transformer块占据了

**[00:01:05]** module and the Transformer block takes up most of the llm basically it's
> 模块，而Transformer块占据了LLM的大部分，基本上它

**[00:01:05]** up most of the llm basically it's
> 占据了LLM的大部分，基本上它

**[00:01:05]** up most of the llm basically it's repeated a lot of times and it's a
> 占据了LLM的大部分，基本上它被重复了很多次，并且它是

**[00:01:05]** repeated a lot of times and it's a
> 被重复了很多次，并且它是

**[00:01:05]** repeated a lot of times and it's a really important aspects of aspect of
> 被重复了很多次，并且它是LLM的一个非常重要的方面

**[00:01:05]** really important aspects of aspect of
> 非常重要的方面

**[00:01:05]** really important aspects of aspect of the llm and um in this section we are
> 非常重要的方面，在本节中，我们将

**[00:01:05]** the llm and um in this section we are
> 在本节中，我们将

**[00:01:05]** the llm and um in this section we are going to look inside what's contained in
> 在本节中，我们将深入探讨Transformer块内部包含的内容

**[00:01:05]** going to look inside what's contained in
> 深入探讨Transformer块内部包含的内容

**[00:01:05]** going to look inside what's contained in this Transformer block so let's go to
> 接下来我们将深入查看这个Transformer块内部包含的内容，让我们进入

**[00:01:05]** this Transformer block so let's go to
> 这个Transformer块，让我们进入

**[00:01:05]** this Transformer block so let's go to section 4.5 here and here we see such a
> 这个Transformer块，让我们进入第4.5节，在这里我们看到这样一个

**[00:01:05]** section 4.5 here and here we see such a
> 第4.5节，在这里我们看到这样一个

**[00:01:05]** section 4.5 here and here we see such a Transformer block
> 第4.5节，在这里我们看到这样一个Transformer块

**[00:01:05]** Transformer block
> Transformer块

**[00:01:05]** Transformer block where it receives embeddings um of token
> Transformer块接收token的embeddings

**[00:01:06]** where it receives embeddings um of token
> 它接收token的embeddings

**[00:01:06]** where it receives embeddings um of token embeddings for example we have uh input
> 它接收token的embeddings，例如我们有输入

**[00:01:06]** embeddings for example we have uh input
> embeddings，例如我们有输入

**[00:01:06]** embeddings for example we have uh input every effort moves you four tokens and
> embeddings，例如我们有输入，每个努力移动四个token

**[00:01:06]** every effort moves you four tokens and
> 每个努力移动四个token

**[00:01:06]** every effort moves you four tokens and they get encoded into 76 768 dimensional
> 每个努力移动四个token，它们被编码为76 768维的

**[00:01:06]** they get encoded into 76 768 dimensional
> 它们被编码为76 768维的

**[00:01:06]** they get encoded into 76 768 dimensional token embeddings as we talked about
> 它们被编码为76 768维的token embeddings，正如我们之前讨论的

**[00:01:06]** token embeddings as we talked about
> token embeddings，正如我们之前讨论的

**[00:01:06]** token embeddings as we talked about earlier and these go into this
> token embeddings，正如我们之前讨论的，这些进入这个

**[00:01:06]** earlier and these go into this
> 这些进入这个

**[00:01:06]** earlier and these go into this Transformer block here and in the
> 这些进入这个Transformer块，在

**[00:01:06]** Transformer block here and in the
> Transformer块中，在

**[00:01:06]** Transformer block here and in the Transformer block we have a layer
> Transformer块中，在Transformer块中我们有一个层

**[00:01:06]** Transformer block we have a layer
> Transformer块中我们有一个层

**[00:01:06]** Transformer block we have a layer normalization layer we have the masked
> Transformer块中我们有一个layer normalization层，我们有来自上一章的

**[00:01:06]** normalization layer we have the masked
> normalization层，我们有来自上一章的

**[00:01:06]** normalization layer we have the masked multi-ad attention module from the
> normalization层，我们有来自上一章的masked multi-head attention模块

**[00:01:06]** multi-ad attention module from the
> multi-head attention模块

**[00:01:06]** multi-ad attention module from the previous chapter we have dropout
> multi-head attention模块，我们有dropout

**[00:01:06]** previous chapter we have dropout
> 我们有dropout

**[00:01:06]** previous chapter we have dropout uh optionally and then here we also you
> 我们有dropout，可选地，然后这里我们还可以

**[00:01:06]** uh optionally and then here we also you
> 可选地，然后这里我们还可以

**[00:01:06]** uh optionally and then here we also you can see we have the shortcut connections
> 可选地，然后这里你还可以看到我们有shortcut connections

**[00:01:06]** can see we have the shortcut connections
> 可以看到我们有shortcut connections

**[00:01:06]** can see we have the shortcut connections so this is like one sub module and then
> 可以看到我们有shortcut connections，所以这就像一个子模块，然后

**[00:01:06]** so this is like one sub module and then
> 所以这就像一个子模块，然后

**[00:01:06]** so this is like one sub module and then we have another layer Norm we have this
> 所以这就像一个子模块，然后我们有另一个layer Norm，我们有这个

**[00:01:06]** we have another layer Norm we have this
> 我们有另一个layer Norm，我们有这个

**[00:01:06]** we have another layer Norm we have this feed forward module where the feed
> 我们有另一个layer Norm，我们有这个feed forward模块，其中feed

**[00:01:06]** feed forward module where the feed
> feed forward模块，其中feed

**[00:01:06]** feed forward module where the feed forward module consists of a linear
> feed forward模块，其中feed forward模块由一个linear

**[00:01:06]** forward module consists of a linear
> forward模块由一个linear

**[00:01:06]** forward module consists of a linear layer a g activation and another linear
> forward模块由一个linear层、一个GELU activation和另一个linear

**[00:01:06]** layer a g activation and another linear
> 层、一个GELU activation和另一个linear

**[00:01:06]** layer a g activation and another linear layer and then optionally we also have
> 层、一个GELU activation和另一个linear层组成，然后可选地我们还有

**[00:01:06]** layer and then optionally we also have
> 层，然后可选地我们还有

**[00:01:06]** layer and then optionally we also have drop out again here and then this feeds
> 层，然后可选地我们还有dropout，然后这输入到

**[00:01:06]** drop out again here and then this feeds
> dropout，然后这输入到

**[00:01:06]** drop out again here and then this feeds to the next Transformer block a number
> dropout，然后这输入到下一个Transformer块，重复若干次

**[00:01:06]** to the next Transformer block a number
> 到下一个Transformer块，重复若干次

**[00:01:06]** to the next Transformer block a number of times so there are usually in a llm
> 到下一个Transformer块，重复若干次，所以在LLM中通常有

**[00:01:06]** of times so there are usually in a llm
> 所以在LLM中通常有

**[00:01:06]** of times so there are usually in a llm there are multiple of these blocks
> 所以在LLM中通常有多个这样的块

**[00:01:06]** there are multiple of these blocks
> 多个这样的块

**[00:01:06]** there are multiple of these blocks following after each other but we will
> 多个这样的块依次排列，但我们将在下一节中更详细地讨论

**[00:01:06]** following after each other but we will
> 依次排列，但我们将在下一节中更详细地讨论

**[00:01:06]** following after each other but we will get to that in more detail in the next
> 依次排列，但我们将在下一节中更详细地讨论，在本节中我们

**[00:01:06]** get to that in more detail in the next
> 在本节中我们

**[00:01:06]** get to that in more detail in the next um section here in this section we are
> 在本节中我们只关注编码这个

**[00:01:07]** um section here in this section we are
> 只关注编码这个

**[00:01:07]** um section here in this section we are just concerned with coding this
> 只关注编码这个Transformer块，所以让我们开始

**[00:01:07]** just concerned with coding this
> Transformer块，所以让我们开始

**[00:01:07]** just concerned with coding this transforma block so let's get to it and
> Transformer块，所以让我们开始看看如何在代码中实现它

**[00:01:07]** transforma block so let's get to it and
> 看看如何在代码中实现它

**[00:01:07]** transforma block so let's get to it and see how we can implement it in code so
> 看看如何在代码中实现它，为此让我创建一个类

**[00:01:07]** see how we can implement it in code so
> 为此让我创建一个类

**[00:01:07]** see how we can implement it in code so for that let me make a class um
> 为此让我创建一个类TransformerBlock，我们将

**[00:01:07]** for that let me make a class um
> TransformerBlock，我们将

**[00:01:07]** for that let me make a class um Transformer block and we are going to
> TransformerBlock，我们将再次使用来自PyTorch的nn.Module

**[00:01:07]** Transformer block and we are going to
> 再次使用来自PyTorch的nn.Module

**[00:01:07]** Transformer block and we are going to use again ann. module from
> 再次使用来自PyTorch的nn.Module，让我实际上复制粘贴一些代码，这样

**[00:01:07]** use again ann. module from
> 让我实际上复制粘贴一些代码，这样

**[00:01:07]** use again ann. module from pytorch and let me just actually co uh
> 让我实际上复制粘贴一些代码，这样速度更快，否则

**[00:01:07]** pytorch and let me just actually co uh
> 速度更快，否则

**[00:01:07]** pytorch and let me just actually co uh copy and paste some of the code so it's
> 速度更快，否则视频总是太长，所以这里我们

**[00:01:07]** copy and paste some of the code so it's
> 视频总是太长，所以这里我们

**[00:01:07]** copy and paste some of the code so it's a bit faster because otherwise the
> 视频总是太长，所以这里我们有构造函数，它再次接收

**[00:01:07]** a bit faster because otherwise the
> 有构造函数，它再次接收

**[00:01:07]** a bit faster because otherwise the videos are always so long so here um we
> a bit faster because otherwise the videos are always so long so here um we

**[00:01:07]** videos are always so long so here um we
> videos are always so long so here um we

**[00:01:07]** videos are always so long so here um we have the Constructor it again receives
> videos are always so long so here um we have the Constructor it again receives

**[00:01:07]** have the Constructor it again receives
> have the Constructor it again receives

**[00:01:07]** have the Constructor it again receives our configuration file then we have the
> 让构造函数再次接收我们的配置文件，然后我们

**[00:01:07]** our configuration file then we have the
> 我们的配置文件，然后我们

**[00:01:07]** our configuration file then we have the multi-ad attention module from the
> 我们的配置文件，然后我们从上一章引入multi-head attention模块

**[00:01:07]** multi-ad attention module from the
> 从上一章引入multi-head attention模块

**[00:01:07]** multi-ad attention module from the previous chapter so I should um ideally
> 从上一章引入multi-head attention模块，所以我应该理想情况下

**[00:01:07]** previous chapter so I should um ideally
> 上一章，所以我应该理想情况下

**[00:01:07]** previous chapter so I should um ideally import it from the previous chapter
> 上一章，所以我应该理想情况下从上一章导入它

**[00:01:07]** import it from the previous chapter
> 从上一章导入它

**[00:01:07]** import it from the previous chapter here so to make this work you would have
> 从上一章导入它到这里，所以要让这个工作，你需要

**[00:01:07]** here so to make this work you would have
> 到这里，所以要让这个工作，你需要

**[00:01:07]** here so to make this work you would have to go in here make a new file for
> 到这里，所以要让这个工作，你需要在这里创建一个新文件

**[00:01:07]** to go in here make a new file for
> 在这里创建一个新文件

**[00:01:07]** to go in here make a new file for example and call this
> 在这里创建一个新文件，例如命名为

**[00:01:07]** example and call this
> 例如命名为

**[00:01:07]** example and call this previous
> 例如命名为previous

**[00:01:07]** previous
> previous

**[00:01:07]** previous chapters
> previous chapters

**[00:01:07]** chapters
> chapters

**[00:01:07]** chapters and in this file
> chapters，在这个文件中

**[00:01:07]** and in this file
> 在这个文件中

**[00:01:07]** and in this file we need to have this multi-ad attention
> 在这个文件中，我们需要有上一章的multi-head attention代码

**[00:01:07]** we need to have this multi-ad attention
> 我们需要有上一章的multi-head attention代码

**[00:01:07]** we need to have this multi-ad attention code from the previous chapter otherwise
> 我们需要有上一章的multi-head attention代码，否则

**[00:01:08]** code from the previous chapter otherwise
> 上一章的代码，否则

**[00:01:08]** code from the previous chapter otherwise yeah it won't
> 上一章的代码，否则它不会

**[00:01:08]** yeah it won't
> 它不会

**[00:01:08]** yeah it won't work so let me copy and paste the code
> 它不会工作，所以让我复制粘贴代码

**[00:01:08]** work so let me copy and paste the code
> 工作，所以让我复制粘贴代码

**[00:01:08]** work so let me copy and paste the code inside here so this is the code from the
> 工作，所以让我把代码复制粘贴到这里，这是来自

**[00:01:08]** inside here so this is the code from the
> 到这里，这是来自

**[00:01:08]** inside here so this is the code from the previous chapter where I have yeah just
> 到这里，这是来自上一章的代码，我刚刚

**[00:01:08]** previous chapter where I have yeah just
> 上一章的代码，我刚刚

**[00:01:08]** previous chapter where I have yeah just copy and pasted some of the functions we
> 上一章的代码，我刚刚复制粘贴了一些函数，我们

**[00:01:08]** copy and pasted some of the functions we
> 复制粘贴了一些函数，我们

**[00:01:08]** copy and pasted some of the functions we have defined the data set from chapter 2
> 复制粘贴了一些函数，我们定义了第2章的数据集

**[00:01:08]** have defined the data set from chapter 2
> 定义了第2章的数据集

**[00:01:08]** have defined the data set from chapter 2 but then also most importantly the
> 定义了第2章的数据集，但最重要的是

**[00:01:08]** but then also most importantly the
> 但最重要的是

**[00:01:08]** but then also most importantly the multihead attention module from chapter
> 但最重要的是来自第3章的multi-head attention模块

**[00:01:08]** multihead attention module from chapter
> 来自第3章的multi-head attention模块

**[00:01:08]** multihead attention module from chapter 3 CU yeah we have to save this because
> 来自第3章的multi-head attention模块，是的，我们必须保存这个，因为

**[00:01:08]** 3 CU yeah we have to save this because
> 是的，我们必须保存这个，因为

**[00:01:08]** 3 CU yeah we have to save this because we are going to import this rather ative
> 是的，我们必须保存这个，因为我们要相对导入这个

**[00:01:08]** we are going to import this rather ative
> 我们要相对导入这个

**[00:01:08]** we are going to import this rather ative um to this notebook so it's a file
> 我们要相对导入这个到笔记本，所以它是一个文件

**[00:01:08]** um to this notebook so it's a file
> 到笔记本，所以它是一个文件

**[00:01:08]** um to this notebook so it's a file previous chapters that contains this
> 到笔记本，所以它是一个文件previous chapters，包含这个

**[00:01:08]** previous chapters that contains this
> previous chapters，包含这个

**[00:01:08]** previous chapters that contains this multi-ad attention um class okay so
> previous chapters，包含这个multi-head attention类，好的，所以

**[00:01:08]** multi-ad attention um class okay so
> multi-head attention类，好的，所以

**[00:01:08]** multi-ad attention um class okay so let's um work with this
> multi-head attention类，好的，所以让我们现在处理这个

**[00:01:08]** let's um work with this
> 让我们现在处理这个

**[00:01:08]** let's um work with this now and let me briefly discuss what we
> 让我们现在处理这个，让我简要讨论一下我们

**[00:01:08]** now and let me briefly discuss what we
> 现在，让我简要讨论一下我们

**[00:01:08]** now and let me briefly discuss what we are looking at here so we have the
> 现在，让我简要讨论一下我们在这里看到的内容，所以我们有

**[00:01:08]** are looking at here so we have the
> 在这里看到的内容，所以我们有

**[00:01:08]** are looking at here so we have the multi-ad attention class and it is using
> 在这里看到的内容，所以我们有multi-head attention类，它正在使用

**[00:01:08]** multi-ad attention class and it is using
> multi-head attention类，它正在使用

**[00:01:08]** multi-ad attention class and it is using the configuration file for the input
> multi-head attention类，它正在使用配置文件作为输入

**[00:01:08]** the configuration file for the input
> 配置文件作为输入

**[00:01:08]** the configuration file for the input embeddings embed them that's 768 the
> 配置文件作为输入embeddings，嵌入维度是768

**[00:01:08]** embeddings embed them that's 768 the
> embeddings，嵌入维度是768

**[00:01:08]** embeddings embed them that's 768 the output Dimensions they're also 768 and
> embeddings，嵌入维度是768，输出维度也是768

**[00:01:08]** output Dimensions they're also 768 and
> 输出维度也是768

**[00:01:08]** output Dimensions they're also 768 and our case so this is what we
> 输出维度也是768，在我们的情况下，所以这是我们

**[00:01:08]** our case so this is what we
> 在我们的情况下，所以这是我们

**[00:01:08]** our case so this is what we defined up here at the beginning of this
> 在我们的情况下，所以这是我们在本章开头定义的

**[00:01:09]** defined up here at the beginning of this
> 在开头定义的

**[00:01:09]** defined up here at the beginning of this um chapter now we Define a context
> 在开头定义的，现在我们定义一个context length

**[00:01:09]** um chapter now we Define a context
> 现在我们定义一个context length

**[00:01:09]** um chapter now we Define a context length how long our context is allowed
> 现在我们定义一个context length，我们的上下文允许有多长

**[00:01:09]** length how long our context is allowed
> 上下文允许有多长

**[00:01:09]** length how long our context is allowed to be then we Define the number of heads
> 上下文允许有多长，然后我们定义head的数量

**[00:01:09]** to be then we Define the number of heads
> 然后我们定义head的数量

**[00:01:09]** to be then we Define the number of heads whether we want to use Dropout or not so
> 然后我们定义head的数量，是否使用Dropout，所以

**[00:01:09]** whether we want to use Dropout or not so
> 是否使用Dropout，所以

**[00:01:09]** whether we want to use Dropout or not so the dropout rate and the qkv bias
> 是否使用Dropout，所以dropout rate和qkv bias

**[00:01:09]** the dropout rate and the qkv bias
> dropout rate和qkv bias

**[00:01:09]** the dropout rate and the qkv bias whether we want to use the bias unit in
> dropout rate和qkv bias，是否在

**[00:01:09]** whether we want to use the bias unit in
> 是否在

**[00:01:09]** whether we want to use the bias unit in the linear layers but this is all things
> 是否要在线性层中使用偏置单元，但这些内容

**[00:01:09]** the linear layers but this is all things
> 线性层，但这些内容

**[00:01:09]** the linear layers but this is all things we covered in chapter 3 so I don't want
> 线性层，但这些内容我们在第3章已经讲过，所以我不想

**[00:01:09]** we covered in chapter 3 so I don't want
> 我们在第3章已经讲过，所以我不想

**[00:01:09]** we covered in chapter 3 so I don't want to go into too much detail here um more
> 我们在第3章已经讲过，所以我不想在这里过多赘述，嗯，更有趣的是

**[00:01:09]** to go into too much detail here um more
> 在这里过多赘述，嗯，更有趣的是

**[00:01:09]** to go into too much detail here um more interestingly we have now this feed
> 在这里过多赘述，嗯，更有趣的是，现在我们有了这个前馈

**[00:01:09]** interestingly we have now this feed
> 现在我们有了这个前馈

**[00:01:09]** interestingly we have now this feed forward module which we defined in a
> 现在我们有了这个前馈模块，我们在

**[00:01:09]** forward module which we defined in a
> 模块，我们在

**[00:01:09]** forward module which we defined in a previous section and then we also have
> 模块，我们在前一节中定义过，然后我们还有

**[00:01:09]** previous section and then we also have
> 前一节中定义过，然后我们还有

**[00:01:09]** previous section and then we also have two places where we have layer Norm so
> 前一节中定义过，然后我们还有两个地方使用了layer Norm，所以

**[00:01:09]** two places where we have layer Norm so
> 两个地方使用了layer Norm，所以

**[00:01:09]** two places where we have layer Norm so the reason why we instantiate two
> 两个地方使用了layer Norm，所以为什么我们要实例化两个

**[00:01:09]** the reason why we instantiate two
> 为什么我们要实例化两个

**[00:01:09]** the reason why we instantiate two separate layer Norm objects is that they
> 为什么我们要实例化两个独立的layer Norm对象，是因为它们

**[00:01:09]** separate layer Norm objects is that they
> 独立的layer Norm对象，是因为它们

**[00:01:09]** separate layer Norm objects is that they have trainable parameters and um we are
> 独立的layer Norm对象，是因为它们有可训练参数，嗯，我们

**[00:01:09]** have trainable parameters and um we are
> 有可训练参数，嗯，我们

**[00:01:09]** have trainable parameters and um we are using them at two different places and
> 有可训练参数，嗯，我们在两个不同的地方使用它们，并且

**[00:01:09]** using them at two different places and
> 在两个不同的地方使用它们，并且

**[00:01:09]** using them at two different places and in both places we want the network to
> 在两个不同的地方使用它们，在这两个地方我们都希望网络

**[00:01:09]** in both places we want the network to
> 在这两个地方我们都希望网络

**[00:01:09]** in both places we want the network to optimize the parameters so we won't
> 在这两个地方我们都希望网络优化这些参数，所以我们不会

**[00:01:09]** optimize the parameters so we won't
> 优化这些参数，所以我们不会

**[00:01:09]** optimize the parameters so we won't share the parameters it it won't be like
> 优化这些参数，所以我们不会共享参数，它不会像

**[00:01:09]** share the parameters it it won't be like
> 共享参数，它不会像

**[00:01:09]** share the parameters it it won't be like one module that is used in two places it
> 共享参数，它不会像是一个模块在两个地方使用，它

**[00:01:09]** one module that is used in two places it
> 是一个模块在两个地方使用，它

**[00:01:09]** one module that is used in two places it will be two distinct layer normalization
> 是一个模块在两个地方使用，它会是两个不同的layer normalization

**[00:01:09]** will be two distinct layer normalization
> 会是两个不同的layer normalization

**[00:01:09]** will be two distinct layer normalization modules and then yeah um the drop out
> 会是两个不同的layer normalization模块，然后，嗯，drop out

**[00:01:09]** modules and then yeah um the drop out
> 模块，然后，嗯，drop out

**[00:01:09]** modules and then yeah um the drop out for another place so there are multiple
> 模块，然后，嗯，drop out在另一个地方，所以有多个

**[00:01:10]** for another place so there are multiple
> 在另一个地方，所以有多个

**[00:01:10]** for another place so there are multiple places where we use drop out I think it
> 在另一个地方，所以有多个地方我们使用了drop out，我想

**[00:01:10]** places where we use drop out I think it
> 地方我们使用了drop out，我想

**[00:01:10]** places where we use drop out I think it will become more clear once I implement
> 地方我们使用了drop out，我想一旦我实现

**[00:01:10]** will become more clear once I implement
> 一旦我实现

**[00:01:10]** will become more clear once I implement the forward
> 一旦我实现前向传播，就会更清楚

**[00:01:10]** method let's do it like this and in the
> 方法，让我们这样做，并且在

**[00:01:10]** method let's do it like this and in the
> 方法，让我们这样做，并且在

**[00:01:10]** method let's do it like this and in the forward method um yeah one thing that we
> 方法，让我们这样做，并且在forward方法中，嗯，有一件事我们

**[00:01:10]** forward method um yeah one thing that we
> forward方法中，嗯，有一件事我们

**[00:01:10]** forward method um yeah one thing that we need is of course the input input X and
> forward方法中，嗯，有一件事我们需要的是输入，输入X，以及

**[00:01:10]** need is of course the input input X and
> 需要的是输入，输入X，以及

**[00:01:10]** need is of course the input input X and then let me also copy the code here so
> 需要的是输入，输入X，然后让我也把代码复制到这里，这样

**[00:01:10]** then let me also copy the code here so
> 然后让我也把代码复制到这里，这样

**[00:01:10]** then let me also copy the code here so it's a bit um faster in the video so
> 然后让我也把代码复制到这里，这样在视频中会快一点，所以

**[00:01:10]** it's a bit um faster in the video so
> 在视频中会快一点，所以

**[00:01:10]** it's a bit um faster in the video so here what we are looking at is how we
> 在视频中会快一点，所以这里我们看的是我们如何

**[00:01:10]** here what we are looking at is how we
> 这里我们看的是我们如何

**[00:01:10]** here what we are looking at is how we are using the things we have defined
> 这里我们看的是我们如何使用在Transformer块中定义的内容

**[00:01:10]** are using the things we have defined
> 使用在Transformer块中定义的内容

**[00:01:10]** are using the things we have defined here in the Transformer block and this
> 使用在Transformer块中定义的内容，这

**[00:01:10]** here in the Transformer block and this
> 在Transformer块中定义的内容，这

**[00:01:10]** here in the Transformer block and this will be mimicking what you are seeing
> 在Transformer块中定义的内容，这将模仿你在这里看到的

**[00:01:10]** will be mimicking what you are seeing
> 将模仿你在这里看到的

**[00:01:10]** will be mimicking what you are seeing here we have a layer Norm multi-ad
> 将模仿你在这里看到的，这里我们有一个layer Norm、multi-ad

**[00:01:10]** here we have a layer Norm multi-ad
> 这里我们有一个layer Norm、multi-ad

**[00:01:10]** here we have a layer Norm multi-ad attention optionally Dropout we have the
> 这里我们有一个layer Norm、multi-ad attention，可选地使用Dropout，我们有

**[00:01:10]** attention optionally Dropout we have the
> attention，可选地使用Dropout，我们有

**[00:01:10]** attention optionally Dropout we have the shortcut connection here another layer
> attention，可选地使用Dropout，我们有这里的快捷连接，另一个layer

**[00:01:10]** shortcut connection here another layer
> 快捷连接，另一个layer

**[00:01:10]** shortcut connection here another layer Norm then the feed forward module and
> 快捷连接，另一个layer Norm，然后是前馈模块，以及

**[00:01:10]** Norm then the feed forward module and
> Norm，然后是前馈模块，以及

**[00:01:10]** Norm then the feed forward module and then uh optionally Dropout again and
> Norm，然后是前馈模块，然后，嗯，再次可选地使用Dropout，以及

**[00:01:10]** then uh optionally Dropout again and
> 然后，嗯，再次可选地使用Dropout，以及

**[00:01:10]** then uh optionally Dropout again and then a shortcut connection so this is
> 然后，嗯，再次可选地使用Dropout，以及一个快捷连接，所以这是

**[00:01:10]** then a shortcut connection so this is
> 一个快捷连接，所以这是

**[00:01:10]** then a shortcut connection so this is the first layer Norm here so first I
> 一个快捷连接，所以这是第一个layer Norm，所以首先我

**[00:01:10]** the first layer Norm here so first I
> 第一个layer Norm，所以首先我

**[00:01:10]** the first layer Norm here so first I should say we are saving this shortcut
> 第一个layer Norm，所以首先我应该说我们保存了这个快捷

**[00:01:10]** should say we are saving this shortcut
> 应该说我们保存了这个快捷

**[00:01:10]** should say we are saving this shortcut connection because we are going to add
> 应该说我们保存了这个快捷连接，因为我们稍后要添加

**[00:01:10]** connection because we are going to add
> 连接，因为我们稍后要添加

**[00:01:10]** connection because we are going to add this later so this is um what we are
> 连接，因为我们稍后要添加它，所以这是，嗯，我们

**[00:01:10]** this later so this is um what we are
> 它，所以这是，嗯，我们

**[00:01:10]** this later so this is um what we are creating here this first shortcut then
> 它，所以这是，嗯，我们在这里创建的第一个快捷连接

**[00:01:11]** creating here this first shortcut then
> 首先在这里创建这个快捷连接

**[00:01:11]** creating here this first shortcut then we have layer nor one which you can see
> 首先在这里创建这个快捷连接，然后我们有layer norm 1，你可以看到

**[00:01:11]** we have layer nor one which you can see
> 我们有layer norm 1，你可以看到

**[00:01:11]** we have layer nor one which you can see here then we have the attention which
> 我们有layer norm 1，在这里然后我们有attention

**[00:01:11]** here then we have the attention which
> 在这里然后我们有attention

**[00:01:11]** here then we have the attention which corresponds to this one um the Dropout
> 在这里然后我们有attention，它对应这个，嗯，Dropout

**[00:01:11]** corresponds to this one um the Dropout
> 它对应这个，嗯，Dropout

**[00:01:11]** corresponds to this one um the Dropout for this um shortcut connection here so
> 它对应这个，嗯，Dropout，用于这个快捷连接，所以

**[00:01:11]** for this um shortcut connection here so
> 用于这个快捷连接，所以

**[00:01:11]** for this um shortcut connection here so I'm just calling this shortcut
> 用于这个快捷连接，所以我只是称它为快捷

**[00:01:11]** I'm just calling this shortcut
> 我只是称它为快捷

**[00:01:11]** I'm just calling this shortcut connection because it's uh before it
> 我只是称它为快捷连接，因为它是在它之前

**[00:01:11]** connection because it's uh before it
> 连接，因为它是在它之前

**[00:01:11]** connection because it's uh before it goes in here you can also give it a
> 连接，因为它是在它之前进入这里，你也可以给它一个

**[00:01:11]** goes in here you can also give it a
> 进入这里，你也可以给它一个

**[00:01:11]** goes in here you can also give it a different name it's just you know
> 进入这里，你也可以给它一个不同的名字，这只是你知道

**[00:01:11]** different name it's just you know
> 不同的名字，这只是你知道

**[00:01:11]** different name it's just you know something I came up with not very
> 不同的名字，这只是你知道我想出来的，不是很

**[00:01:11]** something I came up with not very
> 我想出来的，不是很

**[00:01:11]** something I came up with not very creative here um and then we are adding
> 我想出来的，不是很创意，嗯，然后我们添加

**[00:01:11]** creative here um and then we are adding
> 创意，嗯，然后我们添加

**[00:01:11]** creative here um and then we are adding this shortcut connection back so this is
> 创意，嗯，然后我们添加这个快捷连接回来，所以这是

**[00:01:11]** this shortcut connection back so this is
> 这个快捷连接回来，所以这是

**[00:01:11]** this shortcut connection back so this is um this part corresponds to this first
> 这个快捷连接回来，所以这是嗯，这部分对应第一个

**[00:01:11]** um this part corresponds to this first
> 嗯，这部分对应第一个

**[00:01:11]** um this part corresponds to this first block here that you can see inside the
> 嗯，这部分对应第一个块，你可以在

**[00:01:11]** block here that you can see inside the
> 块，你可以在

**[00:01:11]** block here that you can see inside the Transformer block now next we have
> 块，你可以在Transformer块内部看到，接下来我们有

**[00:01:11]** Transformer block now next we have
> Transformer块，接下来我们有

**[00:01:11]** Transformer block now next we have another layer Norm so the layer
> Transformer块，接下来我们有另一个layer Norm，所以layer

**[00:01:11]** another layer Norm so the layer
> 另一个layer Norm，所以layer

**[00:01:11]** another layer Norm so the layer Norm oops is here again before that we
> 另一个layer Norm，所以layer Norm，哎呀，在这里再次出现，在此之前我们

**[00:01:11]** Norm oops is here again before that we
> Norm，哎呀，在这里再次出现，在此之前我们

**[00:01:11]** Norm oops is here again before that we save the shortcut connection the
> Norm，哎呀，在这里再次出现，在此之前我们保存快捷连接

**[00:01:11]** save the shortcut connection the
> 保存快捷连接

**[00:01:11]** save the shortcut connection the shortcut connection you can see here
> 保存快捷连接，快捷连接你可以在这里看到

**[00:01:11]** shortcut connection you can see here
> 快捷连接你可以在这里看到

**[00:01:11]** shortcut connection you can see here right so then we have a second layer
> 快捷连接你可以在这里看到，对吧，然后我们有第二个layer

**[00:01:11]** right so then we have a second layer
> 对吧，然后我们有第二个layer

**[00:01:11]** right so then we have a second layer Norm then the feed forward module which
> 对吧，然后我们有第二个layer Norm，然后是feed forward模块

**[00:01:11]** Norm then the feed forward module which
> Norm，然后是feed forward模块

**[00:01:11]** Norm then the feed forward module which you can see here this is calling the
> Norm，然后是feed forward模块，你可以在这里看到，这是调用

**[00:01:11]** you can see here this is calling the
> 你可以在这里看到，这是调用

**[00:01:11]** you can see here this is calling the feed forward module and then yeah we
> 你可以在这里看到，这是调用feed forward模块，然后是的，我们

**[00:01:11]** feed forward module and then yeah we
> feed forward模块，然后是的，我们

**[00:01:11]** feed forward module and then yeah we have another Dropout here and then we
> feed forward模块，然后是的，我们有另一个Dropout在这里，然后我们

**[00:01:12]** have another Dropout here and then we
> 有另一个Dropout在这里，然后我们

**[00:01:12]** have another Dropout here and then we add back the shortcut connection and
> 有另一个Dropout在这里，然后我们添加回快捷连接，并且

**[00:01:12]** add back the shortcut connection and
> 添加回快捷连接，并且

**[00:01:12]** add back the shortcut connection and this is essentially what you are seeing
> 添加回快捷连接，并且这基本上就是你看到的

**[00:01:12]** this is essentially what you are seeing
> 这基本上就是你看到的

**[00:01:12]** this is essentially what you are seeing here this is the Transformer block and
> 这基本上就是你看到的这里，这是Transformer块，并且

**[00:01:12]** here this is the Transformer block and
> 这里，这是Transformer块，并且

**[00:01:12]** here this is the Transformer block and to be honest there's not much really
> 这里，这是Transformer块，并且老实说，并没有太多

**[00:01:12]** to be honest there's not much really
> 老实说，并没有太多

**[00:01:12]** to be honest there's not much really that goes on inside here it is
> 老实说，并没有太多真正在这里发生，它

**[00:01:12]** that goes on inside here it is
> 真正在这里发生，它

**[00:01:12]** that goes on inside here it is essentially just encapsulating multiple
> 真正在这里发生，它基本上只是封装了多个

**[00:01:12]** essentially just encapsulating multiple
> 基本上只是封装了多个

**[00:01:12]** essentially just encapsulating multiple of these ideas and the reason why we
> 基本上只是封装了多个这些想法，而我们之所以

**[00:01:12]** of these ideas and the reason why we
> 这些想法，而我们之所以

**[00:01:12]** of these ideas and the reason why we even create this Transformer block and
> 这些想法，而我们之所以甚至创建这个Transformer块，并且

**[00:01:12]** even create this Transformer block and
> 甚至创建这个Transformer块，并且

**[00:01:12]** even create this Transformer block and why we don't add this um directly to the
> 甚至创建这个Transformer块，并且我们不直接添加这个到

**[00:01:12]** why we don't add this um directly to the
> 我们不直接添加这个到

**[00:01:12]** why we don't add this um directly to the llm is so that we can reuse that so if I
> 我们不直接添加这个到LLM，是为了我们可以重用，所以如果我

**[00:01:12]** llm is so that we can reuse that so if I
> LLM，是为了我们可以重用，所以如果我

**[00:01:12]** llm is so that we can reuse that so if I go up again to to what we have done in
> LLM，是为了我们可以重用，所以如果我再次回到我们在本章开头所做的

**[00:01:12]** go up again to to what we have done in
> 回到我们在本章开头所做的

**[00:01:12]** go up again to to what we have done in the beginning of this chapter so you can
> 回到我们在本章开头所做的，所以你可以

**[00:01:12]** the beginning of this chapter so you can
> 本章开头，所以你可以

**[00:01:12]** the beginning of this chapter so you can see we have these embedding layers then
> 本章开头，所以你可以看到我们有这些embedding层，然后

**[00:01:12]** see we have these embedding layers then
> 看到我们有这些embedding层，然后

**[00:01:12]** see we have these embedding layers then we have um drop out defined here maybe
> 看到我们有这些embedding层，然后我们有嗯，Dropout在这里定义，也许

**[00:01:12]** we have um drop out defined here maybe
> 我们有嗯，Dropout在这里定义，也许

**[00:01:12]** we have um drop out defined here maybe ignore that for now what's interesting
> 我们有嗯，Dropout在这里定义，也许先忽略这个，有趣的是

**[00:01:12]** ignore that for now what's interesting
> 先忽略这个，有趣的是

**[00:01:12]** ignore that for now what's interesting is here that we are repeating the
> 先忽略这个，有趣的是这里我们在重复

**[00:01:12]** is here that we are repeating the
> 这里我们在重复

**[00:01:12]** is here that we are repeating the Transformer block so there is a argument
> 这里我们在重复Transformer块，所以有一个参数

**[00:01:12]** Transformer block so there is a argument
> Transformer block 所以有一个参数

**[00:01:12]** Transformer block so there is a argument here or a value we Define as n layers
> Transformer block 所以这里有一个参数或我们定义为 n layers 的值

**[00:01:12]** here or a value we Define as n layers
> 这里有一个参数或我们定义为 n layers 的值

**[00:01:12]** here or a value we Define as n layers and here we have 12 layers so that means
> 这里有一个参数或我们定义为 n layers 的值，这里我们有 12 层，这意味着

**[00:01:12]** and here we have 12 layers so that means
> 这里我们有 12 层，这意味着

**[00:01:12]** and here we have 12 layers so that means we are repeating this Transformer block
> 这里我们有 12 层，这意味着我们重复这个 Transformer block

**[00:01:12]** we are repeating this Transformer block
> 我们重复这个 Transformer block

**[00:01:12]** we are repeating this Transformer block 12 times and then we have a final layer
> 我们重复这个 Transformer block 12 次，然后有一个最终层

**[00:01:12]** 12 times and then we have a final layer
> 12 次，然后有一个最终层

**[00:01:12]** 12 times and then we have a final layer Norm and U output layer but so that we
> 12 次，然后有一个最终层 Norm 和 U 输出层，但这样我们

**[00:01:12]** Norm and U output layer but so that we
> Norm 和 U 输出层，但这样我们

**[00:01:12]** Norm and U output layer but so that we can make this code very concise we
> Norm 和 U 输出层，但这样我们可以让代码非常简洁，我们

**[00:01:13]** can make this code very concise we
> 可以让代码非常简洁，我们

**[00:01:13]** can make this code very concise we encapsulate this in this Transformer
> 可以让代码非常简洁，我们将其封装在这个 Transformer

**[00:01:13]** encapsulate this in this Transformer
> 封装在这个 Transformer

**[00:01:13]** encapsulate this in this Transformer block class which you can see here which
> 封装在这个 Transformer block 类中，你可以在这里看到，这个

**[00:01:13]** block class which you can see here which
> block 类，你可以在这里看到，这个

**[00:01:13]** block class which you can see here which we have just coded so let's just make
> block 类，你可以在这里看到，这个我们刚刚编码完成，所以让我们

**[00:01:13]** we have just coded so let's just make
> 我们刚刚编码完成，所以让我们

**[00:01:13]** we have just coded so let's just make sure that it works so I executed this
> 我们刚刚编码完成，所以让我们确保它工作，我执行了这个

**[00:01:13]** sure that it works so I executed this
> 确保它工作，我执行了这个

**[00:01:13]** sure that it works so I executed this cell not making the mistake I'm usually
> 确保它工作，我执行了这个单元格，没有犯我通常犯的错误

**[00:01:13]** cell not making the mistake I'm usually
> 单元格，没有犯我通常犯的错误

**[00:01:13]** cell not making the mistake I'm usually making and then we are trying it out in
> 单元格，没有犯我通常犯的错误，然后我们在实践中测试它

**[00:01:13]** making and then we are trying it out in
> 在实践中测试它

**[00:01:13]** making and then we are trying it out in practice I'm using a random seat again
> 在实践中测试它，我再次使用了一个随机种子

**[00:01:13]** practice I'm using a random seat again
> 我再次使用了一个随机种子

**[00:01:13]** practice I'm using a random seat again so that you get the same results that I
> 我再次使用了一个随机种子，这样你就能得到和我一样的结果

**[00:01:13]** so that you get the same results that I
> 这样你就能得到和我一样的结果

**[00:01:13]** so that you get the same results that I do and then let's do a small
> 这样你就能得到和我一样的结果，然后让我们做一个小的

**[00:01:13]** do and then let's do a small
> 让我们做一个小的

**[00:01:13]** do and then let's do a small small example um let's do
> 让我们做一个小的例子，嗯，让我们做

**[00:01:13]** small example um let's do
> 小的例子，嗯，让我们做

**[00:01:13]** small example um let's do two examples with four tokens each so
> 小的例子，嗯，让我们做两个例子，每个有四个 tokens，所以

**[00:01:13]** two examples with four tokens each so
> 两个例子，每个有四个 tokens，所以

**[00:01:13]** two examples with four tokens each so this would be something similar to here
> 两个例子，每个有四个 tokens，所以这类似于这里的

**[00:01:13]** this would be something similar to here
> 这类似于这里的

**[00:01:13]** this would be something similar to here where we have 1 2 3 four tokens and
> 这类似于这里的，我们有 1 2 3 四个 tokens 和

**[00:01:13]** where we have 1 2 3 four tokens and
> 我们有 1 2 3 四个 tokens 和

**[00:01:13]** where we have 1 2 3 four tokens and let's just assume they are embedded in a
> 我们有 1 2 3 四个 tokens，让我们假设它们嵌入在一个

**[00:01:13]** let's just assume they are embedded in a
> 让我们假设它们嵌入在一个

**[00:01:13]** let's just assume they are embedded in a 700 68 dimensional space which
> 让我们假设它们嵌入在一个 768 维空间中，这

**[00:01:13]** 700 68 dimensional space which
> 768 维空间中，这

**[00:01:13]** 700 68 dimensional space which corresponds to what we have to find here
> 768 维空间中，这对应于我们在这里定义的

**[00:01:13]** corresponds to what we have to find here
> 对应于我们在这里定义的

**[00:01:13]** corresponds to what we have to find here we will be working with real data soon
> 对应于我们在这里定义的，我们很快会处理真实数据

**[00:01:13]** we will be working with real data soon
> 我们很快会处理真实数据

**[00:01:13]** we will be working with real data soon so for this part just to keep it small
> 我们很快会处理真实数据，所以这部分为了保持小巧

**[00:01:13]** so for this part just to keep it small
> 所以这部分为了保持小巧

**[00:01:13]** so for this part just to keep it small and self-contained using this random
> 所以这部分为了保持小巧和自包含，使用这个随机

**[00:01:13]** and self-contained using this random
> 和自包含，使用这个随机

**[00:01:13]** and self-contained using this random input but we will be working with real
> 和自包含，使用这个随机输入，但我们很快会处理真实

**[00:01:13]** input but we will be working with real
> 输入，但我们很快会处理真实

**[00:01:13]** input but we will be working with real data soon in chapter 5 so not too far
> 输入，但我们很快会处理真实数据，在第 5 章，所以不远了

**[00:01:14]** data soon in chapter 5 so not too far
> 数据，在第 5 章，所以不远了

**[00:01:14]** data soon in chapter 5 so not too far away and then um we Define the
> 数据，在第 5 章，所以不远了，然后嗯，我们定义

**[00:01:14]** away and then um we Define the
> 然后嗯，我们定义

**[00:01:14]** away and then um we Define the Transformer block
> 然后嗯，我们定义 Transformer block

**[00:01:14]** Transformer block
> Transformer block

**[00:01:14]** Transformer block so we do Transformer block and then we
> Transformer block，所以我们做 Transformer block，然后我们

**[00:01:14]** so we do Transformer block and then we
> 所以我们做 Transformer block，然后我们

**[00:01:14]** so we do Transformer block and then we provide the configuration file that we
> 所以我们做 Transformer block，然后我们提供配置文件，这个

**[00:01:14]** provide the configuration file that we
> 提供配置文件，这个

**[00:01:14]** provide the configuration file that we had at the top of this
> 提供配置文件，这个我们在笔记本顶部有的

**[00:01:14]** had at the top of this
> 我们在笔记本顶部有的

**[00:01:14]** had at the top of this notebook um and then yeah we can use it
> 我们在笔记本顶部有的，嗯，然后是的，我们可以使用它

**[00:01:14]** notebook um and then yeah we can use it
> 嗯，然后是的，我们可以使用它

**[00:01:14]** notebook um and then yeah we can use it we can do output equals block and then
> 嗯，然后是的，我们可以使用它，我们可以做 output = block，然后

**[00:01:14]** we can do output equals block and then
> 我们可以做 output = block，然后

**[00:01:14]** we can do output equals block and then we have the input here and that's
> 我们可以做 output = block，然后我们这里有输入，这就是

**[00:01:14]** we have the input here and that's
> 我们这里有输入，这就是

**[00:01:14]** we have the input here and that's essentially it so I can execute this and
> 我们这里有输入，这就是基本内容，所以我可以执行这个并

**[00:01:14]** essentially it so I can execute this and
> 基本内容，所以我可以执行这个并

**[00:01:14]** essentially it so I can execute this and just take a look at the input shape so
> 基本内容，所以我可以执行这个并查看输入形状

**[00:01:14]** just take a look at the input shape so
> 查看输入形状

**[00:01:14]** just take a look at the input shape so the input shape is 246 uh
> 查看输入形状，所以输入形状是 2 4 6 嗯

**[00:01:14]** the input shape is 246 uh
> 输入形状是 2 4 6 嗯

**[00:01:14]** the input shape is 246 uh 24768 which makes sense and the output
> 输入形状是 2 4 6 嗯 2 4 768，这说得通，输出

**[00:01:14]** 24768 which makes sense and the output
> 2 4 768，这说得通，输出

**[00:01:14]** 24768 which makes sense and the output shape should be similar so here take a
> 2 4 768，这说得通，输出形状应该类似，所以这里看一下

**[00:01:14]** shape should be similar so here take a
> shape should be similar so here take a

**[00:01:14]** shape should be similar so here take a look at the output shape has the similar
> shape should be similar so here take a look at the output shape has the similar

**[00:01:14]** look at the output shape has the similar
> look at the output shape has the similar

**[00:01:14]** look at the output shape has the similar size and why is that important that's
> look at the output shape has the similar size and why is that important that's

**[00:01:14]** size and why is that important that's
> size and why is that important that's

**[00:01:14]** size and why is that important that's because we have multiple of these
> size and why is that important that's because we have multiple of these

**[00:01:14]** because we have multiple of these
> because we have multiple of these

**[00:01:14]** because we have multiple of these Transformer blocks as I mentioned before
> because we have multiple of these Transformer blocks as I mentioned before

**[00:01:14]** Transformer blocks as I mentioned before
> Transformer blocks as I mentioned before

**[00:01:14]** Transformer blocks as I mentioned before and so if we want the next Transformer
> Transformer blocks as I mentioned before and so if we want the next Transformer

**[00:01:14]** and so if we want the next Transformer
> and so if we want the next Transformer

**[00:01:14]** and so if we want the next Transformer block to be compatible with this
> and so if we want the next Transformer block to be compatible with this

**[00:01:14]** block to be compatible with this
> block to be compatible with this

**[00:01:14]** block to be compatible with this Transformer block the input has to be
> block to be compatible with this Transformer block the input has to be

**[00:01:14]** Transformer block the input has to be
> Transformer block the input has to be

**[00:01:14]** Transformer block the input has to be the same size as the output otherwise we
> Transformer block the input has to be the same size as the output otherwise we

**[00:01:14]** the same size as the output otherwise we
> the same size as the output otherwise we

**[00:01:14]** the same size as the output otherwise we will have some complaints in the feed
> the same size as the output otherwise we will have some complaints in the feed

**[00:01:14]** will have some complaints in the feed
> will have some complaints in the feed

**[00:01:14]** will have some complaints in the feed forward network if suddenly the input
> will have some complaints in the feed forward network if suddenly the input

**[00:01:15]** forward network if suddenly the input
> forward network if suddenly the input

**[00:01:15]** forward network if suddenly the input Dimension changes so yeah that is all
> forward network if suddenly the input Dimension changes so yeah that is all

**[00:01:15]** Dimension changes so yeah that is all
> Dimension changes so yeah that is all

**[00:01:15]** Dimension changes so yeah that is all there is for now about the Transformer
> Dimension changes so yeah that is all there is for now about the Transformer

**[00:01:15]** there is for now about the Transformer
> there is for now about the Transformer

**[00:01:15]** there is for now about the Transformer block and in the next section we will
> there is for now about the Transformer block and in the next section we will

**[00:01:15]** block and in the next section we will
> block and in the next section we will

**[00:01:15]** block and in the next section we will put everything together into the GPT
> block and in the next section we will put everything together into the GPT

**[00:01:15]** put everything together into the GPT
> put everything together into the GPT

**[00:01:15]** put everything together into the GPT model
> put everything together into the GPT model

**[00:01:15]** architecture now we have all the
> architecture now we have all the

**[00:01:15]** architecture now we have all the
> architecture now we have all the

**[00:01:15]** architecture now we have all the building blocks together to code the GPT
> architecture now we have all the building blocks together to code the GPT

**[00:01:15]** building blocks together to code the GPT
> building blocks together to code the GPT

**[00:01:15]** building blocks together to code the GPT model architecture before we jump into
> building blocks together to code the GPT model architecture before we jump into

**[00:01:15]** model architecture before we jump into
> model architecture before we jump into

**[00:01:15]** model architecture before we jump into the code let me just show you a visual
> model architecture before we jump into the code let me just show you a visual

**[00:01:15]** the code let me just show you a visual
> the code let me just show you a visual

**[00:01:15]** the code let me just show you a visual showing you the whole GPT model
> the code let me just show you a visual showing you the whole GPT model

**[00:01:15]** showing you the whole GPT model
> showing you the whole GPT model

**[00:01:15]** showing you the whole GPT model architecture in one figure so here what
> showing you the whole GPT model architecture in one figure so here what

**[00:01:15]** architecture in one figure so here what
> architecture in one figure so here what

**[00:01:15]** architecture in one figure so here what you can see is the GPT model
> architecture in one figure so here what you can see is the GPT model

**[00:01:15]** you can see is the GPT model
> you can see is the GPT model

**[00:01:15]** you can see is the GPT model architecture and there are of course
> you can see is the GPT model architecture and there are of course

**[00:01:15]** architecture and there are of course
> architecture and there are of course

**[00:01:15]** architecture and there are of course familiar parts to it the input text that
> architecture and there are of course familiar parts to it the input text that

**[00:01:15]** familiar parts to it the input text that
> familiar parts to it the input text that

**[00:01:15]** familiar parts to it the input text that is tokenized and then embedded with the
> familiar parts to it the input text that is tokenized and then embedded with the

**[00:01:15]** is tokenized and then embedded with the
> is tokenized and then embedded with the

**[00:01:15]** is tokenized and then embedded with the token embedding layers and the position
> is tokenized and then embedded with the token embedding layers and the position

**[00:01:15]** token embedding layers and the position
> token embedding layers and the position

**[00:01:15]** token embedding layers and the position embedding layer and then we have an
> token embedding layers and the position embedding layer and then we have an

**[00:01:15]** embedding layer and then we have an
> embedding layer and then we have an

**[00:01:15]** embedding layer and then we have an optional Dropout and in the center here
> embedding layer and then we have an optional Dropout and in the center here

**[00:01:15]** optional Dropout and in the center here
> optional Dropout and in the center here

**[00:01:15]** optional Dropout and in the center here the box here that is the Transformer
> optional Dropout and in the center here the box here that is the Transformer

**[00:01:15]** the box here that is the Transformer
> the box here that is the Transformer

**[00:01:15]** the box here that is the Transformer block that we have just implemented in
> the box here that is the Transformer block that we have just implemented in

**[00:01:15]** block that we have just implemented in
> block that we have just implemented in

**[00:01:15]** block that we have just implemented in the previous section and you can see
> block that we have just implemented in the previous section and you can see

**[00:01:15]** the previous section and you can see
> the previous section and you can see

**[00:01:15]** the previous section and you can see it's repeated 12 times times why 12
> the previous section and you can see it's repeated 12 times times why 12

**[00:01:15]** it's repeated 12 times times why 12
> it's repeated 12 times times why 12

**[00:01:15]** it's repeated 12 times times why 12 that's an uh design decision you as the
> it's repeated 12 times times why 12 that's an uh design decision you as the

**[00:01:15]** that's an uh design decision you as the
> that's an uh design decision you as the

**[00:01:15]** that's an uh design decision you as the designer you can choose this number but
> that's an uh design decision you as the designer you can choose this number but

**[00:01:16]** designer you can choose this number but
> designer you can choose this number but

**[00:01:16]** designer you can choose this number but I'm using 12 here because that's what
> designer you can choose this number but I'm using 12 here because that's what

**[00:01:16]** I'm using 12 here because that's what
> I'm using 12 here because that's what

**[00:01:16]** I'm using 12 here because that's what the original GPT team used for their
> I'm using 12 here because that's what the original GPT team used for their

**[00:01:16]** the original GPT team used for their
> the original GPT team used for their

**[00:01:16]** the original GPT team used for their smallest model but I will also show you
> the original GPT team used for their smallest model but I will also show you

**[00:01:16]** smallest model but I will also show you
> smallest model but I will also show you

**[00:01:16]** smallest model but I will also show you later that we can change this number and
> smallest model but I will also show you later that we can change this number and

**[00:01:16]** later that we can change this number and
> later that we can change this number and

**[00:01:16]** later that we can change this number and have different differently sized GPD
> later that we can change this number and have different differently sized GPD

**[00:01:16]** have different differently sized GPD
> have different differently sized GPD

**[00:01:16]** have different differently sized GPD models um yeah and in the transform
> have different differently sized GPD models um yeah and in the transform

**[00:01:16]** models um yeah and in the transform
> models um yeah and in the transform

**[00:01:16]** models um yeah and in the transform block you see all the familiar parts
> 模型嗯是的，在transformer块中你可以看到所有熟悉的组件

**[00:01:16]** block you see all the familiar parts
> 块中你可以看到所有熟悉的组件

**[00:01:16]** block you see all the familiar parts that we implemented in the previous
> 块中你可以看到所有我们在上一节实现的熟悉组件

**[00:01:16]** that we implemented in the previous
> 我们在上一节实现的

**[00:01:16]** that we implemented in the previous section then we have a layer norm and an
> 我们在上一节实现的，然后我们有layer norm和一个

**[00:01:16]** section then we have a layer norm and an
> 节然后我们有layer norm和一个

**[00:01:16]** section then we have a layer norm and an output layer now what's interesting is
> 节然后我们有layer norm和一个输出层，现在有趣的是

**[00:01:16]** output layer now what's interesting is
> 输出层现在有趣的是

**[00:01:16]** output layer now what's interesting is that the output layer does an output put
> 输出层现在有趣的是输出层输出的是

**[00:01:16]** that the output layer does an output put
> 输出层输出的是

**[00:01:16]** that the output layer does an output put words it's outputting numbers so what do
> 输出层输出的是词语，它输出的是数字，那么这些

**[00:01:16]** words it's outputting numbers so what do
> 数字是什么意思呢，如果你仔细

**[00:01:16]** words it's outputting numbers so what do these numbers mean um if you pay close
> 数字是什么意思呢，如果你仔细看，可能会发现这里有

**[00:01:16]** these numbers mean um if you pay close
> 注意看，你可能会发现这里有

**[00:01:16]** these numbers mean um if you pay close attention you might see that there are
> 注意看，你可能会发现这里有四个输入token，并且这里有四行

**[00:01:16]** attention you might see that there are
> 四个输入token，并且这里有四行

**[00:01:16]** attention you might see that there are four input tokens and there are four
> 四个输入token，并且这里有四行，所以每一行对应

**[00:01:16]** four input tokens and there are four
> 每一行对应

**[00:01:16]** four input tokens and there are four rows here so each row here corresponds
> 每一行对应一个输入token，然而

**[00:01:16]** rows here so each row here corresponds
> 一个输入token，然而

**[00:01:16]** rows here so each row here corresponds to one of the input tokens however
> 一个输入token，然而它不是一个词，你可以看到这是一个向量

**[00:01:16]** to one of the input tokens however
> 这是一个向量

**[00:01:16]** to one of the input tokens however instead of being a word you can see this
> 这是一个向量，所以每个token由

**[00:01:16]** instead of being a word you can see this
> 一个向量表示，并且这个向量

**[00:01:16]** instead of being a word you can see this is a vector here so each token is
> 一个向量表示，并且这个向量是50,2157维的，这些维度

**[00:01:16]** is a vector here so each token is
> 是50,2157维的，这些维度

**[00:01:16]** is a vector here so each token is represented by a vector and this Vector
> 是50,2157维的，这些维度从哪里来呢？在输入层

**[00:01:16]** represented by a vector and this Vector
> 维度从哪里来呢？在输入层

**[00:01:16]** represented by a vector and this Vector is 50,2 157 dimensional where do these
> 维度从哪里来呢？在输入层我们做的是

**[00:01:16]** is 50,2 157 dimensional where do these
> 输入层我们做的是

**[00:01:16]** is 50,2 157 dimensional where do these Dimensions come from so here in the
> 输入层我们做的是将对应50257个词汇表条目的token ID

**[00:01:16]** Dimensions come from so here in the
> 将对应50257个词汇表条目的token ID

**[00:01:16]** Dimensions come from so here in the input layer what we do is we
> 将对应50257个词汇表条目的token ID转换为一个embedding向量

**[00:01:16]** input layer what we do is we
> 50257个词汇表条目转换为一个embedding向量

**[00:01:16]** input layer what we do is we convert a token ID uh that corresponds
> 50257个词汇表条目转换为一个embedding向量，所以我们从巨大的

**[00:01:16]** convert a token ID uh that corresponds
> 向量，所以我们从巨大的

**[00:01:16]** convert a token ID uh that corresponds to
> 向量，所以我们从巨大的词汇空间进入embedding空间

**[00:01:16]** to
> 词汇空间进入embedding空间

**[00:01:16]** to 50257 entry vocabulary into an embedding
> 词汇空间进入embedding空间，而在输出层我们则从

**[00:01:17]** 50257 entry vocabulary into an embedding
> 而在输出层我们则从

**[00:01:17]** 50257 entry vocabulary into an embedding Vector so we go from this huge
> 而在输出层我们则从embedding空间回到词汇空间

**[00:01:17]** Vector so we go from this huge
> embedding空间回到词汇空间

**[00:01:17]** Vector so we go from this huge vocabulary space into an embedding space
> embedding空间回到词汇空间，所以还有一件事

**[00:01:17]** vocabulary space into an embedding space
> 词汇空间，所以还有一件事

**[00:01:17]** vocabulary space into an embedding space and here at the output layer we go back
> 词汇空间，所以还有一件事我们需要做，才能将这些向量

**[00:01:17]** and here at the output layer we go back
> 我们需要做，才能将这些向量

**[00:01:17]** and here at the output layer we go back from an embedding space into the
> 我们需要做，才能将这些向量转换回词语，但我们

**[00:01:17]** from an embedding space into the
> 转换回词语，但我们

**[00:01:17]** from an embedding space into the vocabulary space so there is one more
> 转换回词语，但我们把这个部分留到下一节

**[00:01:17]** vocabulary space so there is one more
> 把这个部分留到下一节

**[00:01:17]** vocabulary space so there is one more thing that we have to do to convert
> 把这个部分留到下一节，这里我们只专注于

**[00:01:17]** thing that we have to do to convert
> 这里我们只专注于

**[00:01:17]** thing that we have to do to convert these vectors back into words but we
> 这里我们只专注于编码架构本身，所以是的

**[00:01:17]** these vectors back into words but we
> 编码架构本身，所以是的

**[00:01:17]** these vectors back into words but we will leave that part for the next
> 编码架构本身，所以是的，让我们开始吧，让我们开始

**[00:01:17]** will leave that part for the next
> 让我们开始吧，让我们开始

**[00:01:17]** will leave that part for the next section here we just focused on on
> 让我们开始吧，让我们开始处理架构，因为

**[00:01:17]** section here we just focused on on
> 处理架构，因为

**[00:01:17]** section here we just focused on on coding the architecture itself so yeah
> 处理架构，因为我懒，或者也许是因为我

**[00:01:17]** coding the architecture itself so yeah
> 我懒，或者也许是因为我

**[00:01:17]** coding the architecture itself so yeah let's jump in and let's um yeah get
> 我懒，或者也许是因为我已经想清楚了，这取决于你

**[00:01:17]** let's jump in and let's um yeah get
> 已经想清楚了，这取决于你

**[00:01:17]** let's jump in and let's um yeah get working here on the architecture and
> 已经想清楚了，这取决于你怎么看我，嗯，我会

**[00:01:17]** working here on the architecture and
> 怎么看我，嗯，我会

**[00:01:17]** working here on the architecture and because I'm lazy or maybe because I
> 怎么看我，嗯，我会直接复制粘贴我们之前实现的stummy gbt模型类

**[00:01:17]** because I'm lazy or maybe because I
> 直接复制粘贴我们之前实现的stummy gbt模型类

**[00:01:17]** because I'm lazy or maybe because I thought it all through depends on how
> 直接复制粘贴我们之前实现的stummy gbt模型类，所以让我

**[00:01:17]** thought it all through depends on how
> 类，所以让我

**[00:01:17]** thought it all through depends on how you would want to think of me um I will
> 类，所以让我直接复制粘贴这个，让我的生活

**[00:01:17]** you would want to think of me um I will
> 直接复制粘贴这个，让我的生活

**[00:01:17]** you would want to think of me um I will just copy and paste the stummy gbt model
> 直接复制粘贴这个，让我的生活更轻松一点，所以是的，我在这里

**[00:01:17]** just copy and paste the stummy gbt model
> 更轻松一点，所以是的，我在这里

**[00:01:17]** just copy and paste the stummy gbt model class that we implemented earlier so let
> 更轻松一点，所以是的，我在这里复制粘贴它，然后

**[00:01:17]** class that we implemented earlier so let
> 复制粘贴它，然后

**[00:01:17]** class that we implemented earlier so let me just copy and paste this to make my
> 复制粘贴它，然后

**[00:01:17]** me just copy and paste this to make my
> me just copy and paste this to make my

**[00:01:17]** me just copy and paste this to make my life here a bit easier so yeah I'm uh
> me just copy and paste this to make my life here a bit easier so yeah I'm uh

**[00:01:17]** life here a bit easier so yeah I'm uh
> life here a bit easier so yeah I'm uh

**[00:01:17]** life here a bit easier so yeah I'm uh copy and pasting it here and then what
> life here a bit easier so yeah I'm uh copy and pasting it here and then what

**[00:01:17]** copy and pasting it here and then what
> copy and pasting it here and then what

**[00:01:17]** copy and pasting it here and then what what we can do is we can replace all the
> 将其复制粘贴到这里，然后我们可以做的是替换所有

**[00:01:17]** what we can do is we can replace all the
> 我们可以做的是替换所有

**[00:01:17]** what we can do is we can replace all the dummy Parts because now we are going to
> 我们可以做的是替换所有虚拟部分，因为现在我们要

**[00:01:17]** dummy Parts because now we are going to
> 虚拟部分，因为现在我们要

**[00:01:17]** dummy Parts because now we are going to implement the real architecture can just
> 虚拟部分，因为现在我们要实现真正的架构，可以只需

**[00:01:18]** implement the real architecture can just
> 实现真正的架构，可以只需

**[00:01:18]** implement the real architecture can just delete those um we get rid of
> 实现真正的架构，可以只需删除那些嗯，我们去掉

**[00:01:18]** delete those um we get rid of
> 删除那些嗯，我们去掉

**[00:01:18]** delete those um we get rid of placeholder because it's no longer a
> 删除那些嗯，我们去掉占位符，因为它不再是

**[00:01:18]** placeholder because it's no longer a
> 占位符，因为它不再是

**[00:01:18]** placeholder because it's no longer a placeholder because our dummy
> 占位符，因为它不再是占位符，因为我们的虚拟

**[00:01:18]** placeholder because our dummy
> 占位符，因为我们的虚拟

**[00:01:18]** placeholder because our dummy Transformer block is now the real
> 占位符，因为我们的虚拟Transformer块现在是真正的

**[00:01:18]** Transformer block is now the real
> Transformer块现在是真正的

**[00:01:18]** Transformer block is now the real Transformer block and then I can do the
> Transformer块现在是真正的Transformer块，然后我可以做

**[00:01:18]** Transformer block and then I can do the
> Transformer块，然后我可以做

**[00:01:18]** Transformer block and then I can do the same thing here for the layer norm and
> Transformer块，然后我可以做同样的事情，对于layer norm和

**[00:01:18]** same thing here for the layer norm and
> 同样的事情，对于layer norm和

**[00:01:18]** same thing here for the layer norm and if I see it correctly we don't have any
> 同样的事情，对于layer norm和，如果我没看错，我们没有任何

**[00:01:18]** if I see it correctly we don't have any
> 如果我没看错，我们没有任何

**[00:01:18]** if I see it correctly we don't have any other placeholders here anymore so this
> 如果我没看错，我们没有任何其他占位符了，所以这个

**[00:01:18]** other placeholders here anymore so this
> 其他占位符了，所以这个

**[00:01:18]** other placeholders here anymore so this one should be the working GPT model
> 其他占位符了，所以这个应该是可用的GPT模型

**[00:01:18]** one should be the working GPT model
> 应该是可用的GPT模型

**[00:01:18]** one should be the working GPT model architecture we can actually try it out
> 应该是可用的GPT模型架构，我们可以实际测试一下

**[00:01:18]** architecture we can actually try it out
> 架构，我们可以实际测试一下

**[00:01:18]** architecture we can actually try it out so let me initialize a random seat here
> 架构，我们可以实际测试一下，所以让我在这里初始化一个随机种子

**[00:01:18]** so let me initialize a random seat here
> 所以让我在这里初始化一个随机种子

**[00:01:18]** so let me initialize a random seat here so you get the same results I do and
> 所以让我在这里初始化一个随机种子，这样你得到的结果和我一样，然后

**[00:01:18]** so you get the same results I do and
> 这样你得到的结果和我一样，然后

**[00:01:18]** so you get the same results I do and then we can instantiate the model so
> 这样你得到的结果和我一样，然后我们可以实例化模型，所以

**[00:01:18]** then we can instantiate the model so
> 然后我们可以实例化模型，所以

**[00:01:18]** then we can instantiate the model so I'm going to initialize it with the
> 然后我们可以实例化模型，所以我将用

**[00:01:18]** I'm going to initialize it with the
> 我将用

**[00:01:18]** I'm going to initialize it with the configuration file we had earlier at the
> 我将用我们之前在本章开头有的配置文件来初始化它

**[00:01:18]** configuration file we had earlier at the
> 配置文件来初始化它

**[00:01:18]** configuration file we had earlier at the beginning of this chapter um and let's
> 配置文件来初始化它，嗯，让我们

**[00:01:18]** beginning of this chapter um and let's
> 嗯，让我们

**[00:01:18]** beginning of this chapter um and let's actually use some real data now so we
> 嗯，让我们现在实际使用一些真实数据，所以

**[00:01:18]** actually use some real data now so we
> 现在实际使用一些真实数据，所以

**[00:01:18]** actually use some real data now so we had somewhere real
> 现在实际使用一些真实数据，所以我们之前有真实

**[00:01:18]** had somewhere real
> 有真实

**[00:01:18]** had somewhere real data let me scroll up a bit
> 有真实数据，让我向上滚动一点

**[00:01:18]** data let me scroll up a bit
> 数据，让我向上滚动一点

**[00:01:18]** data let me scroll up a bit here so here um we defined batch as two
> 数据，让我向上滚动一点，这里，嗯，我们定义了batch为两个

**[00:01:19]** here so here um we defined batch as two
> 这里，嗯，我们定义了batch为两个

**[00:01:19]** here so here um we defined batch as two inputs input text that were yeah
> 这里，嗯，我们定义了batch为两个输入，输入文本被

**[00:01:19]** inputs input text that were yeah
> 输入，输入文本被

**[00:01:19]** inputs input text that were yeah embedded into token embeddings let me
> 输入，输入文本被嵌入到token embeddings中，让我

**[00:01:19]** embedded into token embeddings let me
> 嵌入到token embeddings中，让我

**[00:01:19]** embedded into token embeddings let me just copy and paste here um so that we
> 嵌入到token embeddings中，让我复制粘贴到这里，嗯，这样我们

**[00:01:19]** just copy and paste here um so that we
> 复制粘贴到这里，嗯，这样我们

**[00:01:19]** just copy and paste here um so that we can make sure this actually still
> 复制粘贴到这里，嗯，这样我们可以确保这实际上仍然

**[00:01:19]** can make sure this actually still
> 可以确保这实际上仍然

**[00:01:19]** can make sure this actually still works so yeah we have batch um two
> 可以确保这实际上仍然有效，所以是的，我们有batch嗯两个

**[00:01:19]** works so yeah we have batch um two
> 有效，所以是的，我们有batch嗯两个

**[00:01:19]** works so yeah we have batch um two inputs and we can then compute the
> 有效，所以是的，我们有batch嗯两个输入，然后我们可以计算

**[00:01:19]** inputs and we can then compute the
> 输入，然后我们可以计算

**[00:01:19]** inputs and we can then compute the output
> 输入，然后我们可以计算输出

**[00:01:19]** output
> 输出

**[00:01:19]** output here oops model
> 输出，这里，哎呀，model

**[00:01:19]** here oops model
> 这里，哎呀，model

**[00:01:19]** here oops model batch and then we can double check maybe
> 这里，哎呀，model batch，然后我们可以双重检查一下

**[00:01:19]** batch and then we can double check maybe
> 然后我们可以双重检查一下

**[00:01:19]** batch and then we can double check maybe the input this should be 2 * 4 right
> 然后我们可以双重检查一下输入，这应该是2 * 4，对吧

**[00:01:19]** the input this should be 2 * 4 right
> 输入，这应该是2 * 4，对吧

**[00:01:19]** the input this should be 2 * 4 right oops should be 2 * 4 to examples with
> 输入，这应该是2 * 4，哎呀，应该是2 * 4，两个示例，每个有

**[00:01:19]** oops should be 2 * 4 to examples with
> 哎呀，应该是2 * 4，两个示例，每个有

**[00:01:19]** oops should be 2 * 4 to examples with four tokens and then output should be 2
> 哎呀，应该是2 * 4，两个示例，每个有四个token，然后输出应该是2

**[00:01:19]** four tokens and then output should be 2
> 四个token，然后输出应该是2

**[00:01:19]** four tokens and then output should be 2 * 4 * the vocabulary size so let's
> 四个token，然后输出应该是2 * 4 * 词汇表大小，所以让我们

**[00:01:19]** * 4 * the vocabulary size so let's
> * 4 * 词汇表大小，所以让我们

**[00:01:19]** * 4 * the vocabulary size so let's double check so 2 * 4 and then each of
> * 4 * 词汇表大小，所以让我们双重检查一下，2 * 4，然后每个

**[00:01:19]** double check so 2 * 4 and then each of
> 双重检查一下，2 * 4，然后每个

**[00:01:19]** double check so 2 * 4 and then each of the tokens is embedded in a
> 双重检查一下，2 * 4，然后每个token被嵌入到一个

**[00:01:19]** the tokens is embedded in a
> token被嵌入到一个

**[00:01:19]** the tokens is embedded in a 50257 dimensional space and we will take
> token被嵌入到一个50257维空间中，我们将

**[00:01:19]** 50257 dimensional space and we will take
> 50257维空间中，我们将

**[00:01:19]** 50257 dimensional space and we will take care of converting this in the in the
> 50257维空间中，我们将处理在

**[00:01:19]** care of converting this in the in the
> 处理在

**[00:01:19]** care of converting this in the in the next video for now I wanted to show you
> 关于如何转换这部分，我将在下一个视频中展示，现在我想先给你看

**[00:01:19]** next video for now I wanted to show you
> 下一个视频中展示，现在我想先给你看

**[00:01:19]** next video for now I wanted to show you one more thing so here I have this yeah
> 下一个视频中展示，现在我想先给你看另一件事，这里我有一个

**[00:01:19]** one more thing so here I have this yeah
> 另一件事，这里我有一个

**[00:01:19]** one more thing so here I have this yeah file sorry the dictionary that I defined
> 另一件事，这里我有一个文件，抱歉，是我在本章开头定义的字典

**[00:01:20]** file sorry the dictionary that I defined
> 文件，抱歉，是我在本章开头定义的字典

**[00:01:20]** file sorry the dictionary that I defined at the beginning of this chapter and I
> 文件，抱歉，是我在本章开头定义的字典，我给它命名为124 M，这代表

**[00:01:20]** at the beginning of this chapter and I
> 我给它命名为124 M，这代表

**[00:01:20]** at the beginning of this chapter and I gave it this naming 124 M which stands
> 我给它命名为124 M，这代表1.24亿个参数，那么

**[00:01:20]** gave it this naming 124 M which stands
> 1.24亿个参数，那么

**[00:01:20]** gave it this naming 124 M which stands for 124 million parameters and so where
> 1.24亿个参数，那么这个数字从何而来？我们可以

**[00:01:20]** for 124 million parameters and so where
> 这个数字从何而来？我们可以

**[00:01:20]** for 124 million parameters and so where does this number come from we can
> 这个数字从何而来？我们实际上可以计算参数的数量

**[00:01:20]** does this number come from we can
> 实际上可以计算参数的数量

**[00:01:20]** does this number come from we can actually calculate the number of
> 实际上可以计算参数的数量，在PyTorch中有一个

**[00:01:20]** actually calculate the number of
> 在PyTorch中有一个

**[00:01:20]** actually calculate the number of parameters and in pytorch there is a
> 在PyTorch中有一个函数或方法叫做number of

**[00:01:20]** parameters and in pytorch there is a
> 函数或方法叫做number of

**[00:01:20]** parameters and in pytorch there is a function or method called number of
> 函数或方法叫做number of elements，这里number number of elements

**[00:01:20]** function or method called number of
> elements，这里number number of elements

**[00:01:20]** function or method called number of elements here number number of elements
> elements，这里number number of elements，所以我们可以计算

**[00:01:20]** elements here number number of elements
> 所以我们可以计算

**[00:01:20]** elements here number number of elements so what we can do is we can compute for
> 所以我们可以计算，例如batch中的元素数量

**[00:01:20]** so what we can do is we can compute for
> 例如batch中的元素数量

**[00:01:20]** so what we can do is we can compute for example the number of elements in batch
> 例如batch中的元素数量，这应该是8，对吧？我们有2

**[00:01:20]** example the number of elements in batch
> 这应该是8，对吧？我们有2

**[00:01:20]** example the number of elements in batch and this should be eight right we have 2
> 这应该是8，对吧？我们有2 * 4个元素，所以这里只是一个

**[00:01:20]** and this should be eight right we have 2
> * 4个元素，所以这里只是一个

**[00:01:20]** and this should be eight right we have 2 * 4 elements so here this is just like a
> * 4个元素，所以这里只是一个方便的方法来计算给定tensor中有多少

**[00:01:20]** * 4 elements so here this is just like a
> 方便的方法来计算给定tensor中有多少

**[00:01:20]** * 4 elements so here this is just like a convenient way to compute how many
> 方便的方法来计算给定tensor中有多少值，所以我们可以

**[00:01:20]** convenient way to compute how many
> 值，所以我们可以

**[00:01:20]** convenient way to compute how many values are in a given tensor so we can
> 值，所以我们可以用它来求和模型中所有元素

**[00:01:20]** values are in a given tensor so we can
> 用它来求和模型中所有元素

**[00:01:20]** values are in a given tensor so we can use that um to sum over all the elements
> 用它来求和模型中所有元素，所以对于model中的p

**[00:01:20]** use that um to sum over all the elements
> 所以对于model中的p

**[00:01:20]** use that um to sum over all the elements in the model so for p in model
> 所以对于model中的p parameters，parameters就是所有

**[00:01:20]** in the model so for p in model
> parameters，parameters就是所有

**[00:01:20]** in the model so for p in model parameters so parameters are all the
> parameters，parameters就是所有权重矩阵，例如，以及偏置

**[00:01:20]** parameters so parameters are all the
> 权重矩阵，例如，以及偏置

**[00:01:20]** parameters so parameters are all the weight matrices for example and bias
> 权重矩阵，例如，以及偏置向量，这些都在模型中，或者说是

**[00:01:20]** weight matrices for example and bias
> 向量，这些都在模型中，或者说是

**[00:01:20]** weight matrices for example and bias vectors that are in the model or the
> 向量，这些都在模型中，或者说是可训练的东西，我们基本上

**[00:01:20]** vectors that are in the model or the
> 可训练的东西，我们基本上

**[00:01:20]** vectors that are in the model or the trainable um things that we have
> 可训练的东西，我们基本上有，包括qkv

**[00:01:20]** trainable um things that we have
> 有，包括qkv

**[00:01:20]** trainable um things that we have basically that's um including the qkv
> 有，包括qkv矩阵，还有layer norm和

**[00:01:20]** basically that's um including the qkv
> 矩阵，还有layer norm和

**[00:01:20]** basically that's um including the qkv matrices but also the layer norm and
> 矩阵，还有layer norm和所有东西，所以嗯，然后我们

**[00:01:21]** matrices but also the layer norm and
> 所有东西，所以嗯，然后我们

**[00:01:21]** matrices but also the layer norm and everything so um yeah and then we
> 所有东西，所以嗯，然后我们可以打印出来，哦，我打错了一个字

**[00:01:21]** everything so um yeah and then we
> 可以打印出来，哦，我打错了一个字

**[00:01:21]** everything so um yeah and then we can print this out uh oh I made a typle
> 可以打印出来，哦，我打错了一个字，这里应该是num，是的，你可以看到

**[00:01:21]** can print this out uh oh I made a typle
> 这里应该是num，是的，你可以看到

**[00:01:21]** can print this out uh oh I made a typle here should be num and yeah you can see
> 这里应该是num，是的，你可以看到这是一个很大的数字，为了更容易展示

**[00:01:21]** here should be num and yeah you can see
> 这是一个很大的数字，为了更容易展示

**[00:01:21]** here should be num and yeah you can see it's a large number to maybe show this a
> 这是一个很大的数字，为了更容易展示，让我们用F string

**[00:01:21]** it's a large number to maybe show this a
> 让我们用F string

**[00:01:21]** it's a large number to maybe show this a bit easier let's use um F string um
> 让我们用F string，看看我能不能写对，F

**[00:01:21]** bit easier let's use um F string um
> 看看我能不能写对，F

**[00:01:21]** bit easier let's use um F string um let's see if I get this right uh F
> 看看我能不能写对，F strings，如果有一段时间没用，我

**[00:01:21]** let's see if I get this right uh F
> strings，如果有一段时间没用，我

**[00:01:21]** let's see if I get this right uh F strings if I don't use them in a while I
> strings，如果有一段时间没用，我通常会忘记语法，但我想如果

**[00:01:21]** strings if I don't use them in a while I
> 通常会忘记语法，但我想如果

**[00:01:21]** strings if I don't use them in a while I usually forget the syntax but I think if
> 通常会忘记语法，但我想如果我在这里加一个逗号，是的，我们可以看到

**[00:01:21]** usually forget the syntax but I think if
> 我在这里加一个逗号，是的，我们可以看到

**[00:01:21]** usually forget the syntax but I think if I add a comma here yeah so we can see
> 我在这里加一个逗号，是的，我们可以看到这里有1.63亿个参数

**[00:01:21]** I add a comma here yeah so we can see
> 这里有1.63亿个参数

**[00:01:21]** I add a comma here yeah so we can see it's 163 million parameters that we have
> 这里有1.63亿个参数，那么，这个数字

**[00:01:21]** it's 163 million parameters that we have
> 那么，这个数字

**[00:01:21]** it's 163 million parameters that we have here and um so how does this number
> 那么，这个数字如何与124匹配？我刚才告诉你我们有

**[00:01:21]** here and um so how does this number
> 如何与124匹配？我刚才告诉你我们有

**[00:01:21]** here and um so how does this number match with 124 I just told you we have
> 如何与124匹配？我刚才告诉你我们有1.24亿个参数，还有一件事

**[00:01:21]** match with 124 I just told you we have
> 1.24亿个参数，还有一件事

**[00:01:21]** match with 124 I just told you we have 124 million parameters and there's one
> 1.24亿个参数，还有一件事我还没有解释，所以

**[00:01:21]** 124 million parameters and there's one
> 我还没有解释，所以

**[00:01:21]** 124 million parameters and there's one thing I have not explained um yet so
> 我还没有解释，所以在原始的GPT模型中

**[00:01:21]** thing I have not explained um yet so
> 在原始的GPT模型中

**[00:01:21]** thing I have not explained um yet so there is in the original GPD model
> 在原始的GPT模型中

**[00:01:21]** there is in the original GPD model
> there is in the original GPD model

**[00:01:21]** there is in the original GPD model architecture some weight sharing going
> 在原始GPD模型架构中存在一些权重共享

**[00:01:21]** architecture some weight sharing going
> 架构中存在一些权重共享

**[00:01:21]** architecture some weight sharing going on so I told you before that this one
> 架构中存在一些权重共享，我之前告诉过你，这个

**[00:01:21]** on so I told you before that this one
> 我之前告诉过你，这个

**[00:01:21]** on so I told you before that this one maps from the vocabulary space into the
> 我之前告诉过你，这个从词汇空间映射到

**[00:01:21]** maps from the vocabulary space into the
> 从词汇空间映射到

**[00:01:21]** maps from the vocabulary space into the embedding space and then this one goes
> 从词汇空间映射到embedding空间，然后这个

**[00:01:22]** embedding space and then this one goes
> embedding空间，然后这个

**[00:01:22]** embedding space and then this one goes the other way around it Maps back from
> embedding空间，然后这个反向操作，从

**[00:01:22]** the other way around it Maps back from
> 反向操作，从

**[00:01:22]** the other way around it Maps back from the embedding space into the vocabulary
> 反向操作，从embedding空间映射回词汇

**[00:01:22]** the embedding space into the vocabulary
> embedding空间映射回词汇

**[00:01:22]** the embedding space into the vocabulary space so what we can do is we can take a
> embedding空间映射回词汇空间，所以我们可以

**[00:01:22]** space so what we can do is we can take a
> 空间，所以我们可以

**[00:01:22]** space so what we can do is we can take a look at these weight matrices right so
> 空间，所以我们可以查看这些权重矩阵，对吧

**[00:01:22]** look at these weight matrices right so
> 查看这些权重矩阵，对吧

**[00:01:22]** look at these weight matrices right so we could type model and then um what is
> 查看这些权重矩阵，对吧，我们可以输入model然后嗯

**[00:01:22]** we could type model and then um what is
> 我们可以输入model然后嗯

**[00:01:22]** we could type model and then um what is it
> 我们可以输入model然后嗯，是什么

**[00:01:22]** it
> 是什么

**[00:01:22]** it token edding so let's take a look um at
> 是什么token embedding，让我们看一下

**[00:01:22]** token edding so let's take a look um at
> token embedding，让我们看一下

**[00:01:22]** token edding so let's take a look um at the weight Matrix and see how big it is
> token embedding，让我们看一下权重矩阵，看看它有多大

**[00:01:22]** the weight Matrix and see how big it is
> 权重矩阵，看看它有多大

**[00:01:22]** the weight Matrix and see how big it is so we can see it's
> 权重矩阵，看看它有多大，我们可以看到它是

**[00:01:22]** so we can see it's
> 我们可以看到它是

**[00:01:22]** so we can see it's 50257 *
> 我们可以看到它是50257 *

**[00:01:22]** 50257 *
> 50257 *

**[00:01:22]** 50257 * 768 and if we look at the output
> 50257 * 768，如果我们看输出

**[00:01:22]** 768 and if we look at the output
> 768，如果我们看输出

**[00:01:22]** 768 and if we look at the output here uh let me copy this one and then I
> 768，如果我们看输出，嗯，让我复制这个，然后

**[00:01:22]** here uh let me copy this one and then I
> 让我复制这个，然后

**[00:01:22]** here uh let me copy this one and then I will replace this with
> 让我复制这个，然后我将把这个替换为

**[00:01:22]** outad this has exactly the same size so
> 输出，它的大小完全相同，所以

**[00:01:22]** outad this has exactly the same size so
> 输出，它的大小完全相同，所以

**[00:01:22]** outad this has exactly the same size so one trick because let's say we assume
> 输出，它的大小完全相同，所以有一个技巧，因为假设我们

**[00:01:22]** one trick because let's say we assume
> 有一个技巧，因为假设我们

**[00:01:22]** one trick because let's say we assume this is learning a good mapping we're
> 有一个技巧，因为假设这正在学习一个好的映射，我们

**[00:01:22]** this is learning a good mapping we're
> 这正在学习一个好的映射，我们

**[00:01:22]** this is learning a good mapping we're mapping from vocabulary space into
> 这正在学习一个好的映射，我们从词汇空间映射到

**[00:01:22]** mapping from vocabulary space into
> 从词汇空间映射到

**[00:01:22]** mapping from vocabulary space into embedding space and we might assume that
> 从词汇空间映射到embedding空间，我们可能假设

**[00:01:22]** embedding space and we might assume that
> embedding空间，我们可能假设

**[00:01:22]** embedding space and we might assume that the same in Reverse might work well so
> embedding空间，我们可能假设反向的相同映射也可能效果很好，所以

**[00:01:22]** the same in Reverse might work well so
> 反向的相同映射也可能效果很好，所以

**[00:01:22]** the same in Reverse might work well so mapping from embedding space to
> 反向的相同映射也可能效果很好，所以从embedding空间映射到

**[00:01:23]** mapping from embedding space to
> 从embedding空间映射到

**[00:01:23]** mapping from embedding space to vocabulary space so we could actually
> 从embedding空间映射到词汇空间，所以我们实际上可以

**[00:01:23]** vocabulary space so we could actually
> 词汇空间，所以我们实际上可以

**[00:01:23]** vocabulary space so we could actually share the same weights so we are not
> 词汇空间，所以我们实际上可以共享相同的权重，所以我们没有

**[00:01:23]** share the same weights so we are not
> 共享相同的权重，所以我们没有

**[00:01:23]** share the same weights so we are not doing it here for Simplicity but we
> 共享相同的权重，这里为了简单起见我们没有这样做，但

**[00:01:23]** doing it here for Simplicity but we
> 这里为了简单起见我们没有这样做，但

**[00:01:23]** doing it here for Simplicity but we could technically reuse the same weights
> 这里为了简单起见我们没有这样做，但技术上我们可以重用相同的权重

**[00:01:23]** could technically reuse the same weights
> 技术上我们可以重用相同的权重

**[00:01:23]** could technically reuse the same weights in the output head here um so because
> 技术上我们可以重用相同的权重在输出头这里，嗯，因为

**[00:01:23]** in the output head here um so because
> 在输出头这里，嗯，因为

**[00:01:23]** in the output head here um so because they have the same dimensions and it's
> 在输出头这里，嗯，因为它们具有相同的维度，并且这是

**[00:01:23]** they have the same dimensions and it's
> 它们具有相同的维度，并且这是

**[00:01:23]** they have the same dimensions and it's actually design a design decision some
> 它们具有相同的维度，并且这实际上是一个设计决策，有些

**[00:01:23]** actually design a design decision some
> 实际上是一个设计决策，有些

**[00:01:23]** actually design a design decision some people do that some people don't if you
> 实际上是一个设计决策，有些人这样做，有些人不这样做，如果你

**[00:01:23]** people do that some people don't if you
> 有些人这样做，有些人不这样做，如果你

**[00:01:23]** people do that some people don't if you share the weights yeah you have a fewer
> 有些人这样做，有些人不这样做，如果你共享权重，是的，你的模型参数会更少

**[00:01:23]** share the weights yeah you have a fewer
> 共享权重，是的，你的模型参数会更少

**[00:01:23]** share the weights yeah you have a fewer parameters in the model because these
> 共享权重，是的，你的模型参数会更少，因为这些

**[00:01:23]** parameters in the model because these
> 参数会更少，因为这些

**[00:01:23]** parameters in the model because these are really large matrices but there are
> 参数会更少，因为这些都是非常大的矩阵，但有很多

**[00:01:23]** are really large matrices but there are
> 非常大的矩阵，但有很多

**[00:01:23]** are really large matrices but there are architectures or many many architectures
> 非常大的矩阵，但有很多架构，很多很多架构

**[00:01:23]** architectures or many many architectures
> 架构，很多很多架构

**[00:01:23]** architectures or many many architectures that don't do that so it's really it
> 架构，很多很多架构不这样做，所以这真的

**[00:01:23]** that don't do that so it's really it
> 不这样做，所以这真的

**[00:01:23]** that don't do that so it's really it really depends so llama models by meta
> 不这样做，所以这真的取决于，例如Meta AI的llama模型

**[00:01:23]** really depends so llama models by meta
> 取决于，例如Meta AI的llama模型

**[00:01:23]** really depends so llama models by meta AI for example llama 3 and 3.1 they
> 取决于，例如Meta AI的llama 3和3.1，它们

**[00:01:23]** AI for example llama 3 and 3.1 they
> AI，例如llama 3和3.1，它们

**[00:01:23]** AI for example llama 3 and 3.1 they don't do the weight sharing but 3.2 is
> AI，例如llama 3和3.1，它们不进行权重共享，但3.2

**[00:01:23]** don't do the weight sharing but 3.2 is
> 不进行权重共享，但3.2

**[00:01:23]** don't do the weight sharing but 3.2 is doing the weight sharing it's really you
> 不进行权重共享，但3.2正在进行权重共享，这真的取决于你

**[00:01:23]** doing the weight sharing it's really you
> 进行权重共享实际上就是

**[00:01:23]** doing the weight sharing it's really you know almost arbitrary it depends if you
> 进行权重共享实际上几乎是任意的，取决于你是否

**[00:01:23]** know almost arbitrary it depends if you
> 几乎是任意的，取决于你是否

**[00:01:23]** know almost arbitrary it depends if you want to save on parameters you can share
> 几乎是任意的，取决于你是否想节省参数，可以共享

**[00:01:23]** want to save on parameters you can share
> 想节省参数，可以共享

**[00:01:23]** want to save on parameters you can share the weights but otherwise in my personal
> 想节省参数，可以共享权重，但就我个人

**[00:01:23]** the weights but otherwise in my personal
> 权重，但就我个人

**[00:01:23]** the weights but otherwise in my personal experience it's not always leading to
> 权重，但就我个人经验而言，这并不总能带来

**[00:01:23]** experience it's not always leading to
> 经验而言，这并不总能带来

**[00:01:23]** experience it's not always leading to better results so it's really like a
> 经验而言，这并不总能带来更好的结果，所以这实际上是一种

**[00:01:23]** better results so it's really like a
> 更好的结果，所以这实际上是一种

**[00:01:23]** better results so it's really like a trade-off between parameter size and you
> 更好的结果，所以这实际上是一种参数大小与

**[00:01:23]** trade-off between parameter size and you
> 参数大小与

**[00:01:23]** trade-off between parameter size and you know uh better training Dynamics but it
> 参数大小与更好的训练动态之间的权衡，但这

**[00:01:23]** know uh better training Dynamics but it
> 更好的训练动态之间的权衡，但这

**[00:01:23]** know uh better training Dynamics but it really really depends on the
> 更好的训练动态之间的权衡，但这真的取决于

**[00:01:23]** really really depends on the
> 真的取决于

**[00:01:23]** really really depends on the architecture but um yeah what what I
> 真的取决于架构，但嗯，是的，我

**[00:01:24]** architecture but um yeah what what I
> 架构，但嗯，是的，我

**[00:01:24]** architecture but um yeah what what I wanted to tell you is that the number of
> 架构，但嗯，是的，我想告诉你的是，这里的参数数量

**[00:01:24]** wanted to tell you is that the number of
> 我想告诉你的是，这里的参数数量

**[00:01:24]** wanted to tell you is that the number of parameters here how how it's usually
> 我想告诉你的是，这里的参数数量通常是如何

**[00:01:24]** parameters here how how it's usually
> 参数数量通常是如何

**[00:01:24]** parameters here how how it's usually computed for gpd2 models because they
> 参数数量通常是如何为GPT2模型计算的，因为它们

**[00:01:24]** computed for gpd2 models because they
> 为GPT2模型计算的，因为它们

**[00:01:24]** computed for gpd2 models because they usually originally used the weight
> 为GPT2模型计算的，因为它们通常最初使用权重

**[00:01:24]** usually originally used the weight
> 通常最初使用权重

**[00:01:24]** usually originally used the weight sharing is without
> 通常最初使用权重共享时，是不

**[00:01:24]** sharing is without
> 是不

**[00:01:24]** sharing is without um counting it twice so what I mean is
> 是不重复计算的，所以我的意思是

**[00:01:24]** um counting it twice so what I mean is
> 不重复计算的，所以我的意思是

**[00:01:24]** um counting it twice so what I mean is here we count it once and we count it um
> 不重复计算的，所以我的意思是这里我们计算一次，而这里我们计算

**[00:01:24]** here we count it once and we count it um
> 这里我们计算一次，而这里我们计算

**[00:01:24]** here we count it once and we count it um twice here so sorry let's focus on those
> 这里我们计算一次，而这里我们计算两次，所以抱歉，让我们专注于这些

**[00:01:24]** twice here so sorry let's focus on those
> 两次，所以抱歉，让我们专注于这些

**[00:01:24]** twice here so sorry let's focus on those here and here if those are sh
> 两次，所以抱歉，让我们专注于这些这里和这里，如果这些是共享的

**[00:01:24]** here and here if those are sh
> 这里和这里，如果这些是共享的

**[00:01:24]** here and here if those are sh we are over counting so what we could do
> 这里和这里，如果这些是共享的，我们就重复计算了，所以我们可以做的是

**[00:01:24]** we are over counting so what we could do
> 我们就重复计算了，所以我们可以做的是

**[00:01:24]** we are over counting so what we could do is we could
> 我们就重复计算了，所以我们可以做的是

**[00:01:24]** is we could
> 我们可以

**[00:01:24]** is we could subtract um let's maybe do it here we
> 我们可以减去，嗯，也许我们在这里做

**[00:01:24]** subtract um let's maybe do it here we
> 减去，嗯，也许我们在这里做

**[00:01:24]** subtract um let's maybe do it here we can
> 减去，嗯，也许我们在这里可以

**[00:01:24]** can
> 可以

**[00:01:24]** can subtract
> 减去

**[00:01:24]** subtract
> 减去

**[00:01:24]** subtract this one
> 减去这个

**[00:01:24]** here let's see if this works so yeah so
> 这里，让我们看看这是否有效，所以是的，所以

**[00:01:24]** here let's see if this works so yeah so
> 这里，让我们看看这是否有效，所以是的，所以

**[00:01:24]** here let's see if this works so yeah so you will get the 124 million that is
> 这里，让我们看看这是否有效，所以是的，所以你会得到1.24亿，这就是

**[00:01:24]** you will get the 124 million that is
> 你会得到1.24亿，这就是

**[00:01:24]** you will get the 124 million that is usually quoted when people talk about
> 你会得到1.24亿，这就是人们谈论GPT2模型时通常引用的数字

**[00:01:24]** usually quoted when people talk about
> 人们谈论GPT2模型时通常引用的数字

**[00:01:24]** usually quoted when people talk about gpt2 models um I also have the paper
> 人们谈论GPT2模型时通常引用的数字，嗯，我也打开了这篇论文

**[00:01:24]** gpt2 models um I also have the paper
> 嗯，我也打开了这篇论文

**[00:01:24]** gpt2 models um I also have the paper open here so here in this paper this is
> 嗯，我也打开了这篇论文，所以在这篇论文中，这是

**[00:01:24]** open here so here in this paper this is
> 所以在这篇论文中，这是

**[00:01:24]** open here so here in this paper this is the original gpd2 paper they should have
> 所以在这篇论文中，这是原始的GPT2论文，他们应该有一个

**[00:01:25]** the original gpd2 paper they should have
> 原始的GPT2论文，他们应该有一个

**[00:01:25]** the original gpd2 paper they should have a table somewhere let me see where they
> 原始的GPT2论文，他们应该在某处有一个表格，让我看看他们在哪里

**[00:01:25]** a table somewhere let me see where they
> 在某处有一个表格，让我看看他们在哪里

**[00:01:25]** a table somewhere let me see where they talk about the model sizes yeah right
> 在某处有一个表格，让我看看他们在哪里讨论模型大小，是的，就在

**[00:01:25]** talk about the model sizes yeah right
> 讨论模型大小，是的，就在

**[00:01:25]** talk about the model sizes yeah right here so here they have four different
> 讨论模型大小，是的，就在这儿，所以这里有四种不同的

**[00:01:25]** here so here they have four different
> 这儿，所以这里有四种不同的

**[00:01:25]** here so here they have four different model sizes and this is the one we just
> 这儿，所以这里有四种不同的模型大小，这就是我们刚刚

**[00:01:25]** model sizes and this is the one we just
> 模型大小，这就是我们刚刚

**[00:01:25]** model sizes and this is the one we just implemented with the 12 layers and the
> 模型大小，这就是我们刚刚实现的，有12层和

**[00:01:25]** implemented with the 12 layers and the
> 实现的，有12层和

**[00:01:25]** implemented with the 12 layers and the 768 dimensional
> 实现的，有12层和768维

**[00:01:25]** 768 dimensional
> 768维

**[00:01:25]** 768 dimensional embeddings um you might notice this is
> 768维的嵌入，嗯，你可能会注意到这不是

**[00:01:25]** embeddings um you might notice this is
> 嵌入，嗯，你可能会注意到这不是

**[00:01:25]** embeddings um you might notice this is not 124 and this is because they had a
> 嵌入，嗯，你可能会注意到这不是1.24亿，这是因为他们有一个

**[00:01:25]** not 124 and this is because they had a
> 不是1.24亿，这是因为他们有一个

**[00:01:25]** not 124 and this is because they had a calculation error I think so I think
> 不是1.24亿，这是因为他们有一个计算错误，我想，所以我认为

**[00:01:25]** calculation error I think so I think
> 计算错误，我想，所以我认为

**[00:01:25]** calculation error I think so I think they talked about it somewhere I think
> 计算错误，我想，所以我认为他们在某处讨论过这个，我想

**[00:01:25]** they talked about it somewhere I think
> 他们在某处讨论过这个，我想

**[00:01:25]** they talked about it somewhere I think maybe in the code repository but they
> 他们可能在代码仓库的某个地方讨论过这个问题

**[00:01:25]** maybe in the code repository but they
> 可能在代码仓库里，但他们

**[00:01:25]** maybe in the code repository but they haven't updated the paper which is why
> 可能在代码仓库里，但他们没有更新论文，这就是为什么

**[00:01:25]** haven't updated the paper which is why
> 没有更新论文，这就是为什么

**[00:01:25]** haven't updated the paper which is why the number looks a bit off but it should
> 没有更新论文，这就是为什么数字看起来有点不对，但应该

**[00:01:25]** the number looks a bit off but it should
> 数字看起来有点不对，但应该

**[00:01:25]** the number looks a bit off but it should be 124 million and they also have two
> 数字看起来有点不对，但应该是1.24亿，而且他们还有两个

**[00:01:25]** be 124 million and they also have two
> 是1.24亿，而且他们还有两个

**[00:01:25]** be 124 million and they also have two different sizes here and yeah I'm maybe
> 是1.24亿，而且他们还有两个不同的尺寸，嗯，我可能

**[00:01:25]** different sizes here and yeah I'm maybe
> 不同的尺寸，嗯，我可能

**[00:01:25]** different sizes here and yeah I'm maybe leaving this as an exercise for you the
> 不同的尺寸，嗯，我可能把这个留作练习给你们

**[00:01:25]** leaving this as an exercise for you the
> 留作练习给你们

**[00:01:25]** leaving this as an exercise for you the viewer um so to implement those actually
> 留作练习给你们这些观众，嗯，要实现那些

**[00:01:25]** viewer um so to implement those actually
> 观众，嗯，要实现那些

**[00:01:25]** viewer um so to implement those actually it should be relatively simple once you
> 观众，嗯，要实现那些实际上应该相对简单，一旦你

**[00:01:25]** it should be relatively simple once you
> 实际上应该相对简单，一旦你

**[00:01:25]** it should be relatively simple once you have the architecture coded so you could
> 实际上应该相对简单，一旦你编写好架构代码，你就可以

**[00:01:25]** have the architecture coded so you could
> 编写好架构代码，你就可以

**[00:01:25]** have the architecture coded so you could copy and paste this
> 编写好架构代码，你就可以复制粘贴这个

**[00:01:25]** copy and paste this
> 复制粘贴这个

**[00:01:25]** copy and paste this here and then change some of the values
> 复制粘贴这个到这里，然后改变一些值

**[00:01:25]** here and then change some of the values
> 到这里，然后改变一些值

**[00:01:25]** here and then change some of the values here to make the architecture larger to
> 到这里，然后改变一些值，使架构更大，以

**[00:01:25]** here to make the architecture larger to
> 使架构更大，以

**[00:01:25]** here to make the architecture larger to implement the other ones you are seeing
> 使架构更大，以实现你在这里看到的其他

**[00:01:25]** implement the other ones you are seeing
> 实现你在这里看到的其他

**[00:01:25]** implement the other ones you are seeing here in this table I should tell you the
> 实现你在这里看到的其他表格中的内容，我应该告诉你

**[00:01:26]** here in this table I should tell you the
> 表格中的内容，我应该告诉你

**[00:01:26]** here in this table I should tell you the table doesn't tell you the whole story
> 表格中的内容，我应该告诉你这个表格并没有说明全部情况

**[00:01:26]** table doesn't tell you the whole story
> 表格并没有说明全部情况

**[00:01:26]** table doesn't tell you the whole story it talks about the number of layers and
> 表格并没有说明全部情况，它提到了层数和

**[00:01:26]** it talks about the number of layers and
> 它提到了层数和

**[00:01:26]** it talks about the number of layers and the embedding Dimensions so here the
> 它提到了层数和embedding维度，所以这里的

**[00:01:26]** the embedding Dimensions so here the
> embedding维度，所以这里的

**[00:01:26]** the embedding Dimensions so here the layers and the embedding Dimension where
> embedding维度，这里的层数和embedding维度，其中

**[00:01:26]** layers and the embedding Dimension where
> 层数和embedding维度，其中

**[00:01:26]** layers and the embedding Dimension where layers is maybe a bit unfortunate um as
> 层数和embedding维度，其中“层”这个术语可能有点不幸，嗯，因为

**[00:01:26]** layers is maybe a bit unfortunate um as
> “层”这个术语可能有点不幸，嗯，因为

**[00:01:26]** layers is maybe a bit unfortunate um as the term because this really um refers
> “层”这个术语可能有点不幸，嗯，因为这实际上指的是Transformer

**[00:01:26]** the term because this really um refers
> 这实际上指的是Transformer

**[00:01:26]** the term because this really um refers to the number of times the Transformer
> 这实际上指的是Transformer块重复的次数，所以它被用在

**[00:01:26]** to the number of times the Transformer
> 块重复的次数，所以它被用在

**[00:01:26]** to the number of times the Transformer block is repeated so it's um used right
> 块重复的次数，所以它被用在这里，嗯，所以它不一定指其他

**[00:01:26]** block is repeated so it's um used right
> 这里，嗯，所以它不一定指其他

**[00:01:26]** block is repeated so it's um used right here um so it's not necessarily other
> 这里，嗯，所以它不一定指其他层，它特指

**[00:01:26]** here um so it's not necessarily other
> 层，它特指

**[00:01:26]** here um so it's not necessarily other layers it's really specific to the
> 层，它特指Transformer块，所以你也可以

**[00:01:26]** layers it's really specific to the
> Transformer块，所以你也可以

**[00:01:26]** layers it's really specific to the Transformer block so you could also
> Transformer块，所以你也可以直接称之为Transformer块的数量

**[00:01:26]** Transformer block so you could also
> 直接称之为Transformer块的数量

**[00:01:26]** Transformer block so you could also maybe just call it number of Transformer
> 直接称之为Transformer块的数量，以更清晰，或者n_blocks或

**[00:01:26]** maybe just call it number of Transformer
> 以更清晰，或者n_blocks或

**[00:01:26]** maybe just call it number of Transformer blocks to be more clear or n blocks or
> 以更清晰，或者n_blocks或类似的东西，但嗯，我

**[00:01:26]** blocks to be more clear or n blocks or
> 类似的东西，但嗯，我

**[00:01:26]** blocks to be more clear or n blocks or something like that but yeah I'm
> 类似的东西，但嗯，我在这里坚持用n_layers，因为

**[00:01:26]** something like that but yeah I'm
> 在这里坚持用n_layers，因为

**[00:01:26]** something like that but yeah I'm sticking here with n layers because
> 在这里坚持用n_layers，因为这是这篇论文中使用的，并且

**[00:01:26]** sticking here with n layers because
> 这是这篇论文中使用的，并且

**[00:01:26]** sticking here with n layers because that's what's used in this um paper and
> 这是这篇论文中使用的，并且你知道，这是当今LLM术语中的惯例

**[00:01:26]** that's what's used in this um paper and
> 你知道，这是当今LLM术语中的惯例

**[00:01:26]** that's what's used in this um paper and what's you know a convention nowadays in
> 你知道，这是当今LLM术语中的惯例，嗯，但还有一件事

**[00:01:26]** what's you know a convention nowadays in
> 嗯，但还有一件事

**[00:01:26]** what's you know a convention nowadays in llm terminology um but there's one more
> 嗯，但还有一件事缺失，而且注意力头的数量也不同

**[00:01:26]** llm terminology um but there's one more
> 缺失，而且注意力头的数量也不同

**[00:01:26]** llm terminology um but there's one more thing missing and also the number of
> 缺失，而且注意力头的数量也不同，所以他们在论文中没有

**[00:01:26]** thing missing and also the number of
> 所以他们在论文中没有

**[00:01:26]** thing missing and also the number of heads is different so they didn't
> 所以他们在论文中没有描述这一点，但你可以

**[00:01:26]** heads is different so they didn't
> 描述这一点，但你可以

**[00:01:26]** heads is different so they didn't describe that in the paper but you can
> 描述这一点，但你可以看到，实际上当你下载

**[00:01:26]** describe that in the paper but you can
> 看到，实际上当你下载

**[00:01:26]** describe that in the paper but you can see that actually when you download the
> 看到，实际上当你下载模型权重时，我们将在第5章中这样做

**[00:01:26]** see that actually when you download the
> 模型权重时，我们将在第5章中这样做

**[00:01:26]** see that actually when you download the model weights and we will be doing that
> 模型权重时，我们将在第5章中这样做，他们改变了

**[00:01:26]** model weights and we will be doing that
> 他们改变了

**[00:01:26]** model weights and we will be doing that in chapter 5 that they also changed the
> 他们改变了注意力头的数量，为了方便你们

**[00:01:26]** in chapter 5 that they also changed the
> 注意力头的数量，为了方便你们

**[00:01:26]** in chapter 5 that they also changed the number of heads and for your convenience
> 注意力头的数量，为了方便你们，我会直接复制粘贴这些数字

**[00:01:27]** number of heads and for your convenience
> 我会直接复制粘贴这些数字

**[00:01:27]** number of heads and for your convenience I will just copy and paste these numbers
> 我会直接复制粘贴这些数字

**[00:01:27]** I will just copy and paste these numbers
> I will just copy and paste these numbers

**[00:01:27]** I will just copy and paste these numbers here so here um I have the four
> 我直接把这些数字复制粘贴过来，所以这里我有四个

**[00:01:27]** here so here um I have the four
> 所以这里我有四个

**[00:01:27]** here so here um I have the four different model sizes GPT to small which
> 所以这里我有四个不同的模型尺寸：GPT-2 small

**[00:01:27]** different model sizes GPT to small which
> 不同的模型尺寸：GPT-2 small

**[00:01:27]** different model sizes GPT to small which we just implemented 12 layers 12 heads
> 不同的模型尺寸：GPT-2 small，我们刚刚实现了12层、12个head

**[00:01:27]** we just implemented 12 layers 12 heads
> 我们刚刚实现了12层、12个head

**[00:01:27]** we just implemented 12 layers 12 heads and 768 dimensions for the embeddings
> 我们刚刚实现了12层、12个head，以及768维的embedding

**[00:01:27]** and 768 dimensions for the embeddings
> 以及768维的embedding

**[00:01:27]** and 768 dimensions for the embeddings and then medium large and XL correspond
> 以及768维的embedding，然后medium、large和XL对应

**[00:01:27]** and then medium large and XL correspond
> 然后medium、large和XL对应

**[00:01:27]** and then medium large and XL correspond to these three models and yeah as an
> 然后medium、large和XL对应这三个模型，嗯，作为

**[00:01:27]** to these three models and yeah as an
> 这三个模型，嗯，作为

**[00:01:27]** to these three models and yeah as an exercise I encourage you to try out
> 这三个模型，嗯，作为练习，我鼓励你尝试

**[00:01:27]** exercise I encourage you to try out
> 练习，我鼓励你尝试

**[00:01:27]** exercise I encourage you to try out these mods models um you only have to
> 练习，我鼓励你尝试这些模型，你只需要

**[00:01:27]** these mods models um you only have to
> 这些模型，你只需要

**[00:01:27]** these mods models um you only have to change this um vocabul so the dictionary
> 这些模型，你只需要修改这里的词汇表，也就是字典

**[00:01:27]** change this um vocabul so the dictionary
> 修改这里的词汇表，也就是字典

**[00:01:27]** change this um vocabul so the dictionary here and then you can reinitialize
> 修改这里的词汇表，也就是字典，然后你可以重新初始化

**[00:01:27]** here and then you can reinitialize
> 然后你可以重新初始化

**[00:01:27]** here and then you can reinitialize different sized models and you can also
> 然后你可以重新初始化不同尺寸的模型，你还可以

**[00:01:27]** different sized models and you can also
> 不同尺寸的模型，你还可以

**[00:01:27]** different sized models and you can also compute the number of parameters the
> 不同尺寸的模型，你还可以计算参数数量、

**[00:01:27]** compute the number of parameters the
> 计算参数数量、

**[00:01:27]** compute the number of parameters the size of the parameters the sorry the
> 计算参数数量、参数大小，抱歉，是

**[00:01:27]** size of the parameters the sorry the
> 参数大小，抱歉，是

**[00:01:27]** size of the parameters the sorry the number of the parameters and the size of
> 参数大小，抱歉，是参数数量和模型大小

**[00:01:27]** number of the parameters and the size of
> 参数数量和模型大小

**[00:01:27]** number of the parameters and the size of the model using this code and yeah just
> 参数数量和模型大小，使用这段代码，嗯，只是

**[00:01:27]** the model using this code and yeah just
> 使用这段代码，嗯，只是

**[00:01:27]** the model using this code and yeah just find out how big these models are as an
> 使用这段代码，嗯，只是作为可选练习，看看这些模型有多大

**[00:01:27]** find out how big these models are as an
> 作为可选练习，看看这些模型有多大

**[00:01:27]** find out how big these models are as an optional exercise and then in the next
> 作为可选练习，看看这些模型有多大，然后在下一个

**[00:01:27]** optional exercise and then in the next
> 可选练习，然后在下一个

**[00:01:27]** optional exercise and then in the next video uh I will go over how we convert
> 可选练习，然后在下一个视频中，我会讲解如何将

**[00:01:27]** video uh I will go over how we convert
> 视频中，我会讲解如何将

**[00:01:27]** video uh I will go over how we convert the output back into words
> 视频中，我会讲解如何将输出转换回单词

**[00:01:27]** let's now talk about how we can use the
> 现在我们来讨论如何使用

**[00:01:28]** let's now talk about how we can use the
> 现在我们来讨论如何使用

**[00:01:28]** let's now talk about how we can use the GPT model to generate actual text so in
> 现在我们来讨论如何使用GPT模型生成实际文本，所以在

**[00:01:28]** GPT model to generate actual text so in
> GPT模型生成实际文本，所以在

**[00:01:28]** GPT model to generate actual text so in the previous section we implemented this
> GPT模型生成实际文本，所以在上一节中，我们实现了这个

**[00:01:28]** the previous section we implemented this
> 上一节中，我们实现了这个

**[00:01:28]** the previous section we implemented this GPT model architecture and we have seen
> 上一节中，我们实现了这个GPT模型架构，并且我们看到

**[00:01:28]** GPT model architecture and we have seen
> GPT模型架构，并且我们看到

**[00:01:28]** GPT model architecture and we have seen that it outputs this tensor so it
> GPT模型架构，并且我们看到它输出这个tensor，所以它

**[00:01:28]** that it outputs this tensor so it
> 它输出这个tensor，所以它

**[00:01:28]** that it outputs this tensor so it doesn't look like text at all yet so
> 它输出这个tensor，所以它看起来还不像文本，所以

**[00:01:28]** doesn't look like text at all yet so
> 看起来还不像文本，所以

**[00:01:28]** doesn't look like text at all yet so there's something else we have to do now
> 看起来还不像文本，所以我们现在还需要做其他事情

**[00:01:28]** there's something else we have to do now
> 我们现在还需要做其他事情

**[00:01:28]** there's something else we have to do now so one thing I mentioned before is that
> 我们现在还需要做其他事情，所以之前我提到过一件事

**[00:01:28]** so one thing I mentioned before is that
> 之前我提到过一件事

**[00:01:28]** so one thing I mentioned before is that if we have four input tokens we get a
> 之前我提到过一件事，如果我们有四个输入token，我们会得到一个

**[00:01:28]** if we have four input tokens we get a
> 如果我们有四个输入token，我们会得到一个

**[00:01:28]** if we have four input tokens we get a tensor here consisting of four rows but
> 如果我们有四个输入token，我们会得到一个包含四行的tensor，但是

**[00:01:28]** tensor here consisting of four rows but
> 包含四行的tensor，但是

**[00:01:28]** tensor here consisting of four rows but each row is 50, 257 dimensional so how
> 包含四行的tensor，但是每一行是50,257维的，那么如何

**[00:01:28]** each row is 50, 257 dimensional so how
> 每一行是50,257维的，那么如何

**[00:01:28]** each row is 50, 257 dimensional so how do we get these 50257 Dimensions back
> 每一行是50,257维的，那么如何将这些50257维转换回

**[00:01:28]** do we get these 50257 Dimensions back
> 将这些50257维转换回

**[00:01:28]** do we get these 50257 Dimensions back into words that's the focus of this
> 将这些50257维转换回单词，这就是本节的重点

**[00:01:28]** into words that's the focus of this
> 单词，这就是本节的重点

**[00:01:28]** into words that's the focus of this section and so one thing before we get
> 单词，这就是本节的重点，所以在开始之前

**[00:01:28]** section and so one thing before we get
> 所以在开始之前

**[00:01:28]** section and so one thing before we get there I wanted to mention is llms
> 所以在开始之前，我想提一下，LLM

**[00:01:28]** there I wanted to mention is llms
> 我想提一下，LLM

**[00:01:28]** there I wanted to mention is llms generate one word at a time so if we
> 我想提一下，LLM一次生成一个单词，所以如果我们

**[00:01:28]** generate one word at a time so if we
> 一次生成一个单词，所以如果我们

**[00:01:28]** generate one word at a time so if we have an input hello I am it might
> 一次生成一个单词，所以如果我们有输入"hello I am"，它可能会

**[00:01:28]** have an input hello I am it might
> 有输入"hello I am"，它可能会

**[00:01:28]** have an input hello I am it might generate the next word a and then we
> 有输入"hello I am"，它可能会生成下一个单词"a"，然后我们

**[00:01:28]** generate the next word a and then we
> 生成下一个单词"a"，然后我们

**[00:01:28]** generate the next word a and then we concatenate the a back to the input
> 生成下一个单词"a"，然后我们将"a"拼接回输入

**[00:01:28]** concatenate the a back to the input
> 将"a"拼接回输入

**[00:01:28]** concatenate the a back to the input hello I'm a and then in the next round
> 将"a"拼接回输入"hello I'm a"，然后在下一轮

**[00:01:28]** hello I'm a and then in the next round
> "hello I'm a"，然后在下一轮

**[00:01:28]** hello I'm a and then in the next round it generates the next word and so forth
> "hello I'm a"，然后在下一轮它生成下一个单词，依此类推

**[00:01:28]** it generates the next word and so forth
> 它生成下一个单词，依此类推

**[00:01:28]** it generates the next word and so forth so it's actually an iterative process
> 它生成下一个单词，依此类推，所以这实际上是一个迭代过程

**[00:01:28]** so it's actually an iterative process
> 所以这实际上是一个迭代过程

**[00:01:28]** so it's actually an iterative process where we are always generating one word
> 所以这实际上是一个迭代过程，我们总是生成一个词

**[00:01:29]** where we are always generating one word
> 我们总是生成一个词

**[00:01:29]** where we are always generating one word or one token and then we append this
> 我们总是生成一个词或一个token，然后我们将其追加

**[00:01:29]** or one token and then we append this
> 或一个token，然后我们将其追加

**[00:01:29]** or one token and then we append this token to the input and we put the input
> 或一个token，然后我们把这个token追加到输入中，再把输入

**[00:01:29]** token to the input and we put the input
> token追加到输入中，再把输入

**[00:01:29]** token to the input and we put the input again through the llm and so forth and
> token追加到输入中，再次通过LLM处理，以此类推

**[00:01:29]** again through the llm and so forth and
> 再次通过LLM处理，以此类推

**[00:01:29]** again through the llm and so forth and if you have used uh web application like
> 再次通过LLM处理，以此类推，如果你用过像

**[00:01:29]** if you have used uh web application like
> 如果你用过像

**[00:01:29]** if you have used uh web application like cgpd for example if you type in a query
> 如果你用过像ChatGPT这样的网络应用，比如你输入一个查询

**[00:01:29]** cgpd for example if you type in a query
> ChatGPT这样的网络应用，比如你输入一个查询

**[00:01:29]** cgpd for example if you type in a query and you pay close attention to the
> ChatGPT这样的网络应用，比如你输入一个查询，并仔细观察

**[00:01:29]** and you pay close attention to the
> 并仔细观察

**[00:01:29]** and you pay close attention to the output you might see there's one token
> 并仔细观察输出，你可能会看到一次只有一个token

**[00:01:29]** output you might see there's one token
> 输出，你可能会看到一次只有一个token

**[00:01:29]** output you might see there's one token or word appearing at a time when it
> 输出，你可能会看到一次只有一个token或词出现，当它

**[00:01:29]** or word appearing at a time when it
> 或词出现，当它

**[00:01:29]** or word appearing at a time when it generates the output and that's because
> 或词出现，当它生成输出时，这是因为

**[00:01:29]** generates the output and that's because
> 生成输出时，这是因为

**[00:01:29]** generates the output and that's because it's an iterative process it's not
> 生成输出时，这是因为这是一个迭代过程，它并不是

**[00:01:29]** it's an iterative process it's not
> 这是一个迭代过程，它并不是

**[00:01:29]** it's an iterative process it's not generating the whole response all at
> 这是一个迭代过程，它并不是一次性生成整个回复

**[00:01:29]** generating the whole response all at
> 一次性生成整个回复

**[00:01:29]** generating the whole response all at once it's doing it step by step and so
> 一次性生成整个回复，而是逐步进行的，所以

**[00:01:29]** once it's doing it step by step and so
> 而是逐步进行的，所以

**[00:01:29]** once it's doing it step by step and so how does that work um here is an
> 而是逐步进行的，那么这是如何工作的呢？这里有一个

**[00:01:29]** how does that work um here is an
> 那么这是如何工作的呢？这里有一个

**[00:01:29]** how does that work um here is an overview um it's maybe very technical
> 那么这是如何工作的呢？这里有一个概述，可能技术性很强

**[00:01:29]** overview um it's maybe very technical
> 概述，可能技术性很强

**[00:01:29]** overview um it's maybe very technical but we will get through this I promise
> 概述，可能技术性很强，但我们会讲清楚，我保证

**[00:01:29]** but we will get through this I promise
> 但我们会讲清楚，我保证

**[00:01:29]** but we will get through this I promise you so what you are seeing here is again
> 但我们会讲清楚，我保证。你现在看到的是

**[00:01:29]** you so what you are seeing here is again
> 你现在看到的是

**[00:01:29]** you so what you are seeing here is again the input hello I am and for now please
> 你现在看到的是输入"hello I am"，现在请

**[00:01:29]** the input hello I am and for now please
> 输入"hello I am"，现在请

**[00:01:29]** the input hello I am and for now please ignore anything about below the dash
> 输入"hello I am"，现在请忽略虚线以下的内容

**[00:01:29]** ignore anything about below the dash
> 忽略虚线以下的内容

**[00:01:29]** ignore anything about below the dash line so the input might be hello I am
> 忽略虚线以下的内容。所以输入可能是"hello I am"

**[00:01:29]** line so the input might be hello I am
> 所以输入可能是"hello I am"

**[00:01:29]** line so the input might be hello I am and then similar to what we have done or
> 所以输入可能是"hello I am"，然后类似于我们在

**[00:01:29]** and then similar to what we have done or
> 然后类似于我们在

**[00:01:29]** and then similar to what we have done or learned in Chapter 2 first we convert
> 然后类似于我们在第2章所做或学到的，首先我们将

**[00:01:29]** learned in Chapter 2 first we convert
> 第2章所学到的，首先我们将

**[00:01:29]** learned in Chapter 2 first we convert the tokens um from a string
> 第2章所学到的，首先我们将tokens从字符串

**[00:01:29]** the tokens um from a string
> tokens从字符串

**[00:01:29]** the tokens um from a string representation into a token ID into an
> tokens从字符串表示转换为token ID，转换为

**[00:01:29]** representation into a token ID into an
> 转换为token ID，转换为

**[00:01:29]** representation into a token ID into an integer representation so we are
> 转换为token ID，转换为整数表示，所以我们

**[00:01:29]** integer representation so we are
> 整数表示，所以我们

**[00:01:29]** integer representation so we are converting text into token IDs and then
> 整数表示，所以我们把文本转换为token IDs，然后

**[00:01:30]** converting text into token IDs and then
> 把文本转换为token IDs，然后

**[00:01:30]** converting text into token IDs and then we put these token IDs into GPT uh the
> 把文本转换为token IDs，然后把这些token IDs输入到GPT

**[00:01:30]** we put these token IDs into GPT uh the
> 把这些token IDs输入到GPT

**[00:01:30]** we put these token IDs into GPT uh the GPT model we just coded in the previous
> 把这些token IDs输入到我们之前编码的GPT模型中

**[00:01:30]** GPT model we just coded in the previous
> 我们之前编码的GPT模型中

**[00:01:30]** GPT model we just coded in the previous section and then like we have seen it
> 我们之前编码的GPT模型中，然后像我们看到的

**[00:01:30]** section and then like we have seen it
> 然后像我们看到的

**[00:01:30]** section and then like we have seen it outputs this tensor here and it has four
> 然后像我们看到的，它输出这个tensor，它有四行

**[00:01:30]** outputs this tensor here and it has four
> 它输出这个tensor，它有四行

**[00:01:30]** outputs this tensor here and it has four rows and each of the rows corresponds to
> 它输出这个tensor，它有四行，每一行对应

**[00:01:30]** rows and each of the rows corresponds to
> 每一行对应

**[00:01:30]** rows and each of the rows corresponds to one of the input um words here and if
> 每一行对应输入中的一个词，如果

**[00:01:30]** one of the input um words here and if
> 输入中的一个词，如果

**[00:01:30]** one of the input um words here and if you recall long time ago but if you
> 输入中的一个词，如果你还记得很久以前，但如果你

**[00:01:30]** you recall long time ago but if you
> 你记得很久以前，但如果你

**[00:01:30]** you recall long time ago but if you recall from chapter 2 the way we set up
> 你记得很久以前，但如果你记得第2章，我们设置

**[00:01:30]** recall from chapter 2 the way we set up
> 记得第2章，我们设置

**[00:01:30]** recall from chapter 2 the way we set up our data ler was that the targets are
> 记得第2章，我们设置data loader的方式是，目标

**[00:01:30]** our data ler was that the targets are
> data loader的方式是，目标

**[00:01:30]** our data ler was that the targets are the inputs shifted by one position so
> data loader的方式是，目标是输入向右偏移一个位置

**[00:01:30]** the inputs shifted by one position so
> 输入向右偏移一个位置

**[00:01:30]** the inputs shifted by one position so the way we are going to train the llm is
> 输入向右偏移一个位置，所以我们训练LLM的方式是

**[00:01:30]** the way we are going to train the llm is
> 我们训练LLM的方式是

**[00:01:30]** the way we are going to train the llm is that for each token it will associate
> 我们训练LLM的方式是，对于每个token，它会关联

**[00:01:30]** that for each token it will associate
> 对于每个token，它会关联

**[00:01:30]** that for each token it will associate the next token because of the plus one
> 对于每个token，它会关联下一个token，因为我们在

**[00:01:30]** the next token because of the plus one
> 下一个token，因为我们在

**[00:01:30]** the next token because of the plus one shifting that we implemented in the
> 下一个token，因为我们在实现中做了加一偏移

**[00:01:30]** shifting that we implemented in the
> 我们在目标中实现的偏移

**[00:01:30]** shifting that we implemented in the targets so what that ultimately means
> 我们在目标中实现的偏移，所以最终这意味着

**[00:01:30]** targets so what that ultimately means
> 目标中实现的偏移，所以最终这意味着

**[00:01:30]** targets so what that ultimately means is at each position this Vector will
> 目标中实现的偏移，所以最终这意味着在每个位置，这个Vector将

**[00:01:30]** is at each position this Vector will
> 在每个位置，这个Vector将

**[00:01:30]** is at each position this Vector will contain information about the next
> 在每个位置，这个Vector将包含关于下一个位置的信息

**[00:01:30]** contain information about the next
> 包含关于下一个位置的信息

**[00:01:30]** contain information about the next position about the plus one position and
> 包含关于下一个位置的信息，即加一位置的信息

**[00:01:30]** position about the plus one position and
> 关于加一位置的信息

**[00:01:30]** position about the plus one position and so that means that the last here the
> 关于加一位置的信息，所以这意味着最后一个，这里的

**[00:01:30]** so that means that the last here the
> 所以这意味着最后一个，这里的

**[00:01:30]** so that means that the last here the last Vector which corresponds to the
> 所以这意味着最后一个，这里的最后一个Vector，对应于输入中的

**[00:01:30]** last Vector which corresponds to the
> 最后一个Vector，对应于输入中的

**[00:01:30]** last Vector which corresponds to the last Vector in the input will contain
> 最后一个Vector，对应于输入中的最后一个Vector，将包含

**[00:01:30]** last Vector in the input will contain
> 输入中的最后一个Vector，将包含

**[00:01:30]** last Vector in the input will contain information about the next word we want
> 输入中的最后一个Vector，将包含关于我们想要生成的下一个词的信息

**[00:01:30]** information about the next word we want
> 关于我们想要生成的下一个词的信息

**[00:01:30]** information about the next word we want to generate so the last row contains the
> 关于我们想要生成的下一个词的信息，所以最后一行包含

**[00:01:31]** to generate so the last row contains the
> 要生成的下一个词的信息，所以最后一行包含

**[00:01:31]** to generate so the last row contains the information we need for the next word
> 要生成的下一个词的信息，所以最后一行包含我们为下一个词所需的信息

**[00:01:31]** information we need for the next word
> 我们为下一个词所需的信息

**[00:01:31]** information we need for the next word but we need to now convert this Vector
> 我们为下一个词所需的信息，但我们现在需要将这个Vector

**[00:01:31]** but we need to now convert this Vector
> 但我们现在需要将这个Vector

**[00:01:31]** but we need to now convert this Vector back into the yeah into the word that we
> 但我们现在需要将这个Vector转换回，是的，转换回我们想要生成的词

**[00:01:31]** back into the yeah into the word that we
> 转换回我们想要生成的词

**[00:01:31]** back into the yeah into the word that we want to generate how does that work one
> 转换回我们想要生成的词，这如何工作？一个

**[00:01:31]** want to generate how does that work one
> 想要生成的词，这如何工作？一个

**[00:01:31]** want to generate how does that work one thing I wanted to mention is this last
> 想要生成的词，这如何工作？我想提到的一点是，这最后一行

**[00:01:31]** thing I wanted to mention is this last
> 我想提到的一点是，这最后一行

**[00:01:31]** thing I wanted to mention is this last row in general these last rows here they
> 我想提到的一点是，这最后一行，一般来说，这些最后一行

**[00:01:31]** row in general these last rows here they
> 一般来说，这些最后一行

**[00:01:31]** row in general these last rows here they are also called logits um it's not a llm
> 一般来说，这些最后一行也被称为logits，嗯，这不是一个LLM术语

**[00:01:31]** are also called logits um it's not a llm
> 也被称为logits，嗯，这不是一个LLM术语

**[00:01:31]** are also called logits um it's not a llm term it's a general deep learning term I
> 也被称为logits，嗯，这不是一个LLM术语，而是一个通用的深度学习术语

**[00:01:31]** term it's a general deep learning term I
> 而是一个通用的深度学习术语

**[00:01:31]** term it's a general deep learning term I don't want to go into too much detail
> 而是一个通用的深度学习术语，我不想深入细节

**[00:01:31]** don't want to go into too much detail
> 我不想深入细节

**[00:01:31]** don't want to go into too much detail why they are called logits that would go
> 我不想深入细节，解释为什么它们被称为logits，那会

**[00:01:31]** why they are called logits that would go
> 解释为什么它们被称为logits，那会

**[00:01:31]** why they are called logits that would go into a statistics lecture here but uh in
> 解释为什么它们被称为logits，那会进入统计学讲座，但嗯

**[00:01:31]** into a statistics lecture here but uh in
> 进入统计学讲座，但嗯

**[00:01:31]** into a statistics lecture here but uh in general it's um convention to call the
> 进入统计学讲座，但嗯，一般来说，约定俗成地称

**[00:01:31]** general it's um convention to call the
> 一般来说，约定俗成地称

**[00:01:31]** general it's um convention to call the output of of the last linear layer so
> 一般来说，约定俗成地称最后一个线性层的输出

**[00:01:31]** output of of the last linear layer so
> 最后一个线性层的输出

**[00:01:31]** output of of the last linear layer so the output of the last linear layer in a
> 最后一个线性层的输出，所以神经网络中最后一个线性层的输出

**[00:01:31]** the output of the last linear layer in a
> 神经网络中最后一个线性层的输出

**[00:01:31]** the output of the last linear layer in a neur network is usually called um the
> 神经网络中最后一个线性层的输出通常被称为logits

**[00:01:31]** neur network is usually called um the
> 通常被称为logits

**[00:01:31]** neur network is usually called um the logits so the values that come out of
> 通常被称为logits，所以从这个线性层出来的值

**[00:01:31]** logits so the values that come out of
> 从这个线性层出来的值

**[00:01:31]** logits so the values that come out of this linear layer here so we call them
> 从这个线性层出来的值，我们称它们为

**[00:01:31]** this linear layer here so we call them
> 我们称它们为

**[00:01:31]** this linear layer here so we call them logits and so here we are looking at the
> 我们称它们为logits，所以这里我们正在查看

**[00:01:31]** logits and so here we are looking at the
> logits，所以这里我们正在查看

**[00:01:31]** logits and so here we are looking at the logits of the last token so we are just
> logits，所以这里我们正在查看最后一个token的logits，所以我们只是

**[00:01:31]** logits of the last token so we are just
> 最后一个token的logits，所以我们只是

**[00:01:31]** logits of the last token so we are just you know copy pasting if you pay
> 最后一个token的logits，所以我们只是，你知道，复制粘贴，如果你注意

**[00:01:31]** you know copy pasting if you pay
> 你知道，复制粘贴，如果你注意

**[00:01:31]** you know copy pasting if you pay attention these are the same values
> 你知道，复制粘贴，如果你注意，这些是相同的值

**[00:01:31]** attention these are the same values
> 这些是相同的值

**[00:01:31]** attention these are the same values we're just copy and pasting here in this
> 这些是相同的值，我们只是在这个图中复制粘贴

**[00:01:31]** we're just copy and pasting here in this
> 我们只是在这个图中复制粘贴

**[00:01:31]** we're just copy and pasting here in this figure the last row which we call logits
> 我们只是在这个图中复制粘贴最后一行，我们称之为logits

**[00:01:32]** figure the last row which we call logits
> 最后一行，我们称之为logits

**[00:01:32]** figure the last row which we call logits and then we apply the softmax function
> 最后一行，我们称之为logits，然后我们应用softmax函数

**[00:01:32]** and then we apply the softmax function
> 然后我们应用softmax函数

**[00:01:32]** and then we apply the softmax function to normalize them so they range from
> 然后我们应用softmax函数来归一化它们，使它们范围从

**[00:01:32]** to normalize them so they range from
> 来归一化它们，使它们范围从

**[00:01:32]** to normalize them so they range from minus infinity to Infinity so it could
> 来归一化它们，使它们范围从负无穷到正无穷，所以可能

**[00:01:32]** minus infinity to Infinity so it could
> 负无穷到正无穷，所以可能

**[00:01:32]** minus infinity to Infinity so it could be very negative small values to very
> 负无穷到正无穷，所以可能从非常小的负值到非常大的正值

**[00:01:32]** be very negative small values to very
> 从非常小的负值到非常大的正值

**[00:01:32]** be very negative small values to very large values and we convert them into
> 从非常小的负值到非常大的正值，我们将它们转换为

**[00:01:32]** large values and we convert them into
> 我们将它们转换为

**[00:01:32]** large values and we convert them into probabilities which are all strictly
> 我们将它们转换为概率，这些概率都是严格

**[00:01:32]** probabilities which are all strictly
> 概率，这些概率都是严格

**[00:01:32]** probabilities which are all strictly positive and some up to one and by the
> 概率，这些概率都是严格为正且总和为一，顺便说一句

**[00:01:32]** positive and some up to one and by the
> 为正且总和为一，顺便说一句

**[00:01:32]** positive and some up to one and by the way this is an optional step this is not
> 为正且总和为一，顺便说一句，这是一个可选步骤，这不是

**[00:01:32]** way this is an optional step this is not
> 这是一个可选步骤，不是

**[00:01:32]** way this is an optional step this is not necessary you can actually skip this
> 这是一个可选步骤，不是必需的，你实际上可以跳过

**[00:01:32]** necessary you can actually skip this
> 必需的，你实际上可以跳过

**[00:01:32]** necessary you can actually skip this step
> 必需的，你实际上可以跳过这一步

**[00:01:32]** step
> 步骤

**[00:01:32]** step because the smallest value always
> 步骤，因为最小值总是

**[00:01:32]** because the smallest value always
> 因为最小值总是

**[00:01:32]** because the smallest value always corresponds to the smallest probability
> 因为最小值总是对应最小的概率

**[00:01:32]** corresponds to the smallest probability
> 对应最小的概率

**[00:01:32]** corresponds to the smallest probability and the largest value always corresponds
> 对应最小的概率，而最大值总是对应

**[00:01:32]** and the largest value always corresponds
> 而最大值总是对应

**[00:01:32]** and the largest value always corresponds to the largest probability so there is
> 而最大值总是对应最大的概率，所以没有

**[00:01:32]** to the largest probability so there is
> 对应最大的概率，所以没有

**[00:01:32]** to the largest probability so there is no need for the soft Max but it makes it
> 对应最大的概率，所以不需要softmax，但它让

**[00:01:32]** no need for the soft Max but it makes it
> 不需要softmax，但它让

**[00:01:32]** no need for the soft Max but it makes it a bit easier to interpret and so what
> 不需要softmax，但它让解释起来更容易一些，所以我们

**[00:01:32]** a bit easier to interpret and so what
> 解释起来更容易一些，所以我们

**[00:01:32]** a bit easier to interpret and so what we're looking at is the normalized
> 解释起来更容易一些，所以我们看到的是归一化的

**[00:01:32]** we're looking at is the normalized
> 我们看到的是归一化的

**[00:01:32]** we're looking at is the normalized Logics as probabilities and we are
> 我们看到的是归一化的logits作为概率，我们

**[00:01:32]** Logics as probabilities and we are
> logits作为概率，我们

**[00:01:32]** Logics as probabilities and we are interested in the so maybe one step back
> logits作为概率，我们感兴趣的是，所以也许退一步

**[00:01:32]** interested in the so maybe one step back
> 感兴趣的是，所以也许退一步

**[00:01:32]** interested in the so maybe one step back um I should say again this is 50, 257
> 感兴趣的是，所以也许退一步，嗯，我再说一遍，这是50,257

**[00:01:32]** um I should say again this is 50, 257
> 嗯，我再说一遍，这是50,257

**[00:01:32]** um I should say again this is 50, 257 dimensional so we have
> 嗯，我再说一遍，这是50,257维的，所以我们有

**[00:01:32]** dimensional so we have
> 维的，所以我们有

**[00:01:32]** dimensional so we have 50,2 the seven values here and each
> 维的，所以我们有50,257个值在这里，每个

**[00:01:32]** 50,2 the seven values here and each
> 50,257个值在这里，每个

**[00:01:32]** 50,2 the seven values here and each value corresponds to a token ID in the
> 50,257个值在这里，每个值对应词汇表中的一个token ID

**[00:01:32]** value corresponds to a token ID in the
> 值对应词汇表中的一个token ID

**[00:01:32]** value corresponds to a token ID in the vocabulary and now our goal is to train
> 值对应词汇表中的一个token ID，现在我们的目标是训练

**[00:01:33]** vocabulary and now our goal is to train
> 词汇表，现在我们的目标是训练

**[00:01:33]** vocabulary and now our goal is to train the network such that the correct next
> 词汇表，现在我们的目标是训练网络，使得我们想要生成的正确下一个

**[00:01:33]** the network such that the correct next
> 网络，使得我们想要生成的正确下一个

**[00:01:33]** the network such that the correct next token that we want to generate has the
> 网络，使得我们想要生成的正确下一个token具有

**[00:01:33]** token that we want to generate has the
> token具有

**[00:01:33]** token that we want to generate has the highest probability we are not training
> token具有最高的概率，我们这里没有训练

**[00:01:33]** highest probability we are not training
> 最高的概率，我们这里没有训练

**[00:01:33]** highest probability we are not training anything here this will be done in the
> 最高的概率，我们这里没有训练任何东西，这将在

**[00:01:33]** anything here this will be done in the
> 任何东西，这将在

**[00:01:33]** anything here this will be done in the next chapter but then one let's assume
> 任何东西，这将在下一章中完成，但假设

**[00:01:33]** next chapter but then one let's assume
> 下一章中完成，但假设

**[00:01:33]** next chapter but then one let's assume we have a well-trained GPT model it will
> 下一章中完成，但假设我们有一个训练好的GPT模型，它会

**[00:01:33]** we have a well-trained GPT model it will
> 我们有一个训练好的GPT模型，它会

**[00:01:33]** we have a well-trained GPT model it will associate the token we want to generate
> 我们有一个训练好的GPT模型，它会将我们想要生成的token

**[00:01:33]** associate the token we want to generate
> 将我们想要生成的token

**[00:01:33]** associate the token we want to generate with the highest probability so let's
> 将我们想要生成的token与最高概率关联起来，所以让我们

**[00:01:33]** with the highest probability so let's
> 与最高概率关联起来，所以让我们

**[00:01:33]** with the highest probability so let's just assume this is a train model and it
> 与最高概率关联起来，所以让我们假设这是一个训练好的模型，它

**[00:01:33]** just assume this is a train model and it
> 假设这是一个训练好的模型，它

**[00:01:33]** just assume this is a train model and it will have the highest probability in
> 假设这是一个训练好的模型，它会在位置257处具有最高概率

**[00:01:33]** will have the highest probability in
> 会在位置257处具有最高概率

**[00:01:33]** will have the highest probability in position 257 so this is the first
> 会在位置257处具有最高概率，所以这是第一个

**[00:01:33]** position 257 so this is the first
> 位置257，所以这是第一个

**[00:01:33]** position 257 so this is the first position position zero this is the last
> 位置257，所以这是第一个位置，位置0，这是最后一个

**[00:01:33]** position position zero this is the last
> 位置，位置0，这是最后一个

**[00:01:33]** position position zero this is the last position position
> 位置，位置0，这是最后一个位置，位置

**[00:01:33]** position position
> 位置，位置

**[00:01:33]** position position 50257 and everything else is dotted out
> 位置50257，其他所有东西都用点表示

**[00:01:33]** 50257 and everything else is dotted out
> 50257，其他所有东西都用点表示

**[00:01:33]** 50257 and everything else is dotted out so otherwise it won't fit and this model
> 50257，其他所有东西都用点表示，否则放不下，这个模型

**[00:01:33]** so otherwise it won't fit and this model
> 否则放不下，这个模型

**[00:01:33]** so otherwise it won't fit and this model happens to put the largest probability
> 否则放不下，这个模型恰好将最大的概率

**[00:01:33]** happens to put the largest probability
> 恰好将最大的概率

**[00:01:33]** happens to put the largest probability score here into this position uh 257 and
> 恰好将最大的概率分数放在了这个位置，嗯，257，并且

**[00:01:33]** score here into this position uh 257 and
> 分数放在了这个位置，嗯，257，并且

**[00:01:33]** score here into this position uh 257 and if we take our tokenizer from chapter 2
> 分数放在了这个位置，嗯，257，如果我们从第2章拿出tokenizer

**[00:01:33]** if we take our tokenizer from chapter 2
> 如果我们从第2章拿出tokenizer

**[00:01:33]** if we take our tokenizer from chapter 2 and we use tokenizer decode put in the
> 如果我们从第2章拿出tokenizer，并使用tokenizer decode输入

**[00:01:33]** and we use tokenizer decode put in the
> 并使用tokenizer decode输入

**[00:01:33]** and we use tokenizer decode put in the token ID 2 57 it will generate the next
> 并使用tokenizer decode输入token ID 257，它会生成下一个

**[00:01:33]** token ID 2 57 it will generate the next
> token ID 257，它会生成下一个

**[00:01:33]** token ID 2 57 it will generate the next token a so this is the desired token we
> token ID 257，它会生成下一个token a，所以这是我们想要的token

**[00:01:33]** token a so this is the desired token we
> token a，所以这是我们想要的token

**[00:01:33]** token a so this is the desired token we want to generate
> token a，所以这是我们想要生成的token

**[00:01:34]** want to generate
> 想要生成的token

**[00:01:34]** want to generate now uh how does this actually well maybe
> 想要生成的token，现在，嗯，这实际上是如何工作的，也许

**[00:01:34]** now uh how does this actually well maybe
> 现在，嗯，这实际上是如何工作的，也许

**[00:01:34]** now uh how does this actually well maybe one more thing here is we then take this
> 现在，嗯，这实际上是如何工作的，也许还有一点是，我们然后取这个

**[00:01:34]** one more thing here is we then take this
> 这里还有一点，我们接下来会把这个

**[00:01:34]** one more thing here is we then take this generated token and append it to the
> 这里还有一点，我们接下来会把这个生成的token附加到

**[00:01:34]** generated token and append it to the
> 生成的token附加到

**[00:01:34]** generated token and append it to the input for the next round so this is only
> 生成的token附加到下一轮的输入中，所以这只是

**[00:01:34]** input for the next round so this is only
> 下一轮的输入中，所以这只是

**[00:01:34]** input for the next round so this is only showing you one round and like I
> 下一轮的输入中，所以这只是向你展示了一轮，就像我

**[00:01:34]** showing you one round and like I
> 向你展示了一轮，就像我

**[00:01:34]** showing you one round and like I mentioned before we do this in an
> 向你展示了一轮，就像我之前提到的，我们以

**[00:01:34]** mentioned before we do this in an
> 之前提到的，我们以

**[00:01:34]** mentioned before we do this in an iterative fashion um as long as we want
> 之前提到的，我们以迭代的方式来做，只要我们需要

**[00:01:34]** iterative fashion um as long as we want
> 迭代的方式，只要我们需要

**[00:01:34]** iterative fashion um as long as we want to generate more text um so every time
> 迭代的方式，只要我们需要生成更多文本，所以每次

**[00:01:34]** to generate more text um so every time
> 生成更多文本，所以每次

**[00:01:34]** to generate more text um so every time we generate the next token it's added to
> 生成更多文本，所以每次我们生成下一个token时，它会被添加到

**[00:01:34]** we generate the next token it's added to
> 我们生成下一个token时，它会被添加到

**[00:01:34]** we generate the next token it's added to the input and then we put this whole
> 我们生成下一个token时，它会被添加到输入中，然后我们把整个

**[00:01:34]** the input and then we put this whole
> 输入中，然后我们把整个

**[00:01:34]** the input and then we put this whole input again through this Pipeline and so
> 输入中，然后我们把整个输入再次通过这个Pipeline，以此

**[00:01:34]** input again through this Pipeline and so
> 输入再次通过这个Pipeline，以此

**[00:01:34]** input again through this Pipeline and so forth but yeah to make it maybe more
> 输入再次通过这个Pipeline，以此类推，但为了让它更

**[00:01:34]** forth but yeah to make it maybe more
> 类推，但为了让它更

**[00:01:34]** forth but yeah to make it maybe more clear how does that actually work out in
> 类推，但为了让它更清楚，这在代码中实际上是如何工作的呢？所以让我进入notebook，

**[00:01:34]** clear how does that actually work out in
> 清楚，这在代码中实际上是如何工作的呢？所以让我进入notebook，

**[00:01:34]** clear how does that actually work out in code so let me go to the notebook and
> 清楚，这在代码中实际上是如何工作的呢？所以让我进入notebook，让我来编写代码，这样我们可以定义一个

**[00:01:34]** code so let me go to the notebook and
> 让我来编写代码，这样我们可以定义一个

**[00:01:34]** code so let me go to the notebook and let me um code this so we can define a
> 让我来编写代码，这样我们可以定义一个函数，称之为generate，比如text

**[00:01:34]** let me um code this so we can define a
> 函数，称之为generate，比如text

**[00:01:34]** let me um code this so we can define a function call it generate let's say text
> 函数，称之为generate，比如text，简单版本，为什么简单？因为我们将

**[00:01:34]** function call it generate let's say text
> 简单版本，为什么简单？因为我们将

**[00:01:34]** function call it generate let's say text um simple why simple because we will be
> 简单版本，为什么简单？因为我们在后续章节中会实现更复杂的

**[00:01:34]** um simple why simple because we will be
> 在后续章节中会实现更复杂的

**[00:01:34]** um simple why simple because we will be implementing more sophisticated um
> 在后续章节中会实现更复杂的版本，其中包含一些额外的采样过程，

**[00:01:34]** implementing more sophisticated um
> 版本，其中包含一些额外的采样过程，

**[00:01:34]** implementing more sophisticated um versions of that in later chapters that
> 版本，其中包含一些额外的采样过程，但最简单的版本，也就是

**[00:01:34]** versions of that in later chapters that
> 但最简单的版本，也就是

**[00:01:34]** versions of that in later chapters that contain some additional samp procedures
> 但最简单的版本，也就是我在图中向你展示的那个，工作方式

**[00:01:34]** contain some additional samp procedures
> 我在图中向你展示的那个，工作方式

**[00:01:34]** contain some additional samp procedures but the simplest version which is the
> 我在图中向你展示的那个，工作方式如下：我们有一个模型，我们

**[00:01:34]** but the simplest version which is the
> 如下：我们有一个模型，我们

**[00:01:34]** but the simplest version which is the one I showed you in the figure works as
> 如下：我们有一个模型，我们将其传递给函数，我们有一段文本，

**[00:01:34]** one I showed you in the figure works as
> 将其传递给函数，我们有一段文本，

**[00:01:34]** one I showed you in the figure works as follows so we have the model that we
> 将其传递给函数，我们有一段文本，称之为idx，即索引token，然后

**[00:01:35]** follows so we have the model that we
> 称之为idx，即索引token，然后

**[00:01:35]** follows so we have the model that we passed to our function we have the text
> 称之为idx，即索引token，然后我们传递想要生成的Max new

**[00:01:35]** passed to our function we have the text
> 我们传递想要生成的Max new

**[00:01:35]** passed to our function we have the text let's call them idx the index tokens and
> 我们传递想要生成的Max new tokens的数量，我们必须

**[00:01:35]** let's call them idx the index tokens and
> tokens的数量，我们必须

**[00:01:35]** let's call them idx the index tokens and then we um pass the number of Max new
> tokens的数量，我们必须定义一个数字，否则它会

**[00:01:35]** then we um pass the number of Max new
> 定义一个数字，否则它会

**[00:01:35]** then we um pass the number of Max new tokens we want to generate we have to
> 定义一个数字，否则它会无限生成更多token，然后

**[00:01:35]** tokens we want to generate we have to
> 无限生成更多token，然后

**[00:01:35]** tokens we want to generate we have to define a number otherwise it will be
> 无限生成更多token，然后还有，嗯，LLM支持的上下文

**[00:01:35]** define a number otherwise it will be
> 还有，嗯，LLM支持的上下文

**[00:01:35]** define a number otherwise it will be infinitely generating more tokens and
> 还有，嗯，LLM支持的上下文大小，然后我们

**[00:01:35]** infinitely generating more tokens and
> 大小，然后我们

**[00:01:35]** infinitely generating more tokens and then also the yeah supported context
> 大小，然后我们有一个for循环，所以嗯

**[00:01:35]** then also the yeah supported context
> 有一个for循环，所以嗯

**[00:01:35]** then also the yeah supported context size that the llm supports and then we
> 有一个for循环，所以嗯，这个for循环是针对我们想要

**[00:01:35]** size that the llm supports and then we
> 这个for循环是针对我们想要

**[00:01:35]** size that the llm supports and then we have a for Loop so um
> 这个for循环是针对我们想要生成的token数量，

**[00:01:35]** have a for Loop so um
> 生成的token数量，

**[00:01:35]** have a for Loop so um the for Loop is for how many tokens we
> 生成的token数量，然后我们不断生成，

**[00:01:35]** the for Loop is for how many tokens we
> 然后我们不断生成，

**[00:01:35]** the for Loop is for how many tokens we want to
> 然后我们不断生成，所以这里最终的目标是，我们将

**[00:01:35]** want to
> 所以这里最终的目标是，我们将

**[00:01:35]** want to generate and then we keep generating and
> 所以这里最终的目标是，我们将先暂时把它点出来，

**[00:01:35]** generate and then we keep generating and
> 先暂时把它点出来，

**[00:01:35]** generate and then we keep generating and so here the goal at the end so we will
> 先暂时把它点出来，然后我们将

**[00:01:35]** so here the goal at the end so we will
> 然后我们将

**[00:01:35]** so here the goal at the end so we will let's dot this out for the moment and
> 然后我们将返回修改后的索引，所以这里我们

**[00:01:35]** let's dot this out for the moment and
> 返回修改后的索引，所以这里我们

**[00:01:35]** let's dot this out for the moment and then we will
> 返回修改后的索引，所以这里我们传入索引，然后这里我们不断

**[00:01:35]** then we will
> 传入索引，然后这里我们不断

**[00:01:35]** then we will return the modified index so here we
> 传入索引，然后这里我们不断向索引追加内容，然后我们

**[00:01:35]** return the modified index so here we
> 向索引追加内容，然后我们

**[00:01:35]** return the modified index so here we pass in the index then here we keep
> 向索引追加内容，然后我们将其返回，所以这类似于

**[00:01:35]** pass in the index then here we keep
> pass in the index then here we keep

**[00:01:35]** pass in the index then here we keep appending to the index and then we we
> pass in the index then here we keep appending to the index and then we we

**[00:01:35]** appending to the index and then we we
> appending to the index and then we we

**[00:01:35]** appending to the index and then we we return it so that is similar to what
> appending to the index and then we we return it so that is similar to what

**[00:01:35]** return it so that is similar to what
> 返回它使其类似于

**[00:01:35]** return it so that is similar to what you're seeing here uh in the beginning
> 返回它使其类似于你一开始在这里看到的

**[00:01:35]** you're seeing here uh in the beginning
> 你一开始在这里看到的

**[00:01:35]** you're seeing here uh in the beginning in the first round we have this input
> 你一开始在这里看到的，在第一轮中我们有这个输入

**[00:01:35]** in the first round we have this input
> 在第一轮中我们有这个输入

**[00:01:35]** in the first round we have this input and then we generate the next word a and
> 在第一轮中我们有这个输入，然后我们生成下一个词a和

**[00:01:35]** and then we generate the next word a and
> 然后我们生成下一个词a和

**[00:01:35]** and then we generate the next word a and we add it to the um input and so forth
> 然后我们生成下一个词a和，并将其添加到输入中，依此类推

**[00:01:35]** we add it to the um input and so forth
> 将其添加到输入中，依此类推

**[00:01:35]** we add it to the um input and so forth in an interative fashion um maybe this
> 将其添加到输入中，依此类推，以迭代的方式，也许这

**[00:01:36]** in an interative fashion um maybe this
> 以迭代的方式，也许这

**[00:01:36]** in an interative fashion um maybe this is a bit abstract so what is
> 以迭代的方式，也许这有点抽象，所以什么是

**[00:01:36]** is a bit abstract so what is
> 有点抽象，所以什么是

**[00:01:36]** is a bit abstract so what is index so index uh correspond to the
> 有点抽象，所以什么是索引，索引对应于

**[00:01:36]** index so index uh correspond to the
> 索引对应于

**[00:01:36]** index so index uh correspond to the Token IDs or token index positions
> 索引对应于Token ID或token索引位置

**[00:01:36]** Token IDs or token index positions
> Token ID或token索引位置

**[00:01:36]** Token IDs or token index positions because um the token ID corresponds to
> Token ID或token索引位置，因为token ID对应于

**[00:01:36]** because um the token ID corresponds to
> 因为token ID对应于

**[00:01:36]** because um the token ID corresponds to the index position in this tensor so
> 因为token ID对应于这个tensor中的索引位置，所以

**[00:01:36]** the index position in this tensor so
> 这个tensor中的索引位置，所以

**[00:01:36]** the index position in this tensor so index position means 0 1 2 3 4 5 6 7 up
> 这个tensor中的索引位置，索引位置指的是0 1 2 3 4 5 6 7直到

**[00:01:36]** index position means 0 1 2 3 4 5 6 7 up
> 索引位置指的是0 1 2 3 4 5 6 7直到

**[00:01:36]** index position means 0 1 2 3 4 5 6 7 up to 50, 257 and that's the same as the
> 索引位置指的是0 1 2 3 4 5 6 7直到50,257，这与

**[00:01:36]** to 50, 257 and that's the same as the
> 到50,257，这与

**[00:01:36]** to 50, 257 and that's the same as the token ID so the token ID corresponds to
> 到50,257，这与token ID相同，所以token ID对应于

**[00:01:36]** token ID so the token ID corresponds to
> token ID对应于

**[00:01:36]** token ID so the token ID corresponds to the index position if I say token ID or
> token ID对应于索引位置，如果我说token ID或

**[00:01:36]** the index position if I say token ID or
> 索引位置，如果我说token ID或

**[00:01:36]** the index position if I say token ID or index here in this case means the same
> 索引位置，如果我说token ID或索引，在这种情况下指的是同一件事

**[00:01:36]** index here in this case means the same
> 索引，在这种情况下指的是同一件事

**[00:01:36]** index here in this case means the same thing because we're talking about the
> 索引，在这种情况下指的是同一件事，因为我们讨论的是

**[00:01:36]** thing because we're talking about the
> 因为我们讨论的是

**[00:01:36]** thing because we're talking about the index position of this probability
> 因为我们讨论的是这个概率tensor的索引位置

**[00:01:36]** index position of this probability
> 这个概率tensor的索引位置

**[00:01:36]** index position of this probability tensor okay uh so just to have something
> 这个概率tensor的索引位置，好的，所以为了有东西

**[00:01:36]** tensor okay uh so just to have something
> 好的，所以为了有东西

**[00:01:36]** tensor okay uh so just to have something to work with we have this start context
> 好的，所以为了有东西可以操作，我们有这个起始上下文

**[00:01:36]** to work with we have this start context
> 我们有这个起始上下文

**[00:01:36]** to work with we have this start context just um the first text we start with uh
> 我们有这个起始上下文，只是第一个文本，我们从

**[00:01:36]** just um the first text we start with uh
> 只是第一个文本，我们从

**[00:01:36]** just um the first text we start with uh we encode this with our tokenizer into
> 只是第一个文本，我们从，我们用tokenizer将其编码为

**[00:01:36]** we encode this with our tokenizer into
> 我们用tokenizer将其编码为

**[00:01:36]** we encode this with our tokenizer into token IDs so let me just show it here so
> 我们用tokenizer将其编码为token ID，所以让我在这里展示一下

**[00:01:36]** token IDs so let me just show it here so
> token ID，所以让我在这里展示一下

**[00:01:36]** token IDs so let me just show it here so these are our token IDs and we can also
> token ID，所以让我在这里展示一下，这些是我们的token ID，我们也可以

**[00:01:36]** these are our token IDs and we can also
> 这些是我们的token ID，我们也可以

**[00:01:36]** these are our token IDs and we can also call them indexes um yeah and so and
> 这些是我们的token ID，我们也可以称它们为索引，嗯，是的，然后

**[00:01:36]** call them indexes um yeah and so and
> 称它们为索引，嗯，是的，然后

**[00:01:36]** call them indexes um yeah and so and then we also we have because our model
> 称它们为索引，嗯，是的，然后我们还有，因为我们的模型

**[00:01:36]** then we also we have because our model
> 然后我们还有，因为我们的模型

**[00:01:36]** then we also we have because our model only works with pyo tensors we have to
> 然后我们还有，因为我们的模型只适用于PyTorch tensor，我们必须

**[00:01:37]** only works with pyo tensors we have to
> 只适用于PyTorch tensor，我们必须

**[00:01:37]** only works with pyo tensors we have to convert this list into a pytorch tensor
> 只适用于PyTorch tensor，我们必须将这个列表转换为PyTorch tensor

**[00:01:37]** convert this list into a pytorch tensor
> 将这个列表转换为PyTorch tensor

**[00:01:37]** convert this list into a pytorch tensor so can maybe just show it to you so this
> 将这个列表转换为PyTorch tensor，所以也许可以展示给你看，所以这个

**[00:01:37]** so can maybe just show it to you so this
> 所以也许可以展示给你看，所以这个

**[00:01:37]** so can maybe just show it to you so this is our pytorch tensor and I added the
> 所以也许可以展示给你看，所以这是我们的PyTorch tensor，我添加了

**[00:01:37]** is our pytorch tensor and I added the
> 这是我们的PyTorch tensor，我添加了

**[00:01:37]** is our pytorch tensor and I added the batch Dimension here so with this UNS
> 这是我们的PyTorch tensor，我添加了batch维度，所以通过这个unsqueeze

**[00:01:37]** batch Dimension here so with this UNS
> batch维度，所以通过这个unsqueeze

**[00:01:37]** batch Dimension here so with this UNS squeeze I just added an additional
> batch维度，所以通过这个unsqueeze，我添加了一个额外的

**[00:01:37]** squeeze I just added an additional
> 我添加了一个额外的

**[00:01:37]** squeeze I just added an additional Dimension so you can see this is now a
> 我添加了一个额外的维度，所以你可以看到现在这是一个

**[00:01:37]** Dimension so you can see this is now a
> 维度，所以你可以看到现在这是一个

**[00:01:37]** Dimension so you can see this is now a 1x4 dimensional one so because yeah our
> 维度，所以你可以看到现在这是一个1x4维的，因为，是的，我们的

**[00:01:37]** 1x4 dimensional one so because yeah our
> 1x4维的，因为，是的，我们的

**[00:01:37]** 1x4 dimensional one so because yeah our model
> 1x4维的，因为，是的，我们的模型

**[00:01:37]** model
> 模型

**[00:01:37]** model expects batches so we only have a batch
> 模型期望batch，所以我们只有一个batch

**[00:01:37]** expects batches so we only have a batch
> 期望batch，所以我们只有一个batch

**[00:01:37]** expects batches so we only have a batch size of one but it's just for
> 期望batch，所以我们只有一个batch大小为1，但这只是为了

**[00:01:37]** size of one but it's just for
> 大小为1，但这只是为了

**[00:01:37]** size of one but it's just for compatibility reasons and then we can
> 大小为1，但这只是为了兼容性原因，然后我们可以

**[00:01:37]** compatibility reasons and then we can
> 兼容性原因，然后我们可以

**[00:01:37]** compatibility reasons and then we can use that model later for training okay
> 兼容性原因，然后我们可以稍后使用该模型进行训练，好的

**[00:01:37]** use that model later for training okay
> 稍后使用该模型进行训练，好的

**[00:01:37]** use that model later for training okay now let's fill in the interesting part
> 稍后使用该模型进行训练，好的，现在让我们填写有趣的部分

**[00:01:37]** now let's fill in the interesting part
> 现在让我们填写有趣的部分

**[00:01:37]** now let's fill in the interesting part here so we are going to code now our
> 现在让我们填写有趣的部分，所以我们将编写我们的

**[00:01:37]** here so we are going to code now our
> 现在我们开始编写代码

**[00:01:37]** here so we are going to code now our token generation
> 现在我们开始编写token生成

**[00:01:37]** token generation
> token生成

**[00:01:37]** token generation so first we're going to truncate so what
> token生成，首先我们要进行截断

**[00:01:37]** so first we're going to truncate so what
> 首先我们要进行截断

**[00:01:37]** so first we're going to truncate so what we're doing here is we are
> 首先我们要进行截断，这里我们要做的是

**[00:01:37]** we're doing here is we are
> 这里我们要做的是

**[00:01:37]** we're doing here is we are truncating it up to a size that the
> 这里我们要做的是将其截断到模型支持的尺寸

**[00:01:37]** truncating it up to a size that the
> 将其截断到模型支持的尺寸

**[00:01:37]** truncating it up to a size that the model supports in our case that if we
> 将其截断到模型支持的尺寸，在我们的例子中，如果我们

**[00:01:37]** model supports in our case that if we
> 模型支持的尺寸，在我们的例子中，如果我们

**[00:01:37]** model supports in our case that if we have something like this that's not an
> 模型支持的尺寸，在我们的例子中，如果我们有这样的情况，那没问题

**[00:01:37]** have something like this that's not an
> 有这样的情况，那没问题

**[00:01:37]** have something like this that's not an issue it's just four tokens in our model
> 有这样的情况，那没问题，它只是我们模型中的四个token

**[00:01:37]** issue it's just four tokens in our model
> 没问题，它只是我们模型中的四个token

**[00:01:37]** issue it's just four tokens in our model supports um what have we defined um up
> 没问题，它只是我们模型中的四个token，模型支持，我们定义了

**[00:01:38]** supports um what have we defined um up
> 模型支持，我们定义了

**[00:01:38]** supports um what have we defined um up to 1,24 tokens but let's say we copy and
> 模型支持，我们定义了最多1,024个token，但假设我们复制并

**[00:01:38]** to 1,24 tokens but let's say we copy and
> 最多1,024个token，但假设我们复制并

**[00:01:38]** to 1,24 tokens but let's say we copy and paste a large document you know 50,000
> 最多1,024个token，但假设我们复制并粘贴一个大文档，比如50,000个

**[00:01:38]** paste a large document you know 50,000
> 粘贴一个大文档，比如50,000个

**[00:01:38]** paste a large document you know 50,000 tokens so that the model doesn't crash
> 粘贴一个大文档，比如50,000个token，这样模型就不会崩溃

**[00:01:38]** tokens so that the model doesn't crash
> token，这样模型就不会崩溃

**[00:01:38]** tokens so that the model doesn't crash we just um truncate it and pass only the
> token，这样模型就不会崩溃，我们只需截断并只传递

**[00:01:38]** we just um truncate it and pass only the
> 我们只需截断并只传递

**[00:01:38]** we just um truncate it and pass only the first 1,24
> 我们只需截断并只传递前1,024个

**[00:01:38]** first 1,24
> 前1,024个

**[00:01:38]** first 1,24 tokens um it's just like a sanity check
> 前1,024个token，这只是一个合理性检查

**[00:01:38]** tokens um it's just like a sanity check
> token，这只是一个合理性检查

**[00:01:38]** tokens um it's just like a sanity check and then um let me double check if I
> token，这只是一个合理性检查，然后让我再检查一下

**[00:01:38]** and then um let me double check if I
> 然后让我再检查一下

**[00:01:38]** and then um let me double check if I have 1 2 3 4
> 然后让我再检查一下，我是否有1 2 3 4

**[00:01:38]** have 1 2 3 4
> 我是否有1 2 3 4

**[00:01:38]** have 1 2 3 4 oops 1 2 3 4 that I just have the right
> 我是否有1 2 3 4，哦，1 2 3 4，我是否有正确的

**[00:01:38]** oops 1 2 3 4 that I just have the right
> 哦，1 2 3 4，我是否有正确的

**[00:01:38]** oops 1 2 3 4 that I just have the right spacing because it looked a bit off it's
> 哦，1 2 3 4，我是否有正确的间距，因为看起来有点不对

**[00:01:38]** spacing because it looked a bit off it's
> 间距，因为看起来有点不对

**[00:01:38]** spacing because it looked a bit off it's just a python convention and then we
> 间距，因为看起来有点不对，这只是Python的惯例，然后我们

**[00:01:38]** just a python convention and then we
> 这只是Python的惯例，然后我们

**[00:01:38]** just a python convention and then we compute the logits of the
> 这只是Python的惯例，然后我们计算模型的logits

**[00:01:38]** compute the logits of the
> 计算模型的logits

**[00:01:38]** compute the logits of the model uh of the sorry of the
> 计算模型的logits，呃，抱歉，是输入的

**[00:01:38]** model uh of the sorry of the
> 模型，呃，抱歉，是输入的

**[00:01:38]** model uh of the sorry of the inputs so this is corresponding to this
> 模型，呃，抱歉，是输入的logits，所以这对应于此

**[00:01:38]** inputs so this is corresponding to this
> logits，所以这对应于此

**[00:01:38]** inputs so this is corresponding to this one
> logits，所以这对应于此

**[00:01:38]** one
> 此

**[00:01:38]** one here in py toch though one more thing I
> 此，在PyTorch中，还有一件事

**[00:01:38]** here in py toch though one more thing I
> 在PyTorch中，还有一件事

**[00:01:38]** here in py toch though one more thing I should add um is
> 在PyTorch中，还有一件事我应该补充

**[00:01:38]** should add um is
> 我应该补充

**[00:01:38]** should add um is the no grat context so in pytorch by
> 我应该补充，就是no_grad上下文，因为在PyTorch中默认情况下

**[00:01:38]** the no grat context so in pytorch by
> 就是no_grad上下文，因为在PyTorch中默认情况下

**[00:01:38]** the no grat context so in pytorch by default
> 就是no_grad上下文，因为在PyTorch中默认情况下

**[00:01:38]** default
> 默认情况下

**[00:01:38]** default it is always building this computation
> 默认情况下，它总是在后台构建这个计算图

**[00:01:38]** it is always building this computation
> 它总是在后台构建这个计算图

**[00:01:38]** it is always building this computation graph of the model in the background
> 它总是在后台构建这个计算图，用于反向传播算法

**[00:01:38]** graph of the model in the background
> 计算图，用于反向传播算法

**[00:01:38]** graph of the model in the background that is used for the back propagation
> 计算图，用于反向传播算法，即训练算法

**[00:01:39]** that is used for the back propagation
> 即训练算法

**[00:01:39]** that is used for the back propagation algorithm the training algorithm um and
> 即训练算法，如果我们不训练模型，这非常

**[00:01:39]** algorithm the training algorithm um and
> 如果我们不训练模型，这非常

**[00:01:39]** algorithm the training algorithm um and if we don't train the model this is very
> 如果我们不训练模型，这非常低效，所以这里我抑制了

**[00:01:39]** if we don't train the model this is very
> 低效，所以这里我抑制了

**[00:01:39]** if we don't train the model this is very inefficient so here I'm suppressing the
> 低效，所以这里我抑制了梯度计算的生成

**[00:01:39]** inefficient so here I'm suppressing the
> 梯度计算的生成

**[00:01:39]** inefficient so here I'm suppressing the generation of this yeah gradient
> 梯度计算的生成，即计算图

**[00:01:39]** generation of this yeah gradient
> 即计算图

**[00:01:39]** generation of this yeah gradient computation the computation graph
> 即计算图，因为我们不是在这里训练

**[00:01:39]** computation the computation graph
> 因为我们不是在这里训练

**[00:01:39]** computation the computation graph because we are not training here we're
> 因为我们不是在这里训练，我们只是使用模型，所以这是一个很好的实践

**[00:01:39]** because we are not training here we're
> 我们只是使用模型，所以这是一个很好的实践

**[00:01:39]** because we are not training here we're just using the model so it's just a back
> 我们只是使用模型，所以这是一个很好的实践，可以节省内存

**[00:01:39]** just using the model so it's just a back
> 可以节省内存

**[00:01:39]** just using the model so it's just a back good practice to save uh memory so it's
> 可以节省内存，所以计算强度更低

**[00:01:39]** good practice to save uh memory so it's
> 所以计算强度更低

**[00:01:39]** good practice to save uh memory so it's less compute intensive
> 所以计算强度更低，然后我们正在做的是

**[00:01:39]** less compute intensive
> less compute intensive

**[00:01:39]** less compute intensive and then what we are doing is we are um
> less compute intensive and then what we are doing is we are um

**[00:01:39]** and then what we are doing is we are um
> 然后我们正在做的是，嗯

**[00:01:39]** and then what we are doing is we are um grapping the first oh sorry um the last
> 然后我们正在做的是，嗯，获取第一个，哦抱歉，嗯，最后一个

**[00:01:39]** grapping the first oh sorry um the last
> 获取第一个，哦抱歉，嗯，最后一个

**[00:01:39]** grapping the first oh sorry um the last row so here what I'm doing is I'm just
> 获取第一个，哦抱歉，嗯，最后一行，所以这里我做的就是

**[00:01:39]** row so here what I'm doing is I'm just
> 行，所以这里我做的就是

**[00:01:39]** row so here what I'm doing is I'm just yeah getting the last row of the
> 行，所以这里我做的就是，是的，获取logits的最后一行

**[00:01:39]** logits uh let me see here this is what
> 嗯，让我看看，这里是

**[00:01:39]** logits uh let me see here this is what
> 嗯，让我看看，这里是

**[00:01:39]** logits uh let me see here this is what we're doing we're extracting the last
> 嗯，让我看看，这里是我们正在做的，我们提取最后一个

**[00:01:39]** we're doing we're extracting the last
> 我们正在做的，我们提取最后一个

**[00:01:39]** we're doing we're extracting the last row okay um let's just it like this I
> 我们正在做的，我们提取最后一行，好的，嗯，就这样吧

**[00:01:39]** row okay um let's just it like this I
> 行，好的，嗯，就这样吧

**[00:01:39]** row okay um let's just it like this I not uh um next um what we are going to
> 行，好的，嗯，就这样吧，我，不是，嗯，接下来，嗯，我们要做的是

**[00:01:39]** not uh um next um what we are going to
> 不是，嗯，接下来，嗯，我们要做的是

**[00:01:39]** not uh um next um what we are going to do is now we are Computing the
> 不是，嗯，接下来，嗯，我们要做的是计算概率

**[00:01:39]** do is now we are Computing the
> 做的是计算概率

**[00:01:39]** do is now we are Computing the probabilities so we use torch. softmax
> 做的是计算概率，所以我们使用torch. softmax

**[00:01:40]** probabilities so we use torch. softmax
> 概率，所以我们使用torch. softmax

**[00:01:40]** probabilities so we use torch. softmax similar to what we' have done with the
> 概率，所以我们使用torch. softmax，类似于我们之前对multi-ad所做的

**[00:01:40]** similar to what we' have done with the
> 类似于我们之前对multi-ad所做的

**[00:01:40]** similar to what we' have done with the multi-ad
> 类似于我们之前对multi-ad所做的

**[00:01:40]** multi-ad
> multi-ad

**[00:01:40]** multi-ad tension um yeah so we'll do that for the
> multi-ad注意力，嗯，是的，所以我们将对最后一个维度这样做

**[00:01:40]** tension um yeah so we'll do that for the
> 注意力，嗯，是的，所以我们将对最后一个维度这样做

**[00:01:40]** tension um yeah so we'll do that for the last Dimension and then
> 注意力，嗯，是的，所以我们将对最后一个维度这样做，然后

**[00:01:40]** last Dimension and then
> 最后一个维度，然后

**[00:01:40]** last Dimension and then we look up so here maybe one more thing
> 最后一个维度，然后我们查找，所以这里可能还有一件事

**[00:01:40]** we look up so here maybe one more thing
> 我们查找，所以这里可能还有一件事

**[00:01:40]** we look up so here maybe one more thing to explain the
> 我们查找，所以这里可能还有一件事要解释

**[00:01:40]** to explain the
> 要解释

**[00:01:40]** to explain the arcx um we are using the arcm ARX
> 要解释arcx，嗯，我们使用的是arcm ARX

**[00:01:40]** arcx um we are using the arcm ARX
> arcx，嗯，我们使用的是arcm ARX

**[00:01:40]** arcx um we are using the arcm ARX function in pytorch so arcx is the let
> arcx，嗯，我们使用的是PyTorch中的arcx函数，所以arcx是，让我

**[00:01:40]** function in pytorch so arcx is the let
> PyTorch中的函数，所以arcx是，让我

**[00:01:40]** function in pytorch so arcx is the let me just move that down here the arcx
> PyTorch中的函数，所以arcx是，让我把它移到这里，arcx

**[00:01:40]** me just move that down here the arcx
> 让我把它移到这里，arcx

**[00:01:40]** me just move that down here the arcx function we are going to use next is a
> 让我把它移到这里，arcx函数，我们接下来要使用的是

**[00:01:40]** function we are going to use next is a
> 函数，我们接下来要使用的是

**[00:01:40]** function we are going to use next is a function in pyo that looks up the index
> 函数，我们接下来要使用的是pyo中的一个函数，它查找索引位置

**[00:01:40]** function in pyo that looks up the index
> pyo中的一个函数，它查找索引位置

**[00:01:40]** function in pyo that looks up the index position so let me make a
> pyo中的一个函数，它查找索引位置，所以让我做一个

**[00:01:40]** position so let me make a
> 位置，所以让我做一个

**[00:01:40]** position so let me make a arbitrary example
> 位置，所以让我做一个任意的例子

**[00:01:40]** arbitrary example
> 任意的例子

**[00:01:40]** arbitrary example tensor let's do something like this and
> 任意的例子tensor，让我们做类似这样的事情，然后

**[00:01:40]** tensor let's do something like this and
> tensor，让我们做类似这样的事情，然后

**[00:01:40]** tensor let's do something like this and so what arcx is doing it's finding the
> tensor，让我们做类似这样的事情，所以arcx的作用是找到

**[00:01:40]** so what arcx is doing it's finding the
> 所以arcx的作用是找到

**[00:01:40]** so what arcx is doing it's finding the index position with the highest value
> 所以arcx的作用是找到具有最高值的索引位置

**[00:01:40]** index position with the highest value
> 具有最高值的索引位置

**[00:01:40]** index position with the highest value here the highest value is the first
> 具有最高值的索引位置，这里最高值是第一个

**[00:01:40]** here the highest value is the first
> 值，这里最高值是第一个

**[00:01:40]** here the highest value is the first value so it should return a zero now if
> 值，这里最高值是第一个值，所以它应该返回0，现在如果

**[00:01:40]** value so it should return a zero now if
> 值，所以它应该返回0，现在如果

**[00:01:40]** value so it should return a zero now if I put this 14 or let me just put a
> 值，所以它应该返回0，现在如果我把这个14，或者让我放一个

**[00:01:40]** I put this 14 or let me just put a
> 我把这个14，或者让我放一个

**[00:01:40]** I put this 14 or let me just put a larger one here it should return 0 1 2 3
> 我把这个14，或者让我放一个更大的在这里，它应该返回0 1 2 3

**[00:01:40]** larger one here it should return 0 1 2 3
> 更大的在这里，它应该返回0 1 2 3

**[00:01:40]** larger one here it should return 0 1 2 3 4 four so it is returning the index
> 更大的在这里，它应该返回0 1 2 3 4，所以它返回的是索引

**[00:01:41]** 4 four so it is returning the index
> 4，所以它返回的是索引

**[00:01:41]** 4 four so it is returning the index position of the highest value
> 4，所以它返回的是最高值的索引位置

**[00:01:41]** position of the highest value
> 最高值的索引位置

**[00:01:41]** position of the highest value in the Probus here it's what we're
> 最高值的索引位置，在概率中，这里是我们正在做的

**[00:01:41]** in the Probus here it's what we're
> 在概率中，这里是我们正在做的

**[00:01:41]** in the Probus here it's what we're doing and let's call that the next index
> 在概率中，这里是我们正在做的，让我们称之为下一个索引

**[00:01:41]** doing and let's call that the next index
> 做的，让我们称之为下一个索引

**[00:01:41]** doing and let's call that the next index of what we want to
> 做的，让我们称之为我们想要生成的下一个索引

**[00:01:41]** of what we want to
> 我们想要生成的

**[00:01:41]** of what we want to generate so here we are going over the
> 我们想要生成的，所以这里我们遍历

**[00:01:41]** generate so here we are going over the
> 生成，所以这里我们遍历

**[00:01:41]** generate so here we are going over the last Dimension because um the First
> 生成，所以这里我们遍历最后一个维度，因为，嗯，第一个

**[00:01:41]** last Dimension because um the First
> 最后一个维度，因为，嗯，第一个

**[00:01:41]** last Dimension because um the First Dimension is the batch mention so we
> 最后一个维度，因为，嗯，第一个维度是batch维度，所以我们

**[00:01:41]** Dimension is the batch mention so we
> 维度是batch维度，所以我们

**[00:01:41]** Dimension is the batch mention so we don't want to get the largest um we
> 维度是batch维度，所以我们不想获取最大的，嗯，我们

**[00:01:41]** don't want to get the largest um we
> 不想获取最大的，嗯，我们

**[00:01:41]** don't want to get the largest um we don't want to mix up different batches
> 不想获取最大的，嗯，我们不想混淆不同的batches

**[00:01:41]** don't want to mix up different batches
> 不想混淆不同的batches

**[00:01:41]** don't want to mix up different batches we want to have this for each so here we
> 不想混淆不同的batches，我们希望每个都有，所以这里我们

**[00:01:41]** we want to have this for each so here we
> 希望每个都有，所以这里我们

**[00:01:41]** we want to have this for each so here we are only looking at one batch but we
> 我们希望每个批次都这样处理，这里我们只看一个批次，但

**[00:01:41]** are only looking at one batch but we
> 我们只看一个批次，但

**[00:01:41]** are only looking at one batch but we have maybe two rows if we have a batch
> 我们只看一个批次，但如果有批次大小为2，我们可能有两行

**[00:01:41]** have maybe two rows if we have a batch
> 如果有批次大小为2，我们可能有两行

**[00:01:41]** have maybe two rows if we have a batch size of two so we don't want to mix
> 如果有批次大小为2，我们可能有两行，所以我们不想混合

**[00:01:41]** size of two so we don't want to mix
> 大小为2，所以我们不想混合

**[00:01:41]** size of two so we don't want to mix between batches we want to keep them
> 大小为2，所以我们不想混合不同批次，我们希望保持它们

**[00:01:41]** between batches we want to keep them
> 不同批次，我们希望保持它们

**[00:01:41]** between batches we want to keep them independent so we are looking at the
> 不同批次，我们希望保持它们独立，所以我们看的是

**[00:01:41]** independent so we are looking at the
> 独立，所以我们看的是

**[00:01:41]** independent so we are looking at the last
> 独立，所以我们看的是最后一个

**[00:01:41]** last
> 最后一个

**[00:01:41]** last Dimension um and let's also
> 最后一个维度，嗯，让我们也

**[00:01:41]** Dimension um and let's also
> 维度，嗯，让我们也

**[00:01:41]** Dimension um and let's also do keep them true so that the dimension
> 维度，嗯，让我们也设置keepdim=True，这样维度

**[00:01:41]** do keep them true so that the dimension
> 设置keepdim=True，这样维度

**[00:01:41]** do keep them true so that the dimension matches the one that we have as the
> 设置keepdim=True，这样维度与我们输入的维度匹配

**[00:01:41]** matches the one that we have as the
> 与我们输入的维度匹配

**[00:01:41]** matches the one that we have as the input and then we can nicely I hope
> 与我们输入的维度匹配，然后我们可以很好地，我希望

**[00:01:41]** input and then we can nicely I hope
> 输入，然后我们可以很好地，我希望

**[00:01:41]** input and then we can nicely I hope concatenate let's do index
> 输入，然后我们可以很好地，我希望拼接起来，让我们做index

**[00:01:41]** concatenate let's do index
> 拼接起来，让我们做index

**[00:01:41]** concatenate let's do index index next
> 拼接起来，让我们做index，index next

**[00:01:42]** index next
> index next

**[00:01:42]** index next and yeah the dimensions is always
> index next，是的，维度总是

**[00:01:42]** and yeah the dimensions is always
> 是的，维度总是

**[00:01:42]** and yeah the dimensions is always something you have to keep in mind to
> 是的，维度总是你需要牢记的东西，以确保

**[00:01:42]** something you have to keep in mind to
> 你需要牢记的东西，以确保

**[00:01:42]** something you have to keep in mind to make sure everything stays the same and
> 你需要牢记的东西，以确保一切保持一致，并且

**[00:01:42]** make sure everything stays the same and
> 一切保持一致，并且

**[00:01:42]** make sure everything stays the same and if I didn't make a mistake here this
> 一切保持一致，并且如果我没有犯错误，这里

**[00:01:42]** if I didn't make a mistake here this
> 如果我没有犯错误，这里

**[00:01:42]** if I didn't make a mistake here this should be adding the next generated
> 如果我没有犯错误，这里应该会添加下一个生成的

**[00:01:42]** should be adding the next generated
> 应该会添加下一个生成的

**[00:01:42]** should be adding the next generated token which you see here back to the
> 应该会添加下一个生成的token，你在这里看到它被添加回

**[00:01:42]** token which you see here back to the
> token，你在这里看到它被添加回

**[00:01:42]** token which you see here back to the input here basically we are we're not
> token，你在这里看到它被添加回输入，基本上我们并没有

**[00:01:42]** input here basically we are we're not
> 输入，基本上我们并没有

**[00:01:42]** input here basically we are we're not converting back to text here that is
> 输入，基本上我们并没有在这里转换回文本，那是

**[00:01:42]** converting back to text here that is
> 在这里转换回文本，那是

**[00:01:42]** converting back to text here that is what we will do later with a tokenizer
> 在这里转换回文本，那是我们稍后会用tokenizer做的事情

**[00:01:42]** what we will do later with a tokenizer
> 我们稍后会用tokenizer做的事情

**[00:01:42]** what we will do later with a tokenizer we are here operating in this token um
> 我们稍后会用tokenizer做的事情，我们在这里操作的是这个token的

**[00:01:42]** we are here operating in this token um
> 我们在这里操作的是这个token的

**[00:01:42]** we are here operating in this token um ID
> 我们在这里操作的是这个token的ID

**[00:01:42]** ID
> ID

**[00:01:42]** ID Dimension okay so let's hope I didn't
> ID维度，好的，希望我没有

**[00:01:42]** Dimension okay so let's hope I didn't
> 维度，好的，希望我没有

**[00:01:42]** Dimension okay so let's hope I didn't make a mistake and let's try this out in
> 维度，好的，希望我没有犯错误，让我们在实践中试试看

**[00:01:42]** make a mistake and let's try this out in
> 犯错误，让我们在实践中试试看

**[00:01:42]** make a mistake and let's try this out in practice so for that um we are going to
> 犯错误，让我们在实践中试试看，为此，嗯，我们将

**[00:01:42]** practice so for that um we are going to
> 为此，嗯，我们将

**[00:01:42]** practice so for that um we are going to use this text and then we uh will
> 为此，嗯，我们将使用这段文本，然后我们嗯会

**[00:01:42]** use this text and then we uh will
> 使用这段文本，然后我们嗯会

**[00:01:42]** use this text and then we uh will generate the output we'll use our
> 使用这段文本，然后我们嗯会生成输出，我们将使用我们的

**[00:01:42]** generate the output we'll use our
> 生成输出，我们将使用我们的

**[00:01:42]** generate the output we'll use our generate text simple and let me just
> 生成输出，我们将使用我们的generate_text_simple函数，让我

**[00:01:42]** generate text simple and let me just
> generate_text_simple函数，让我

**[00:01:42]** generate text simple and let me just copy and paste here what we have it's
> generate_text_simple函数，让我复制粘贴这里已有的内容，这样

**[00:01:42]** copy and paste here what we have it's
> 复制粘贴这里已有的内容，这样

**[00:01:42]** copy and paste here what we have it's maybe easier this
> 复制粘贴这里已有的内容，这样可能更容易

**[00:01:42]** maybe easier this
> 可能更容易

**[00:01:42]** maybe easier this way and then
> 这样可能更容易，然后

**[00:01:42]** we assign the model here the index is
> 我们在这里分配模型，index是

**[00:01:43]** we assign the model here the index is
> 我们在这里分配模型，index是

**[00:01:43]** we assign the model here the index is our um encoded
> 我们在这里分配模型，index是我们的嗯编码后的

**[00:01:43]** our um encoded
> 我们的嗯编码后的

**[00:01:43]** our um encoded tensor Max new tokens how many do we
> 我们的嗯编码后的tensor，Max new tokens，我们想要多少个

**[00:01:43]** tensor Max new tokens how many do we
> tensor，Max new tokens，我们想要多少个

**[00:01:43]** tensor Max new tokens how many do we want you can experiment with this let's
> tensor，Max new tokens，我们想要多少个，你可以实验一下，让我们

**[00:01:43]** want you can experiment with this let's
> 想要多少个，你可以实验一下，让我们

**[00:01:43]** want you can experiment with this let's do six and the context size um yeah
> 想要多少个，你可以实验一下，让我们设为6，context size嗯是的

**[00:01:43]** do six and the context size um yeah
> 设为6，context size嗯是的

**[00:01:43]** do six and the context size um yeah let's just reuse this here this should
> 设为6，context size嗯是的，让我们直接复用这里的，这应该是

**[00:01:43]** be context length
> context length

**[00:01:43]** and hopefully this works if I haven't
> 希望这能工作，如果我没有

**[00:01:43]** and hopefully this works if I haven't
> 希望这能工作，如果我没有

**[00:01:43]** and hopefully this works if I haven't made a mistake I did of
> 希望这能工作，如果我没有犯错误，我当然犯了

**[00:01:43]** made a mistake I did of
> 犯错误，我当然犯了

**[00:01:43]** made a mistake I did of course oh I think I forgot a colon of
> 犯错误，我当然犯了，哦，我想我忘了加冒号了

**[00:01:43]** course oh I think I forgot a colon of
> 课程，哦，我想我忘了加冒号

**[00:01:43]** course oh I think I forgot a colon of course so this would not work yeah so I
> 课程，哦，我想我忘了加冒号，当然，这样不行，是的，所以我

**[00:01:43]** course so this would not work yeah so I
> 课程，这样不行，是的，所以我

**[00:01:43]** course so this would not work yeah so I need a colon
> 课程，这样不行，是的，所以我需要加冒号

**[00:01:43]** here keep
> 这里保留

**[00:01:43]** here keep
> 这里保留

**[00:01:43]** here keep them without the
> 这里保留它们，不加

**[00:01:43]** underscore there we go now we have
> 下划线，好了，现在我们有了

**[00:01:43]** underscore there we go now we have
> 下划线，好了，现在我们有了

**[00:01:43]** underscore there we go now we have something in let's take a look of how
> 下划线，好了，现在我们有了内容，让我们看看它是如何

**[00:01:43]** something in let's take a look of how
> 内容，让我们看看它是如何

**[00:01:43]** something in let's take a look of how that looks like so we have now token IDs
> 内容，让我们看看它看起来如何，现在我们有了token IDs

**[00:01:43]** that looks like so we have now token IDs
> 看起来如何，现在我们有了token IDs

**[00:01:43]** that looks like so we have now token IDs now the last thing of course is to use
> 看起来如何，现在我们有了token IDs，当然，最后一步是使用

**[00:01:44]** now the last thing of course is to use
> 当然，最后一步是使用

**[00:01:44]** now the last thing of course is to use our decode method to convert token IDs
> 当然，最后一步是使用我们的decode方法将token IDs转换

**[00:01:44]** our decode method to convert token IDs
> 我们的decode方法将token IDs转换

**[00:01:44]** our decode method to convert token IDs back
> 我们的decode方法将token IDs转换回

**[00:01:44]** back
> 回

**[00:01:44]** back into words you can see this does not
> 回单词，你可以看到这并不

**[00:01:44]** into words you can see this does not
> 单词，你可以看到这并不

**[00:01:44]** into words you can see this does not work the reason is this is a tensor and
> 单词，你可以看到这不起作用，原因是这是一个tensor

**[00:01:44]** work the reason is this is a tensor and
> 不起作用，原因是这是一个tensor

**[00:01:44]** work the reason is this is a tensor and our tokenizer work works with python
> 不起作用，原因是这是一个tensor，而我们的tokenizer适用于Python

**[00:01:44]** our tokenizer work works with python
> 我们的tokenizer适用于Python

**[00:01:44]** our tokenizer work works with python lists so we have to convert this back
> 我们的tokenizer适用于Python列表，所以我们必须将其转换回

**[00:01:44]** lists so we have to convert this back
> 列表，所以我们必须将其转换回

**[00:01:44]** lists so we have to convert this back into uh python list so first we can do
> 列表，所以我们必须将其转换回Python列表，首先我们可以

**[00:01:44]** into uh python list so first we can do
> Python列表，首先我们可以

**[00:01:44]** into uh python list so first we can do maybe a squeeze
> Python列表，首先我们可以做一个squeeze

**[00:01:44]** maybe a squeeze
> 一个squeeze

**[00:01:44]** maybe a squeeze to get rid of one
> 一个squeeze来去掉一个

**[00:01:44]** to get rid of one
> 去掉一个

**[00:01:44]** to get rid of one dimension so this got rid of one of the
> 去掉一个维度，这样就去掉了其中一个

**[00:01:44]** dimension so this got rid of one of the
> 维度，这样就去掉了其中一个

**[00:01:44]** dimension so this got rid of one of the dimensions and then we can maybe do a
> 维度，这样就去掉了其中一个维度，然后我们可以做一个

**[00:01:44]** dimensions and then we can maybe do a
> 维度，然后我们可以做一个

**[00:01:44]** dimensions and then we can maybe do a two list I always forget if we need an
> 维度，然后我们可以做一个to list，我总忘记是否需要加

**[00:01:44]** two list I always forget if we need an
> to list，我总忘记是否需要加

**[00:01:44]** two list I always forget if we need an underscore or not let's just
> to list，我总忘记是否需要加下划线，我们直接

**[00:01:44]** underscore or not let's just
> 下划线，我们直接

**[00:01:44]** underscore or not let's just see yeah this worked so we have a list
> 下划线，我们直接看看，是的，这成功了，所以我们有了一个列表

**[00:01:44]** see yeah this worked so we have a list
> 看看，是的，这成功了，所以我们有了一个列表

**[00:01:44]** see yeah this worked so we have a list now and we can see what happens here so
> 看看，是的，这成功了，所以我们有了一个列表，现在我们可以看看这里发生了什么

**[00:01:44]** now and we can see what happens here so
> 现在我们可以看看这里发生了什么

**[00:01:44]** now and we can see what happens here so we can see it generated something but
> 现在我们可以看看这里发生了什么，我们可以看到它生成了某些东西，但

**[00:01:44]** we can see it generated something but
> 我们可以看到它生成了某些东西，但

**[00:01:44]** we can see it generated something but this does not look like what we want to
> 我们可以看到它生成了某些东西，但这看起来完全不像我们想要

**[00:01:44]** this does not look like what we want to
> 这看起来完全不像我们想要

**[00:01:44]** this does not look like what we want to generate at all you can see this of
> 这看起来完全不像我们想要生成的内容，你可以看到这

**[00:01:44]** generate at all you can see this of
> 生成的内容，你可以看到这

**[00:01:44]** generate at all you can see this of course is our in input context this
> 生成的内容，你可以看到这当然是我们输入的上下文，这

**[00:01:44]** course is our in input context this
> 当然是我们输入的上下文，这

**[00:01:44]** course is our in input context this looks correct but this for example does
> 当然是我们输入的上下文，这看起来正确，但例如这

**[00:01:44]** looks correct but this for example does
> 看起来正确，但例如这

**[00:01:44]** looks correct but this for example does not look like at all what we have seen
> 看起来正确，但例如这完全不像我们之前看到的

**[00:01:45]** not look like at all what we have seen
> 完全不像我们之前看到的

**[00:01:45]** not look like at all what we have seen here it doesn't say hello I'm a model
> 完全不像我们之前看到的，它没有说“你好，我是一个模型”

**[00:01:45]** here it doesn't say hello I'm a model
> 它没有说“你好，我是一个模型”

**[00:01:45]** here it doesn't say hello I'm a model ready to help it is more like
> 它没有说“你好，我是一个模型，随时准备提供帮助”，它更像是

**[00:01:45]** ready to help it is more like
> 随时准备提供帮助”，它更像是

**[00:01:45]** ready to help it is more like gibberish and the reason is we have
> 随时准备提供帮助”，它更像是胡言乱语，原因是

**[00:01:45]** gibberish and the reason is we have
> 胡言乱语，原因是

**[00:01:45]** gibberish and the reason is we have initialized our model with random
> 胡言乱语，原因是我们用随机权重初始化了模型

**[00:01:45]** initialized our model with random
> 用随机权重初始化了模型

**[00:01:45]** initialized our model with random weights and we have not trained the
> 用随机权重初始化了模型，并且我们还没有训练

**[00:01:45]** weights and we have not trained the
> 并且我们还没有训练

**[00:01:45]** weights and we have not trained the model yet so the model training will be
> 并且我们还没有训练模型，所以模型训练将是

**[00:01:45]** model yet so the model training will be
> 模型，所以模型训练将是

**[00:01:45]** model yet so the model training will be actually the topic of chapter 5 and once
> 模型，所以模型训练实际上是第5章的主题，一旦

**[00:01:45]** actually the topic of chapter 5 and once
> 实际上是第5章的主题，一旦

**[00:01:45]** actually the topic of chapter 5 and once we have completed chapter 5 hopefully
> 实际上是第5章的主题，一旦我们完成了第5章，希望

**[00:01:45]** we have completed chapter 5 hopefully
> 我们完成了第5章，希望

**[00:01:45]** we have completed chapter 5 hopefully this will actually generate useful text
> 我们完成了第5章，希望这能实际生成有用的文本

**[00:01:45]** this will actually generate useful text
> 这能实际生成有用的文本

**[00:01:45]** this will actually generate useful text but yeah again we have this generate
> 这能实际生成有用的文本，但是的，我们又有这个generate

**[00:01:45]** but yeah again we have this generate
> 但是的，我们又有这个generate

**[00:01:45]** but yeah again we have this generate text method and we can actually reuse
> 但是的，我们又有这个generate text方法，我们实际上可以在训练后重用

**[00:01:45]** text method and we can actually reuse
> text方法，我们实际上可以在训练后重用

**[00:01:45]** text method and we can actually reuse that after the training to see that this
> text方法，我们实际上可以在训练后重用，以验证这一点

**[00:01:45]** that after the training to see that this
> 在训练之后，我们会发现这

**[00:01:45]** that after the training to see that this actually will help us generate um
> 在训练之后，我们会发现这实际上能帮助我们生成

**[00:01:45]** actually will help us generate um
> 实际上能帮助我们生成

**[00:01:45]** actually will help us generate um coherent text but yeah stay tuned for
> 实际上能帮助我们生成连贯的文本，但请继续关注

**[00:01:45]** coherent text but yeah stay tuned for
> 连贯的文本，但请继续关注

**[00:01:45]** coherent text but yeah stay tuned for the next chapter on training the model
> 连贯的文本，但请继续关注下一章关于模型训练的内容

---

## 🎯 关键要点

- [ ] 要点 1
- [ ] 要点 2
- [ ] 要点 3
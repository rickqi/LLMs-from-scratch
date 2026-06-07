# 第 3 章：PyTorch Buffers 深入理解（视频笔记）

> 🎬 [原视频](https://www.youtube.com/watch?v=PetlIokI9Ao)
> 📅 中英双语字幕（DeepSeek 翻译）

---

## 中英双语字幕

**[00:00:00]** yeah hi everyone so today I want to talk
> 大家好，今天我想聊聊

**[00:00:00]** yeah hi everyone so today I want to talk
> 大家好，今天我想聊聊

**[00:00:00]** yeah hi everyone so today I want to talk about pytorch buffers what pytorch
> 大家好，今天我想聊聊PyTorch buffers，什么是PyTorch

**[00:00:00]** about pytorch buffers what pytorch
> 什么是PyTorch buffers

**[00:00:00]** about pytorch buffers what pytorch buffers are when we use them and how
> 什么是PyTorch buffers，什么时候使用它们，以及它们

**[00:00:00]** buffers are when we use them and how
> 什么时候使用它们，以及它们

**[00:00:00]** buffers are when we use them and how they are actually useful and the reason
> 什么时候使用它们，以及它们到底有什么用处。我之所以想聊这个，是因为我觉得

**[00:00:00]** they are actually useful and the reason
> 到底有什么用处。我之所以想聊这个，是因为我觉得

**[00:00:00]** they are actually useful and the reason I want to talk about it is it's I think
> 到底有什么用处。我之所以想聊这个，是因为我觉得这是一个非常重要的概念，如果你

**[00:00:00]** I want to talk about it is it's I think
> 这是一个非常重要的概念，如果你

**[00:00:00]** I want to talk about it is it's I think a very important concept if you
> 这是一个非常重要的概念，如果你在PyTorch中实现更大的模型，比如

**[00:00:00]** a very important concept if you
> 在PyTorch中实现更大的模型，比如

**[00:00:00]** a very important concept if you implement larger models in pytorch for
> 在PyTorch中实现更大的模型，比如从头实现一个大型

**[00:00:00]** implement larger models in pytorch for
> 从头实现一个大型

**[00:00:00]** implement larger models in pytorch for example if you implement a large
> 从头实现一个大型语言模型，你可能会

**[00:00:00]** example if you implement a large
> 语言模型，你可能会

**[00:00:00]** example if you implement a large language model from scratch you might
> 语言模型，你可能会遇到一个叫PyTorch buffer的概念

**[00:00:00]** language model from scratch you might
> 遇到一个叫PyTorch buffer的概念

**[00:00:00]** language model from scratch you might encounter a concept called a pyo buffer
> 遇到一个叫PyTorch buffer的概念。在代码中，它大致看起来像

**[00:00:00]** encounter a concept called a pyo buffer
> 在代码中，它大致看起来像

**[00:00:00]** encounter a concept called a pyo buffer so in code this would look roughly like
> 在代码中，它大致看起来像这样，有一行代码是

**[00:00:00]** so in code this would look roughly like
> 这样，有一行代码是

**[00:00:00]** so in code this would look roughly like this where you have a line that is like
> 这样，有一行代码是self.register_buffer，在一个PyTorch module中

**[00:00:00]** this where you have a line that is like
> self.register_buffer，在一个PyTorch module中

**[00:00:00]** this where you have a line that is like self uh register buffer in a pyo module
> self.register_buffer，在一个PyTorch module中。这实际上是一个非常有用

**[00:00:00]** self uh register buffer in a pyo module
> 这实际上是一个非常有用

**[00:00:00]** self uh register buffer in a pyo module and this is actually quite useful
> 这实际上是一个非常有用的概念，你马上就会看到。所以这是一个

**[00:00:00]** and this is actually quite useful
> 的概念，你马上就会看到。所以这是一个

**[00:00:00]** and this is actually quite useful concept as you will see so this is a
> 的概念，你马上就会看到。所以这是一个比较随性的视频，但我会通过

**[00:00:00]** concept as you will see so this is a
> 比较随性的视频，但我会通过

**[00:00:00]** concept as you will see so this is a pretty spontaneous video but I will show
> 比较随性的视频，但我会通过一些动手编码示例来展示

**[00:00:00]** pretty spontaneous video but I will show
> 一些动手编码示例来展示

**[00:00:00]** pretty spontaneous video but I will show you with some Hands-On coding examples
> 一些动手编码示例来展示何时以及如何使用这些buffers，以及

**[00:00:00]** you with some Hands-On coding examples
> 何时以及如何使用这些buffers，以及

**[00:00:00]** you with some Hands-On coding examples when and how to use these buffers and
> 何时以及如何使用这些buffers，以及它们的优势是什么。我会尽量

**[00:00:00]** when and how to use these buffers and
> 它们的优势是什么。我会尽量

**[00:00:00]** when and how to use these buffers and what the advantages are so I will try to
> 它们的优势是什么。我会尽量简短，但也会尽量

**[00:00:00]** what the advantages are so I will try to
> 简短，但也会尽量

**[00:00:00]** what the advantages are so I will try to keep it brief but um I will also try to
> 简短，但也会尽量让它有趣且

**[00:00:00]** keep it brief but um I will also try to
> 让它有趣且

**[00:00:00]** keep it brief but um I will also try to you know make it interesting and
> 让它有趣且相关。为此，我们将使用一个

**[00:00:00]** you know make it interesting and
> 相关。为此，我们将使用一个

**[00:00:00]** you know make it interesting and relevant so for that we will be using a
> 相关。为此，我们将使用一个不带buffers的因果注意力PyTorch

**[00:00:00]** relevant so for that we will be using a
> 不带buffers的因果注意力PyTorch

**[00:00:00]** relevant so for that we will be using a causal attention without buffers pyou
> 不带buffers的因果注意力PyTorch module，这从技术上讲是

**[00:00:00]** causal attention without buffers pyou
> module，这从技术上讲是

**[00:00:00]** causal attention without buffers pyou module that would be technically part of
> module，这从技术上讲是大型语言模型的一部分。所以你不需要

**[00:00:01]** module that would be technically part of
> 大型语言模型的一部分。所以你不需要

**[00:00:01]** module that would be technically part of a large language model so you don't have
> 大型语言模型的一部分。所以你不需要详细了解这里的self-attention机制是如何工作的，唯一

**[00:00:01]** a large language model so you don't have
> 详细了解这里的self-attention机制是如何工作的，唯一

**[00:00:01]** a large language model so you don't have to understand in detail how the self
> 详细了解这里的self-attention机制是如何工作的，唯一重要的方面是我们有一个

**[00:00:01]** to understand in detail how the self
> 重要的方面是我们有一个

**[00:00:01]** to understand in detail how the self attention mechanism here works the only
> 重要的方面是我们有一个PyTorch module，哦，PyTorch module在这里

**[00:00:01]** attention mechanism here works the only
> PyTorch module，哦，PyTorch module在这里

**[00:00:01]** attention mechanism here works the only important aspect is that we have a yeah
> PyTorch module，哦，PyTorch module在这里。然后我们会看到如何，或者说为什么

**[00:00:01]** important aspect is that we have a yeah
> 然后我们会看到如何，或者说为什么

**[00:00:01]** important aspect is that we have a yeah py to module oops uh py to module here
> 然后我们会看到如何，或者说为什么我们会想使用一个buffer。所以再次强调，

**[00:00:01]** py to module oops uh py to module here
> 我们会想使用一个buffer。所以再次强调，

**[00:00:01]** py to module oops uh py to module here and we will then see how we or why we
> 我们会想使用一个buffer。所以再次强调，这里的细节并不是

**[00:00:01]** and we will then see how we or why we
> 这里的细节并不是

**[00:00:01]** and we will then see how we or why we would want to use a buffer so again the
> 这里的细节并不是这个视频的重点，只是为了

**[00:00:01]** would want to use a buffer so again the
> 这个视频的重点，只是为了

**[00:00:01]** would want to use a buffer so again the details here they are not super
> 这个视频的重点，只是为了让我们有一个可以操作的动手代码

**[00:00:01]** details here they are not super
> 让我们有一个可以操作的动手代码

**[00:00:01]** details here they are not super important for this video it's just so
> 让我们有一个可以操作的动手代码。那么，我们如何运行

**[00:00:01]** important for this video it's just so
> 那么，我们如何运行

**[00:00:01]** important for this video it's just so that we have a Hands-On codee that we
> 那么，我们如何运行这段代码呢？通过下面的例子

**[00:00:01]** that we have a Hands-On codee that we
> 这段代码呢？通过下面的例子

**[00:00:01]** that we have a Hands-On codee that we can work with so um how we would run
> 这段代码呢？通过下面的例子。这里同样只是一些代码，

**[00:00:01]** can work with so um how we would run
> 这里同样只是一些代码，

**[00:00:01]** can work with so um how we would run this code is with this following example
> 这里同样只是一些代码，你不需要真正逐行跟进

**[00:00:01]** this code is with this following example
> 你不需要真正逐行跟进

**[00:00:01]** this code is with this following example so here again it's just some code that
> 你不需要真正逐行跟进。真正重要的部分是

**[00:00:01]** so here again it's just some code that
> 真正重要的部分是

**[00:00:01]** so here again it's just some code that you don't have to really follow in
> 真正重要的部分是我们有一些可以操作的代码

**[00:00:01]** you don't have to really follow in
> 我们有一些可以操作的代码

**[00:00:01]** you don't have to really follow in detail the really the important part is
> 我们有一些可以操作的代码

**[00:00:01]** detail the really the important part is
> detail the really the important part is

**[00:00:01]** detail the really the important part is here that we have some code that we can
> detail the really the important part is here that we have some code that we can

**[00:00:01]** here that we have some code that we can
> here that we have some code that we can

**[00:00:01]** here that we have some code that we can run so here but still maybe because it's
> 这里我们有一些可以运行的代码，但可能因为它是

**[00:00:01]** run so here but still maybe because it's
> 运行在这里，但可能因为它是

**[00:00:01]** run so here but still maybe because it's interesting so here we have some inputs
> 运行在这里，但可能因为它很有趣，所以这里我们有一些输入

**[00:00:01]** interesting so here we have some inputs
> 有趣，所以这里我们有一些输入

**[00:00:01]** interesting so here we have some inputs so we have 1 2 3 4 five six inputs that
> 有趣，所以这里我们有一些输入，我们有1 2 3 4 五 六个输入

**[00:00:01]** so we have 1 2 3 4 five six inputs that
> 所以我们有1 2 3 4 五 六个输入

**[00:00:01]** so we have 1 2 3 4 five six inputs that we arrange as a mini batch so like in
> 所以我们有1 2 3 4 五 六个输入，我们将其排列为一个mini batch，就像在

**[00:00:01]** we arrange as a mini batch so like in
> 我们将其排列为一个mini batch，就像在

**[00:00:01]** we arrange as a mini batch so like in regular deep learning this just like a a
> 我们将其排列为一个mini batch，就像在常规深度学习中一样，这就像一个

**[00:00:01]** regular deep learning this just like a a
> 常规深度学习中一样，这就像一个

**[00:00:01]** regular deep learning this just like a a mini batch then we have some
> 常规深度学习中一样，这就像一个mini batch，然后我们有一些

**[00:00:01]** mini batch then we have some
> mini batch，然后我们有一些

**[00:00:01]** mini batch then we have some hyperparameter settings you don't have
> mini batch，然后我们有一些超参数设置，你不需要

**[00:00:02]** hyperparameter settings you don't have
> 超参数设置，你不需要

**[00:00:02]** hyperparameter settings you don't have to worry really about what these are
> 超参数设置，你不需要真正担心这些是什么

**[00:00:02]** to worry really about what these are
> 真正担心这些是什么

**[00:00:02]** to worry really about what these are what's important is that we are
> 真正担心这些是什么，重要的是我们正在

**[00:00:02]** what's important is that we are
> 重要的是我们正在

**[00:00:02]** what's important is that we are instantiating a new causal attention
> 重要的是我们正在实例化一个新的causal attention

**[00:00:02]** instantiating a new causal attention
> 实例化一个新的causal attention

**[00:00:02]** instantiating a new causal attention without buffers object here and so if
> 实例化一个新的没有buffers的causal attention对象，所以如果

**[00:00:02]** without buffers object here and so if
> 没有buffers的对象，所以如果

**[00:00:02]** without buffers object here and so if you were looking for the buffer this one
> 没有buffers的对象，所以如果你在找buffer，这个

**[00:00:02]** you were looking for the buffer this one
> 你在找buffer，这个

**[00:00:02]** you were looking for the buffer this one doesn't have the buffer yet I will
> 你在找buffer，这个还没有buffer，我会

**[00:00:02]** doesn't have the buffer yet I will
> 还没有buffer，我会

**[00:00:02]** doesn't have the buffer yet I will introduce it in a moment I wanted to
> 还没有buffer，我会稍后介绍它。我想先

**[00:00:02]** introduce it in a moment I wanted to
> 稍后介绍它。我想先

**[00:00:02]** introduce it in a moment I wanted to show you first what happens if we don't
> 稍后介绍它。我想先向你展示如果我们不使用buffers会发生什么，然后我希望这

**[00:00:02]** show you first what happens if we don't
> 向你展示如果我们不使用buffers会发生什么，然后我希望这

**[00:00:02]** show you first what happens if we don't use buffers and then I hope this
> 向你展示如果我们不使用buffers会发生什么，然后我希望这能激励使用嗯是的buffers，所以

**[00:00:02]** use buffers and then I hope this
> 能激励使用嗯是的buffers，所以

**[00:00:02]** use buffers and then I hope this motivates the use of um yeah buffers so
> 能激励使用嗯是的buffers，所以现在我们正在实例化嗯或者

**[00:00:02]** motivates the use of um yeah buffers so
> 现在我们正在实例化嗯或者

**[00:00:02]** motivates the use of um yeah buffers so here we are now instantiating um or
> 现在我们正在实例化嗯或者抱歉，我们在

**[00:00:02]** here we are now instantiating um or
> 抱歉，我们在

**[00:00:02]** here we are now instantiating um or sorry here we are using after
> 抱歉，我们在实例化对象之后，我们正在

**[00:00:02]** sorry here we are using after
> 实例化对象之后，我们正在

**[00:00:02]** sorry here we are using after instantiating the object here we are
> 实例化对象之后，我们正在对我们的batch使用这个对象，如果

**[00:00:02]** instantiating the object here we are
> 对我们的batch使用这个对象，如果

**[00:00:02]** instantiating the object here we are using the object on our badge and if
> 对我们的batch使用这个对象，如果一切正确，我们应该得到

**[00:00:02]** using the object on our badge and if
> 一切正确，我们应该得到

**[00:00:02]** using the object on our badge and if everything is correct here we should get
> 一切正确，我们应该得到一些数字，这些数字

**[00:00:02]** everything is correct here we should get
> 一些数字，这些数字

**[00:00:02]** everything is correct here we should get some numbers again what these numbers
> 一些数字，这些数字是什么意思不用担心，这只是

**[00:00:02]** some numbers again what these numbers
> 是什么意思不用担心，这只是

**[00:00:02]** some numbers again what these numbers mean don't worry about it it's just
> 是什么意思不用担心，这只是表明这个attention机制

**[00:00:02]** mean don't worry about it it's just
> 表明这个attention机制

**[00:00:02]** mean don't worry about it it's just showing that this attention mechanism
> 表明这个attention机制可以说在输入上起作用了，所以现在我们

**[00:00:02]** showing that this attention mechanism
> 可以说在输入上起作用了，所以现在我们

**[00:00:02]** showing that this attention mechanism let's say works on the input so now we
> 可以说在输入上起作用了，所以现在我们有一些代码可以处理，通常

**[00:00:02]** let's say works on the input so now we
> 有一些代码可以处理，通常

**[00:00:02]** let's say works on the input so now we have some code to work with and usually
> 有一些代码可以处理，通常你可能知道，当你在深度学习中工作时

**[00:00:02]** have some code to work with and usually
> 你可能知道，当你在深度学习中工作时

**[00:00:02]** have some code to work with and usually as you might know when you work in deep
> 你可能知道，当你在深度学习中工作时，尤其是当你处理

**[00:00:02]** as you might know when you work in deep
> 尤其是当你处理

**[00:00:02]** as you might know when you work in deep learning especially when you work with
> 尤其是当你处理大型语言模型时，你可能希望

**[00:00:02]** learning especially when you work with
> 大型语言模型时，你可能希望

**[00:00:02]** learning especially when you work with large language models you probably want
> 大型语言模型时，你可能希望在GPU上运行东西

**[00:00:02]** large language models you probably want
> 在GPU上运行东西

**[00:00:02]** large language models you probably want to run things on a GPU
> 在GPU上运行东西，所以首先是的，我们想检查这个

**[00:00:03]** to run things on a GPU
> 所以首先是的，我们想检查这个

**[00:00:03]** to run things on a GPU so first yeah we want to check if this
> 所以首先是的，我们想检查这个机器是否有GPU，所以让我们做一些

**[00:00:03]** so first yeah we want to check if this
> 机器是否有GPU，所以让我们做一些

**[00:00:03]** so first yeah we want to check if this machine has a GPU so let's do some
> 机器是否有GPU，所以让我们做一些简单的检查，机器有GPU，让我们

**[00:00:03]** machine has a GPU so let's do some
> 简单的检查，机器有GPU，让我们

**[00:00:03]** machine has a GPU so let's do some simple check so machine has GPU let's
> 简单的检查，机器有GPU，让我们说，我们可以嗯说torch

**[00:00:03]** simple check so machine has GPU let's
> 说，我们可以嗯说torch

**[00:00:03]** simple check so machine has GPU let's say and we can um say torch
> 说，我们可以嗯说torch Cuda

**[00:00:03]** say and we can um say torch
> Cuda

**[00:00:03]** say and we can um say torch Cuda
> Cuda 是

**[00:00:03]** Cuda
> 是

**[00:00:03]** Cuda is
> 是 可用的，所以这应该显示我嗯

**[00:00:03]** is
> 可用的，所以这应该显示我嗯

**[00:00:03]** is available so this should show me yeah
> 可用的，所以这应该显示我嗯，这台机器有GPU，如果你在

**[00:00:03]** available so this should show me yeah
> 这台机器有GPU，如果你在

**[00:00:03]** available so this should show me yeah this machine has a GPU if you're running
> 这台机器有GPU，如果你在没有GPU的机器上运行这个，那么

**[00:00:03]** this machine has a GPU if you're running
> 没有GPU的机器上运行这个，那么

**[00:00:03]** this machine has a GPU if you're running this on a machine without a GPU then
> 没有GPU的机器上运行这个，那么

**[00:00:03]** this on a machine without a GPU then
> this on a machine without a GPU then

**[00:00:03]** this on a machine without a GPU then this buffer might actually not be that
> 在没有GPU的机器上，这个缓冲区可能实际上没那么

**[00:00:03]** this buffer might actually not be that
> 这个缓冲区可能实际上没那么

**[00:00:03]** this buffer might actually not be that useful so this is kind of like a
> 这个缓冲区可能实际上没那么有用，所以这有点像是一个

**[00:00:03]** useful so this is kind of like a
> 有用，所以这有点像是一个

**[00:00:03]** useful so this is kind of like a requirement um for I would say the
> 有用，所以这有点像是一个要求，嗯，对于我来说

**[00:00:03]** requirement um for I would say the
> 要求，嗯，对于我来说

**[00:00:03]** requirement um for I would say the usefulness of this particular video or
> 要求，嗯，对于我来说，这个特定视频或缓冲区解释的有用性

**[00:00:03]** usefulness of this particular video or
> 这个特定视频或缓冲区解释的有用性

**[00:00:03]** usefulness of this particular video or buffer explanation so now if we want to
> 这个特定视频或缓冲区解释的有用性，所以现在如果我们想

**[00:00:03]** buffer explanation so now if we want to
> 缓冲区解释，所以现在如果我们想

**[00:00:03]** buffer explanation so now if we want to run things on the
> 缓冲区解释，所以现在如果我们想在GPU上运行东西

**[00:00:03]** run things on the
> 在GPU上运行东西

**[00:00:03]** run things on the GPU what we usually do is we transfer
> 在GPU上运行东西，我们通常做的是将数据转移到GPU，比如说，哦，batch

**[00:00:03]** GPU what we usually do is we transfer
> 将数据转移到GPU，比如说，哦，batch

**[00:00:03]** GPU what we usually do is we transfer the data to the GPU let's say oops batch
> 将数据转移到GPU，比如说，哦，batch U batch到ca，所以这会是，哦，是的

**[00:00:03]** the data to the GPU let's say oops batch
> U batch到ca，所以这会是，哦，是的

**[00:00:03]** the data to the GPU let's say oops batch U batch to ca so this would be oops Yeah
> U batch到ca，所以这会是，哦，是的，对我来说现场编码有时很难

**[00:00:03]** U batch to ca so this would be oops Yeah
> 对我来说现场编码有时很难

**[00:00:03]** U batch to ca so this would be oops Yeah it's hard to life code for me sometimes
> 对我来说现场编码有时很难，因为我的桌子高度不同

**[00:00:03]** it's hard to life code for me sometimes
> 因为我的桌子高度不同

**[00:00:03]** it's hard to life code for me sometimes because my desk is a different height
> 因为我的桌子高度不同，因为麦克风，然后

**[00:00:03]** because my desk is a different height
> 因为麦克风，然后

**[00:00:03]** because my desk is a different height because of the microphone and then
> 因为麦克风，然后打字变得有点挑战性

**[00:00:04]** because of the microphone and then
> 打字变得有点挑战性

**[00:00:04]** because of the microphone and then typing becomes a bit more challenging
> 打字变得有点挑战性，总之，假设我们将数据转移到GPU

**[00:00:04]** typing becomes a bit more challenging
> 总之，假设我们将数据转移到GPU

**[00:00:04]** typing becomes a bit more challenging anyway so let's say we transfer the data
> 总之，假设我们将数据转移到GPU，然后我们对py模块做同样的事情

**[00:00:04]** anyway so let's say we transfer the data
> 然后我们对py模块做同样的事情

**[00:00:04]** anyway so let's say we transfer the data to the GPU and then we do the same thing
> 然后我们对py模块做同样的事情，所以我们也在这里做to Cuda，注意它是，哦

**[00:00:04]** to the GPU and then we do the same thing
> 所以我们也在这里做to Cuda，注意它是，哦

**[00:00:04]** to the GPU and then we do the same thing for the py module so we
> 所以我们也在这里做to Cuda，注意它是，哦，抱歉，这是一个特定的pych东西

**[00:00:04]** for the py module so we
> 抱歉，这是一个特定的pych东西

**[00:00:04]** for the py module so we do to Cuda here too note that it's oh
> 抱歉，这是一个特定的pych东西，对于数据或t Source，你必须做

**[00:00:04]** do to Cuda here too note that it's oh
> 对于数据或t Source，你必须做

**[00:00:04]** do to Cuda here too note that it's oh sorry it's like a particular pych thing
> 对于数据或t Source，你必须像那样赋值，否则它

**[00:00:04]** sorry it's like a particular pych thing
> 像那样赋值，否则它

**[00:00:04]** sorry it's like a particular pych thing that for data or t Source you have to do
> 像那样赋值，否则它不会被转移，或者换句话说

**[00:00:04]** that for data or t Source you have to do
> 不会被转移，或者换句话说

**[00:00:04]** that for data or t Source you have to do the assignment like that otherwise it
> 不会被转移，或者换句话说，它会被转移，但不会被

**[00:00:04]** the assignment like that otherwise it
> 它会被转移，但不会被

**[00:00:04]** the assignment like that otherwise it won't be transferred or in other words
> 它会被转移，但不会被赋值给任何东西，所以对于模块

**[00:00:04]** won't be transferred or in other words
> 赋值给任何东西，所以对于模块

**[00:00:04]** won't be transferred or in other words it will be transferred but they're not
> 赋值给任何东西，所以对于模块，这是一个模块，我们不需要

**[00:00:04]** it will be transferred but they're not
> 这是一个模块，我们不需要

**[00:00:04]** it will be transferred but they're not assigned to anything so so for modules
> 这是一个模块，我们不需要那样做，所以这里的模块我们只需

**[00:00:04]** assigned to anything so so for modules
> 那样做，所以这里的模块我们只需

**[00:00:04]** assigned to anything so so for modules though so this is a module we don't have
> 那样做，所以这里的模块我们只需像那样转移，它只是

**[00:00:04]** though so this is a module we don't have
> 像那样转移，它只是

**[00:00:04]** though so this is a module we don't have to do that so the module here we just
> 像那样转移，它只是pytorch的一个特定之处

**[00:00:04]** to do that so the module here we just
> pytorch的一个特定之处

**[00:00:04]** to do that so the module here we just transfer it like that it's just
> pytorch的一个特定之处，好的，让我们做

**[00:00:04]** transfer it like that it's just
> 好的，让我们做

**[00:00:04]** transfer it like that it's just something in particular about pytorch
> 好的，让我们做那个，嗯，是的，所以我们现在可以看到

**[00:00:04]** something in particular about pytorch
> 那个，嗯，是的，所以我们现在可以看到

**[00:00:04]** something in particular about pytorch okay so let's do
> 那个，嗯，是的，所以我们现在可以看到我们已经将东西转移到GPU

**[00:00:04]** okay so let's do
> 已经将东西转移到GPU

**[00:00:04]** okay so let's do that and um yeah so we can see now we
> 已经将东西转移到GPU，所以我们现在可以做的，例如，我们

**[00:00:04]** that and um yeah so we can see now we
> 所以我们现在可以做的，例如，我们

**[00:00:04]** that and um yeah so we can see now we have transferred things to the GPU
> 所以我们现在可以做的，例如，我们可以运行这段代码，嗯，是的，也许

**[00:00:04]** have transferred things to the GPU
> 可以运行这段代码，嗯，是的，也许

**[00:00:04]** have transferred things to the GPU so what we can do now is for example we
> 可以运行这段代码，嗯，是的，也许看看一切是否正常工作，所以对于

**[00:00:04]** so what we can do now is for example we
> 看看一切是否正常工作，所以对于

**[00:00:04]** so what we can do now is for example we could run this code and um yeah maybe
> 看看一切是否正常工作，所以例如，我们可以做的是，嗯，通常

**[00:00:04]** could run this code and um yeah maybe
> 例如，我们可以做的是，嗯，通常

**[00:00:04]** could run this code and um yeah maybe see if everything works correctly so for
> 例如，我们可以做的是，嗯，通常当你进行前向传播时，我建议

**[00:00:04]** see if everything works correctly so for
> 当你进行前向传播时，我建议

**[00:00:04]** see if everything works correctly so for example what we can do is uh usually
> 当你进行前向传播时，我建议使用no Grant，这样在构建计算图时不会浪费

**[00:00:05]** example what we can do is uh usually
> 使用no Grant，这样在构建计算图时不会浪费

**[00:00:05]** example what we can do is uh usually when you do a forward pass I recommend
> 使用no Grant，这样在构建计算图时不会浪费内存，所以如果我们不训练，我们可以

**[00:00:05]** when you do a forward pass I recommend
> 内存，所以如果我们不训练，我们可以

**[00:00:05]** when you do a forward pass I recommend doing uh no Grant so it doesn't waste
> 内存，所以如果我们不训练，我们可以做这个no Grant，然后嗯，上下文

**[00:00:05]** doing uh no Grant so it doesn't waste
> 做这个no Grant，然后嗯，上下文

**[00:00:05]** doing uh no Grant so it doesn't waste memory when constructing a computation
> 做这个no Grant，然后嗯，上下文返回，假设

**[00:00:05]** memory when constructing a computation
> 返回，假设

**[00:00:05]** memory when constructing a computation graph so if we are not training we can
> 返回，假设

**[00:00:05]** graph so if we are not training we can
> graph so if we are not training we can

**[00:00:05]** graph so if we are not training we can do this no Grant and then um context
> graph so if we are not training we can do this no Grant and then um context

**[00:00:05]** do this no Grant and then um context
> do this no Grant and then um context

**[00:00:05]** do this no Grant and then um context back let's say
> do this no Grant and then um context back let's say

**[00:00:05]** back let's say
> back let's say

**[00:00:05]** back let's say equals uh CA without buffer and then the
> 回到，假设等于没有缓冲区的CA，然后

**[00:00:05]** equals uh CA without buffer and then the
> 等于没有缓冲区的CA，然后

**[00:00:05]** equals uh CA without buffer and then the batch so this is just calling this um
> 等于没有缓冲区的CA，然后batch，所以这只是调用这个

**[00:00:05]** batch so this is just calling this um
> batch，所以这只是调用这个

**[00:00:05]** batch so this is just calling this um like we have here
> batch，所以这只是调用这个，就像我们这里有的

**[00:00:05]** like we have here
> 就像我们这里有的

**[00:00:05]** like we have here above now though it should run on the
> 就像我们这里有的上面，但现在它应该在

**[00:00:05]** above now though it should run on the
> 上面，但现在它应该在

**[00:00:05]** above now though it should run on the GPU because we just transferred things
> 上面，但现在它应该在GPU上运行，因为我们刚刚转移了东西

**[00:00:05]** GPU because we just transferred things
> GPU，因为我们刚刚转移了东西

**[00:00:05]** GPU because we just transferred things to the GPU so let's see what
> GPU，因为我们刚刚转移了东西到GPU，所以让我们看看

**[00:00:05]** to the GPU so let's see what
> 到GPU，所以让我们看看

**[00:00:05]** to the GPU so let's see what happens oops I made a typo that is what
> 到GPU，所以让我们看看会发生什么，哦，我打错了，这就是

**[00:00:05]** happens oops I made a typo that is what
> 会发生什么，哦，我打错了，这就是

**[00:00:05]** happens oops I made a typo that is what happened let's fix that okay so what
> 会发生什么，哦，我打错了，这就是发生的事情，让我们修复它，好的，那么

**[00:00:05]** happened let's fix that okay so what
> 发生的事情，让我们修复它，好的，那么

**[00:00:05]** happened let's fix that okay so what happened here so we have a runtime error
> 发生的事情，让我们修复它，好的，那么这里发生了什么，所以我们有一个运行时错误

**[00:00:05]** happened here so we have a runtime error
> 这里发生了什么，所以我们有一个运行时错误

**[00:00:05]** happened here so we have a runtime error and if we look at this error if we
> 这里发生了什么，所以我们有一个运行时错误，如果我们查看这个错误，如果我们

**[00:00:05]** and if we look at this error if we
> 如果我们查看这个错误，如果我们

**[00:00:05]** and if we look at this error if we scroll to the bottom it says expected
> 如果我们查看这个错误，如果我们滚动到底部，它说期望

**[00:00:05]** scroll to the bottom it says expected
> 滚动到底部，它说期望

**[00:00:05]** scroll to the bottom it says expected self and mask so self here would be
> 滚动到底部，它说期望self和mask，所以这里的self应该是

**[00:00:05]** self and mask so self here would be
> self和mask，所以这里的self应该是

**[00:00:05]** self and mask so self here would be referring to this CA without buffer um
> self和mask，所以这里的self应该是指这个没有缓冲区的CA

**[00:00:05]** referring to this CA without buffer um
> 指这个没有缓冲区的CA

**[00:00:05]** referring to this CA without buffer um expected self and mask to be on the same
> 指这个没有缓冲区的CA，期望self和mask在同一个

**[00:00:05]** expected self and mask to be on the same
> 期望self和mask在同一个

**[00:00:05]** expected self and mask to be on the same device but got mask on CPU and self Cuda
> 期望self和mask在同一个设备上，但mask在CPU上，而self在Cuda上

**[00:00:05]** device but got mask on CPU and self Cuda
> 设备上，但mask在CPU上，而self在Cuda上

**[00:00:05]** device but got mask on CPU and self Cuda sorry and self on Cuda so this means
> 设备上，但mask在CPU上，而self在Cuda上，抱歉，self在Cuda上，所以这意味着

**[00:00:06]** sorry and self on Cuda so this means
> 抱歉，self在Cuda上，所以这意味着

**[00:00:06]** sorry and self on Cuda so this means somehow something got transferred to the
> 抱歉，self在Cuda上，所以这意味着不知何故，有些东西被转移到了

**[00:00:06]** somehow something got transferred to the
> 不知何故，有些东西被转移到了

**[00:00:06]** somehow something got transferred to the GPU and something doesn't didn't get
> 不知何故，有些东西被转移到了GPU，而有些东西没有被

**[00:00:06]** GPU and something doesn't didn't get
> GPU，而有些东西没有被

**[00:00:06]** GPU and something doesn't didn't get transferred so one might think okay
> GPU，而有些东西没有被转移，所以人们可能会想，好吧

**[00:00:06]** transferred so one might think okay
> 转移，所以人们可能会想，好吧

**[00:00:06]** transferred so one might think okay there's maybe something wrong here but
> 转移，所以人们可能会想，好吧，这里可能有些问题，但是

**[00:00:06]** there's maybe something wrong here but
> 这里可能有些问题，但是

**[00:00:06]** there's maybe something wrong here but let's say everything is correct here we
> 这里可能有些问题，但是假设这里一切都是正确的，我们

**[00:00:06]** let's say everything is correct here we
> 假设这里一切都是正确的，我们

**[00:00:06]** let's say everything is correct here we can actually inspect this a bit further
> 假设这里一切都是正确的，我们实际上可以进一步检查这个

**[00:00:06]** can actually inspect this a bit further
> 实际上可以进一步检查这个

**[00:00:06]** can actually inspect this a bit further so what what we could do is so if we
> 实际上可以进一步检查这个，所以我们可以做的是，如果我们

**[00:00:06]** so what what we could do is so if we
> 所以我们可以做的是，如果我们

**[00:00:06]** so what what we could do is so if we look at this class above again so we
> 所以我们可以做的是，如果我们再次查看上面的这个类，所以我们

**[00:00:06]** look at this class above again so we
> 再次查看上面的这个类，所以我们

**[00:00:06]** look at this class above again so we could check are all these things on the
> 再次查看上面的这个类，所以我们可以检查所有这些是否都在

**[00:00:06]** could check are all these things on the
> 可以检查所有这些是否都在

**[00:00:06]** could check are all these things on the GPU and I will actually tell you exactly
> 可以检查所有这些是否都在GPU上，我会确切地告诉你

**[00:00:06]** GPU and I will actually tell you exactly
> GPU上，我会确切地告诉你

**[00:00:06]** GPU and I will actually tell you exactly the places we will check here and so one
> GPU上，我会确切地告诉你我们要检查的地方，所以一个

**[00:00:06]** the places we will check here and so one
> 我们要检查的地方，所以一个

**[00:00:06]** the places we will check here and so one place we will check is just for
> 我们要检查的地方，所以一个我们要检查的地方只是作为

**[00:00:06]** place we will check is just for
> 我们要检查的地方只是作为

**[00:00:06]** place we will check is just for reference this uh W query which is a
> 我们要检查的地方只是作为参考，这个W query，它是一个

**[00:00:06]** reference this uh W query which is a
> 参考，这个W query，它是一个

**[00:00:06]** reference this uh W query which is a linear layer which contains weights and
> 参考，这个W query，它是一个线性层，包含权重和

**[00:00:06]** linear layer which contains weights and
> 线性层，包含权重和

**[00:00:06]** linear layer which contains weights and then we have I will draw attention to
> 线性层，包含权重和，然后我会提请注意

**[00:00:06]** then we have I will draw attention to
> 然后我会提请注意

**[00:00:06]** then we have I will draw attention to this this triou here so triou stands for
> 然后我会提请注意这个triou，所以triou代表

**[00:00:06]** this this triou here so triou stands for
> 这个triou，所以triou代表

**[00:00:06]** this this triou here so triou stands for triangular um like the upper triangular
> 这个triou，所以triou代表三角形，比如上三角

**[00:00:06]** triangular um like the upper triangular
> 三角形，比如上三角

**[00:00:06]** triangular um like the upper triangular Matrix and this is essentially just
> 三角形，比如上三角矩阵，这基本上只是

**[00:00:06]** Matrix and this is essentially just
> 矩阵，这基本上只是

**[00:00:06]** Matrix and this is essentially just creating a mask that we use down here by
> 矩阵，这基本上只是创建一个我们在这里下面使用的mask

**[00:00:06]** creating a mask that we use down here by
> 创建一个我们在这里下面使用的mask

**[00:00:06]** creating a mask that we use down here by the way if you're curious why we are
> 创建一个我们在这里下面使用的mask，顺便说一下，如果你好奇为什么我们

**[00:00:06]** the way if you're curious why we are
> 如果你好奇为什么我们

**[00:00:06]** the way if you're curious why we are doing it like this there's also um was a
> 如果你好奇为什么我们这样做，还有一个

**[00:00:06]** doing it like this there's also um was a
> 这样做，还有一个

**[00:00:06]** doing it like this there's also um was a good question and a GitHub discussion um
> 这样做，还有一个很好的问题和GitHub讨论

**[00:00:06]** good question and a GitHub discussion um
> 很好的问题和GitHub讨论

**[00:00:06]** good question and a GitHub discussion um I will probably link this in this video
> 很好的问题和GitHub讨论，我可能会在这个视频中链接这个

**[00:00:07]** I will probably link this in this video
> 我可能会在这个视频中链接这个

**[00:00:07]** I will probably link this in this video because I've been doing this for
> 我可能会在这个视频中链接这个，因为我一直在这样做

**[00:00:07]** because I've been doing this for
> 因为我一直在这样做

**[00:00:07]** because I've been doing this for efficiency reasons you could technically
> 因为我一直出于效率原因这样做，技术上你可以

**[00:00:07]** efficiency reasons you could technically
> 效率原因，技术上你可以

**[00:00:07]** efficiency reasons you could technically down here have the triou every time in
> 效率原因，技术上你可以在这里每次都在前向传播中定义triou，但这会非常

**[00:00:07]** down here have the triou every time in
> 在这里每次都在前向传播中定义triou，但这会非常

**[00:00:07]** down here have the triou every time in the forward but this would be very
> 在这里每次都在前向传播中定义triou，但这会非常昂贵，所以你会，我会说损失

**[00:00:07]** the forward but this would be very
> 昂贵，所以你会，我会说损失

**[00:00:07]** the forward but this would be very expensive so you would I would say lose
> 昂贵，所以你会，我会说损失30%的效率，如果你把

**[00:00:07]** expensive so you would I would say lose
> 30%的效率，如果你把

**[00:00:07]** expensive so you would I would say lose 30% of the efficiency if you would move
> 30%的效率，如果你把triou从上面移到这儿，并且

**[00:00:07]** 30% of the efficiency if you would move
> triou从上面移到这儿，并且

**[00:00:07]** 30% of the efficiency if you would move the triou from up here to here and
> triou从上面移到这儿，并且每次传递都重新定义它，所以我

**[00:00:07]** the triou from up here to here and
> 每次传递都重新定义它，所以我

**[00:00:07]** the triou from up here to here and redefine it every um every pass so I I
> 每次传递都重新定义它，所以我可以在录制后在我的视频中链接一个基准测试

**[00:00:07]** redefine it every um every pass so I I
> 可以在录制后在我的视频中链接一个基准测试

**[00:00:07]** redefine it every um every pass so I I can link a benchmark in my video after
> 可以在录制后在我的视频中链接一个基准测试，但现在请先跟着我，所以

**[00:00:07]** can link a benchmark in my video after
> 现在请先跟着我，所以

**[00:00:07]** can link a benchmark in my video after recording but so for now stay with me so
> 现在请先跟着我，所以我们有这个mask，我们将查看它

**[00:00:07]** recording but so for now stay with me so
> 我们查看这个mask，我们将查看它

**[00:00:07]** recording but so for now stay with me so we have this mask here that we will look
> 我们查看这个mask，我们将查看它，并且我们假设有一个

**[00:00:07]** we have this mask here that we will look
> 并且我们假设有一个

**[00:00:07]** we have this mask here that we will look at and we have let's say one of the
> 并且我们假设有一个权重矩阵，我也可以直接选

**[00:00:07]** at and we have let's say one of the
> 权重矩阵，我也可以直接选

**[00:00:07]** at and we have let's say one of the weight matrices I could also just pick
> 权重矩阵，我也可以直接选那些，但因为是第一个

**[00:00:07]** weight matrices I could also just pick
> 那些，但因为是第一个

**[00:00:07]** weight matrices I could also just pick those but yeah because it's the first
> 那些，但因为是第一个，这里的第一行，让我们实际上

**[00:00:07]** those but yeah because it's the first
> 这里的第一行，让我们实际上

**[00:00:07]** those but yeah because it's the first one here the first row let's actually
> 这里的第一行，让我们实际上看看那个，所以那个和

**[00:00:07]** one here the first row let's actually
> 看看那个，所以那个和

**[00:00:07]** one here the first row let's actually take a look at that one so that one and
> 看看那个，所以那个和那个，看看它们是否真的

**[00:00:07]** take a look at that one so that one and
> 那个，看看它们是否真的

**[00:00:07]** take a look at that one so that one and that one and see if those are actually
> 那个，看看它们是否真的在GPU上，所以让我们看看

**[00:00:07]** that one and see if those are actually
> 在GPU上，所以让我们看看

**[00:00:07]** that one and see if those are actually in fact on the GPU so let's see so what
> 在GPU上，所以让我们看看我们能做什么，例如类似于

**[00:00:07]** in fact on the GPU so let's see so what
> 我们能做什么，例如类似于

**[00:00:07]** in fact on the GPU so let's see so what we can do is for example similar to what
> 我们能做什么，例如类似于我们之前做过的，我们只需做一个简单的

**[00:00:07]** we can do is for example similar to what
> 我们之前做过的，我们只需做一个简单的

**[00:00:07]** we can do is for example similar to what we've done before we just do a simple
> 我们之前做过的，我们只需做一个简单的print，我通常喜欢这样做

**[00:00:07]** we've done before we just do a simple
> print，我通常喜欢这样做

**[00:00:07]** we've done before we just do a simple print I usually like to do it like that
> print，我通常喜欢这样做，嗯，这样我知道什么属于什么

**[00:00:07]** print I usually like to do it like that
> 嗯，这样我知道什么属于什么

**[00:00:07]** print I usually like to do it like that um so I know what belongs to what so
> 嗯，这样我知道什么属于什么，所以在这里加上名字，然后我可以

**[00:00:07]** um so I know what belongs to what so
> 所以在这里加上名字，然后我可以

**[00:00:07]** um so I know what belongs to what so just adding the name here and then I can
> 所以在这里加上名字，然后我可以做，嗯，用我们的buffer做CA，然后

**[00:00:07]** just adding the name here and then I can
> 用我们的buffer做CA，然后

**[00:00:07]** just adding the name here and then I can do um CA with our buffer and then
> 用我们的buffer做CA，然后正是那个query，哦query，然后让我们

**[00:00:08]** do um CA with our buffer and then
> 正是那个query，哦query，然后让我们

**[00:00:08]** do um CA with our buffer and then exactly that query oops query and let's
> 正是那个query，哦query，然后让我们看看权重，所以我们做

**[00:00:08]** exactly that query oops query and let's
> 看看权重，所以我们做

**[00:00:08]** exactly that query oops query and let's say we look at the weight so we do
> 看看权重，所以我们做device，这样我们可以检查，哦

**[00:00:08]** say we look at the weight so we do
> device，这样我们可以检查，哦

**[00:00:08]** say we look at the weight so we do device and with that we can check oh
> device，这样我们可以检查，哦这实际上是一个CUDA设备，所以

**[00:00:08]** device and with that we can check oh
> 这实际上是一个CUDA设备，所以

**[00:00:08]** device and with that we can check oh this is actually an Auda device so
> 这实际上是一个CUDA设备，所以一切应该没问题，那么

**[00:00:08]** this is actually an Auda device so
> 一切应该没问题，那么

**[00:00:08]** this is actually an Auda device so everything should be fine here so what
> 一切应该没问题，那么另一个呢，嗯，让我们

**[00:00:08]** everything should be fine here so what
> 另一个呢，嗯，让我们

**[00:00:08]** everything should be fine here so what now about what is the other one um let's
> 另一个呢，嗯，让我们叫它mask device，所以这会是

**[00:00:08]** now about what is the other one um let's
> 叫它mask device，所以这会是

**[00:00:08]** now about what is the other one um let's call it mask device and so this would be
> 叫它mask device，所以这对应这里的这个，那么

**[00:00:08]** call it mask device and so this would be
> 对应这里的这个，那么

**[00:00:08]** call it mask device and so this would be corresponding to to this here so what
> 对应这里的这个，那么这里发生了什么，让我们看看

**[00:00:08]** corresponding to to this here so what
> 这里发生了什么，让我们看看

**[00:00:08]** corresponding to to this here so what happened here let's see
> 这里发生了什么，让我们看看，嗯，下面这里，所以我们做CA，哦CA没有

**[00:00:08]** happened here let's see
> 嗯，下面这里，所以我们做CA，哦CA没有

**[00:00:08]** happened here let's see um down here so we do CA oops CA without
> 嗯，下面这里，所以我们做CA，哦CA没有buffer，然后mask，然后

**[00:00:08]** um down here so we do CA oops CA without
> buffer，然后mask，然后

**[00:00:08]** um down here so we do CA oops CA without buffer and then mask and then
> buffer，然后mask，然后device，所以我们在这里可以看到的是

**[00:00:08]** buffer and then mask and then
> device，所以我们在这里可以看到的是

**[00:00:08]** buffer and then mask and then device so what we can see here is that
> device，所以我们在这里可以看到的是，我们的mask device没有被转移

**[00:00:08]** device so what we can see here is that
> 我们的mask device没有被转移

**[00:00:08]** device so what we can see here is that our mask device did not get transferred
> 我们的mask device没有被转移，由于某种原因，当我

**[00:00:08]** our mask device did not get transferred
> 到GPU，由于某种原因，当我

**[00:00:08]** our mask device did not get transferred to the GPU for some reason when I
> 到GPU，由于某种原因，当我执行这段代码时，不知何故mask

**[00:00:08]** to the GPU for some reason when I
> 执行这段代码时，不知何故mask

**[00:00:08]** to the GPU for some reason when I executed this code here somehow The Mask
> 执行这段代码时，不知何故mask固执地，是的，留在了CPU上，所以为什么

**[00:00:08]** executed this code here somehow The Mask
> 固执地，是的，留在了CPU上，所以为什么

**[00:00:08]** executed this code here somehow The Mask stubbornly yeah stayed on the CPU so why
> 固执地，是的，留在了CPU上，所以为什么会这样，原因是

**[00:00:08]** stubbornly yeah stayed on the CPU so why
> 会这样，原因是

**[00:00:08]** stubbornly yeah stayed on the CPU so why is that and so the reason
> 会这样，原因是

**[00:00:08]** is that and so the reason
> is that and so the reason

**[00:00:08]** is that and so the reason is
> 原因在于

**[00:00:09]** is
> 是

**[00:00:09]** is because these are weight parameters so
> 因为这些是权重参数

**[00:00:09]** because these are weight parameters so
> 因为这些是权重参数

**[00:00:09]** because these are weight parameters so um essentially linear the weight um I'm
> 因为这些是权重参数，嗯，本质上是线性权重

**[00:00:09]** um essentially linear the weight um I'm
> 嗯，本质上是线性权重

**[00:00:09]** um essentially linear the weight um I'm not sure if we can show this actually
> 嗯，本质上是线性权重，我不确定能否展示这个

**[00:00:09]** not sure if we can show this actually
> 不确定能否展示这个

**[00:00:09]** not sure if we can show this actually let's just
> 不确定能否展示这个，我们试试

**[00:00:09]** try so you can see this is a parameter
> 试试看，你可以看到这是一个参数

**[00:00:09]** try so you can see this is a parameter
> 试试看，你可以看到这是一个参数

**[00:00:09]** try so you can see this is a parameter and all the parameters get transferred
> 试试看，你可以看到这是一个参数，所有参数都会被

**[00:00:09]** and all the parameters get transferred
> 所有参数都会被

**[00:00:09]** and all the parameters get transferred by pytorch automatically to the GPU when
> 所有参数都会被PyTorch自动传输到GPU，当

**[00:00:09]** by pytorch automatically to the GPU when
> 当PyTorch自动传输到GPU

**[00:00:09]** by pytorch automatically to the GPU when we execute the two Cuda call now the
> 当PyTorch自动传输到GPU，当我们执行两个Cuda调用时，现在

**[00:00:09]** we execute the two Cuda call now the
> 当我们执行两个Cuda调用时

**[00:00:09]** we execute the two Cuda call now the mask on the other hand
> 当我们执行两个Cuda调用时，另一方面，mask

**[00:00:09]** mask on the other hand
> 另一方面，mask

**[00:00:09]** mask on the other hand let's try it out here the mask here is a
> 另一方面，mask，我们在这里试试，这里的mask是一个

**[00:00:09]** let's try it out here the mask here is a
> 我们在这里试试，这里的mask是一个

**[00:00:09]** let's try it out here the mask here is a torch tensor and torch tensors they
> 我们在这里试试，这里的mask是一个torch tensor，而torch tensors

**[00:00:09]** torch tensor and torch tensors they
> torch tensor和torch tensors

**[00:00:09]** torch tensor and torch tensors they don't get automatically transferred now
> torch tensor和torch tensors不会自动传输，现在

**[00:00:09]** don't get automatically transferred now
> 不会自动传输

**[00:00:09]** don't get automatically transferred now in order to fix that what we could do is
> 不会自动传输，为了解决这个问题，我们可以

**[00:00:09]** in order to fix that what we could do is
> 为了解决这个问题，我们可以

**[00:00:09]** in order to fix that what we could do is we could go
> 为了解决这个问题，我们可以

**[00:00:09]** we could go
> 我们可以

**[00:00:09]** we could go here um and do a torch dot or just an N
> 我们可以在这里，用torch dot或者直接用N

**[00:00:09]** here um and do a torch dot or just an N
> 在这里，用torch dot或者直接用N

**[00:00:09]** here um and do a torch dot or just an N do parameter here and make this a
> 在这里，用torch dot或者直接用N，在这里做参数，并把它变成

**[00:00:09]** do parameter here and make this a
> 做参数，并把它变成

**[00:00:09]** do parameter here and make this a parameter however it's also just a
> 做参数，并把它变成参数，然而这也只是

**[00:00:09]** parameter however it's also just a
> 参数，然而这也只是

**[00:00:09]** parameter however it's also just a convention or not a convention but even
> 参数，然而这也只是一种约定，或者说不是约定，但即使是

**[00:00:09]** convention or not a convention but even
> 约定，或者说不是约定，但即使是

**[00:00:09]** convention or not a convention but even like implementation wise the fact that
> 约定，或者说不是约定，但即使是实现层面，事实上

**[00:00:09]** like implementation wise the fact that
> 实现层面，事实上

**[00:00:09]** like implementation wise the fact that parameters are usually for things you
> 实现层面，事实上参数通常用于那些你

**[00:00:10]** parameters are usually for things you
> 参数通常用于那些你

**[00:00:10]** parameters are usually for things you want to learn during training so they
> 参数通常用于那些你想在训练中学习的东西，所以它们

**[00:00:10]** want to learn during training so they
> 想在训练中学习的东西，所以它们

**[00:00:10]** want to learn during training so they get passed to the optimizer so we don't
> 想在训练中学习的东西，所以它们会被传递给优化器，因此我们不想

**[00:00:10]** get passed to the optimizer so we don't
> 会被传递给优化器，因此我们不想

**[00:00:10]** get passed to the optimizer so we don't want to to learn this mask it's like a
> 会被传递给优化器，因此我们不想学习这个mask，它就像一个

**[00:00:10]** want to to learn this mask it's like a
> 学习这个mask，它就像一个

**[00:00:10]** want to to learn this mask it's like a fixed mask so there's no no point in in
> 学习这个mask，它就像一个固定的mask，所以没有必要

**[00:00:10]** fixed mask so there's no no point in in
> 固定的mask，所以没有必要

**[00:00:10]** fixed mask so there's no no point in in learning that so we actually don't want
> 固定的mask，所以没有必要学习它，因此我们实际上不想

**[00:00:10]** learning that so we actually don't want
> 学习它，因此我们实际上不想

**[00:00:10]** learning that so we actually don't want that so for that what we want is
> 学习它，因此我们实际上不想那样做，所以对于这个，我们想要的是

**[00:00:10]** that so for that what we want is
> 那样做，所以对于这个，我们想要的是

**[00:00:10]** that so for that what we want is actually we want to make this a buffer
> 那样做，所以对于这个，我们想要的是实际上把它变成一个buffer

**[00:00:10]** actually we want to make this a buffer
> 实际上把它变成一个buffer

**[00:00:10]** actually we want to make this a buffer so what I mean is what we do is we
> 实际上把它变成一个buffer，我的意思是，我们

**[00:00:10]** so what I mean is what we do is we
> 我的意思是，我们

**[00:00:10]** so what I mean is what we do is we change it
> 我的意思是，我们把它改成

**[00:00:10]** change it
> 改成

**[00:00:10]** change it to let's say comment this out to
> 改成，比如说，注释掉这个，改成

**[00:00:10]** to let's say comment this out to
> 比如说，注释掉这个，改成

**[00:00:10]** to let's say comment this out to self
> 比如说，注释掉这个，改成self

**[00:00:10]** self
> self

**[00:00:10]** self register oops buffer
> self register，哦，buffer

**[00:00:10]** register oops buffer
> register，哦，buffer

**[00:00:10]** register oops buffer and then it's a bit awkward but we have
> register，哦，buffer，然后这有点别扭，但我们得

**[00:00:10]** and then it's a bit awkward but we have
> 然后这有点别扭，但我们得

**[00:00:10]** and then it's a bit awkward but we have to provide it as a string so we will
> 然后这有点别扭，但我们得把它作为字符串提供，所以我们将

**[00:00:10]** to provide it as a string so we will
> 把它作为字符串提供，所以我们将

**[00:00:10]** to provide it as a string so we will provide this as a string here which was
> 把它作为字符串提供，所以我们将在这里提供一个字符串，它之前是

**[00:00:10]** provide this as a string here which was
> 在这里提供一个字符串，它之前是

**[00:00:10]** provide this as a string here which was previously an attribute and this will
> 在这里提供一个字符串，它之前是一个属性，这将会

**[00:00:10]** previously an attribute and this will
> 一个属性，这将会

**[00:00:10]** previously an attribute and this will become an attribute so we will have that
> 一个属性，这将会变成一个属性，所以我们将拥有

**[00:00:10]** become an attribute so we will have that
> 变成一个属性，所以我们将拥有

**[00:00:10]** become an attribute so we will have that and then let me just copy
> 变成一个属性，所以我们将拥有它，然后让我复制一下

**[00:00:10]** and then let me just copy
> 然后让我复制一下

**[00:00:10]** and then let me just copy this so this would be essentially now a
> 然后让我复制一下，这现在本质上就是一个

**[00:00:10]** this so this would be essentially now a
> 这现在本质上就是一个

**[00:00:10]** this so this would be essentially now a registered buffer so let's run this run
> 这现在本质上就是一个注册的buffer，所以让我们运行这个，运行

**[00:00:10]** registered buffer so let's run this run
> 注册了buffer，让我们运行这个

**[00:00:10]** registered buffer so let's run this run this so this still works like before we
> 注册了buffer，让我们运行这个，这个仍然像之前一样工作

**[00:00:10]** this so this still works like before we
> 这个仍然像之前一样工作

**[00:00:10]** this so this still works like before we have everything still set up everything
> 这个仍然像之前一样工作，我们仍然设置好了一切

**[00:00:10]** have everything still set up everything
> 仍然设置好了一切

**[00:00:10]** have everything still set up everything is still running on the GP
> 仍然设置好了一切，一切仍在GPU上运行

**[00:00:10]** is still running on the GP
> 仍在GPU上运行

**[00:00:10]** is still running on the GP now let's do the transfer again so we
> 仍在GPU上运行，现在让我们再次进行转移

**[00:00:11]** now let's do the transfer again so we
> 现在让我们再次进行转移

**[00:00:11]** now let's do the transfer again so we transferred and now so maybe that's a
> 现在让我们再次进行转移，我们转移了，现在这可能有点

**[00:00:11]** transferred and now so maybe that's a
> 转移了，现在这可能有点

**[00:00:11]** transferred and now so maybe that's a bit misleading because now it has a
> 转移了，现在这可能有点误导，因为它现在有一个

**[00:00:11]** bit misleading because now it has a
> 有点误导，因为它现在有一个

**[00:00:11]** bit misleading because now it has a buffer so let's actually change this so
> 有点误导，因为它现在有一个buffer，所以让我们实际更改这个

**[00:00:11]** buffer so let's actually change this so
> buffer，所以让我们实际更改这个

**[00:00:11]** buffer so let's actually change this so to make it not
> buffer，所以让我们实际更改这个，使其不

**[00:00:11]** misleading yeah it's a pretty
> 误导，是的，这是一个相当

**[00:00:11]** misleading yeah it's a pretty
> 误导，是的，这是一个相当

**[00:00:11]** misleading yeah it's a pretty spontaneous video but it's good to do
> 误导，是的，这是一个相当随意的视频，但做好事情是好的

**[00:00:11]** spontaneous video but it's good to do
> 随意的视频，但做好事情是好的

**[00:00:11]** spontaneous video but it's good to do things properly here could have made a
> 随意的视频，但做好事情是好的，本来可以创建一个

**[00:00:11]** things properly here could have made a
> 本来可以创建一个

**[00:00:11]** things properly here could have made a second class but it's what it is so
> 本来可以创建一个第二类，但就是这样

**[00:00:11]** second class but it's what it is so
> 第二类，但就是这样

**[00:00:11]** second class but it's what it is so okay uh it's not defined oh yeah there
> 第二类，但就是这样，好的，它没有定义，哦，是的

**[00:00:11]** okay uh it's not defined oh yeah there
> 好的，它没有定义，哦，是的

**[00:00:11]** okay uh it's not defined oh yeah there we
> 好的，它没有定义，哦，是的，我们

**[00:00:11]** we
> 我们

**[00:00:11]** we go so this still works works or machine
> 我们开始，所以这个仍然有效，机器

**[00:00:11]** go so this still works works or machine
> 开始，所以这个仍然有效，机器

**[00:00:11]** go so this still works works or machine is using GPU still and we transfer it to
> 开始，所以这个仍然有效，机器仍然在使用GPU，我们将其转移到

**[00:00:11]** is using GPU still and we transfer it to
> 仍然在使用GPU，我们将其转移到

**[00:00:11]** is using GPU still and we transfer it to the GPU here and now let's try the with
> 仍然在使用GPU，我们将其转移到这里的GPU，现在让我们尝试使用

**[00:00:11]** buffer so what happened um things worked
> buffer，所以发生了什么，嗯，事情成功了

**[00:00:11]** buffer so what happened um things worked
> buffer，所以发生了什么，嗯，事情成功了

**[00:00:11]** buffer so what happened um things worked let's print this
> buffer，所以发生了什么，嗯，事情成功了，让我们打印这个

**[00:00:11]** let's print this
> 让我们打印这个

**[00:00:11]** let's print this out it looks like we fixed the
> 让我们打印这个，看起来我们修复了

**[00:00:11]** out it looks like we fixed the
> 看起来我们修复了

**[00:00:11]** out it looks like we fixed the problem so now we can run the code on
> 看起来我们修复了问题，所以现在我们可以运行代码在

**[00:00:11]** problem so now we can run the code on
> 问题，所以现在我们可以运行代码在

**[00:00:11]** problem so now we can run the code on the GPU thanks to the buffer so if we
> 问题，所以现在我们可以运行代码在GPU上，多亏了buffer，所以如果我们

**[00:00:11]** the GPU thanks to the buffer so if we
> GPU上，多亏了buffer，所以如果我们

**[00:00:11]** the GPU thanks to the buffer so if we double check
> GPU上，多亏了buffer，所以如果我们再次检查

**[00:00:12]** double check
> 再次检查

**[00:00:12]** double check similar to before the weight is still on
> 再次检查，和之前类似，权重仍然在

**[00:00:12]** similar to before the weight is still on
> 和之前类似，权重仍然在

**[00:00:12]** similar to before the weight is still on the GPU before we saw that the weight is
> 和之前类似，权重仍然在GPU上，之前我们看到权重在

**[00:00:12]** the GPU before we saw that the weight is
> GPU上，之前我们看到权重在

**[00:00:12]** the GPU before we saw that the weight is on the CP sorry the mask is on the CPU
> GPU上，之前我们看到权重在CPU上，抱歉，mask在CPU上

**[00:00:12]** on the CP sorry the mask is on the CPU
> CPU上，抱歉，mask在CPU上

**[00:00:12]** on the CP sorry the mask is on the CPU now with our buffer let's see what
> CPU上，抱歉，mask在CPU上，现在有了我们的buffer，让我们看看

**[00:00:12]** now with our buffer let's see what
> 现在有了我们的buffer，让我们看看

**[00:00:12]** now with our buffer let's see what happens you can see now it's also
> 现在有了我们的buffer，让我们看看发生了什么，你可以看到现在它也被

**[00:00:12]** happens you can see now it's also
> 发生了什么，你可以看到现在它也被

**[00:00:12]** happens you can see now it's also transferred to the GPU so weights is
> 发生了什么，你可以看到现在它也被转移到了GPU，所以权重是

**[00:00:12]** transferred to the GPU so weights is
> 转移到了GPU，所以权重是

**[00:00:12]** transferred to the GPU so weights is still a parameter but
> 转移到了GPU，所以权重仍然是一个parameter，但是

**[00:00:12]** still a parameter but
> 仍然是一个parameter，但是

**[00:00:12]** still a parameter but this mask should be a buffer now no it's
> 仍然是一个parameter，但是这个mask现在应该是一个buffer，不，它

**[00:00:12]** this mask should be a buffer now no it's
> 这个mask现在应该是一个buffer，不，它

**[00:00:12]** this mask should be a buffer now no it's not actually interesting okay it's still
> 这个mask现在应该是一个buffer，不，它实际上不是，有趣，好的，它仍然是

**[00:00:12]** not actually interesting okay it's still
> 实际上不是，有趣，好的，它仍然是

**[00:00:12]** not actually interesting okay it's still a tensor but it's registered as a buffer
> 实际上不是，有趣，好的，它仍然是一个tensor，但被注册为buffer

**[00:00:12]** a tensor but it's registered as a buffer
> 一个tensor，但被注册为buffer

**[00:00:12]** a tensor but it's registered as a buffer so in that sense it got transferred to
> 一个tensor，但被注册为buffer，所以从这个意义上说，它被转移到了

**[00:00:12]** so in that sense it got transferred to
> 所以从这个意义上说，它被转移到了

**[00:00:12]** so in that sense it got transferred to the GPU so the nice thing is also later
> 所以从这个意义上说，它被转移到了GPU，所以好处是以后

**[00:00:12]** the GPU so the nice thing is also later
> GPU，所以好处是以后

**[00:00:12]** the GPU so the nice thing is also later if I decide I don't want this on the GPU
> GPU，所以好处是以后如果我决定不再需要它在GPU上

**[00:00:12]** if I decide I don't want this on the GPU
> 如果我决定不再需要它在GPU上

**[00:00:12]** if I decide I don't want this on the GPU anymore what I can do is I can go here
> 如果我决定不再需要它在GPU上，我可以在这里

**[00:00:12]** anymore what I can do is I can go here
> 我可以在这里

**[00:00:12]** anymore what I can do is I can go here CA with buffer say I want this back on
> 我可以在这里用buffer说，我想把它放回

**[00:00:12]** CA with buffer say I want this back on
> 用buffer说，我想把它放回

**[00:00:12]** CA with buffer say I want this back on the
> 用buffer说，我想把它放回

**[00:00:12]** the
> 放回

**[00:00:12]** the CPU and then I should see
> CPU上，然后我应该看到

**[00:00:12]** CPU and then I should see
> CPU上，然后我应该看到

**[00:00:12]** CPU and then I should see that this is still this is now on the
> CPU上，然后我应该看到这个仍然在，现在它在

**[00:00:12]** that this is still this is now on the
> 这仍然是在

**[00:00:12]** that this is still this is now on the CPU so I can go basically back and forth
> 这仍然是在CPU上，所以我可以基本来回切换

**[00:00:12]** CPU so I can go basically back and forth
> CPU上，所以我可以基本来回切换

**[00:00:12]** CPU so I can go basically back and forth and it should get transferred as you can
> CPU上，所以我可以基本来回切换，它应该会被传输，正如你

**[00:00:13]** and it should get transferred as you can
> 它应该会被传输，正如你

**[00:00:13]** and it should get transferred as you can see so yeah so this was a short
> 它应该会被传输，正如你所见，所以是的，这是一个简短的

**[00:00:13]** see so yeah so this was a short
> 所见，所以是的，这是一个简短的

**[00:00:13]** see so yeah so this was a short spontaneous video on explaining what
> 所见，所以是的，这是一个简短的自发视频，用于解释

**[00:00:13]** spontaneous video on explaining what
> 自发视频，用于解释

**[00:00:13]** spontaneous video on explaining what buffers are and why they are useful so
> 自发视频，用于解释什么是buffers以及它们为什么有用，所以

**[00:00:13]** buffers are and why they are useful so
> buffers以及它们为什么有用，所以

**[00:00:13]** buffers are and why they are useful so this allows us conveniently to transfer
> buffers以及它们为什么有用，所以这让我们可以方便地传输

**[00:00:13]** this allows us conveniently to transfer
> 这让我们可以方便地传输

**[00:00:13]** this allows us conveniently to transfer all the things in a module that we want
> 这让我们可以方便地传输模块中所有我们想要

**[00:00:13]** all the things in a module that we want
> 模块中所有我们想要

**[00:00:13]** all the things in a module that we want on a GPU to the GPU without it parts of
> 模块中所有我们想要放到GPU上的内容，而不会出现部分

**[00:00:13]** on a GPU to the GPU without it parts of
> 放到GPU上的内容，而不会出现部分

**[00:00:13]** on a GPU to the GPU without it parts of it will be transferred other parts won't
> 放到GPU上的内容，而不会出现部分被传输、其他部分不被传输的情况

**[00:00:13]** it will be transferred other parts won't
> 被传输、其他部分不被传输的情况

**[00:00:13]** it will be transferred other parts won't and then yeah the code won't work as
> 被传输、其他部分不被传输的情况，然后代码就无法按预期运行

**[00:00:13]** and then yeah the code won't work as
> 然后代码就无法按预期运行

**[00:00:13]** and then yeah the code won't work as intended so yeah I hope this was useful
> 然后代码就无法按预期运行，所以是的，我希望这有用

**[00:00:13]** intended so yeah I hope this was useful
> 所以是的，我希望这有用

**[00:00:13]** intended so yeah I hope this was useful a short and sweet I guess lesson on
> 所以是的，我希望这是一个简短而精炼的课程，关于

**[00:00:13]** a short and sweet I guess lesson on
> 一个简短而精炼的课程，关于

**[00:00:13]** a short and sweet I guess lesson on understanding py torch buffers I hope
> 一个简短而精炼的课程，关于理解PyTorch buffers，我希望

**[00:00:13]** understanding py torch buffers I hope
> 理解PyTorch buffers，我希望

**[00:00:13]** understanding py torch buffers I hope yeah you F this useful
> 理解PyTorch buffers，我希望你觉得这有用

---

## 🎯 关键要点

- [ ] 要点 1
- [ ] 要点 2
- [ ] 要点 3
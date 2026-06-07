# 第 1 章：LLM 开发生命周期概述（视频笔记）

> 🎬 [原视频](https://www.youtube.com/watch?v=kPGTx4wcm_w)
> 📅 中英双语字幕（DeepSeek 翻译）

---

## 中英双语字幕

**[00:00:00]** yeah hi everyone so today I want to talk
> 大家好，今天我想谈谈

**[00:00:00]** yeah hi everyone so today I want to talk
> 大家好，今天我想谈谈

**[00:00:00]** yeah hi everyone so today I want to talk about developing an llm a large language
> 大家好，今天我想谈谈开发一个LLM，一个大型语言

**[00:00:00]** about developing an llm a large language
> 关于开发一个LLM，一个大型语言

**[00:00:00]** about developing an llm a large language model and in particular I want to talk
> 关于开发一个LLM，一个大型语言模型，特别想谈谈

**[00:00:00]** model and in particular I want to talk
> 模型，特别想谈谈

**[00:00:00]** model and in particular I want to talk about the three stages building training
> 模型，特别想谈谈构建、训练

**[00:00:00]** about the three stages building training
> 关于构建、训练

**[00:00:00]** about the three stages building training and fine-tuning large language models so
> 关于构建、训练和微调大型语言模型的三个阶段

**[00:00:00]** and fine-tuning large language models so
> 和微调大型语言模型的三个阶段

**[00:00:00]** and fine-tuning large language models so but before we begin with the building
> 和微调大型语言模型的三个阶段，但在开始构建

**[00:00:00]** but before we begin with the building
> 但在开始构建

**[00:00:00]** but before we begin with the building aspect of large language models I wanted
> 但在开始构建大型语言模型之前，我想

**[00:00:00]** aspect of large language models I wanted
> 大型语言模型之前，我想

**[00:00:00]** aspect of large language models I wanted to briefly go over the different use
> 大型语言模型之前，我想简要回顾一下不同的使用

**[00:00:00]** to briefly go over the different use
> 简要回顾一下不同的使用

**[00:00:00]** to briefly go over the different use cases or the different ways we are using
> 简要回顾一下不同的使用场景或我们使用

**[00:00:00]** cases or the different ways we are using
> 场景或我们使用

**[00:00:00]** cases or the different ways we are using large language models so the yeah maybe
> 场景或我们使用大型语言模型的不同方式，所以，嗯，也许

**[00:00:00]** large language models so the yeah maybe
> 大型语言模型的不同方式，所以，嗯，也许

**[00:00:00]** large language models so the yeah maybe most popular way to use large language
> 大型语言模型的不同方式，所以，嗯，也许目前最流行的使用大型语言

**[00:00:00]** most popular way to use large language
> 最流行的使用大型语言

**[00:00:00]** most popular way to use large language models these days is via a public or
> 最流行的使用大型语言模型的方式是通过公共或

**[00:00:00]** models these days is via a public or
> 模型的方式是通过公共或

**[00:00:00]** models these days is via a public or proprietary service like a public API or
> 模型的方式是通过公共或专有服务，比如公共API或

**[00:00:00]** proprietary service like a public API or
> 专有服务，比如公共API或

**[00:00:00]** proprietary service like a public API or proprietary um you know like API so for
> 专有服务，比如公共API或专有的，嗯，你知道，像API，所以

**[00:00:00]** proprietary um you know like API so for
> 专有的，嗯，你知道，像API，所以

**[00:00:00]** proprietary um you know like API so for example chat GPT or um yeah perplexity
> 专有的，嗯，你知道，像API，例如Chat GPT或，嗯，是的，Perplexity

**[00:00:00]** example chat GPT or um yeah perplexity
> 例如Chat GPT或，嗯，是的，Perplexity

**[00:00:00]** example chat GPT or um yeah perplexity Gemini and um others so we would
> 例如Chat GPT或，嗯，是的，Perplexity、Gemini等等，我们会

**[00:00:00]** Gemini and um others so we would
> Gemini等等，我们会

**[00:00:00]** Gemini and um others so we would basically go to website ask um or give
> Gemini等等，我们会基本上访问网站，询问或给出

**[00:00:00]** basically go to website ask um or give
> 基本上访问网站，询问或给出

**[00:00:00]** basically go to website ask um or give the LM instruction or ask at a question
> 基本上访问网站，询问或给出LLM指令或提问

**[00:00:00]** the LM instruction or ask at a question
> LLM指令或提问

**[00:00:00]** the LM instruction or ask at a question and then it uh yeah Returns the answer
> LLM指令或提问，然后它，嗯，返回答案

**[00:00:00]** and then it uh yeah Returns the answer
> 然后它，嗯，返回答案

**[00:00:00]** and then it uh yeah Returns the answer so this would be one way of using a lot
> 然后它，嗯，返回答案，所以这是使用大型语言

**[00:00:01]** so this would be one way of using a lot
> 所以这是使用大型语言

**[00:00:01]** so this would be one way of using a lot language model the other one is yeah
> 所以这是使用大型语言模型的一种方式，另一种是，嗯，

**[00:00:01]** language model the other one is yeah
> 模型的一种方式，另一种是，嗯，

**[00:00:01]** language model the other one is yeah nowaday is also very popular thanks to
> 模型的一种方式，另一种是，嗯，现在也非常流行，感谢

**[00:00:01]** nowaday is also very popular thanks to
> 现在也非常流行，感谢

**[00:00:01]** nowaday is also very popular thanks to open source and open weights so that's
> 现在也非常流行，感谢开源和开放权重，所以那是

**[00:00:01]** open source and open weights so that's
> 开源和开放权重，所以那是

**[00:00:01]** open source and open weights so that's running a custom large language model
> 开源和开放权重，所以那是本地运行自定义大型语言模型

**[00:00:01]** running a custom large language model
> 本地运行自定义大型语言模型

**[00:00:01]** running a custom large language model locally so here's an example running a
> 本地运行自定义大型语言模型，这里有一个例子，运行一个

**[00:00:01]** locally so here's an example running a
> 本地运行自定义大型语言模型，这里有一个例子，运行一个

**[00:00:01]** locally so here's an example running a llama 3 Model 8 billion Lama 3 model and
> 本地运行自定义大型语言模型，这里有一个例子，运行一个Llama 3模型，80亿参数的Llama 3模型，并且

**[00:00:01]** llama 3 Model 8 billion Lama 3 model and
> Llama 3模型，80亿参数的Llama 3模型，并且

**[00:00:01]** llama 3 Model 8 billion Lama 3 model and yeah I can also similarly give it my
> Llama 3模型，80亿参数的Llama 3模型，并且，嗯，我也可以类似地给它我的

**[00:00:01]** yeah I can also similarly give it my
> 嗯，我也可以类似地给它我的

**[00:00:01]** yeah I can also similarly give it my query and my terminal here and yeah
> 嗯，我也可以类似地给它我的查询，在我的终端这里，并且，嗯，

**[00:00:01]** query and my terminal here and yeah
> 查询，在我的终端这里，并且，嗯，

**[00:00:01]** query and my terminal here and yeah interact with the llm yeah let's say
> 查询，在我的终端这里，并且，嗯，与LLM交互，嗯，比如说

**[00:00:01]** interact with the llm yeah let's say
> 与LLM交互，嗯，比如说

**[00:00:01]** interact with the llm yeah let's say locally and here I'm using a particular
> 与LLM交互，嗯，比如说本地，这里我使用一个特定的

**[00:00:01]** locally and here I'm using a particular
> 本地，这里我使用一个特定的

**[00:00:01]** locally and here I'm using a particular tool called libt that I help um
> 本地，这里我使用一个特定的工具，叫做libt，我帮助，嗯，

**[00:00:01]** tool called libt that I help um
> 工具，叫做libt，我帮助，嗯，

**[00:00:01]** tool called libt that I help um developing so another way then is also
> 工具，叫做libt，我帮助开发的，所以另一种方式也是

**[00:00:01]** developing so another way then is also
> 开发的，所以另一种方式也是

**[00:00:01]** developing so another way then is also deploying a custom large language model
> 开发的，所以另一种方式也是部署自定义大型语言模型

**[00:00:01]** deploying a custom large language model
> 部署自定义大型语言模型

**[00:00:01]** deploying a custom large language model for example um yeah deploying it on an
> 部署自定义大型语言模型，例如，嗯，是的，将其部署在

**[00:00:01]** for example um yeah deploying it on an
> 例如，嗯，是的，将其部署在

**[00:00:01]** for example um yeah deploying it on an external server or web service and then
> 例如，嗯，是的，将其部署在外部服务器或Web服务上，然后

**[00:00:01]** external server or web service and then
> 外部服务器或Web服务上，然后

**[00:00:01]** external server or web service and then we can also use um the llm Via an API
> 外部服务器或Web服务上，然后我们也可以通过API使用LLM

**[00:00:01]** we can also use um the llm Via an API
> 我们也可以通过API使用LLM

**[00:00:01]** we can also use um the llm Via an API and this is um especially useful or
> 我们也可以通过API使用LLM，这，嗯，特别有用或

**[00:00:01]** and this is um especially useful or
> 这，嗯，特别有用或

**[00:00:01]** and this is um especially useful or interesting if we are developing
> 这，嗯，特别有用或有趣，如果我们正在开发

**[00:00:01]** interesting if we are developing
> 有趣，如果我们正在开发

**[00:00:01]** interesting if we are developing products so going back one slide so here
> 有趣，如果我们正在开发产品，所以回到上一张幻灯片，这里

**[00:00:01]** products so going back one slide so here
> 产品，所以回到上一张幻灯片，这里

**[00:00:01]** products so going back one slide so here it's really more like for me as a
> 产品，所以回到上一张幻灯片，这里对我来说更像是

**[00:00:01]** it's really more like for me as a
> 对我来说更像是

**[00:00:01]** it's really more like for me as a customer I can or as a user I can run my
> 对我来说，更像是作为客户或用户，我可以运行自己的

**[00:00:01]** customer I can or as a user I can run my
> 作为客户或用户，我可以运行自己的

**[00:00:01]** customer I can or as a user I can run my own LM and use it and this one is for
> 作为客户或用户，我可以运行自己的LM并使用它，这是一个

**[00:00:02]** own LM and use it and this one is for
> 自己的LM并使用它，这是一个

**[00:00:02]** own LM and use it and this one is for example an option where I can still run
> 自己的LM并使用它，这是一个例子，我仍然可以在本地或服务器上运行

**[00:00:02]** example an option where I can still run
> 例子，我仍然可以在本地或服务器上运行

**[00:00:02]** example an option where I can still run my llm locally or on server but the
> 例子，我仍然可以在本地或服务器上运行我的LLM，但

**[00:00:02]** my llm locally or on server but the
> 我的LLM在本地或服务器上，但

**[00:00:02]** my llm locally or on server but the difference is that we um have an API
> 我的LLM在本地或服务器上，但区别在于我们有一个API

**[00:00:02]** difference is that we um have an API
> 区别在于我们有一个API

**[00:00:02]** difference is that we um have an API endpoint that we can then use for
> 区别在于我们有一个API端点，然后我们可以用于

**[00:00:02]** endpoint that we can then use for
> 端点，然后我们可以用于

**[00:00:02]** endpoint that we can then use for example in an application it could be uh
> 端点，然后我们可以用于例如应用程序中，可能是

**[00:00:02]** example in an application it could be uh
> 例如应用程序中，可能是

**[00:00:02]** example in an application it could be uh user interface like cat GPT like the UI
> 例如应用程序中，可能是用户界面，比如cat GPT这样的UI

**[00:00:02]** user interface like cat GPT like the UI
> 用户界面，比如cat GPT这样的UI

**[00:00:02]** user interface like cat GPT like the UI but it could also be uh iPhone app or
> 用户界面，比如cat GPT这样的UI，但也可能是iPhone应用或

**[00:00:02]** but it could also be uh iPhone app or
> 但也可能是iPhone应用或

**[00:00:02]** but it could also be uh iPhone app or something like that so I would say yeah
> 但也可能是iPhone应用或类似的东西，所以我会说，是的

**[00:00:02]** something like that so I would say yeah
> 类似的东西，所以我会说，是的

**[00:00:02]** something like that so I would say yeah these are the three common ways of how
> 类似的东西，所以我会说，是的，这是三种常见的方式

**[00:00:02]** these are the three common ways of how
> 这是三种常见的方式

**[00:00:02]** these are the three common ways of how we are using llms like the typical ways
> 这是三种常见的方式，我们如何使用LLM，比如典型的

**[00:00:02]** we are using llms like the typical ways
> 我们如何使用LLM，比如典型的

**[00:00:02]** we are using llms like the typical ways to interact with them um they are all
> 我们如何使用LLM，比如典型的交互方式，它们都是

**[00:00:02]** to interact with them um they are all
> 交互方式，它们都是

**[00:00:02]** to interact with them um they are all different use cases and they also have
> 交互方式，它们都是不同的用例，并且也有

**[00:00:02]** different use cases and they also have
> 不同的用例，并且也有

**[00:00:02]** different use cases and they also have different tradeoffs personally I use all
> 不同的用例，并且也有不同的权衡。我个人使用所有

**[00:00:02]** different tradeoffs personally I use all
> 不同的权衡。我个人使用所有

**[00:00:02]** different tradeoffs personally I use all three of them so for different you know
> 不同的权衡。我个人使用所有这三种，所以针对不同的

**[00:00:02]** three of them so for different you know
> 这三种，所以针对不同的

**[00:00:02]** three of them so for different you know for different tasks or for different
> 这三种，所以针对不同的任务或不同的

**[00:00:02]** for different tasks or for different
> 任务或不同的

**[00:00:02]** for different tasks or for different goals I use different approaches and
> 任务或不同的目标，我使用不同的方法，并且

**[00:00:02]** goals I use different approaches and
> 目标，我使用不同的方法，并且

**[00:00:02]** goals I use different approaches and there is no right or wrong and there is
> 目标，我使用不同的方法，没有对错之分，也没有

**[00:00:02]** there is no right or wrong and there is
> 没有对错之分，也没有

**[00:00:02]** there is no right or wrong and there is no better or worse they all have
> 没有对错之分，也没有好坏之别，它们都有

**[00:00:02]** no better or worse they all have
> 好坏之别，它们都有

**[00:00:02]** no better or worse they all have different kinds of tradeoffs so um but
> 好坏之别，它们都有不同类型的权衡，所以

**[00:00:02]** different kinds of tradeoffs so um but
> 不同类型的权衡，所以

**[00:00:02]** different kinds of tradeoffs so um but today the talk is not about yeah the
> 不同类型的权衡，所以今天的演讲不是关于这些用法的

**[00:00:02]** today the talk is not about yeah the
> 今天的演讲不是关于这些用法的

**[00:00:02]** today the talk is not about yeah the different trade-offs of these usages I
> 今天的演讲不是关于这些用法的不同权衡，我

**[00:00:02]** different trade-offs of these usages I
> 不同权衡，我

**[00:00:02]** different trade-offs of these usages I wanted to yeah pull the curtain back a
> 不同权衡，我想稍微揭开帷幕

**[00:00:02]** wanted to yeah pull the curtain back a
> 想稍微揭开帷幕

**[00:00:02]** wanted to yeah pull the curtain back a bit uh and talk about what goes into
> 想稍微揭开帷幕，谈谈开发LLM需要什么，这样我们才能运行

**[00:00:03]** bit uh and talk about what goes into
> 谈谈开发LLM需要什么，这样我们才能运行

**[00:00:03]** bit uh and talk about what goes into developing llms that we can then run
> 谈谈开发LLM需要什么，这样我们才能运行，就像我之前展示的那样，所以

**[00:00:03]** developing llms that we can then run
> 就像我之前展示的那样，所以

**[00:00:03]** developing llms that we can then run like I've shown you before so what goes
> 就像我之前展示的那样，所以首先创建那个LLM需要什么

**[00:00:03]** like I've shown you before so what goes
> 首先创建那个LLM需要什么

**[00:00:03]** like I've shown you before so what goes into creating that llm in the first
> 首先创建那个LLM需要什么，特别是我想谈谈

**[00:00:03]** into creating that llm in the first
> 特别是我想谈谈

**[00:00:03]** into creating that llm in the first place in particular I want to talk about
> 特别是我想谈谈开发LLM的这些不同阶段

**[00:00:03]** place in particular I want to talk about
> 这些不同阶段

**[00:00:03]** place in particular I want to talk about these different stages of developing an
> 这些不同阶段，所以这包括构建LLM

**[00:00:03]** these different stages of developing an
> 所以这包括构建LLM

**[00:00:03]** these different stages of developing an llm so this involves building the llm
> 所以这包括构建LLM本身，所以这里我们必须准备

**[00:00:03]** llm so this involves building the llm
> 本身，所以这里我们必须准备

**[00:00:03]** llm so this involves building the llm itself so here we have to prepare the
> 本身，所以这里我们必须准备数据、采样数据集等等

**[00:00:03]** itself so here we have to prepare the
> 数据、采样数据集等等

**[00:00:03]** itself so here we have to prepare the data sampling the data set and so forth
> 数据、采样数据集等等，我们必须做一些编码，实现这个

**[00:00:03]** data sampling the data set and so forth
> 我们必须做一些编码，实现这个

**[00:00:03]** data sampling the data set and so forth we have to do some coding implement this
> 我们必须做一些编码，实现这个attention机制，它是LLM的核心

**[00:00:03]** we have to do some coding implement this
> attention机制，它是LLM的核心

**[00:00:03]** we have to do some coding implement this attention mechanism that is at the heart
> attention机制，它是LLM的核心，可以说是LLM的引擎

**[00:00:03]** attention mechanism that is at the heart
> 可以说是LLM的引擎

**[00:00:03]** attention mechanism that is at the heart of the llm it's essentially the motor if
> 可以说是LLM的引擎，然后当然

**[00:00:03]** of the llm it's essentially the motor if
> 然后当然

**[00:00:03]** of the llm it's essentially the motor if you will of the llm and then of course
> 然后当然还要编码整个架构

**[00:00:03]** you will of the llm and then of course
> 还要编码整个架构

**[00:00:03]** you will of the llm and then of course also coding the whole architecture
> 还要编码整个架构，所以如果attention机制

**[00:00:03]** also coding the whole architecture
> 所以如果attention机制

**[00:00:03]** also coding the whole architecture around it so if the attention mechanism
> 所以如果attention机制是汽车的引擎，那么LM

**[00:00:03]** around it so if the attention mechanism
> 是汽车的引擎，那么LM

**[00:00:03]** around it so if the attention mechanism is let's say the motor of a car the LM
> around it so if the attention mechanism is let's say the motor of a car the LM

**[00:00:03]** is let's say the motor of a car the LM
> is let's say the motor of a car the LM

**[00:00:03]** is let's say the motor of a car the LM architecture would be basically
> 比如说，如果把LM架构比作汽车的发动机

**[00:00:03]** architecture would be basically
> 架构基本上就是

**[00:00:03]** architecture would be basically everything surrounding that motor like
> 架构基本上就是围绕发动机的一切，比如

**[00:00:03]** everything surrounding that motor like
> 围绕发动机的一切，比如

**[00:00:03]** everything surrounding that motor like connecting the
> 围绕发动机的一切，比如连接

**[00:00:03]** connecting the
> 连接

**[00:00:03]** connecting the wheels putting driver seats in a
> 连接车轮、安装驾驶座

**[00:00:03]** wheels putting driver seats in a
> 车轮、安装驾驶座

**[00:00:03]** wheels putting driver seats in a steering wheel and everything
> 车轮、安装驾驶座、方向盘等等

**[00:00:03]** steering wheel and everything
> 方向盘等等

**[00:00:03]** steering wheel and everything else um the second stage then is the
> 方向盘等等其他部分。那么第二阶段就是

**[00:00:03]** else um the second stage then is the
> 那么第二阶段就是

**[00:00:03]** else um the second stage then is the pre-training stage um so this would be
> 那么第二阶段就是预训练阶段。这基本上就是

**[00:00:03]** pre-training stage um so this would be
> 预训练阶段。这基本上就是

**[00:00:03]** pre-training stage um so this would be essentially taking an large length Model
> 预训练阶段。这基本上就是拿一个大型语言模型

**[00:00:04]** essentially taking an large length Model
> 拿一个大型语言模型

**[00:00:04]** essentially taking an large length Model A large language model architecture and
> 拿一个大型语言模型架构

**[00:00:04]** A large language model architecture and
> 大型语言模型架构

**[00:00:04]** A large language model architecture and then training it on a data set for that
> 大型语言模型架构，然后在数据集上训练它。为此

**[00:00:04]** then training it on a data set for that
> 然后在数据集上训练它。为此

**[00:00:04]** then training it on a data set for that we have to implement the training Loop
> 然后在数据集上训练它。为此，我们需要实现训练循环

**[00:00:04]** we have to implement the training Loop
> 我们需要实现训练循环

**[00:00:04]** we have to implement the training Loop the model evaluation and um usually we
> 我们需要实现训练循环、模型评估，通常我们

**[00:00:04]** the model evaluation and um usually we
> 模型评估，通常我们

**[00:00:04]** the model evaluation and um usually we also want a mechanism to save and load
> 模型评估，通常我们还需要一个机制来保存和加载

**[00:00:04]** also want a mechanism to save and load
> 还需要一个机制来保存和加载

**[00:00:04]** also want a mechanism to save and load um pre-trained weights so that we can
> 还需要一个机制来保存和加载预训练权重，这样我们就能

**[00:00:04]** um pre-trained weights so that we can
> 预训练权重，这样我们就能

**[00:00:04]** um pre-trained weights so that we can use the large language model later
> 预训练权重，这样我们就能在之后使用这个大型语言模型

**[00:00:04]** use the large language model later
> 在之后使用这个大型语言模型

**[00:00:04]** use the large language model later because usually this um stage is what
> 在之后使用这个大型语言模型。因为通常这个阶段

**[00:00:04]** because usually this um stage is what
> 因为通常这个阶段

**[00:00:04]** because usually this um stage is what forms a so-called Foundation model so
> 因为通常这个阶段会形成所谓的Foundation模型

**[00:00:04]** forms a so-called Foundation model so
> 会形成所谓的Foundation模型

**[00:00:04]** forms a so-called Foundation model so it's usually not our end product it's
> 会形成所谓的Foundation模型。所以它通常不是我们的最终产品

**[00:00:04]** it's usually not our end product it's
> 它通常不是我们的最终产品

**[00:00:04]** it's usually not our end product it's like an intermediate um yeah I would say
> 它通常不是我们的最终产品，而是这个流程中的一个中间产品

**[00:00:04]** like an intermediate um yeah I would say
> 而是这个流程中的一个中间产品

**[00:00:04]** like an intermediate um yeah I would say product in in this pipeline but in
> 而是这个流程中的一个中间产品。但总的来说，它更像是我们之后用于

**[00:00:04]** product in in this pipeline but in
> 但总的来说，它更像是我们之后用于

**[00:00:04]** product in in this pipeline but in general it's more like the foundation or
> 但总的来说，它更像是我们之后用于微调的基础或基座模型

**[00:00:04]** general it's more like the foundation or
> 微调的基础或基座模型

**[00:00:04]** general it's more like the foundation or the base model that we use then for
> 微调的基础或基座模型。而微调

**[00:00:04]** the base model that we use then for
> 而微调

**[00:00:04]** the base model that we use then for fine-tuning and so fine-tuning um yeah
> 而微调可以是不同的事情。比如我们可以

**[00:00:04]** fine-tuning and so fine-tuning um yeah
> 比如我们可以

**[00:00:04]** fine-tuning and so fine-tuning um yeah that can be different things so we could
> 比如我们可以微调一个模型来做分类

**[00:00:04]** that can be different things so we could
> 微调一个模型来做分类

**[00:00:04]** that can be different things so we could for example fine tune a model to do
> 微调一个模型来做分类，比如文本分类

**[00:00:04]** for example fine tune a model to do
> 比如文本分类

**[00:00:04]** for example fine tune a model to do classification you know like text
> 比如文本分类，识别垃圾邮件

**[00:00:04]** classification you know like text
> 识别垃圾邮件

**[00:00:04]** classification you know like text categorization um yeah identifying spam
> 识别垃圾邮件这类事情。但我们也可以

**[00:00:04]** categorization um yeah identifying spam
> 但我们也可以

**[00:00:04]** categorization um yeah identifying spam email and these types of things but we
> 但我们也可以训练一个个人助手或聊天机器人

**[00:00:04]** email and these types of things but we
> 训练一个个人助手或聊天机器人

**[00:00:04]** email and these types of things but we can also yeah train a personal assistant
> 训练一个个人助手或聊天机器人。为此，我们可以使用一个指令数据集

**[00:00:04]** can also yeah train a personal assistant
> 为此，我们可以使用一个指令数据集

**[00:00:04]** can also yeah train a personal assistant a chatbot for that we can use an
> 为此，我们可以使用一个指令数据集。顺便说一下，我展示的所有图表

**[00:00:04]** a chatbot for that we can use an
> 顺便说一下，我展示的所有图表

**[00:00:04]** a chatbot for that we can use an instruction data set for
> 顺便说一下，我展示的所有图表都来自我的书《从零开始构建大型语言模型》

**[00:00:04]** instruction data set for
> 都来自我的书《从零开始构建大型语言模型》

**[00:00:04]** instruction data set for example so um by the way all the figures
> 都来自我的书《从零开始构建大型语言模型》。这些图表是我为这本书制作的

**[00:00:04]** example so um by the way all the figures
> 这些图表是我为这本书制作的

**[00:00:04]** example so um by the way all the figures I'm showing you are for my book uh build
> 这些图表是我为这本书制作的，所以这里只是重复使用它们

**[00:00:05]** I'm showing you are for my book uh build
> 所以这里只是重复使用它们

**[00:00:05]** I'm showing you are for my book uh build a large language model from scratch so
> 所以这里只是重复使用它们，虽然不是全部，但很多都来自那本书

**[00:00:05]** a large language model from scratch so
> 虽然不是全部，但很多都来自那本书

**[00:00:05]** a large language model from scratch so um I made all these figures for my book
> 虽然不是全部，但很多都来自那本书。书中还有更多代码

**[00:00:05]** um I made all these figures for my book
> 书中还有更多代码

**[00:00:05]** um I made all these figures for my book so I'm just reusing them here or not all
> 书中还有更多代码，如果你真的想构建的话。因为这次演讲只是概念性的

**[00:00:05]** so I'm just reusing them here or not all
> 因为这次演讲只是概念性的

**[00:00:05]** so I'm just reusing them here or not all of them but many are from the book um
> 因为这次演讲只是概念性的，所以我在这里

**[00:00:05]** of them but many are from the book um
> of them but many are from the book um

**[00:00:05]** of them but many are from the book um and in the book is also more code if you
> of them but many are from the book um and in the book is also more code if you

**[00:00:05]** and in the book is also more code if you
> and in the book is also more code if you

**[00:00:05]** and in the book is also more code if you actually want to build it because this
> and in the book is also more code if you actually want to build it because this

**[00:00:05]** actually want to build it because this
> actually want to build it because this

**[00:00:05]** actually want to build it because this talk is only conceptual so here I'm
> actually want to build it because this talk is only conceptual so here I'm

**[00:00:05]** talk is only conceptual so here I'm
> talk is only conceptual so here I'm

**[00:00:05]** talk is only conceptual so here I'm going over the concepts and if you're
> 目前只是概念性的讲解，所以我将介绍这些概念，如果你

**[00:00:05]** going over the concepts and if you're
> 正在学习这些概念，并且如果你

**[00:00:05]** going over the concepts and if you're interested in coding something like this
> 正在学习这些概念，并且如果你有兴趣编写类似这样的代码

**[00:00:05]** interested in coding something like this
> 有兴趣编写类似这样的代码

**[00:00:05]** interested in coding something like this yourself um you have more details in the
> 有兴趣编写类似这样的代码，那么你可以在

**[00:00:05]** yourself um you have more details in the
> 书中找到更多细节，同时我还有一个GitHub仓库

**[00:00:05]** yourself um you have more details in the book and also I have a GitHub repository
> 书中找到更多细节，同时我还有一个GitHub仓库

**[00:00:05]** book and also I have a GitHub repository
> 书中找到更多细节，同时我还有一个GitHub仓库，里面包含所有代码示例。所以从

**[00:00:05]** book and also I have a GitHub repository with all the code examples so starting
> 所有代码示例开始，所以从

**[00:00:05]** with all the code examples so starting
> 所有代码示例开始，所以从第一阶段开始，这里构建

**[00:00:05]** with all the code examples so starting with stage one the building here so here
> 第一阶段开始，这里构建

**[00:00:05]** with stage one the building here so here
> 第一阶段开始，这里构建，我们将组装LLM

**[00:00:05]** with stage one the building here so here we will be putting together the llm
> 我们将组装LLM

**[00:00:05]** we will be putting together the llm
> 我们将组装LLM架构本身，但在实现

**[00:00:05]** we will be putting together the llm architecture itself but um even taking a
> 架构本身之前，我们甚至需要退一步

**[00:00:05]** architecture itself but um even taking a
> 架构本身之前，我们甚至需要退一步

**[00:00:05]** architecture itself but um even taking a step back before we implement the
> 架构本身之前，我们甚至需要退一步，先看看

**[00:00:05]** step back before we implement the
> 先看看

**[00:00:05]** step back before we implement the architecture we will take a look at the
> 先看看数据集，数据集看起来

**[00:00:05]** architecture we will take a look at the
> 数据集看起来

**[00:00:05]** architecture we will take a look at the data set how the data set um looks like
> 数据集看起来是什么样的，或者数据集长什么样，然后

**[00:00:05]** data set how the data set um looks like
> 或者数据集长什么样，然后

**[00:00:05]** data set how the data set um looks like or how a data set looks like and then
> 或者数据集长什么样，然后我们如何将该数据集输入到大型

**[00:00:05]** or how a data set looks like and then
> 我们如何将该数据集输入到大型

**[00:00:05]** or how a data set looks like and then how we feed that data set into the large
> 我们如何将该数据集输入到大型语言模型中。实际上，这在我看来是

**[00:00:05]** how we feed that data set into the large
> 在我看来是

**[00:00:05]** how we feed that data set into the large language model and this is actually um
> 在我看来是理解LLM工作原理的一个好方法，因为

**[00:00:05]** language model and this is actually um
> 理解LLM工作原理的一个好方法，因为

**[00:00:05]** language model and this is actually um in my opinion a good way of yeah of
> 理解LLM工作原理的一个好方法，因为理解LLM的工作原理在某种程度上

**[00:00:05]** in my opinion a good way of yeah of
> 理解LLM的工作原理在某种程度上

**[00:00:05]** in my opinion a good way of yeah of understanding how an llm works because
> 理解LLM的工作原理在某种程度上需要理解它处理什么

**[00:00:05]** understanding how an llm works because
> 需要理解它处理什么

**[00:00:05]** understanding how an llm works because understanding how an llm Works kind of
> 需要理解它处理什么，数据集是什么样的。所以

**[00:00:05]** understanding how an llm Works kind of
> 数据集是什么样的。所以

**[00:00:05]** understanding how an llm Works kind of requires to understand what it works
> 数据集是什么样的。所以如果我们理解如何将数据

**[00:00:06]** requires to understand what it works
> 如果我们理解如何将数据

**[00:00:06]** requires to understand what it works with what does the data set look like so
> 如果我们理解如何将数据输入到LLM中，我们就更接近

**[00:00:06]** with what does the data set look like so
> 输入到LLM中，我们就更接近

**[00:00:06]** with what does the data set look like so if we understand um how we feed the data
> 输入到LLM中，我们就更接近理解LLM架构的机制

**[00:00:06]** if we understand um how we feed the data
> 理解LLM架构的机制

**[00:00:06]** if we understand um how we feed the data into the llm we are already closer to
> 理解LLM架构的机制等等。所以

**[00:00:06]** into the llm we are already closer to
> 等等。所以

**[00:00:06]** into the llm we are already closer to understanding yeah how the mechanism the
> 等等。所以这就是为什么在最左边这里，我们从

**[00:00:06]** understanding yeah how the mechanism the
> 最左边这里，我们从

**[00:00:06]** understanding yeah how the mechanism the llm architecture works and so forth um
> 最左边这里，我们从数据集准备和采样开始

**[00:00:06]** llm architecture works and so forth um
> 数据集准备和采样开始

**[00:00:06]** llm architecture works and so forth um so yeah this is the reason why at the
> 数据集准备和采样开始，在真正进行

**[00:00:06]** so yeah this is the reason why at the
> 解释它，在真正进行

**[00:00:06]** so yeah this is the reason why at the very left here we are starting with the
> 解释它，在真正进行预训练之前解释它，因为我确实认为这

**[00:00:06]** very left here we are starting with the
> 预训练之前解释它，因为我确实认为这

**[00:00:06]** very left here we are starting with the data set preparation and sampling um
> 预训练之前解释它，因为我确实认为这有助于理解LLM的工作原理。所以

**[00:00:06]** data set preparation and sampling um
> 有助于理解LLM的工作原理。所以

**[00:00:06]** data set preparation and sampling um explaining it um before we actually do
> 有助于理解LLM的工作原理。所以LLM本质上

**[00:00:06]** explaining it um before we actually do
> LLM本质上

**[00:00:06]** explaining it um before we actually do the pre-training because I do think it
> LLM本质上你可以把它看作一个深度神经

**[00:00:06]** the pre-training because I do think it
> 你可以把它看作一个深度神经

**[00:00:06]** the pre-training because I do think it helps um understanding a bit um how an
> 你可以把它看作一个深度神经网络，如果你之前做过深度学习

**[00:00:06]** helps um understanding a bit um how an
> 网络，如果你之前做过深度学习

**[00:00:06]** helps um understanding a bit um how an llm works so um yeah LM is essentially
> 网络，如果你之前做过深度学习的话。特别是，我认为如果你

**[00:00:06]** llm works so um yeah LM is essentially
> 的话。特别是，我认为如果你

**[00:00:06]** llm works so um yeah LM is essentially you can think of it as a deep neural
> 的话。特别是，我认为如果你正在观看这个视频，你可能已经看过

**[00:00:06]** you can think of it as a deep neural
> 正在观看这个视频，你可能已经看过

**[00:00:06]** you can think of it as a deep neural network if you have done deep learning
> 正在观看这个视频，你可能已经看过我在YouTube上的其他一些视频，所以

**[00:00:06]** network if you have done deep learning
> 我在YouTube上的其他一些视频，所以

**[00:00:06]** network if you have done deep learning before so especially I think if you're
> 我在YouTube上的其他一些视频，所以你可能有一些机器学习和深度学习的背景

**[00:00:06]** before so especially I think if you're
> 你可能有一些机器学习和深度学习的背景

**[00:00:06]** before so especially I think if you're watching this video you have maybe seen
> 你可能有一些机器学习和深度学习的背景。所以目前，我认为

**[00:00:06]** watching this video you have maybe seen
> 所以目前，我认为

**[00:00:06]** watching this video you have maybe seen some of my other videos on YouTube so
> 所以目前，我认为思考大型语言模型最简单的方式

**[00:00:06]** some of my other videos on YouTube so
> 思考大型语言模型最简单的方式

**[00:00:06]** some of my other videos on YouTube so you probably um have some background in
> 思考大型语言模型最简单的方式就是把它看作一个大型深度

**[00:00:06]** you probably um have some background in
> 就是把它看作一个大型深度

**[00:00:06]** you probably um have some background in machine learning and deep learning so
> 就是把它看作一个大型深度神经网络

**[00:00:06]** machine learning and deep learning so
> 神经网络

**[00:00:06]** machine learning and deep learning so for now I would say um the simplest way
> 神经网络

**[00:00:06]** for now I would say um the simplest way
> for now I would say um the simplest way

**[00:00:06]** for now I would say um the simplest way to think of a large language model is
> for now I would say um the simplest way to think of a large language model is

**[00:00:06]** to think of a large language model is
> to think of a large language model is

**[00:00:06]** to think of a large language model is really to think of it as a large deep
> to think of a large language model is really to think of it as a large deep

**[00:00:06]** really to think of it as a large deep
> really to think of it as a large deep

**[00:00:06]** really to think of it as a large deep neur Network and this deep new network
> 实际上可以将其视为一个大型深度神经网络，而这个深度神经网络

**[00:00:06]** neur Network and this deep new network
> 神经网络，而这个深度神经网络

**[00:00:06]** neur Network and this deep new network is yeah uh model that is trained to
> 神经网络，而这个深度神经网络是一个经过训练的模型，用于

**[00:00:07]** is yeah uh model that is trained to
> 是一个经过训练的模型，用于

**[00:00:07]** is yeah uh model that is trained to predict the next word in the text so
> 是一个经过训练的模型，用于预测文本中的下一个词，所以

**[00:00:07]** predict the next word in the text so
> 预测文本中的下一个词，所以

**[00:00:07]** predict the next word in the text so that's the first stage when we do the
> 预测文本中的下一个词，这就是我们进行预训练时的第一阶段

**[00:00:07]** that's the first stage when we do the
> 这就是我们进行预训练时的第一阶段

**[00:00:07]** that's the first stage when we do the pre-training we train the model to
> 这就是我们进行预训练时的第一阶段，我们训练模型来

**[00:00:07]** pre-training we train the model to
> 我们训练模型来

**[00:00:07]** pre-training we train the model to predict just the next word in a sentence
> 我们训练模型来预测句子中的下一个词

**[00:00:07]** predict just the next word in a sentence
> 预测句子中的下一个词

**[00:00:07]** predict just the next word in a sentence or in a
> 预测句子中的下一个词，或者

**[00:00:07]** or in a
> 或者

**[00:00:07]** or in a text and yeah we call that also
> 或者文本中的下一个词，我们有时也称之为

**[00:00:07]** text and yeah we call that also
> 文本中的下一个词，我们有时也称之为

**[00:00:07]** text and yeah we call that also sometimes the next word prediction task
> 文本中的下一个词，我们有时也称之为下一个词预测任务

**[00:00:07]** sometimes the next word prediction task
> 下一个词预测任务

**[00:00:07]** sometimes the next word prediction task but um essentially it's the next token
> 下一个词预测任务，但本质上它是下一个token

**[00:00:07]** but um essentially it's the next token
> 但本质上它是下一个token

**[00:00:07]** but um essentially it's the next token prediction task and I will Al explain in
> 但本质上它是下一个token预测任务，我稍后会解释

**[00:00:07]** prediction task and I will Al explain in
> 预测任务，我稍后会解释

**[00:00:07]** prediction task and I will Al explain in a bit what the token is so just to show
> 预测任务，我稍后会解释什么是token，所以为了展示

**[00:00:07]** a bit what the token is so just to show
> 什么是token，所以为了展示

**[00:00:07]** a bit what the token is so just to show you what that looks like so if we have
> 什么是token，所以为了展示它的样子，如果我们有

**[00:00:07]** you what that looks like so if we have
> 它的样子，如果我们有

**[00:00:07]** you what that looks like so if we have an example text here at the top um the
> 它的样子，如果我们有顶部的一个示例文本，嗯

**[00:00:07]** an example text here at the top um the
> 顶部的一个示例文本，嗯

**[00:00:07]** an example text here at the top um the example text is llms learn to predict
> 顶部的一个示例文本，示例文本是“LLMs learn to predict”

**[00:00:07]** example text is llms learn to predict
> 示例文本是“LLMs learn to predict”

**[00:00:07]** example text is llms learn to predict one word at a time so I want to train
> 示例文本是“LLMs learn to predict one word at a time”，所以我想训练

**[00:00:07]** one word at a time so I want to train
> “one word at a time”，所以我想训练

**[00:00:07]** one word at a time so I want to train let's say the model on this text of
> “one word at a time”，所以我想训练模型在这个文本上，当然

**[00:00:07]** let's say the model on this text of
> 模型在这个文本上，当然

**[00:00:07]** let's say the model on this text of course the real world data set is much
> 模型在这个文本上，当然真实世界的数据集要大得多

**[00:00:07]** course the real world data set is much
> 真实世界的数据集要大得多

**[00:00:07]** course the real world data set is much much larger it's billions of words but
> 真实世界的数据集要大得多，有数十亿个词，但

**[00:00:07]** much larger it's billions of words but
> 有数十亿个词，但

**[00:00:07]** much larger it's billions of words but just you know to give a example that
> 有数十亿个词，但只是为了给你一个适合这个幻灯片的例子

**[00:00:07]** just you know to give a example that
> 只是为了给你一个适合这个幻灯片的例子

**[00:00:07]** just you know to give a example that fits into this um slide so on the left
> 只是为了给你一个适合这个幻灯片的例子，所以在左侧

**[00:00:07]** fits into this um slide so on the left
> 所以在左侧

**[00:00:07]** fits into this um slide so on the left hand side if we have a sentence like
> 所以在左侧，如果我们有一个像这样的句子

**[00:00:07]** hand side if we have a sentence like
> 像这样的句子

**[00:00:07]** hand side if we have a sentence like that we take to start with the left word
> 像这样的句子，我们从左边的词开始

**[00:00:07]** that we take to start with the left word
> 我们从左边的词开始

**[00:00:07]** that we take to start with the left word um let's say llms here that is what the
> 我们从左边的词开始，嗯，比如这里的“LLMs”，这就是

**[00:00:07]** um let's say llms here that is what the
> 嗯，比如这里的“LLMs”，这就是

**[00:00:07]** um let's say llms here that is what the llm receives as an input then the learn
> 嗯，比如这里的“LLMs”，这就是LLM接收的输入，然后“learn”

**[00:00:07]** llm receives as an input then the learn
> LLM接收的输入，然后“learn”

**[00:00:07]** llm receives as an input then the learn is the Target that it should predict so
> LLM接收的输入，然后“learn”是它应该预测的目标，所以

**[00:00:08]** is the Target that it should predict so
> 是它应该预测的目标，所以

**[00:00:08]** is the Target that it should predict so that's the next word and everything
> 是它应该预测的目标，所以这就是下一个词，而所有

**[00:00:08]** that's the next word and everything
> 这就是下一个词，而所有

**[00:00:08]** that's the next word and everything right from the target here on the right
> 这就是下一个词，而所有从目标开始右侧的内容

**[00:00:08]** right from the target here on the right
> 从目标开始右侧的内容

**[00:00:08]** right from the target here on the right hand side is For Now hidden from the llm
> 从目标开始右侧的内容目前对LLM是隐藏的

**[00:00:08]** hand side is For Now hidden from the llm
> 目前对LLM是隐藏的

**[00:00:08]** hand side is For Now hidden from the llm so we are just feeding at llms the word
> 目前对LLM是隐藏的，所以我们只向LLM输入“LLMs”这个词

**[00:00:08]** so we are just feeding at llms the word
> 所以我们只向LLM输入“LLMs”这个词

**[00:00:08]** so we are just feeding at llms the word llms and then um it's supposed to
> 所以我们只向LLM输入“LLMs”这个词，然后它应该

**[00:00:08]** llms and then um it's supposed to
> 然后它应该

**[00:00:08]** llms and then um it's supposed to predict the next word learn in this case
> 然后它应该预测下一个词，在这种情况下是“learn”

**[00:00:08]** predict the next word learn in this case
> 预测下一个词，在这种情况下是“learn”

**[00:00:08]** predict the next word learn in this case so um here let's say are two training
> 预测下一个词，在这种情况下是“learn”，所以这里有两个训练

**[00:00:08]** so um here let's say are two training
> 所以这里有两个训练

**[00:00:08]** so um here let's say are two training input examples so the first sample is
> 所以这里有两个训练输入示例，第一个样本是

**[00:00:08]** input examples so the first sample is
> 输入示例，第一个样本是

**[00:00:08]** input examples so the first sample is llms the model um generates or supposed
> 输入示例，第一个样本是“LLMs”，模型生成或应该

**[00:00:08]** llms the model um generates or supposed
> “LLMs”，模型生成或应该

**[00:00:08]** llms the model um generates or supposed to generate learn the second sample is
> “LLMs”，模型生成或应该生成“learn”，第二个样本是

**[00:00:08]** to generate learn the second sample is
> 生成“learn”，第二个样本是

**[00:00:08]** to generate learn the second sample is lm's learn and then the next word it
> 生成“learn”，第二个样本是“LLMs learn”，然后它应该预测的下一个词是“to”，然后我们

**[00:00:08]** lm's learn and then the next word it
> “LLMs learn”，然后它应该预测的下一个词是“to”，然后我们

**[00:00:08]** lm's learn and then the next word it should predict is to and then we
> “LLMs learn”，然后它应该预测的下一个词是“to”，然后我们继续构建数据集

**[00:00:08]** should predict is to and then we
> 继续构建数据集

**[00:00:08]** should predict is to and then we continue on to construct the data set
> 继续构建数据集，这样我们给LLM学习任务

**[00:00:08]** continue on to construct the data set
> 这样我们给LLM学习任务

**[00:00:08]** continue on to construct the data set such that we give the llm learning tasks
> 这样我们给LLM学习任务

**[00:00:08]** such that we give the llm learning tasks
> such that we give the llm learning tasks

**[00:00:08]** such that we give the llm learning tasks where one word is missing and it's
> 因此我们给LLM设计学习任务，其中缺失一个单词

**[00:00:08]** where one word is missing and it's
> 其中缺失一个单词

**[00:00:08]** where one word is missing and it's supposed to generate this next word we I
> 其中缺失一个单词，它需要生成下一个单词

**[00:00:08]** supposed to generate this next word we I
> 需要生成下一个单词

**[00:00:08]** supposed to generate this next word we I will explain later um a bit more let's
> 需要生成下一个单词，我稍后会进一步解释

**[00:00:08]** will explain later um a bit more let's
> 稍后会进一步解释

**[00:00:08]** will explain later um a bit more let's say how how that works how the
> 稍后会进一步解释预训练的工作原理

**[00:00:08]** say how how that works how the
> 工作原理

**[00:00:08]** say how how that works how the pre-training or let's say the training
> 预训练或训练函数的工作原理

**[00:00:08]** pre-training or let's say the training
> 预训练或训练函数

**[00:00:08]** pre-training or let's say the training function works I will say a few more
> 预训练或训练函数的工作原理，我会再多说几句

**[00:00:08]** function works I will say a few more
> 工作原理，我会再多说几句

**[00:00:08]** function works I will say a few more words about that but in general that's
> 工作原理，我会再多说几句，但总的来说

**[00:00:08]** words about that but in general that's
> 但总的来说

**[00:00:08]** words about that but in general that's yeah how we would prepare a data set for
> 但总的来说，这就是我们如何为预训练准备数据集

**[00:00:08]** yeah how we would prepare a data set for
> 这就是我们如何为预训练准备数据集

**[00:00:08]** yeah how we would prepare a data set for pre-training now in practice this would
> 这就是我们如何为预训练准备数据集，在实践中

**[00:00:09]** pre-training now in practice this would
> 预训练，在实践中

**[00:00:09]** pre-training now in practice this would be quite inefficient to feed it just one
> 预训练，在实践中，如果一次只输入一个句子或文本

**[00:00:09]** be quite inefficient to feed it just one
> 如果一次只输入一个

**[00:00:09]** be quite inefficient to feed it just one you know sentence or um text at a time
> 如果一次只输入一个句子或文本，效率会很低

**[00:00:09]** you know sentence or um text at a time
> 句子或文本

**[00:00:09]** you know sentence or um text at a time so in practice what we do is we do a
> 句子或文本，因此实践中我们采用

**[00:00:09]** so in practice what we do is we do a
> 因此实践中我们采用

**[00:00:09]** so in practice what we do is we do a batching like we do in deep learning in
> 因此实践中我们采用批处理，就像深度学习中的做法

**[00:00:09]** batching like we do in deep learning in
> 批处理，就像深度学习中的做法

**[00:00:09]** batching like we do in deep learning in general so what that means is we are
> 批处理，就像深度学习中的一般做法，这意味着

**[00:00:09]** general so what that means is we are
> 一般做法，这意味着

**[00:00:09]** general so what that means is we are putting multiple training inputs
> 一般做法，这意味着我们将多个训练输入

**[00:00:09]** putting multiple training inputs
> 将多个训练输入

**[00:00:09]** putting multiple training inputs together into a batch and usually
> 将多个训练输入组合成一个batch，通常

**[00:00:09]** together into a batch and usually
> 组合成一个batch，通常

**[00:00:09]** together into a batch and usually batches have to have the same length
> 组合成一个batch，通常batch必须具有相同的长度

**[00:00:09]** batches have to have the same length
> batch必须具有相同的长度

**[00:00:09]** batches have to have the same length because they are implemented as um
> batch必须具有相同的长度，因为它们被实现为

**[00:00:09]** because they are implemented as um
> 因为它们被实现为

**[00:00:09]** because they are implemented as um tensors so where we you can think of it
> 因为它们被实现为tensor，你可以将其视为

**[00:00:09]** tensors so where we you can think of it
> tensor，你可以将其视为

**[00:00:09]** tensors so where we you can think of it almost like yeah as a matrix and um so
> tensor，你可以将其视为一个矩阵，因此

**[00:00:09]** almost like yeah as a matrix and um so
> 一个矩阵，因此

**[00:00:09]** almost like yeah as a matrix and um so we have to have the same number of
> 一个矩阵，因此我们必须确保列中的元素数量相同

**[00:00:09]** we have to have the same number of
> 我们必须确保列中的元素数量相同

**[00:00:09]** we have to have the same number of elements in the columns so usually yeah
> 我们必须确保列中的元素数量相同，所以通常

**[00:00:09]** elements in the columns so usually yeah
> 元素数量相同，所以通常

**[00:00:09]** elements in the columns so usually yeah we prepare the data set like that where
> 元素数量相同，所以通常我们这样准备数据集

**[00:00:09]** we prepare the data set like that where
> 我们这样准备数据集

**[00:00:09]** we prepare the data set like that where we take um fixed size inputs sliding it
> 我们这样准备数据集，即采用固定大小的输入，在文本上滑动

**[00:00:09]** we take um fixed size inputs sliding it
> 采用固定大小的输入，在文本上滑动

**[00:00:09]** we take um fixed size inputs sliding it over the text here to create such a
> 采用固定大小的输入，在文本上滑动，以创建这样的batch

**[00:00:09]** over the text here to create such a
> 在文本上滑动，以创建这样的batch

**[00:00:09]** over the text here to create such a batch and we would do that for the whole
> 在文本上滑动，以创建这样的batch，并对整个数据集执行此操作

**[00:00:09]** batch and we would do that for the whole
> batch，并对整个数据集执行此操作

**[00:00:09]** batch and we would do that for the whole data set then of course I'm only showing
> batch，并对整个数据集执行此操作，当然这里我只展示

**[00:00:09]** data set then of course I'm only showing
> 数据集，当然这里我只展示

**[00:00:09]** data set then of course I'm only showing you um is it one two three four um four
> 数据集，当然这里我只展示每行有四个单词

**[00:00:09]** you um is it one two three four um four
> 每行有四个单词

**[00:00:09]** you um is it one two three four um four words in one row but in reality these
> 每行有四个单词，但实际上这些

**[00:00:09]** words in one row but in reality these
> 单词，但实际上这些

**[00:00:09]** words in one row but in reality these are much larger so usually we are
> 单词，但实际上这些规模要大得多，通常我们

**[00:00:09]** are much larger so usually we are
> 规模要大得多，通常我们

**[00:00:09]** are much larger so usually we are working with input length of at least um
> 规模要大得多，通常我们处理的输入长度至少为

**[00:00:10]** working with input length of at least um
> 处理的输入长度至少为

**[00:00:10]** working with input length of at least um I would say uh 256 for a very small
> 处理的输入长度至少为256，对于非常小的模型

**[00:00:10]** I would say uh 256 for a very small
> 256，对于非常小的模型

**[00:00:10]** I would say uh 256 for a very small model 1024 or even larger for
> 256，对于非常小的模型，1024甚至更大

**[00:00:10]** model 1024 or even larger for
> 1024甚至更大

**[00:00:10]** model 1024 or even larger for pre-training so we usually uh yeah give
> 1024甚至更大，用于预训练，所以我们通常

**[00:00:10]** pre-training so we usually uh yeah give
> 预训练，所以我们通常

**[00:00:10]** pre-training so we usually uh yeah give it bigger inputs but again this is a
> 预训练，所以我们通常提供更大的输入，但这只是一个

**[00:00:10]** it bigger inputs but again this is a
> 提供更大的输入，但这只是一个

**[00:00:10]** it bigger inputs but again this is a small text example just to make it fit
> 提供更大的输入，但这只是一个小的文本示例，只是为了适应

**[00:00:10]** small text example just to make it fit
> 小的文本示例，只是为了适应

**[00:00:10]** small text example just to make it fit onto this um slide here so the question
> 小的文本示例，只是为了适应这张幻灯片，所以现在的问题是

**[00:00:10]** onto this um slide here so the question
> 这张幻灯片，所以现在的问题是

**[00:00:10]** onto this um slide here so the question now is if we um yeah if we just predict
> 这张幻灯片，所以现在的问题是，如果我们一次只预测一个单词

**[00:00:10]** now is if we um yeah if we just predict
> 一次只预测一个单词

**[00:00:10]** now is if we um yeah if we just predict one word at a time so if we train the
> 一次只预测一个单词，即训练LLM使其只能预测下一个单词

**[00:00:10]** one word at a time so if we train the
> 即训练LLM使其只能预测下一个单词

**[00:00:10]** one word at a time so if we train the llm so that it can only predict the next
> 即训练LLM使其只能预测下一个单词

**[00:00:10]** llm so that it can only predict the next
> LLM使其只能预测下一个单词

**[00:00:10]** llm so that it can only predict the next word how can an LM actually then
> llm 只能预测下一个词，那么语言模型实际上如何

**[00:00:10]** word how can an LM actually then
> 词，语言模型实际上如何

**[00:00:10]** word how can an LM actually then generate a multi-word output so for
> 词，语言模型实际上如何生成多词输出？例如

**[00:00:10]** generate a multi-word output so for
> 生成多词输出？例如

**[00:00:10]** generate a multi-word output so for example in the case if you used a Chet
> 生成多词输出？例如，如果你使用过 Chet

**[00:00:10]** example in the case if you used a Chet
> 例如，如果你使用过 Chet

**[00:00:10]** example in the case if you used a Chet GPD before it can
> 例如，如果你使用过 Chet GPD，它如何

**[00:00:10]** GPD before it can
> GPD，它如何

**[00:00:10]** GPD before it can generate this whole output um all at
> GPD，它如何一次性生成整个输出

**[00:00:10]** generate this whole output um all at
> 一次性生成整个输出

**[00:00:10]** generate this whole output um all at once how does that work so this is
> 一次性生成整个输出？这是如何工作的？这通常

**[00:00:10]** once how does that work so this is
> 这是如何工作的？这通常

**[00:00:10]** once how does that work so this is usually um still a one-word prediction
> 这是如何工作的？这通常仍然是每次迭代预测一个词的任务

**[00:00:10]** usually um still a one-word prediction
> 仍然是每次迭代预测一个词的任务

**[00:00:10]** usually um still a one-word prediction task per iteration so here for example
> 仍然是每次迭代预测一个词的任务。例如，这里

**[00:00:10]** task per iteration so here for example
> 例如，这里

**[00:00:10]** task per iteration so here for example if I have an input and that is called or
> 例如，这里如果我有一个输入，称为或

**[00:00:10]** if I have an input and that is called or
> 如果我有一个输入，称为或

**[00:00:10]** if I have an input and that is called or the sentence no let's say the text is
> 如果我有一个输入，称为或句子，不，假设文本是

**[00:00:10]** the sentence no let's say the text is
> 句子，不，假设文本是

**[00:00:10]** the sentence no let's say the text is this and um I feed it to the llm the
> 句子，不，假设文本是这个，然后我将其输入到 llm

**[00:00:10]** this and um I feed it to the llm the
> 这个，然后我将其输入到 llm

**[00:00:10]** this and um I feed it to the llm the output here would be this is so it's
> 这个，然后我将其输入到 llm，输出会是“this is”，所以它

**[00:00:10]** output here would be this is so it's
> 输出会是“this is”，所以它

**[00:00:10]** output here would be this is so it's still um generating only the next word
> 输出会是“this is”，所以它仍然只生成下一个词

**[00:00:11]** still um generating only the next word
> 仍然只生成下一个词

**[00:00:11]** still um generating only the next word but what we do then is we pass this um
> 仍然只生成下一个词，但我们随后将这个输出

**[00:00:11]** but what we do then is we pass this um
> 但我们随后将这个输出

**[00:00:11]** but what we do then is we pass this um output here so I hope you can see the
> 但我们随后将这个输出传递回去，所以我希望你能看到

**[00:00:11]** output here so I hope you can see the
> 输出传递回去，所以我希望你能看到

**[00:00:11]** output here so I hope you can see the mous uh you can pass this output here
> 输出传递回去，所以我希望你能看到鼠标，你可以将这个输出

**[00:00:11]** mous uh you can pass this output here
> 鼠标，你可以将这个输出

**[00:00:11]** mous uh you can pass this output here back into the llm and so the context
> 鼠标，你可以将这个输出传回 llm，这样上下文

**[00:00:11]** back into the llm and so the context
> 传回 llm，这样上下文

**[00:00:11]** back into the llm and so the context here is now one word more than before it
> 传回 llm，这样上下文现在比之前多了一个词，它

**[00:00:11]** here is now one word more than before it
> 现在比之前多了一个词，它

**[00:00:11]** here is now one word more than before it passes again through the llm and now
> 现在比之前多了一个词，它再次通过 llm，现在

**[00:00:11]** passes again through the llm and now
> 再次通过 llm，现在

**[00:00:11]** passes again through the llm and now it's this is n so if we added another
> 再次通过 llm，现在是“this is n”，所以如果我们添加另一个

**[00:00:11]** it's this is n so if we added another
> 现在是“this is n”，所以如果我们添加另一个

**[00:00:11]** it's this is n so if we added another word and then again we take this output
> 现在是“this is n”，所以如果我们添加另一个词，然后再次取这个输出

**[00:00:11]** word and then again we take this output
> 词，然后再次取这个输出

**[00:00:11]** word and then again we take this output feed it back to the llm and then yeah uh
> 词，然后再次取这个输出，将其反馈给 llm，然后

**[00:00:11]** feed it back to the llm and then yeah uh
> 将其反馈给 llm，然后

**[00:00:11]** feed it back to the llm and then yeah uh we have one word more and we keep on
> 将其反馈给 llm，然后我们又多了一个词，我们继续

**[00:00:11]** we have one word more and we keep on
> 我们又多了一个词，我们继续

**[00:00:11]** we have one word more and we keep on doing this until um either we generate a
> 我们又多了一个词，我们继续这样做，直到要么我们生成一个

**[00:00:11]** doing this until um either we generate a
> 这样做，直到要么我们生成一个

**[00:00:11]** doing this until um either we generate a so-called end of text token which is a
> 这样做，直到要么我们生成一个所谓的 end of text token，这是一个

**[00:00:11]** so-called end of text token which is a
> 所谓的 end of text token，这是一个

**[00:00:11]** so-called end of text token which is a special token that will stop the
> 所谓的 end of text token，这是一个特殊的 token，会停止

**[00:00:11]** special token that will stop the
> 特殊的 token，会停止

**[00:00:11]** special token that will stop the generation process or we reach a let's
> 特殊的 token，会停止生成过程，要么我们达到一个

**[00:00:11]** generation process or we reach a let's
> 生成过程，要么我们达到一个

**[00:00:11]** generation process or we reach a let's say um system or user specified number
> 生成过程，要么我们达到一个系统或用户指定的

**[00:00:11]** say um system or user specified number
> 系统或用户指定的

**[00:00:11]** say um system or user specified number of tokens
> 系统或用户指定的 token 数量

**[00:00:11]** of tokens
> token 数量

**[00:00:11]** of tokens and uh if you next time if you use chat
> token 数量。如果你下次使用 chat

**[00:00:11]** and uh if you next time if you use chat
> 如果你下次使用 chat

**[00:00:11]** and uh if you next time if you use chat GPT you have to maybe uh if you let's
> 如果你下次使用 chat GPT，你可能需要，如果你

**[00:00:11]** GPT you have to maybe uh if you let's
> GPT，你可能需要，如果你

**[00:00:11]** GPT you have to maybe uh if you let's say pay attention if you put in some
> GPT，你可能需要，如果你注意一下，当你输入一些

**[00:00:11]** say pay attention if you put in some
> 注意一下，当你输入一些

**[00:00:11]** say pay attention if you put in some input text some longer that requires a
> 注意一下，当你输入一些需要较长回答的输入文本时

**[00:00:11]** input text some longer that requires a
> 需要较长回答的输入文本时

**[00:00:11]** input text some longer that requires a longer answer you will see it's also
> 需要较长回答的输入文本时，你会看到它也是

**[00:00:11]** longer answer you will see it's also
> 你会看到它也是

**[00:00:11]** longer answer you will see it's also generating one word at a time in this
> 你会看到它也是逐词生成输出，就像这个视觉提示

**[00:00:11]** generating one word at a time in this
> 逐词生成输出，就像这个视觉提示

**[00:00:11]** generating one word at a time in this output it's like this um visual cue
> 逐词生成输出，就像这个视觉提示一样，它不会一次性生成所有

**[00:00:12]** output it's like this um visual cue
> 一样，它不会一次性生成所有

**[00:00:12]** output it's like this um visual cue almost where it doesn't generate all the
> 一样，它不会一次性生成所有输出，你不会看到

**[00:00:12]** almost where it doesn't generate all the
> 输出，你不会看到

**[00:00:12]** almost where it doesn't generate all the output all at once you don't see the
> 输出，你不会看到文本块一次性出现，它

**[00:00:12]** output all at once you don't see the
> 文本块一次性出现，它

**[00:00:12]** output all at once you don't see the block of text appearing all at once it's
> 文本块一次性出现，它实际上也在向你展示这个逐词

**[00:00:12]** block of text appearing all at once it's
> 实际上也在向你展示这个逐词

**[00:00:12]** block of text appearing all at once it's really um showing you also this um word
> 实际上也在向你展示这个逐词生成的过程

**[00:00:12]** really um showing you also this um word
> really um showing you also this um word

**[00:00:12]** really um showing you also this um word by word uh generation which is I feel
> 实际上，向你展示这种逐词生成的方式，我觉得

**[00:00:12]** by word uh generation which is I feel
> 逐词生成的方式，我觉得

**[00:00:12]** by word uh generation which is I feel like a small gimmick in the UI but it
> 逐词生成的方式，我觉得像是UI上的一个小噱头，但它

**[00:00:12]** like a small gimmick in the UI but it
> 像是UI上的一个小噱头，但它

**[00:00:12]** like a small gimmick in the UI but it also kind of highlights in a way how
> 像是UI上的一个小噱头，但它也在某种程度上凸显了

**[00:00:12]** also kind of highlights in a way how
> 也在某种程度上凸显了

**[00:00:12]** also kind of highlights in a way how llms generate texts um yeah so um this
> 也在某种程度上凸显了LLMs是如何生成文本的。所以，嗯，这就是

**[00:00:12]** llms generate texts um yeah so um this
> LLMs是如何生成文本的。所以，嗯，这就是

**[00:00:12]** llms generate texts um yeah so um this is how we get from the let's say next
> LLMs是如何生成文本的。所以，嗯，这就是我们如何从所谓的下一个

**[00:00:12]** is how we get from the let's say next
> 我们如何从所谓的下一个

**[00:00:12]** is how we get from the let's say next word prediction task to actually
> 我们如何从所谓的下一个词预测任务，真正地

**[00:00:12]** word prediction task to actually
> 词预测任务，真正地

**[00:00:12]** word prediction task to actually generating
> 词预测任务，真正地生成

**[00:00:12]** generating
> 生成

**[00:00:12]** generating outputs so there's one more thing I
> 生成输出。所以还有一件事我

**[00:00:12]** outputs so there's one more thing I
> 输出。所以还有一件事我

**[00:00:12]** outputs so there's one more thing I mentioned earlier um actually we are not
> 输出。所以还有一件事我之前提到过，嗯，实际上我们并不是

**[00:00:12]** mentioned earlier um actually we are not
> 之前提到过，嗯，实际上我们并不是

**[00:00:12]** mentioned earlier um actually we are not generating one word at a time it's uh
> 之前提到过，嗯，实际上我们并不是一次生成一个词，而是

**[00:00:12]** generating one word at a time it's uh
> 一次生成一个词，而是

**[00:00:12]** generating one word at a time it's uh called a token actually so there's a
> 一次生成一个词，而是实际上叫做一个token。所以词和

**[00:00:12]** called a token actually so there's a
> 实际上叫做一个token。所以词和

**[00:00:12]** called a token actually so there's a small distinction between words and
> 实际上叫做一个token。所以词和token之间有一个小的区别。所以如果我们有一些输入文本

**[00:00:12]** small distinction between words and
> token之间有一个小的区别。所以如果我们有一些输入文本

**[00:00:12]** small distinction between words and tokens so if we have some input text
> token之间有一个小的区别。所以如果我们有一些输入文本，内部发生的是

**[00:00:12]** tokens so if we have some input text
> 内部发生的是

**[00:00:12]** tokens so if we have some input text here what happens internally is that
> 内部发生的是，这个输入文本会被token化。所以

**[00:00:12]** here what happens internally is that
> 这个输入文本会被token化。所以

**[00:00:12]** here what happens internally is that this um input text gets tokenized so
> 这个输入文本会被token化。所以基本上，在这个非常简单的

**[00:00:12]** this um input text gets tokenized so
> 基本上，在这个非常简单的

**[00:00:12]** this um input text gets tokenized so it's basically uh in this very simple
> 基本上，在这个非常简单的例子中，是基于空格进行token化的，

**[00:00:12]** it's basically uh in this very simple
> 例子中，是基于空格进行token化的，

**[00:00:12]** it's basically uh in this very simple example tokenized based on white spaces
> 例子中，是基于空格进行token化的，但这只是一个非常简单的例子，

**[00:00:12]** example tokenized based on white spaces
> 但这只是一个非常简单的例子，

**[00:00:12]** example tokenized based on white spaces but that's just a very simple example
> 但这只是一个非常简单的例子，嗯，通常它会稍微复杂一些，

**[00:00:12]** but that's just a very simple example
> 嗯，通常它会稍微复杂一些，

**[00:00:12]** but that's just a very simple example and um usually it's a bit more complic
> 嗯，通常它会稍微复杂一些，但总体概念是，我们

**[00:00:12]** and um usually it's a bit more complic
> 但总体概念是，我们

**[00:00:12]** and um usually it's a bit more complic at it but the general concept is that we
> 但总体概念是，我们将句子分解成单独的

**[00:00:13]** at it but the general concept is that we
> 将句子分解成单独的

**[00:00:13]** at it but the general concept is that we break down the sentence into individual
> 将句子分解成单独的单词tokens或标点符号，在这个例子中

**[00:00:13]** break down the sentence into individual
> 单词tokens或标点符号，在这个例子中

**[00:00:13]** break down the sentence into individual word tokens or punctuation in this case
> 单词tokens或标点符号，在这个例子中。所以从这个输入文本中，我们有1、2、3

**[00:00:13]** word tokens or punctuation in this case
> 所以从这个输入文本中，我们有1、2、3

**[00:00:13]** word tokens or punctuation in this case so we have from this input text 1 two 3
> 所以从这个输入文本中，我们有1、2、3、4、5个tokens，然后我们从中

**[00:00:13]** so we have from this input text 1 two 3
> 4、5个tokens，然后我们从中

**[00:00:13]** so we have from this input text 1 two 3 four five tokens and from that we then
> 4、5个tokens，然后我们从中得到token IDs。所以，嗯，我没有

**[00:00:13]** four five tokens and from that we then
> 得到token IDs。所以，嗯，我没有

**[00:00:13]** four five tokens and from that we then um get token IDs so um this is I'm not
> 得到token IDs。所以，嗯，我没有在这次演示中展示这一点，因为它

**[00:00:13]** um get token IDs so um this is I'm not
> 在这次演示中展示这一点，因为它

**[00:00:13]** um get token IDs so um this is I'm not showing in this presentation because it
> 在这次演示中展示这一点，因为它可能会变成一个三个小时的

**[00:00:13]** showing in this presentation because it
> 可能会变成一个三个小时的

**[00:00:13]** showing in this presentation because it would be maybe more like a three-hour
> 可能会变成一个三个小时的演示，而不是一个45分钟的

**[00:00:13]** would be maybe more like a three-hour
> 演示，而不是一个45分钟的

**[00:00:13]** would be maybe more like a three-hour presentation rather than a um 45 minute
> 演示，而不是一个45分钟的演示。但是，嗯，所以通常有一个

**[00:00:13]** presentation rather than a um 45 minute
> 演示。但是，嗯，所以通常有一个

**[00:00:13]** presentation rather than a um 45 minute presentation but um so there's usually a
> 演示。但是，嗯，所以通常有一个基于训练数据中所有

**[00:00:13]** presentation but um so there's usually a
> 基于训练数据中所有

**[00:00:13]** presentation but um so there's usually a vocabulary that we built based on all
> 基于训练数据中所有唯一词构建的词汇表，

**[00:00:13]** vocabulary that we built based on all
> 唯一词构建的词汇表，

**[00:00:13]** vocabulary that we built based on all the unique words in the training data
> 唯一词构建的词汇表，并且基于那个词汇表，嗯，是的，

**[00:00:13]** the unique words in the training data
> 并且基于那个词汇表，嗯，是的，

**[00:00:13]** the unique words in the training data set and based on that vocabulary um yeah
> 并且基于那个词汇表，嗯，是的，我们分配这些token索引。所以

**[00:00:13]** set and based on that vocabulary um yeah
> 我们分配这些token索引。所以

**[00:00:13]** set and based on that vocabulary um yeah we we assign in these token indes so
> 我们分配这些token索引。所以你可以看到，字母顺序靠前的词

**[00:00:13]** we we assign in these token indes so
> 你可以看到，字母顺序靠前的词

**[00:00:13]** we we assign in these token indes so words you can see with um smaller um
> 你可以看到，字母顺序靠前的词，抱歉，是字母表中

**[00:00:13]** words you can see with um smaller um
> 字母表中

**[00:00:13]** words you can see with um smaller um letters sorry with the letters that come
> 字母表中靠前的字母对应的词，有较小的数字，

**[00:00:13]** letters sorry with the letters that come
> 靠前的字母对应的词，有较小的数字，

**[00:00:13]** letters sorry with the letters that come earli in the
> 靠前的字母对应的词，有较小的数字，而这是一个较大的数字，或者你知道，

**[00:00:13]** earli in the
> 而这是一个较大的数字，或者你知道，

**[00:00:13]** earli in the alphabet um have smaller numbers and
> 而这是一个较大的数字，或者你知道，如果按字母顺序，嗯，

**[00:00:13]** alphabet um have smaller numbers and
> 如果按字母顺序，嗯，

**[00:00:13]** alphabet um have smaller numbers and this is a larger number you or you know
> 如果按字母顺序，嗯，标点符号和数字可能排在前面。

**[00:00:13]** this is a larger number you or you know
> 标点符号和数字可能排在前面。

**[00:00:13]** this is a larger number you or you know like if it's alphabetically um
> this is a larger number you or you know like if it's alphabetically um

**[00:00:13]** like if it's alphabetically um
> like if it's alphabetically um

**[00:00:13]** like if it's alphabetically um punctuation maybe and numbers come first
> like if it's alphabetically um punctuation maybe and numbers come first

**[00:00:13]** punctuation maybe and numbers come first
> punctuation maybe and numbers come first

**[00:00:13]** punctuation maybe and numbers come first and these smaller words or tokens with a
> 标点符号和数字可能排在前面，这些较小的词或token带有

**[00:00:13]** and these smaller words or tokens with a
> 这些较小的词或token带有

**[00:00:13]** and these smaller words or tokens with a at the beginning and then alphabetically
> 这些较小的词或token带有a开头，然后按字母顺序排列

**[00:00:14]** at the beginning and then alphabetically
> 以a开头，然后按字母顺序排列

**[00:00:14]** at the beginning and then alphabetically you can see they get larger and there is
> 以a开头，然后按字母顺序排列，你可以看到它们变得更大，并且有

**[00:00:14]** you can see they get larger and there is
> 你可以看到它们变得更大，并且有

**[00:00:14]** you can see they get larger and there is actually um also a more sophisticated
> 你可以看到它们变得更大，并且实际上还有一个更复杂的

**[00:00:14]** actually um also a more sophisticated
> 实际上还有一个更复杂的

**[00:00:14]** actually um also a more sophisticated process of play um I mean depends on
> 实际上还有一个更复杂的处理过程，嗯，我的意思是取决于

**[00:00:14]** process of play um I mean depends on
> 处理过程，嗯，我的意思是取决于

**[00:00:14]** process of play um I mean depends on what llm you are working with but for
> 处理过程，嗯，我的意思是取决于你使用的LLM，但对于

**[00:00:14]** what llm you are working with but for
> 你使用的LLM，但对于

**[00:00:14]** what llm you are working with but for example GPT originally I think they
> 你使用的LLM，但例如GPT最初，我认为他们

**[00:00:14]** example GPT originally I think they
> 例如GPT最初，我认为他们

**[00:00:14]** example GPT originally I think they still do uh use a modified version of
> 例如GPT最初，我认为他们仍然使用一个修改版本

**[00:00:14]** still do uh use a modified version of
> 仍然使用一个修改版本

**[00:00:14]** still do uh use a modified version of that it's called a bpe tokenizer I think
> 仍然使用一个修改版本，叫做bpe tokenizer，我认为

**[00:00:14]** that it's called a bpe tokenizer I think
> 叫做bpe tokenizer，我认为

**[00:00:14]** that it's called a bpe tokenizer I think it stands for bite pair encoding if I
> 叫做bpe tokenizer，我认为它代表字节对编码，如果我

**[00:00:14]** it stands for bite pair encoding if I
> 它代表字节对编码，如果我

**[00:00:14]** it stands for bite pair encoding if I recall correctly um and there are
> 它代表字节对编码，如果我没记错的话，嗯，还有

**[00:00:14]** recall correctly um and there are
> 没记错的话，嗯，还有

**[00:00:14]** recall correctly um and there are variations of that like sentence piece
> 没记错的话，嗯，还有像sentence piece这样的变体

**[00:00:14]** variations of that like sentence piece
> 像sentence piece这样的变体

**[00:00:14]** variations of that like sentence piece for llama and
> 像sentence piece这样的变体，用于llama和

**[00:00:14]** for llama and
> 用于llama和

**[00:00:14]** for llama and so they can also deal with unknown words
> 用于llama和，所以它们也能处理未知词

**[00:00:14]** so they can also deal with unknown words
> 所以它们也能处理未知词

**[00:00:14]** so they can also deal with unknown words so if you have an unknown word so like
> 所以它们也能处理未知词，所以如果你有一个未知词，比如

**[00:00:14]** so if you have an unknown word so like
> 如果你有一个未知词，比如

**[00:00:14]** so if you have an unknown word so like some word or two words I made up here
> 如果你有一个未知词，比如我在这里编造的一些词或两个词

**[00:00:14]** some word or two words I made up here
> 我在这里编造的一些词或两个词

**[00:00:14]** some word or two words I made up here they would then be sub tokenized so you
> 我在这里编造的一些词或两个词，它们会被子token化，所以

**[00:00:14]** they would then be sub tokenized so you
> 它们会被子token化，所以

**[00:00:14]** they would then be sub tokenized so you can see um it breaks it down like this
> 它们会被子token化，所以你可以看到它像这样分解

**[00:00:14]** can see um it breaks it down like this
> 你可以看到它像这样分解

**[00:00:14]** can see um it breaks it down like this so what I'm trying to say is even if
> 你可以看到它像这样分解，所以我想说的是，即使

**[00:00:14]** so what I'm trying to say is even if
> 所以我想说的是，即使

**[00:00:14]** so what I'm trying to say is even if your words didn't appear in the training
> 所以我想说的是，即使你的词没有出现在训练

**[00:00:14]** your words didn't appear in the training
> 你的词没有出现在训练

**[00:00:14]** your words didn't appear in the training set when you build this bpe tokenizer um
> 你的词没有出现在训练集中，当你构建这个bpe tokenizer时，嗯

**[00:00:14]** set when you build this bpe tokenizer um
> 当你构建这个bpe tokenizer时，嗯

**[00:00:14]** set when you build this bpe tokenizer um or if you use an existing LBP tokenizer
> 当你构建这个bpe tokenizer时，嗯，或者如果你使用一个现有的LBP tokenizer

**[00:00:14]** or if you use an existing LBP tokenizer
> 或者如果你使用一个现有的LBP tokenizer

**[00:00:14]** or if you use an existing LBP tokenizer that has not been seeing such a word it
> 或者如果你使用一个现有的LBP tokenizer，它没有见过这样的词，它

**[00:00:14]** that has not been seeing such a word it
> 它没有见过这样的词，它

**[00:00:14]** that has not been seeing such a word it would still be able to give you tokens
> 它没有见过这样的词，它仍然能够给你token

**[00:00:14]** would still be able to give you tokens
> 仍然能够给你token

**[00:00:14]** would still be able to give you tokens it's just giving you more tokens now so
> 仍然能够给你token，只是现在给你更多的token，所以

**[00:00:15]** it's just giving you more tokens now so
> 只是现在给你更多的token，所以

**[00:00:15]** it's just giving you more tokens now so it's not one token per word it's like um
> 只是现在给你更多的token，所以不是每个词一个token，而是像

**[00:00:15]** it's not one token per word it's like um
> 不是每个词一个token，而是像

**[00:00:15]** it's not one token per word it's like um for this word it's like 1 2 3 four four
> 不是每个词一个token，而是像对于这个词，它是1 2 3 4四个

**[00:00:15]** for this word it's like 1 2 3 four four
> 对于这个词，它是1 2 3 4四个

**[00:00:15]** for this word it's like 1 2 3 four four tokens because it doesn't recognize the
> 对于这个词，它是1 2 3 4四个token，因为它不认识这个

**[00:00:15]** tokens because it doesn't recognize the
> token，因为它不认识这个

**[00:00:15]** tokens because it doesn't recognize the word so it's breaking it down into
> token，因为它不认识这个词，所以它把它分解成

**[00:00:15]** word so it's breaking it down into
> 词，所以它把它分解成

**[00:00:15]** word so it's breaking it down into individual letters here in this case and
> 词，所以它把它分解成单个字母，在这种情况下，并且

**[00:00:15]** individual letters here in this case and
> 单个字母，在这种情况下，并且

**[00:00:15]** individual letters here in this case and this is why to some extent um llms also
> 单个字母，在这种情况下，这就是为什么在某种程度上，LLM也能

**[00:00:15]** this is why to some extent um llms also
> 这就是为什么在某种程度上，LLM也能

**[00:00:15]** this is why to some extent um llms also work with unknown words um they wouldn't
> 这就是为什么在某种程度上，LLM也能处理未知词，嗯，它们不会

**[00:00:15]** work with unknown words um they wouldn't
> 处理未知词，嗯，它们不会

**[00:00:15]** work with unknown words um they wouldn't crash or something but it's yeah it's
> 处理未知词，嗯，它们不会崩溃或什么的，但是的，这

**[00:00:15]** crash or something but it's yeah it's
> 崩溃或什么的，但是的，这

**[00:00:15]** crash or something but it's yeah it's just inefficient basically so you want
> 崩溃或什么的，但是的，这基本上只是效率低下，所以你想要

**[00:00:15]** just inefficient basically so you want
> 基本上只是效率低下，所以你想要

**[00:00:15]** just inefficient basically so you want to essentially um try to represent all
> 基本上只是效率低下，所以你想要本质上嗯尝试表示所有

**[00:00:15]** to essentially um try to represent all
> 本质上嗯尝试表示所有

**[00:00:15]** to essentially um try to represent all the words in in the vocabulary if you
> 本质上嗯尝试表示词汇表中的所有词，如果你

**[00:00:15]** the words in in the vocabulary if you
> 词汇表中的所有词，如果你

**[00:00:15]** the words in in the vocabulary if you can but of course if a user comes up
> 词汇表中的所有词，如果你可以的话，但当然如果用户提出

**[00:00:15]** can but of course if a user comes up
> 可以的话，但当然如果用户提出

**[00:00:15]** can but of course if a user comes up with a new word or something it
> 可以的话，但当然如果用户提出一个新词或什么，它

**[00:00:15]** with a new word or something it
> 一个新词或什么，它

**[00:00:15]** with a new word or something it shouldn't fail or crash okay um so I
> 一个新词或什么，它不应该失败或崩溃，好的，嗯，所以我

**[00:00:15]** shouldn't fail or crash okay um so I
> 不应该失败或崩溃，好的，嗯，所以我

**[00:00:15]** shouldn't fail or crash okay um so I talked about simple data sets in reality
> 不应该失败或崩溃，好的，嗯，所以我谈到了简单的数据集，实际上

**[00:00:15]** talked about simple data sets in reality
> 谈到了简单的数据集，实际上

**[00:00:15]** talked about simple data sets in reality um there are so data sets that are used
> 刚才谈到了现实中的简单数据集，嗯，实际上有很多数据集被使用

**[00:00:15]** um there are so data sets that are used
> 嗯，实际上有很多数据集被使用

**[00:00:15]** um there are so data sets that are used for training LMS are much larger than
> 嗯，实际上有很多用于训练语言模型的数据集比

**[00:00:15]** for training LMS are much larger than
> 用于训练语言模型的数据集比

**[00:00:15]** for training LMS are much larger than the small Snippets I've shown you so
> 用于训练语言模型的数据集比我展示的小片段要大得多

**[00:00:15]** the small Snippets I've shown you so
> 我展示的小片段要大得多

**[00:00:15]** the small Snippets I've shown you so back then for example when um gbd3 was
> 我展示的小片段要大得多，比如当年训练GPT-3时

**[00:00:15]** back then for example when um gbd3 was
> 比如当年训练GPT-3时

**[00:00:15]** back then for example when um gbd3 was trained it was trained on yeah almost 5
> 比如当年训练GPT-3时，它是在近50亿

**[00:00:15]** trained it was trained on yeah almost 5
> 它是在近50亿

**[00:00:15]** trained it was trained on yeah almost 5 billion 500 billion tokens so it's half
> 它是在近50亿、5000亿个token上训练的，所以是半个

**[00:00:15]** billion 500 billion tokens so it's half
> 5000亿个token，所以是半个

**[00:00:15]** billion 500 billion tokens so it's half a trillion tokens this is uh back in
> 5000亿个token，所以是半个万亿token，这是在

**[00:00:15]** a trillion tokens this is uh back in
> 半个万亿token，这是在

**[00:00:15]** a trillion tokens this is uh back in 2020 long time ago and but what's nice
> 半个万亿token，这是在2020年，很久以前了，但好处是

**[00:00:16]** 2020 long time ago and but what's nice
> 2020年，很久以前了，但好处是

**[00:00:16]** 2020 long time ago and but what's nice um about this is also back then
> 2020年，很久以前了，但好处是，嗯，当时

**[00:00:16]** um about this is also back then
> 嗯，当时

**[00:00:16]** um about this is also back then researchers um still
> 嗯，当时的研究人员仍然

**[00:00:16]** researchers um still
> 研究人员仍然

**[00:00:16]** researchers um still showed or roughly showed what they used
> 研究人员仍然展示或大致展示了他们使用的数据

**[00:00:16]** showed or roughly showed what they used
> 展示或大致展示了他们使用的数据

**[00:00:16]** showed or roughly showed what they used for training so they shared a bit of the
> 展示或大致展示了他们用于训练的数据，所以他们分享了一些

**[00:00:16]** for training so they shared a bit of the
> 用于训练的数据，所以他们分享了一些

**[00:00:16]** for training so they shared a bit of the details that went into training that
> 用于训练的数据，所以他们分享了一些训练该模型的细节

**[00:00:16]** details that went into training that
> 训练该模型的细节

**[00:00:16]** details that went into training that model so for example they talk a bit
> 训练该模型的细节，例如他们在论文中稍微谈到了

**[00:00:16]** model so for example they talk a bit
> 例如他们在论文中稍微谈到了

**[00:00:16]** model so for example they talk a bit about um in the paper what these data
> 例如他们在论文中稍微谈到了这些数据集是什么

**[00:00:16]** about um in the paper what these data
> 这些数据集是什么

**[00:00:16]** about um in the paper what these data sets are for example Wikipedia data sets
> 这些数据集是什么，比如维基百科数据集

**[00:00:16]** sets are for example Wikipedia data sets
> 比如维基百科数据集

**[00:00:16]** sets are for example Wikipedia data sets some book data sets uh website crawl
> 比如维基百科数据集、一些书籍数据集、网站爬取

**[00:00:16]** some book data sets uh website crawl
> 一些书籍数据集、网站爬取

**[00:00:16]** some book data sets uh website crawl data set and so forth and uh yeah
> 一些书籍数据集、网站爬取数据集等等，嗯，是的

**[00:00:16]** data set and so forth and uh yeah
> 数据集等等，嗯，是的

**[00:00:16]** data set and so forth and uh yeah nowadays it's less common nowadays it's
> 数据集等等，嗯，是的，如今这不太常见了，如今

**[00:00:16]** nowadays it's less common nowadays it's
> 如今这不太常见了，如今

**[00:00:16]** nowadays it's less common nowadays it's really um they maybe tell you how big
> 如今这不太常见了，如今他们可能只告诉你数据集有多大

**[00:00:16]** really um they maybe tell you how big
> 他们可能只告诉你数据集有多大

**[00:00:16]** really um they maybe tell you how big the data set was but they don't share
> 他们可能只告诉你数据集有多大，但不再分享

**[00:00:16]** the data set was but they don't share
> 但不再分享

**[00:00:16]** the data set was but they don't share any details anymore um so uh but another
> 但不再分享任何细节了，嗯，所以，嗯，但另一个

**[00:00:16]** any details anymore um so uh but another
> 任何细节了，嗯，所以，嗯，但另一个

**[00:00:16]** any details anymore um so uh but another yeah example from uh back then I think
> 任何细节了，嗯，所以，嗯，但另一个例子来自当时，我想

**[00:00:16]** yeah example from uh back then I think
> 例子来自当时，我想

**[00:00:16]** yeah example from uh back then I think this should actually be 2023 I have to
> 例子来自当时，我想这实际上应该是2023年，我得

**[00:00:16]** this should actually be 2023 I have to
> 这实际上应该是2023年，我得

**[00:00:16]** this should actually be 2023 I have to correct this was Lama one by meta Ai and
> 这实际上应该是2023年，我得纠正一下，这是Meta AI的Llama 1

**[00:00:16]** correct this was Lama one by meta Ai and
> 纠正一下，这是Meta AI的Llama 1

**[00:00:16]** correct this was Lama one by meta Ai and here um yeah they trained so my main
> 纠正一下，这是Meta AI的Llama 1，嗯，他们训练了，所以我的主要

**[00:00:16]** here um yeah they trained so my main
> 嗯，他们训练了，所以我的主要

**[00:00:16]** here um yeah they trained so my main point was essentially training sets are
> 嗯，他们训练了，所以我的主要观点是训练集

**[00:00:16]** point was essentially training sets are
> 观点是训练集

**[00:00:16]** point was essentially training sets are getting larger so we are going from 500
> 观点是训练集变得越来越大，所以我们从5000亿

**[00:00:16]** getting larger so we are going from 500
> 变得越来越大，所以我们从5000亿

**[00:00:16]** getting larger so we are going from 500 billion to 1.4 trillion it's um three
> 变得越来越大，所以我们从5000亿增加到1.4万亿，这是

**[00:00:16]** billion to 1.4 trillion it's um three
> 增加到1.4万亿，这是

**[00:00:16]** billion to 1.4 trillion it's um three times larger now um even larger when we
> 增加到1.4万亿，这是现在的三倍大，嗯，甚至更大，当我们

**[00:00:17]** times larger now um even larger when we
> 现在的三倍大，嗯，甚至更大，当我们

**[00:00:17]** times larger now um even larger when we go to Lama 2 so it's 2 trillion and then
> 现在的三倍大，嗯，甚至更大，当我们到Llama 2时，是2万亿，然后

**[00:00:17]** go to Lama 2 so it's 2 trillion and then
> 到Llama 2时，是2万亿，然后

**[00:00:17]** go to Lama 2 so it's 2 trillion and then Lama 3 it's 15 trillion but going back
> 到Llama 2时，是2万亿，然后Llama 3是15万亿，但回到

**[00:00:17]** Lama 3 it's 15 trillion but going back
> Llama 3是15万亿，但回到

**[00:00:17]** Lama 3 it's 15 trillion but going back steps so here we can still see what went
> Llama 3是15万亿，但回到之前，这里我们仍然可以看到数据中包含了什么

**[00:00:17]** steps so here we can still see what went
> 之前，这里我们仍然可以看到数据中包含了什么

**[00:00:17]** steps so here we can still see what went into that data so we have GitHub now um
> 之前，这里我们仍然可以看到数据中包含了什么，现在有GitHub，嗯

**[00:00:17]** into that data so we have GitHub now um
> 现在有GitHub，嗯

**[00:00:17]** into that data so we have GitHub now um archive papers um here um stack Overflow
> 现在有GitHub，嗯，存档论文，嗯，这里有Stack Overflow

**[00:00:17]** archive papers um here um stack Overflow
> 存档论文，嗯，这里有Stack Overflow

**[00:00:17]** archive papers um here um stack Overflow and stack exchange data and so forth um
> 存档论文，嗯，这里有Stack Overflow和Stack Exchange数据等等，嗯

**[00:00:17]** and stack exchange data and so forth um
> 和Stack Exchange数据等等，嗯

**[00:00:17]** and stack exchange data and so forth um newer models they don't actually reveal
> 和Stack Exchange数据等等，嗯，较新的模型实际上不再透露

**[00:00:17]** newer models they don't actually reveal
> 较新的模型实际上不再透露

**[00:00:17]** newer models they don't actually reveal that information anymore it just says um
> 较新的模型实际上不再透露这些信息了，它只说嗯

**[00:00:17]** that information anymore it just says um
> 这些信息了，它只说嗯

**[00:00:17]** that information anymore it just says um a new mix of data from publicly
> 这些信息了，它只说嗯，来自公开可用来源的新数据混合

**[00:00:17]** a new mix of data from publicly
> 来自公开可用来源的新数据混合

**[00:00:17]** a new mix of data from publicly available sources and for Lama 3 it's um
> 来自公开可用来源的新数据混合，对于Llama 3，嗯

**[00:00:17]** available sources and for Lama 3 it's um
> 对于Llama 3，嗯

**[00:00:17]** available sources and for Lama 3 it's um 15 trilon tokens that were all collected
> 可用的来源，对于Llama 3来说，是150亿个token，全部收集自

**[00:00:17]** 15 trilon tokens that were all collected
> 150亿个token，全部收集自

**[00:00:17]** 15 trilon tokens that were all collected from publicly available sources um it's
> 150亿个token，全部收集自公开可用的来源，嗯，这

**[00:00:17]** from publicly available sources um it's
> 来自公开可用的来源，嗯，这

**[00:00:17]** from publicly available sources um it's a bit unfortunate if you um you know as
> 来自公开可用的来源，嗯，这有点遗憾，如果你，你知道，作为

**[00:00:17]** a bit unfortunate if you um you know as
> 有点遗憾，如果你，你知道，作为

**[00:00:17]** a bit unfortunate if you um you know as a researcher want to know a bit more um
> 有点遗憾，如果你，你知道，作为研究人员想了解更多，嗯

**[00:00:17]** a researcher want to know a bit more um
> 研究人员想了解更多，嗯

**[00:00:17]** a researcher want to know a bit more um how the data set look like um for
> 研究人员想了解更多，嗯，关于数据集的样子，嗯，比如

**[00:00:17]** how the data set look like um for
> 关于数据集的样子，嗯，比如

**[00:00:17]** how the data set look like um for example if you are trying to put down or
> 关于数据集的样子，嗯，比如，如果你试图整理或

**[00:00:17]** example if you are trying to put down or
> 比如，如果你试图整理或

**[00:00:17]** example if you are trying to put down or together your own um data set um but I
> 比如，如果你试图整理或构建你自己的数据集，嗯，但我

**[00:00:17]** together your own um data set um but I
> 构建你自己的数据集，嗯，但我

**[00:00:17]** together your own um data set um but I think this is really because yeah some
> 构建你自己的数据集，嗯，但我认为这确实是因为，是的，有些

**[00:00:17]** think this is really because yeah some
> 认为这确实是因为，是的，有些

**[00:00:17]** think this is really because yeah some companies have been sued um by training
> 认为这确实是因为，是的，有些公司被起诉了，嗯，因为训练

**[00:00:17]** companies have been sued um by training
> 公司被起诉了，嗯，因为训练

**[00:00:17]** companies have been sued um by training on data that was protected so basically
> 公司被起诉了，嗯，因为训练在受保护的数据上，所以基本上

**[00:00:17]** on data that was protected so basically
> 在受保护的数据上，所以基本上

**[00:00:17]** on data that was protected so basically we actually shouldn't go out and just
> 在受保护的数据上，所以基本上我们不应该直接出去就

**[00:00:17]** we actually shouldn't go out and just
> 我们不应该直接出去就

**[00:00:17]** we actually shouldn't go out and just train LMS on publicly available data
> 我们不应该直接出去就在公开可用的数据上训练LM

**[00:00:18]** train LMS on publicly available data
> 在公开可用的数据上训练LM

**[00:00:18]** train LMS on publicly available data because publicly available doesn't mean
> 在公开可用的数据上训练LM，因为公开可用并不意味着

**[00:00:18]** because publicly available doesn't mean
> 因为公开可用并不意味着

**[00:00:18]** because publicly available doesn't mean we are supposed or allowed to use that
> 因为公开可用并不意味着我们被期望或被允许使用那些

**[00:00:18]** we are supposed or allowed to use that
> 我们被期望或被允许使用那些

**[00:00:18]** we are supposed or allowed to use that data and so I guess by not saying what
> 我们被期望或被允许使用那些数据，所以我想通过不说明

**[00:00:18]** data and so I guess by not saying what
> 数据，所以我想通过不说明

**[00:00:18]** data and so I guess by not saying what the data set is here um researchers try
> 数据，所以我想通过不说明这里的数据集是什么，嗯，研究人员试图

**[00:00:18]** the data set is here um researchers try
> 这里的数据集是什么，嗯，研究人员试图

**[00:00:18]** the data set is here um researchers try or companies try to avoid um lawsuits so
> 这里的数据集是什么，嗯，研究人员试图或公司试图避免，嗯，诉讼，所以

**[00:00:18]** or companies try to avoid um lawsuits so
> 或公司试图避免，嗯，诉讼，所以

**[00:00:18]** or companies try to avoid um lawsuits so um yeah Al I should say my book um I'm
> 或公司试图避免，嗯，诉讼，所以，嗯，是的，AI，我应该说我的书，嗯，我

**[00:00:18]** um yeah Al I should say my book um I'm
> 嗯，是的，AI，我应该说我的书，嗯，我

**[00:00:18]** um yeah Al I should say my book um I'm only training um the llm on example data
> 嗯，是的，AI，我应该说我的书，嗯，我只在示例数据上训练LLM

**[00:00:18]** only training um the llm on example data
> 只在示例数据上训练LLM

**[00:00:18]** only training um the llm on example data that is in the public domain where um
> 只在示例数据上训练LLM，这些数据属于公共领域，嗯，其中

**[00:00:18]** that is in the public domain where um
> 这些数据属于公共领域，嗯，其中

**[00:00:18]** that is in the public domain where um this is from data that is not
> 这些数据属于公共领域，嗯，其中这些数据来自不受

**[00:00:18]** this is from data that is not
> 这些数据来自不受

**[00:00:18]** this is from data that is not copyrighted so I'm also making sure of
> 这些数据来自不受版权保护的数据，所以我也在确保

**[00:00:18]** copyrighted so I'm also making sure of
> 版权保护的数据，所以我也在确保

**[00:00:18]** copyrighted so I'm also making sure of that because I uh yeah I think um we
> 版权保护的数据，所以我也在确保这一点，因为我，嗯，是的，我认为，嗯，我们

**[00:00:18]** that because I uh yeah I think um we
> 这一点，因为我，嗯，是的，我认为，嗯，我们

**[00:00:18]** that because I uh yeah I think um we should respect that if data is yeah
> 这一点，因为我，嗯，是的，我认为，嗯，我们应该尊重，如果数据是，是的

**[00:00:18]** should respect that if data is yeah
> 我们应该尊重，如果数据是，是的

**[00:00:18]** should respect that if data is yeah let's say not designated for training we
> 我们应该尊重，如果数据是，是的，比如说，没有被指定用于训练，我们

**[00:00:18]** let's say not designated for training we
> 比如说，没有被指定用于训练，我们

**[00:00:18]** let's say not designated for training we probably shouldn't use that um anyways U
> 比如说，没有被指定用于训练，我们可能不应该使用它，嗯，无论如何，嗯

**[00:00:18]** probably shouldn't use that um anyways U
> 可能不应该使用它，嗯，无论如何，嗯

**[00:00:18]** probably shouldn't use that um anyways U so here I just wanted to show you or
> 可能不应该使用它，嗯，无论如何，嗯，所以在这里我只想向你展示或

**[00:00:18]** so here I just wanted to show you or
> 所以在这里我只想向你展示或

**[00:00:18]** so here I just wanted to show you or highlight the sizes of the different
> 所以在这里我只想向你展示或强调不同数据集的大小，这些数据集已被用于

**[00:00:18]** highlight the sizes of the different
> 强调不同数据集的大小，这些数据集已被用于

**[00:00:18]** highlight the sizes of the different data sets that have been used to to
> 强调不同数据集的大小，这些数据集已被用于训练LM，所以趋势是朝着，嗯

**[00:00:18]** data sets that have been used to to
> 训练LM，所以趋势是朝着，嗯

**[00:00:18]** data sets that have been used to to train LM so the trend goes towards um
> 训练LM，所以趋势是朝着，嗯，是的，更多数据，根据scaling

**[00:00:18]** train LM so the trend goes towards um
> 是的，更多数据，根据scaling

**[00:00:18]** train LM so the trend goes towards um yeah more data according to the scaling
> 是的，更多数据，根据scaling laws，对于特定大小，我们仍然没有

**[00:00:18]** yeah more data according to the scaling
> laws，对于特定大小，我们仍然没有

**[00:00:18]** yeah more data according to the scaling lws for a certain size we still haven't
> laws，对于特定大小，我们仍然没有饱和LM的性能，所以随着

**[00:00:18]** lws for a certain size we still haven't
> 饱和LM的性能，所以随着

**[00:00:18]** lws for a certain size we still haven't saturated the performance of LM so with
> 饱和LM的性能，所以随着更多数据，我们仍然可以挤出更多

**[00:00:18]** saturated the performance of LM so with
> 更多数据，我们仍然可以挤出更多

**[00:00:18]** saturated the performance of LM so with more data we can still squeeze out more
> 更多数据，我们仍然可以挤出更多性能，唯一的，嗯，最近的论文，我

**[00:00:19]** more data we can still squeeze out more
> 性能，唯一的，嗯，最近的论文，我

**[00:00:19]** more data we can still squeeze out more performance the only um recent paper I
> 性能，唯一的，嗯，最近的论文，我会说，我，我，我知道的，或者模型

**[00:00:19]** performance the only um recent paper I
> 我会说，我，我，我知道的，或者模型

**[00:00:19]** performance the only um recent paper I would say that I I I'm aware of or model
> 我会说，我，我，我知道的，或者模型，我知道的，它走向了一个有点

**[00:00:19]** would say that I I I'm aware of or model
> 我知道的，它走向了一个有点

**[00:00:19]** would say that I I I'm aware of or model that I'm aware of that went into a bit
> 我知道的，它走向了一个有点不同的趋势，嗯，是fi3模型

**[00:00:19]** that I'm aware of that went into a bit
> 不同的趋势，嗯，是fi3模型

**[00:00:19]** that I'm aware of that went into a bit of a different trend is um the fi3 model
> 不同的趋势，嗯，是fi3模型，由Meta，哦，抱歉，由Microsoft，我是说

**[00:00:19]** of a different trend is um the fi3 model
> 由Meta，哦，抱歉，由Microsoft，我是说

**[00:00:19]** of a different trend is um the fi3 model by meta uh sorry by by Microsoft I mean
> 由Meta，哦，抱歉，由Microsoft，我是说，所以Microsoft的F模型，嗯，他们

**[00:00:19]** by meta uh sorry by by Microsoft I mean
> 所以Microsoft的F模型，嗯，他们

**[00:00:19]** by meta uh sorry by by Microsoft I mean so the F model by Microsoft um they
> by meta uh sorry by by Microsoft I mean so the F model by Microsoft um they

**[00:00:19]** so the F model by Microsoft um they
> so the F model by Microsoft um they

**[00:00:19]** so the F model by Microsoft um they focused more on developing smaller
> 所以微软的F模型，他们更专注于开发更小的

**[00:00:19]** focused more on developing smaller
> 更专注于开发更小的

**[00:00:19]** focused more on developing smaller models with um smaller amounts of data
> 更专注于用更少的数据开发更小的模型

**[00:00:19]** models with um smaller amounts of data
> 用更少的数据开发模型

**[00:00:19]** models with um smaller amounts of data so they
> 用更少的数据开发模型，所以他们

**[00:00:19]** so they
> 所以他们

**[00:00:19]** so they um are argue here um actually even using
> 所以他们在这里论证，实际上即使使用

**[00:00:19]** um are argue here um actually even using
> 在这里论证，实际上即使使用

**[00:00:19]** um are argue here um actually even using less data that would be um considered
> 在这里论证，实际上即使使用被认为最优的更少数据，也会对LLM有益

**[00:00:19]** less data that would be um considered
> 被认为最优的更少数据，也会对LLM有益

**[00:00:19]** less data that would be um considered optimal would actually be beneficial to
> 被认为最优的更少数据，实际上会对LLM有益，因为你留下了一些所谓的

**[00:00:19]** optimal would actually be beneficial to
> 实际上会对LLM有益，因为你留下了一些所谓的

**[00:00:19]** optimal would actually be beneficial to the llm because you leave some let's say
> 实际上会对LLM有益，因为你留下了一些所谓的容量来学习某些行为

**[00:00:19]** the llm because you leave some let's say
> 容量来学习某些行为

**[00:00:19]** the llm because you leave some let's say capacity for learning certain behaviors
> 容量来学习某些行为或推理，例如这里

**[00:00:19]** capacity for learning certain behaviors
> 或推理，例如这里

**[00:00:19]** capacity for learning certain behaviors or reasoning for example so here for
> 或推理，例如这里他们说不包含

**[00:00:19]** or reasoning for example so here for
> 他们说不包含

**[00:00:19]** or reasoning for example so here for example they say they don't ex they
> 他们说不包含某一天英超联赛的比赛结果

**[00:00:19]** example they say they don't ex they
> 某一天英超联赛的比赛结果

**[00:00:19]** example they say they don't ex they don't include the result of a game in
> 某一天英超联赛的比赛结果，因为这是正确的信息

**[00:00:19]** don't include the result of a game in
> 因为这是正确的信息

**[00:00:19]** don't include the result of a game in Premier League in a particular day um
> 因为这是正确的信息，但不一定那么有用

**[00:00:19]** Premier League in a particular day um
> 不一定那么有用

**[00:00:19]** Premier League in a particular day um because this is I mean this is correct
> 不一定那么有用去记住，你知道，我不会说

**[00:00:19]** because this is I mean this is correct
> 去记住，你知道，我不会说

**[00:00:19]** because this is I mean this is correct information but it's not necessarily
> 去记住，你知道，我不会说这对英超球迷来说是琐碎的

**[00:00:20]** information but it's not necessarily
> 这对英超球迷来说是琐碎的

**[00:00:20]** information but it's not necessarily that useful um to
> 这对英超球迷来说是琐碎的，无意冒犯，我也看

**[00:00:20]** that useful um to
> 无意冒犯，我也看

**[00:00:20]** that useful um to memorize you know like I wouldn't say
> 无意冒犯，我也看英超比赛，利物浦，所以我是

**[00:00:20]** memorize you know like I wouldn't say
> 英超比赛，利物浦，所以我是

**[00:00:20]** memorize you know like I wouldn't say it's trivial for Premier League fans uh
> 英超比赛，利物浦，所以我是利物浦球迷，特别是

**[00:00:20]** it's trivial for Premier League fans uh
> 利物浦球迷，特别是

**[00:00:20]** it's trivial for Premier League fans uh no offense here I I watch also sometimes
> 利物浦球迷，特别是，但总的来说，我会说这取决于

**[00:00:20]** no offense here I I watch also sometimes
> 但总的来说，我会说这取决于

**[00:00:20]** no offense here I I watch also sometimes Premier League games uh Liverpool so I'm
> 但总的来说，我会说这取决于你想构建什么，所以除非

**[00:00:20]** Premier League games uh Liverpool so I'm
> 你想构建什么，所以除非

**[00:00:20]** Premier League games uh Liverpool so I'm a Liverpool fan fan in particular but in
> 你想构建什么，所以除非你试图构建一个体育知识库或类似的东西

**[00:00:20]** a Liverpool fan fan in particular but in
> 知识库或类似的东西

**[00:00:20]** a Liverpool fan fan in particular but in uh in general I would say depends on
> 知识库或类似的东西，可能没有必要把所有比赛的数据都喂给LLM

**[00:00:20]** uh in general I would say depends on
> 可能没有必要把所有比赛的数据都喂给LLM

**[00:00:20]** uh in general I would say depends on what you want to build here um so unless
> 可能没有必要把所有比赛的数据都喂给LLM，因为那样你会用其他东西或质量来权衡

**[00:00:20]** what you want to build here um so unless
> 因为那样你会用其他东西或质量来权衡

**[00:00:20]** what you want to build here um so unless you are trying to build a sports
> 因为那样你会用其他东西或质量来权衡，所以他们在这里论证

**[00:00:20]** you are trying to build a sports
> 所以他们在这里论证

**[00:00:20]** you are trying to build a sports knowledge base or something like that
> 所以他们在这里论证，通过不在过多数据上训练模型，他们为推理留下了更多容量

**[00:00:20]** knowledge base or something like that
> 通过不在过多数据上训练模型，他们为推理留下了更多容量

**[00:00:20]** knowledge base or something like that it's maybe not necessary to feed all
> 通过不在过多数据上训练模型，他们为推理留下了更多容量，例如，这是一个假设

**[00:00:20]** it's maybe not necessary to feed all
> 例如，这是一个假设

**[00:00:20]** it's maybe not necessary to feed all that um yeah data about all the games
> 例如，这是一个假设，我认为这是一个有效的论点

**[00:00:20]** that um yeah data about all the games
> 我认为这是一个有效的论点

**[00:00:20]** that um yeah data about all the games into the llm because then you yeah you
> 我认为这是一个有效的论点，但当然，正如其他人所示，更多数据可能更好

**[00:00:20]** into the llm because then you yeah you
> 但当然，正如其他人所示，更多数据可能更好

**[00:00:20]** into the llm because then you yeah you you trade it off with um other things or
> 但当然，正如其他人所示，更多数据可能更好，所以你知道，我认为

**[00:00:20]** you trade it off with um other things or
> 所以你知道，我认为

**[00:00:20]** you trade it off with um other things or qualities so here they argue um that
> 所以你知道，我认为我们会在未来几个月或几年看到一些其他或额外的架构

**[00:00:20]** qualities so here they argue um that
> 我们会在未来几个月或几年看到一些其他或额外的架构

**[00:00:20]** qualities so here they argue um that they leave more capacity for reasoning
> 我们会在未来几个月或几年看到一些其他或额外的架构进行一些研究

**[00:00:20]** they leave more capacity for reasoning
> 进行一些研究

**[00:00:20]** they leave more capacity for reasoning by not training the model on too much
> they leave more capacity for reasoning by not training the model on too much

**[00:00:20]** by not training the model on too much
> by not training the model on too much

**[00:00:20]** by not training the model on too much data for example I mean this is a
> by not training the model on too much data for example I mean this is a

**[00:00:20]** data for example I mean this is a
> data for example I mean this is a

**[00:00:20]** data for example I mean this is a hypothesis um it's yeah a valid argument
> data for example I mean this is a hypothesis um it's yeah a valid argument

**[00:00:20]** hypothesis um it's yeah a valid argument
> hypothesis um it's yeah a valid argument

**[00:00:20]** hypothesis um it's yeah a valid argument I would say but of course also as others
> hypothesis um it's yeah a valid argument I would say but of course also as others

**[00:00:20]** I would say but of course also as others
> I would say but of course also as others

**[00:00:20]** I would say but of course also as others have shown more data can be better so
> I would say but of course also as others have shown more data can be better so

**[00:00:20]** have shown more data can be better so
> have shown more data can be better so

**[00:00:20]** have shown more data can be better so it's you know it's something I think
> have shown more data can be better so it's you know it's something I think

**[00:00:20]** it's you know it's something I think
> it's you know it's something I think

**[00:00:20]** it's you know it's something I think that we will see um trickling out in the
> it's you know it's something I think that we will see um trickling out in the

**[00:00:21]** that we will see um trickling out in the
> that we will see um trickling out in the

**[00:00:21]** that we will see um trickling out in the upcoming months or years when we see
> that we will see um trickling out in the upcoming months or years when we see

**[00:00:21]** upcoming months or years when we see
> upcoming months or years when we see

**[00:00:21]** upcoming months or years when we see some other or additional architectures
> upcoming months or years when we see some other or additional architectures

**[00:00:21]** some other or additional architectures
> some other or additional architectures

**[00:00:21]** some other or additional architectures um doing some investigation here because
> some other or additional architectures um doing some investigation here because

**[00:00:21]** um doing some investigation here because
> um doing some investigation here because

**[00:00:21]** um doing some investigation here because um these are both very recent papers um
> 嗯，我正在做一些调查，因为这两篇都是非常新的论文

**[00:00:21]** um these are both very recent papers um
> 嗯，这两篇都是非常新的论文

**[00:00:21]** um these are both very recent papers um or Lama 3 is not even published as a
> 嗯，这两篇都是非常新的论文，或者Lama 3甚至还没有作为

**[00:00:21]** or Lama 3 is not even published as a
> 或者Lama 3甚至还没有作为

**[00:00:21]** or Lama 3 is not even published as a paper it's just released so we don't
> 或者Lama 3甚至还没有作为论文发表，它只是被发布了，所以我们没有

**[00:00:21]** paper it's just released so we don't
> 论文，它只是被发布了，所以我们没有

**[00:00:21]** paper it's just released so we don't have really that much information about
> 论文，它只是被发布了，所以我们目前还没有太多关于这些的信息

**[00:00:21]** have really that much information about
> 目前还没有太多关于这些的信息

**[00:00:21]** have really that much information about these yet F just came out a few weeks
> 目前还没有太多关于这些的信息，F只是几周前才发布

**[00:00:21]** these yet F just came out a few weeks
> 只是几周前才发布

**[00:00:21]** these yet F just came out a few weeks ago so in that case um yeah we need to
> 只是几周前才发布，所以在这种情况下，嗯，我们需要

**[00:00:21]** ago so in that case um yeah we need to
> 所以在这种情况下，嗯，我们需要

**[00:00:21]** ago so in that case um yeah we need to really see more research to say whether
> 所以在这种情况下，嗯，我们确实需要看到更多研究才能判断这是否

**[00:00:21]** really see more research to say whether
> 确实需要看到更多研究才能判断这是否

**[00:00:21]** really see more research to say whether this is uh
> 确实需要看到更多研究才能判断这是否是

**[00:00:21]** this is uh
> 这是

**[00:00:21]** this is uh actual thing here with a capacity okay
> 这是实际存在的情况，好吧

**[00:00:21]** actual thing here with a capacity okay
> 实际存在的情况，好吧

**[00:00:21]** actual thing here with a capacity okay moving on though um so we talked about
> 实际存在的情况，好吧，继续往下，嗯，我们讨论了

**[00:00:21]** moving on though um so we talked about
> 继续往下，嗯，我们讨论了

**[00:00:21]** moving on though um so we talked about the data we understand a bit now how an
> 继续往下，嗯，我们讨论了数据，现在我们稍微了解了LLM如何

**[00:00:21]** the data we understand a bit now how an
> 数据，现在我们稍微了解了LLM如何

**[00:00:21]** the data we understand a bit now how an llm receives data uh that it learns to
> 数据，现在我们稍微了解了LLM如何接收数据，它学习

**[00:00:21]** llm receives data uh that it learns to
> 接收数据，它学习

**[00:00:21]** llm receives data uh that it learns to predict the next word but um yeah what
> 接收数据，它学习预测下一个词，但是嗯，是的

**[00:00:21]** predict the next word but um yeah what
> 预测下一个词，但是嗯，是的

**[00:00:21]** predict the next word but um yeah what goes into um developing an LM that can
> 预测下一个词，但是嗯，是的，开发一个能够读取这些数据的LM需要什么

**[00:00:21]** goes into um developing an LM that can
> 开发一个能够读取这些数据的LM需要什么

**[00:00:21]** goes into um developing an LM that can read this data what is um let's say the
> 开发一个能够读取这些数据的LM需要什么，嗯，比如说

**[00:00:21]** read this data what is um let's say the
> 读取这些数据，嗯，比如说

**[00:00:21]** read this data what is um let's say the architecture what do the architectures
> 读取这些数据，嗯，比如说架构，这些架构

**[00:00:21]** architecture what do the architectures
> 架构，这些架构

**[00:00:21]** architecture what do the architectures look like here so let's talk about the
> 架构，这些架构看起来是什么样的，所以让我们讨论一下

**[00:00:21]** look like here so let's talk about the
> 看起来是什么样的，所以让我们讨论一下

**[00:00:21]** look like here so let's talk about the architecture and then I will briefly
> 看起来是什么样的，所以让我们讨论一下架构，然后我会简要地

**[00:00:21]** architecture and then I will briefly
> 架构，然后我会简要地

**[00:00:21]** architecture and then I will briefly revisit um the pre-training and then we
> 架构，然后我会简要地重新讨论预训练，然后我们

**[00:00:21]** revisit um the pre-training and then we
> 重新讨论预训练，然后我们

**[00:00:21]** revisit um the pre-training and then we will talk about model evaluation and the
> 重新讨论预训练，然后我们将讨论模型评估和

**[00:00:22]** will talk about model evaluation and the
> 将讨论模型评估和

**[00:00:22]** will talk about model evaluation and the fine-tuning stages
> 将讨论模型评估和微调阶段

**[00:00:22]** fine-tuning stages
> 微调阶段

**[00:00:22]** fine-tuning stages so because I fear this talk might be a
> 微调阶段，因为我担心这个演讲可能会

**[00:00:22]** so because I fear this talk might be a
> 因为我担心这个演讲可能会

**[00:00:22]** so because I fear this talk might be a very long talk that might go many many
> 因为我担心这个演讲可能会非常长，可能会持续很多很多

**[00:00:22]** very long talk that might go many many
> 非常长，可能会持续很多很多

**[00:00:22]** very long talk that might go many many hours I'm not going into too much detail
> 非常长，可能会持续很多很多小时，所以我不会深入讨论

**[00:00:22]** hours I'm not going into too much detail
> 小时，所以我不会深入讨论

**[00:00:22]** hours I'm not going into too much detail of these individual components but I'm
> 小时，所以我不会深入讨论这些单独组件的太多细节，但我

**[00:00:22]** of these individual components but I'm
> 这些单独组件的太多细节，但我

**[00:00:22]** of these individual components but I'm just showing you um yeah how how the
> 这些单独组件的太多细节，但我只是向你们展示，嗯，原始的GPT2和GPT3模型

**[00:00:22]** just showing you um yeah how how the
> 只是向你们展示，嗯，原始的GPT2和GPT3模型

**[00:00:22]** just showing you um yeah how how the original and gpt2 and three models look
> 只是向你们展示，嗯，原始的GPT2和GPT3模型看起来是什么样的

**[00:00:22]** original and gpt2 and three models look
> 看起来是什么样的

**[00:00:22]** original and gpt2 and three models look like so this is the basic architecture
> 看起来是什么样的，所以这是用于开发GPT模型的基本架构

**[00:00:22]** like so this is the basic architecture
> 所以这是用于开发GPT模型的基本架构

**[00:00:22]** like so this is the basic architecture that was used to um develop the GPT
> 所以这是用于开发GPT模型的基本架构

**[00:00:22]** that was used to um develop the GPT
> 用于开发GPT

**[00:00:22]** that was used to um develop the GPT models we don't know exactly how GPT 4
> 用于开发GPT模型，我们不知道GPT 4具体

**[00:00:22]** models we don't know exactly how GPT 4
> 模型，我们不知道GPT 4具体

**[00:00:22]** models we don't know exactly how GPT 4 looks like um so there's no paper about
> 模型，我们不知道GPT 4具体是什么样子，嗯，因为没有关于它的论文

**[00:00:22]** looks like um so there's no paper about
> 是什么样子，嗯，因为没有关于它的论文

**[00:00:22]** looks like um so there's no paper about that but for the first um three GPD
> 是什么样子，嗯，但对于前三个GPT

**[00:00:22]** that but for the first um three GPD
> 但对于前三个GPT

**[00:00:22]** that but for the first um three GPD models is um yeah this type of
> 但对于前三个GPT模型，嗯，是的，是这种类型的

**[00:00:22]** models is um yeah this type of
> 模型，嗯，是的，是这种类型的

**[00:00:22]** models is um yeah this type of architecture here and um there are
> 模型，嗯，是的，是这种架构，并且嗯，有一些

**[00:00:22]** architecture here and um there are
> 架构，并且嗯，有一些

**[00:00:22]** architecture here and um there are certain components I will maybe say
> 架构，并且嗯，有一些组件，我可能会稍后提到

**[00:00:22]** certain components I will maybe say
> 组件，我可能会稍后提到

**[00:00:22]** certain components I will maybe say something about that later that are not
> 组件，我可能会稍后提到一些不再使用的组件，但总的来说，这是

**[00:00:22]** something about that later that are not
> 一些不再使用的组件，但总的来说，这是

**[00:00:22]** something about that later that are not used anymore but in in general it's this
> 一些不再使用的组件，但总的来说，这是嗯，这个

**[00:00:22]** used anymore but in in general it's this
> 嗯，这个

**[00:00:22]** used anymore but in in general it's this um the
> 嗯，这个相同的通用模板，也

**[00:00:22]** um the
> 相同的通用模板，也

**[00:00:22]** um the same cookie cutter template that is also
> 相同的通用模板，也用于其他LLM，特别是

**[00:00:22]** same cookie cutter template that is also
> 用于其他LLM，特别是

**[00:00:22]** same cookie cutter template that is also used for other llms so in particular
> 用于其他LLM，特别是

**[00:00:22]** used for other llms so in particular
> used for other llms so in particular

**[00:00:22]** used for other llms so in particular there's this masked multi-ad attention
> 用于其他LLM，特别是这种掩码多头注意力机制

**[00:00:22]** there's this masked multi-ad attention
> 这种掩码多头注意力机制

**[00:00:22]** there's this masked multi-ad attention module um there are feet forward layers
> 这种掩码多头注意力模块，还有前馈层

**[00:00:22]** module um there are feet forward layers
> 模块，还有前馈层

**[00:00:22]** module um there are feet forward layers um it's essentially two linear layers
> 模块，还有前馈层，它本质上由两个线性层组成

**[00:00:22]** um it's essentially two linear layers
> 它本质上由两个线性层组成

**[00:00:22]** um it's essentially two linear layers usually with uh nonlinear activation um
> 它本质上由两个线性层组成，通常带有非线性激活函数

**[00:00:23]** usually with uh nonlinear activation um
> 通常带有非线性激活函数

**[00:00:23]** usually with uh nonlinear activation um the silu or silu activation they have um
> 通常带有非线性激活函数，比如SiLU或SiLU激活函数

**[00:00:23]** the silu or silu activation they have um
> SiLU或SiLU激活函数

**[00:00:23]** the silu or silu activation they have um it depends also how you structure it but
> SiLU或SiLU激活函数，具体还取决于你的结构设计

**[00:00:23]** it depends also how you structure it but
> 具体还取决于你的结构设计

**[00:00:23]** it depends also how you structure it but it has it can have three linear layers
> 具体还取决于你的结构设计，它也可以有三个线性层

**[00:00:23]** it has it can have three linear layers
> 它也可以有三个线性层

**[00:00:23]** it has it can have three linear layers depending on how you write the code um
> 它也可以有三个线性层，取决于你如何编写代码

**[00:00:23]** depending on how you write the code um
> 取决于你如何编写代码

**[00:00:23]** depending on how you write the code um yeah positional embedding layer it's
> 取决于你如何编写代码，位置嵌入层

**[00:00:23]** yeah positional embedding layer it's
> 位置嵌入层

**[00:00:23]** yeah positional embedding layer it's usually for the input token embedding
> 位置嵌入层通常用于输入token嵌入

**[00:00:23]** usually for the input token embedding
> 通常用于输入token嵌入

**[00:00:23]** usually for the input token embedding layer and yeah this is a layer Norm here
> 通常用于输入token嵌入层，这里还有一个层归一化

**[00:00:23]** layer and yeah this is a layer Norm here
> 层，这里还有一个层归一化

**[00:00:23]** layer and yeah this is a layer Norm here some architectures use RMS Norm but this
> 层，这里还有一个层归一化，有些架构使用RMS归一化

**[00:00:23]** some architectures use RMS Norm but this
> 有些架构使用RMS归一化

**[00:00:23]** some architectures use RMS Norm but this is the overall architecture what's
> 有些架构使用RMS归一化，但这是整体架构

**[00:00:23]** is the overall architecture what's
> 但这是整体架构

**[00:00:23]** is the overall architecture what's interesting is um here in blue that is
> 但这是整体架构，有趣的是，这里用蓝色标出的

**[00:00:23]** interesting is um here in blue that is
> 有趣的是，这里用蓝色标出的

**[00:00:23]** interesting is um here in blue that is the so-called Transformer block
> 有趣的是，这里用蓝色标出的就是所谓的Transformer块

**[00:00:23]** the so-called Transformer block
> 就是所谓的Transformer块

**[00:00:23]** the so-called Transformer block and this is an element that is repeated
> 就是所谓的Transformer块，这是一个重复多次的元素

**[00:00:23]** and this is an element that is repeated
> 这是一个重复多次的元素

**[00:00:23]** and this is an element that is repeated a number of times um and that really
> 这是一个重复多次的元素，具体次数取决于LLM的大小

**[00:00:23]** a number of times um and that really
> 具体次数取决于LLM的大小

**[00:00:23]** a number of times um and that really depends on on the size of the llm but
> 具体次数取决于LLM的大小，但通常我会说至少重复

**[00:00:23]** depends on on the size of the llm but
> 但通常我会说至少重复

**[00:00:23]** depends on on the size of the llm but usually I would say you repeat that at
> 但通常我会说至少重复12到32到64次，取决于

**[00:00:23]** usually I would say you repeat that at
> 12到32到64次，取决于

**[00:00:23]** usually I would say you repeat that at least 12 to 32 64 times depending on the
> 12到32到64次，取决于

**[00:00:23]** least 12 to 32 64 times depending on the
> 取决于

**[00:00:23]** least 12 to 32 64 times depending on the size of the
> 取决于LLM的大小，实际上有一些具体数字

**[00:00:23]** size of the
> LLM的大小，实际上有一些具体数字

**[00:00:23]** size of the llm you actually have some numbers so um
> LLM的大小，实际上有一些具体数字，对于小模型，这里先回顾一下

**[00:00:23]** llm you actually have some numbers so um
> 对于小模型，这里先回顾一下

**[00:00:23]** llm you actually have some numbers so um for the small so here maybe to back up a
> 对于小模型，这里先回顾一下，这里是GPT-2模型的不同尺寸

**[00:00:23]** for the small so here maybe to back up a
> 这里是GPT-2模型的不同尺寸

**[00:00:23]** for the small so here maybe to back up a bit so here are different sizes of the
> 这里是GPT-2模型的不同尺寸，从1.24亿

**[00:00:23]** bit so here are different sizes of the
> 从1.24亿

**[00:00:23]** bit so here are different sizes of the gbt2 model um from 124 million
> 从1.24亿参数到15亿或5亿

**[00:00:23]** gbt2 model um from 124 million
> 参数到15亿或5亿

**[00:00:23]** gbt2 model um from 124 million parameters to 1.5 billion or, 500
> 参数到15亿或5亿，这些架构之间的区别

**[00:00:23]** parameters to 1.5 billion or, 500
> 这些架构之间的区别

**[00:00:23]** parameters to 1.5 billion or, 500 million and the difference between these
> 这些架构之间的区别其实很小，主要就是

**[00:00:24]** million and the difference between these
> 其实很小，主要就是

**[00:00:24]** million and the difference between these architectures is really I would say
> 其实很小，主要就是重复这个Transformer块的次数

**[00:00:24]** architectures is really I would say
> 重复这个Transformer块的次数

**[00:00:24]** architectures is really I would say minor it's really just the number of
> 重复这个Transformer块的次数，从12次到

**[00:00:24]** minor it's really just the number of
> 从12次到

**[00:00:24]** minor it's really just the number of times you repeat this um Transformer
> 从12次到48次，以及多头注意力机制中的头数

**[00:00:24]** times you repeat this um Transformer
> 48次，以及多头注意力机制中的头数

**[00:00:24]** times you repeat this um Transformer block from 12 to
> 48次，以及多头注意力机制中的头数，如果你熟悉的话

**[00:00:24]** block from 12 to
> 如果你熟悉的话

**[00:00:24]** block from 12 to 48 and um also the number of heads in
> 如果你熟悉的话，我可能应该也做个讲座讲这个，但如果你

**[00:00:24]** 48 and um also the number of heads in
> 我可能应该也做个讲座讲这个，但如果你

**[00:00:24]** 48 and um also the number of heads in the multi-ad attention mechanism it's
> 我可能应该也做个讲座讲这个，但如果你熟悉卷积网络

**[00:00:24]** the multi-ad attention mechanism it's
> 熟悉卷积网络

**[00:00:24]** the multi-ad attention mechanism it's essentially if you are familiar I should
> 熟悉卷积网络，如果你考虑卷积层中的通道

**[00:00:24]** essentially if you are familiar I should
> 如果你考虑卷积层中的通道

**[00:00:24]** essentially if you are familiar I should maybe make a lecture on that too but if
> 如果你考虑卷积层中的通道，你可以把

**[00:00:24]** maybe make a lecture on that too but if
> 你可以把

**[00:00:24]** maybe make a lecture on that too but if you're familiar with um convolution
> 你可以把多头注意力中的头看作

**[00:00:24]** you're familiar with um convolution
> 多头注意力中的头看作

**[00:00:24]** you're familiar with um convolution networks if you think about the channels
> you're familiar with um convolution networks if you think about the channels

**[00:00:24]** networks if you think about the channels
> networks if you think about the channels

**[00:00:24]** networks if you think about the channels in a conv convolutional layer you can
> networks if you think about the channels in a conv convolutional layer you can

**[00:00:24]** in a conv convolutional layer you can
> in a conv convolutional layer you can

**[00:00:24]** in a conv convolutional layer you can think um of the
> in a conv convolutional layer you can think um of the

**[00:00:24]** think um of the
> think um of the

**[00:00:24]** think um of the heads and multier tension as an
> think um of the heads and multier tension as an

**[00:00:24]** heads and multier tension as an
> heads and multier tension as an

**[00:00:24]** heads and multier tension as an equivalent so it's basically stacking
> heads和multier attention是等价的，所以基本上就是堆叠

**[00:00:24]** equivalent so it's basically stacking
> 等价，所以基本上就是堆叠

**[00:00:24]** equivalent so it's basically stacking like stacking um channels in a
> 等价，所以基本上就是堆叠，就像堆叠卷积网络中的通道一样

**[00:00:24]** like stacking um channels in a
> 就像堆叠卷积网络中的通道一样

**[00:00:24]** like stacking um channels in a convolution network here you stacking
> 就像堆叠卷积网络中的通道一样，这里你堆叠这些multi-head attention头

**[00:00:24]** convolution network here you stacking
> 这里你堆叠这些multi-head attention头

**[00:00:24]** convolution network here you stacking these multi-ad attention heads and
> 这里你堆叠这些multi-head attention头，真正的区别在于

**[00:00:24]** these multi-ad attention heads and
> 真正的区别在于

**[00:00:24]** these multi-ad attention heads and really the difference
> 真正的区别在于小模型和大模型之间

**[00:00:24]** really the difference
> 小模型和大模型之间

**[00:00:24]** really the difference between the small model and the large
> 小模型和大模型之间，区别只是你重复这个操作的次数

**[00:00:24]** between the small model and the large
> 模型只是你重复这个操作的次数

**[00:00:24]** between the small model and the large model is just the number of times you
> 模型只是你重复这个操作的次数，比如你的堆叠有多深

**[00:00:24]** model is just the number of times you
> 重复这个操作，比如你的堆叠有多深

**[00:00:24]** model is just the number of times you repeat this like how many how how deep
> 重复这个操作，比如你的堆叠有多深，但它是相同的元素

**[00:00:24]** repeat this like how many how how deep
> 你的堆叠，但它是相同的元素

**[00:00:24]** repeat this like how many how how deep your stack is but it's the same element
> 你的堆叠，但它是相同的元素，所以这里的关键思想是

**[00:00:24]** your stack is but it's the same element
> 所以这里的关键思想是

**[00:00:24]** your stack is but it's the same element so so the key idea here is really that
> 所以这里的关键思想是我们重复使用元素，然后只是

**[00:00:24]** so so the key idea here is really that
> 我们重复使用元素，然后只是

**[00:00:24]** so so the key idea here is really that we are reusing elements and we just um
> 我们重复使用元素，然后只是复制这些元素来让模型更大

**[00:00:24]** we are reusing elements and we just um
> 复制这些元素来让模型更大

**[00:00:24]** we are reusing elements and we just um you know duplicating these elements to
> 复制这些元素来让模型更大，另一个小区别是

**[00:00:24]** you know duplicating these elements to
> 另一个小区别是

**[00:00:24]** you know duplicating these elements to make the models larger
> 另一个小区别是embedding维度，例如

**[00:00:24]** make the models larger
> embedding维度，例如

**[00:00:24]** make the models larger another small difference here is the
> embedding维度，例如最小的模型有

**[00:00:25]** another small difference here is the
> 最小的模型有

**[00:00:25]** another small difference here is the embedding dimension for example the
> 最小的模型有768，较大的模型有大约1,600

**[00:00:25]** embedding dimension for example the
> 768，较大的模型有大约1,600

**[00:00:25]** embedding dimension for example the smallest one has
> 768，较大的模型有大约1,600，现在也常见到4,000

**[00:00:25]** smallest one has
> 现在也常见到4,000

**[00:00:25]** smallest one has 768 the larger one has um 1,600 I think
> 现在也常见到4,000到8,000

**[00:00:25]** 768 the larger one has um 1,600 I think
> 到8,000

**[00:00:25]** 768 the larger one has um 1,600 I think nowadays um it's also common to go 4,000
> 到8,000，例如，所以以

**[00:00:25]** nowadays um it's also common to go 4,000
> 例如，所以以

**[00:00:25]** nowadays um it's also common to go 4,000 to 8,000 for
> 例如，以Llama为例，抱歉，Meta 2

**[00:00:25]** to 8,000 for
> 以Llama为例，抱歉，Meta 2

**[00:00:25]** to 8,000 for example so um just to take an example
> 以Llama为例，抱歉，Meta AI的Llama 2模型，70亿参数版本

**[00:00:25]** example so um just to take an example
> Meta AI的Llama 2模型，70亿参数版本

**[00:00:25]** example so um just to take an example from uh llama so the meta 2 sorry The
> Meta AI的Llama 2模型，70亿参数版本，他们所做的

**[00:00:25]** from uh llama so the meta 2 sorry The
> 他们所做的

**[00:00:25]** from uh llama so the meta 2 sorry The Meta AI Lama 2 model the 7 billion
> 他们所做的，是用RMS norm替换了layer Norm

**[00:00:25]** Meta AI Lama 2 model the 7 billion
> 是用RMS norm替换了layer Norm

**[00:00:25]** Meta AI Lama 2 model the 7 billion version so what they done what they've
> 是用RMS norm替换了layer Norm，RMS norm是一种均方根

**[00:00:25]** version so what they done what they've
> RMS norm是一种均方根

**[00:00:25]** version so what they done what they've done here is they replaced layer Norm
> RMS norm是一种均方根Norm，基本上大多数

**[00:00:25]** done here is they replaced layer Norm
> 基本上大多数

**[00:00:25]** done here is they replaced layer Norm with RMS norm and RMS s Norm a root mean
> 基本上大多数现代架构都在使用

**[00:00:25]** with RMS norm and RMS s Norm a root mean
> 现代架构都在使用

**[00:00:25]** with RMS norm and RMS s Norm a root mean Square Norm is basically what most I
> 现代架构都在使用，它基本上是一个归一化层

**[00:00:25]** Square Norm is basically what most I
> 它基本上是一个归一化层

**[00:00:25]** Square Norm is basically what most I would say most modern architectures use
> 它基本上是一个归一化层，类似于batch Norm

**[00:00:25]** would say most modern architectures use
> 类似于batch Norm

**[00:00:25]** would say most modern architectures use um it's basically an normalization layer
> 类似于batch Norm，它有点像batch Norm，但

**[00:00:25]** um it's basically an normalization layer
> 它有点像batch Norm，但

**[00:00:25]** um it's basically an normalization layer similar if you're familiar with batch
> 它有点像batch Norm，但它在多GPU训练中效果更好

**[00:00:25]** similar if you're familiar with batch
> 在多GPU训练中效果更好

**[00:00:25]** similar if you're familiar with batch Norm it's kind of like batch Norm but it
> 在多GPU训练中效果更好，基本上因为你可以

**[00:00:25]** Norm it's kind of like batch Norm but it
> 基本上因为你可以

**[00:00:25]** Norm it's kind of like batch Norm but it works better for um multi-gpu training
> 基本上因为你可以，是的，你不需要在batch size上收集

**[00:00:25]** works better for um multi-gpu training
> 是的，你不需要在batch size上收集

**[00:00:25]** works better for um multi-gpu training essentially because uh you can have so
> 是的，你不需要在batch size上收集，所以它基本上与batch size无关

**[00:00:25]** essentially because uh you can have so
> batch size，所以它基本上与batch size无关

**[00:00:25]** essentially because uh you can have so yeah you don't have to gather over the
> batch size，所以它基本上与batch size无关，我想说的是

**[00:00:25]** yeah you don't have to gather over the
> 我想说的是

**[00:00:25]** yeah you don't have to gather over the batch size basically so it's bad patch
> 我想说的是，是的，所以你用SwiGLU代替

**[00:00:25]** batch size basically so it's bad patch
> 用SwiGLU代替

**[00:00:25]** batch size basically so it's bad patch size independent is what I'm trying to
> 用SwiGLU代替GELU激活函数，在7B模型的情况下重复32次

**[00:00:25]** size independent is what I'm trying to
> GELU激活函数，在7B模型的情况下重复32次

**[00:00:25]** size independent is what I'm trying to say um yeah so you have the CEO instead
> GELU激活函数，在7B模型的情况下重复32次，你没有

**[00:00:25]** say um yeah so you have the CEO instead
> 你没有

**[00:00:25]** say um yeah so you have the CEO instead of G activate here you repeat it 32
> 你没有Dropout，是的，对于embedding

**[00:00:26]** of G activate here you repeat it 32
> Dropout，是的，对于embedding

**[00:00:26]** of G activate here you repeat it 32 times in the case of 7B model you don't
> of G activate here you repeat it 32 times in the case of 7B model you don't

**[00:00:26]** times in the case of 7B model you don't
> times in the case of 7B model you don't

**[00:00:26]** times in the case of 7B model you don't have
> times in the case of 7B model you don't have

**[00:00:26]** have
> have

**[00:00:26]** have Dropout and yeah for the embedding
> have Dropout and yeah for the embedding

**[00:00:26]** Dropout and yeah for the embedding
> Dropout and yeah for the embedding

**[00:00:26]** Dropout and yeah for the embedding layers um usually we use um for POS for
> Dropout，对于embedding层，我们通常用于POS

**[00:00:26]** layers um usually we use um for POS for
> 层，我们通常用于POS

**[00:00:26]** layers um usually we use um for POS for the positional embedding we use uh rope
> 层，我们通常用于POS，对于positional embedding，我们使用rope

**[00:00:26]** the positional embedding we use uh rope
> 对于positional embedding，我们使用rope

**[00:00:26]** the positional embedding we use uh rope I think it stands for rotational
> 对于positional embedding，我们使用rope，我认为它代表旋转

**[00:00:26]** I think it stands for rotational
> 我认为它代表旋转

**[00:00:26]** I think it stands for rotational positional embeddings so the first GPT
> 我认为它代表旋转位置嵌入，所以第一个GPT

**[00:00:26]** positional embeddings so the first GPT
> 位置嵌入，所以第一个GPT

**[00:00:26]** positional embeddings so the first GPT model has absolute um an absolute
> 位置嵌入，所以第一个GPT模型有绝对的位置嵌入，而较新的

**[00:00:26]** model has absolute um an absolute
> 模型有绝对的位置嵌入，而较新的

**[00:00:26]** model has absolute um an absolute position embedding and yeah newer newer
> 模型有绝对的位置嵌入，而较新的模型使用相对位置

**[00:00:26]** position embedding and yeah newer newer
> 模型使用相对位置

**[00:00:26]** position embedding and yeah newer newer models they use a relative positional
> 模型使用相对位置嵌入，并且大小有点

**[00:00:26]** models they use a relative positional
> 嵌入，并且大小有点

**[00:00:26]** models they use a relative positional embedding and the um the size is a bit
> 嵌入，并且大小有点更大，之前我们有1,28个token，现在

**[00:00:26]** embedding and the um the size is a bit
> 更大，之前我们有1,28个token，现在

**[00:00:26]** embedding and the um the size is a bit bigger before we had 1,28 tokens now we
> 更大，之前我们有1,28个token，现在有4,000个，是四倍，是的，三倍

**[00:00:26]** bigger before we had 1,28 tokens now we
> 有4,000个，是四倍，是的，三倍

**[00:00:26]** bigger before we had 1,28 tokens now we have 4,000 it's four times yeah three
> 有4,000个，是四倍，是的，三倍大，基本上比三倍多一点

**[00:00:26]** have 4,000 it's four times yeah three
> 大，基本上比三倍多一点

**[00:00:26]** have 4,000 it's four times yeah three times large basically a bit more than
> 大，基本上比三倍多一点，但是的，嗯，这基本上是

**[00:00:26]** times large basically a bit more than
> 但是的，嗯，这基本上是

**[00:00:26]** times large basically a bit more than three times but yeah um yeah and this is
> 但是的，嗯，这基本上是我认为的主要

**[00:00:26]** three times but yeah um yeah and this is
> 我认为的主要

**[00:00:26]** three times but yeah um yeah and this is essentially I would say the main
> 我认为的主要区别，嗯，这里的要点是

**[00:00:26]** essentially I would say the main
> 区别，嗯，这里的要点是

**[00:00:26]** essentially I would say the main difference um the takeaway here is
> 区别，嗯，这里的要点是大多数LLM仍然非常

**[00:00:26]** difference um the takeaway here is
> 大多数LLM仍然非常

**[00:00:26]** difference um the takeaway here is really that most LMS are still very
> 大多数LLM仍然非常相似，嗯，它们是非常

**[00:00:26]** really that most LMS are still very
> 相似，嗯，它们是非常

**[00:00:26]** really that most LMS are still very similar to each other um they are very
> 相似，嗯，它们是非常小的变化，我可以说在一些LLM中

**[00:00:26]** similar to each other um they are very
> 小的变化，我可以说在一些LLM中

**[00:00:26]** similar to each other um they are very small I would say changes in some llms
> 小的变化，我可以说在一些LLM中像这些，这也是为什么在L GPT

**[00:00:26]** small I would say changes in some llms
> 像这些，这也是为什么在L GPT

**[00:00:26]** small I would say changes in some llms like these and that's also why in L GPT
> 像这些，这也是为什么在L GPT中，例如，我们实现了许多

**[00:00:26]** like these and that's also why in L GPT
> 中，例如，我们实现了许多

**[00:00:26]** like these and that's also why in L GPT for example we yeah we Implement a lot
> 中，例如，我们实现了许多基于相同基础架构的LLM

**[00:00:26]** for example we yeah we Implement a lot
> 基于相同基础架构的LLM

**[00:00:26]** for example we yeah we Implement a lot of llms based on the same base
> 基于相同基础架构的LLM，因为实际上这些是

**[00:00:27]** of llms based on the same base
> 因为实际上这些是

**[00:00:27]** of llms based on the same base architecture because really these are
> 因为实际上这些是相对较小的变化，嗯，这不是

**[00:00:27]** architecture because really these are
> 相对较小的变化，嗯，这不是

**[00:00:27]** architecture because really these are relatively small changes um it's not
> 相对较小的变化，嗯，这不是真正的，我可以说，重新发明

**[00:00:27]** relatively small changes um it's not
> 真正的，我可以说，重新发明

**[00:00:27]** relatively small changes um it's not really um I would say Reinventing the
> 真正的，我可以说，重新发明轮子，而且它效果很好，所以人们仍然

**[00:00:27]** really um I would say Reinventing the
> 轮子，而且它效果很好，所以人们仍然

**[00:00:27]** really um I would say Reinventing the wheel and it works well so people still
> 轮子，而且它效果很好，所以人们仍然继续使用那个架构构建

**[00:00:27]** wheel and it works well so people still
> 继续使用那个架构构建

**[00:00:27]** wheel and it works well so people still yeah um keep building with that
> 继续使用那个架构构建，这里一切都回到

**[00:00:27]** yeah um keep building with that
> 这里一切都回到

**[00:00:27]** yeah um keep building with that architecture here the it all goes back
> 这里一切都回到原始的GPT架构，基本上

**[00:00:27]** architecture here the it all goes back
> 原始的GPT架构，基本上

**[00:00:27]** architecture here the it all goes back to the original GPD architecture B
> 原始的GPT架构，基本上，嗯，所以现在我们熟悉了架构，让我们

**[00:00:27]** to the original GPD architecture B
> 熟悉了架构，让我们

**[00:00:27]** to the original GPD architecture B basically um yeah so now that we are
> 熟悉了架构，让我们简要回到预训练，所以

**[00:00:27]** basically um yeah so now that we are
> 简要回到预训练，所以

**[00:00:27]** basically um yeah so now that we are familiar with the architecture let's
> 简要回到预训练，所以预训练本质上就是

**[00:00:27]** familiar with the architecture let's
> 预训练本质上就是

**[00:00:27]** familiar with the architecture let's briefly return to the pre-training so
> 预训练本质上就是创建这个所谓的Foundation model

**[00:00:27]** briefly return to the pre-training so
> 创建这个所谓的Foundation model

**[00:00:27]** briefly return to the pre-training so the pre-training is essentially what
> 创建这个所谓的Foundation model，然后用于后续的微调

**[00:00:27]** the pre-training is essentially what
> 然后用于后续的微调

**[00:00:27]** the pre-training is essentially what creates this so-called Foundation model
> 然后用于后续的微调，所以我不打算详细讲解训练循环

**[00:00:27]** creates this so-called Foundation model
> 所以我不打算详细讲解训练循环

**[00:00:27]** creates this so-called Foundation model which is then used for fine-tuning later
> 所以我不打算详细讲解训练循环，因为嗯，它看起来会相当

**[00:00:27]** which is then used for fine-tuning later
> 因为嗯，它看起来会相当

**[00:00:27]** which is then used for fine-tuning later so I'm not going over the training Loop
> 因为嗯，它看起来会相当技术性，如果你不熟悉

**[00:00:27]** so I'm not going over the training Loop
> 技术性，如果你不熟悉

**[00:00:27]** so I'm not going over the training Loop because um yeah it would look pretty
> 技术性，如果你不熟悉训练循环，它看起来会

**[00:00:27]** because um yeah it would look pretty
> 训练循环，它看起来会

**[00:00:27]** because um yeah it would look pretty technical if you're not familiar with
> 训练循环，它看起来会非常无聊，如果你熟悉

**[00:00:27]** technical if you're not familiar with
> 非常无聊，如果你熟悉

**[00:00:27]** technical if you're not familiar with training loops and it look it would look
> 非常无聊，如果你熟悉训练循环，因为它真的

**[00:00:27]** training loops and it look it would look
> 训练循环，因为它真的

**[00:00:27]** training loops and it look it would look very boring to you if you are familiar
> 训练循环，因为它真的就像标准的深度学习

**[00:00:27]** very boring to you if you are familiar
> 就像标准的深度学习

**[00:00:27]** very boring to you if you are familiar with training Loops because it's really
> 就像标准的深度学习

**[00:00:27]** with training Loops because it's really
> with training Loops because it's really

**[00:00:27]** with training Loops because it's really yeah just like a standard deep learning
> with training Loops because it's really yeah just like a standard deep learning

**[00:00:27]** yeah just like a standard deep learning
> yeah just like a standard deep learning

**[00:00:27]** yeah just like a standard deep learning training Loop there's really nothing
> 是的，就像标准的深度学习训练循环，实际上没什么特别的

**[00:00:27]** training Loop there's really nothing
> 训练循环，实际上没什么特别的

**[00:00:27]** training Loop there's really nothing different from training I would say
> 训练循环，我认为和训练没什么不同

**[00:00:27]** different from training I would say
> 我认为和训练没什么不同

**[00:00:27]** different from training I would say Evolution networks multilayer pums you
> 我认为和训练没什么不同，进化网络、多层感知机

**[00:00:28]** Evolution networks multilayer pums you
> 进化网络、多层感知机

**[00:00:28]** Evolution networks multilayer pums you use the same atom Optimizer same uh
> 进化网络、多层感知机，你使用相同的优化器，呃

**[00:00:28]** use the same atom Optimizer same uh
> 你使用相同的优化器，呃

**[00:00:28]** use the same atom Optimizer same uh learning rate schedulers like one one um
> 你使用相同的优化器，呃，学习率调度器，比如

**[00:00:28]** learning rate schedulers like one one um
> 学习率调度器，比如

**[00:00:28]** learning rate schedulers like one one um cycle coine scheduler and so forth you
> 学习率调度器，比如循环余弦调度器等等

**[00:00:28]** cycle coine scheduler and so forth you
> 循环余弦调度器等等

**[00:00:28]** cycle coine scheduler and so forth you use the same cross entropy loss so it's
> 循环余弦调度器等等，你使用相同的交叉熵损失

**[00:00:28]** use the same cross entropy loss so it's
> 你使用相同的交叉熵损失

**[00:00:28]** use the same cross entropy loss so it's all the same basically that we would use
> 你使用相同的交叉熵损失，所以基本上和我们训练时用的都一样

**[00:00:28]** all the same basically that we would use
> 基本上和我们训练时用的都一样

**[00:00:28]** all the same basically that we would use when training um yeah convolution
> 基本上和我们训练时用的都一样，嗯，训练卷积网络

**[00:00:28]** when training um yeah convolution
> 嗯，训练卷积网络

**[00:00:28]** when training um yeah convolution networks recurrent new network I
> 嗯，训练卷积网络、循环神经网络

**[00:00:28]** networks recurrent new network I
> 循环神经网络

**[00:00:28]** networks recurrent new network I recurrents are maybe bit different but
> 循环神经网络，循环网络可能有点不同

**[00:00:28]** recurrents are maybe bit different but
> 循环网络可能有点不同

**[00:00:28]** recurrents are maybe bit different but convolution networks multilayer
> 循环网络可能有点不同，但卷积网络、多层

**[00:00:28]** convolution networks multilayer
> 卷积网络、多层

**[00:00:28]** convolution networks multilayer perceptrons and so forth and uh I would
> 卷积网络、多层感知机等等，呃，我认为

**[00:00:28]** perceptrons and so forth and uh I would
> 感知机等等，呃，我认为

**[00:00:28]** perceptrons and so forth and uh I would say the the only difference is really
> 感知机等等，呃，我认为唯一的区别实际上是

**[00:00:28]** say the the only difference is really
> 唯一的区别实际上是

**[00:00:28]** say the the only difference is really that we do that on a larger scale on
> 唯一的区别实际上是我们在更大规模上做这件事

**[00:00:28]** that we do that on a larger scale on
> 我们在更大规模上做这件事

**[00:00:28]** that we do that on a larger scale on multiple gpus and here the tricky part
> 我们在更大规模上做这件事，在多个GPU上，这里棘手的部分

**[00:00:28]** multiple gpus and here the tricky part
> 多个GPU上，这里棘手的部分

**[00:00:28]** multiple gpus and here the tricky part is more like um yeah the hardware access
> 多个GPU上，这里棘手的部分更像是，嗯，硬件访问

**[00:00:28]** is more like um yeah the hardware access
> 更像是，嗯，硬件访问

**[00:00:28]** is more like um yeah the hardware access and and that type of
> 更像是，嗯，硬件访问以及诸如此类的

**[00:00:28]** and and that type of
> 以及诸如此类的

**[00:00:28]** and and that type of stuff um yeah what what do the labels
> 以及诸如此类的东西，嗯，标签是什么样的

**[00:00:28]** stuff um yeah what what do the labels
> 东西，嗯，标签是什么样的

**[00:00:28]** stuff um yeah what what do the labels look like we talked about this before
> 东西，嗯，标签是什么样的，我们之前讨论过

**[00:00:28]** look like we talked about this before
> 我们之前讨论过

**[00:00:28]** look like we talked about this before already so the target um so when you
> 我们之前讨论过，所以目标，嗯，当你

**[00:00:28]** already so the target um so when you
> 所以目标，嗯，当你

**[00:00:28]** already so the target um so when you train for example in classic um deep
> 所以目标，嗯，当你训练例如经典的深度学习

**[00:00:28]** train for example in classic um deep
> 训练例如经典的深度学习

**[00:00:28]** train for example in classic um deep learning a convolution network is
> 训练例如经典的深度学习中的卷积网络时

**[00:00:28]** learning a convolution network is
> 中的卷积网络时

**[00:00:28]** learning a convolution network is usually for some prediction task for
> 中的卷积网络时，通常是用于某个预测任务

**[00:00:28]** usually for some prediction task for
> 通常是用于某个预测任务

**[00:00:28]** usually for some prediction task for example um let's say classifying cats
> 通常是用于某个预测任务，比如，嗯，分类猫

**[00:00:28]** example um let's say classifying cats
> 比如，嗯，分类猫

**[00:00:28]** example um let's say classifying cats versus dogs in image data and uh in LMS
> 比如，嗯，分类猫与狗的图像数据，而在LLM中

**[00:00:28]** versus dogs in image data and uh in LMS
> 与狗的图像数据，而在LLM中

**[00:00:28]** versus dogs in image data and uh in LMS of course we we deal with text so what
> 与狗的图像数据，而在LLM中，当然我们处理的是文本

**[00:00:29]** of course we we deal with text so what
> 当然我们处理的是文本

**[00:00:29]** of course we we deal with text so what is the class label here or the thing we
> 当然我们处理的是文本，那么这里的类别标签是什么，或者当我们谈论

**[00:00:29]** is the class label here or the thing we
> 那么这里的类别标签是什么，或者当我们谈论

**[00:00:29]** is the class label here or the thing we want to predict when we talk about um
> 那么这里的类别标签是什么，或者当我们谈论标准训练循环时我们想要预测的东西

**[00:00:29]** want to predict when we talk about um
> 标准训练循环时我们想要预测的东西

**[00:00:29]** want to predict when we talk about um standard training Loop so here that's
> 标准训练循环时我们想要预测的东西，所以这里

**[00:00:29]** standard training Loop so here that's
> 所以这里

**[00:00:29]** standard training Loop so here that's really the next word in a text so like
> 所以这里实际上是文本中的下一个词

**[00:00:29]** really the next word in a text so like
> 实际上是文本中的下一个词

**[00:00:29]** really the next word in a text so like we talked about before um the next word
> 实际上是文本中的下一个词，就像我们之前讨论过的，下一个词

**[00:00:29]** we talked about before um the next word
> 就像我们之前讨论过的，下一个词

**[00:00:29]** we talked about before um the next word is what we want the llm to predict so
> 就像我们之前讨论过的，下一个词是我们希望LLM预测的

**[00:00:29]** is what we want the llm to predict so
> 是我们希望LLM预测的

**[00:00:29]** is what we want the llm to predict so when we prepare the data we have the
> 是我们希望LLM预测的，所以当我们准备数据时，我们有

**[00:00:29]** when we prepare the data we have the
> 所以当我们准备数据时，我们有

**[00:00:29]** when we prepare the data we have the input here and the target is really the
> 所以当我们准备数据时，我们有输入，而目标实际上是

**[00:00:29]** input here and the target is really the
> 输入，而目标实际上是

**[00:00:29]** input here and the target is really the same as the input but shifted by one
> 输入，而目标实际上与输入相同，但偏移了一个

**[00:00:29]** same as the input but shifted by one
> 与输入相同，但偏移了一个

**[00:00:29]** same as the input but shifted by one position so when the model is in the
> 与输入相同，但偏移了一个位置，所以当模型处于

**[00:00:29]** position so when the model is in the
> 位置，所以当模型处于

**[00:00:29]** position so when the model is in the heart of the the
> 位置，所以当模型处于核心位置时

**[00:00:29]** heart of the the
> 核心位置时

**[00:00:29]** heart of the the um the next word would be the for
> 核心位置时，嗯，下一个词会是"the"

**[00:00:29]** um the next word would be the for
> 嗯，下一个词会是"the"

**[00:00:29]** um the next word would be the for example so really just shifting
> 嗯，下一个词会是"the"，例如，所以实际上只是偏移

**[00:00:29]** example so really just shifting
> 例如，所以实际上只是偏移

**[00:00:29]** example so really just shifting everything by by
> 举例来说，实际上只是将所有内容偏移

**[00:00:29]** everything by by
> 将所有内容偏移

**[00:00:29]** everything by by one um yeah and so usually we train for
> 将所有内容偏移一个位置，嗯，是的，通常我们训练

**[00:00:29]** one um yeah and so usually we train for
> 一个位置，嗯，是的，通常我们训练

**[00:00:29]** one um yeah and so usually we train for one to two EPO that's usually a good
> 一个位置，嗯，是的，通常我们训练一到两个Epoch，这通常是一个好的

**[00:00:29]** one to two EPO that's usually a good
> 一到两个Epoch，这通常是一个好的

**[00:00:29]** one to two EPO that's usually a good sweet spot actually most um llms are not
> 一到两个Epoch，这通常是一个好的最佳点，实际上大多数LLM并没有

**[00:00:29]** sweet spot actually most um llms are not
> 最佳点，实际上大多数LLM并没有

**[00:00:29]** sweet spot actually most um llms are not even trained for a full Epoch or
> 最佳点，实际上大多数LLM甚至没有训练完整一个Epoch，或者

**[00:00:29]** even trained for a full Epoch or
> 甚至没有训练完整一个Epoch，或者

**[00:00:29]** even trained for a full Epoch or sometimes you might read something like
> 甚至没有训练完整一个Epoch，或者有时你可能会读到类似

**[00:00:29]** sometimes you might read something like
> 有时你可能会读到类似

**[00:00:29]** sometimes you might read something like 1.1 EPO or something like that so just
> 有时你可能会读到类似1.1 Epoch这样的内容，所以只是

**[00:00:29]** 1.1 EPO or something like that so just
> 1.1 Epoch这样的内容，所以只是

**[00:00:29]** 1.1 EPO or something like that so just to take a step back an epoc is one pass
> 1.1 Epoch这样的内容，所以只是退一步讲，一个Epoch是一次遍历

**[00:00:30]** to take a step back an epoc is one pass
> 退一步讲，一个Epoch是一次遍历

**[00:00:30]** to take a step back an epoc is one pass over the training set but usually these
> 退一步讲，一个Epoch是一次遍历整个训练集，但通常这些

**[00:00:30]** over the training set but usually these
> 整个训练集，但通常这些

**[00:00:30]** over the training set but usually these um training sets are super large and the
> 整个训练集，但通常这些训练集非常大，并且

**[00:00:30]** um training sets are super large and the
> 训练集非常大，并且

**[00:00:30]** um training sets are super large and the models are distributed over multiple
> 训练集非常大，并且模型分布在多个

**[00:00:30]** models are distributed over multiple
> 模型分布在多个

**[00:00:30]** models are distributed over multiple machines that it's not really super I
> 模型分布在多个机器上，以至于我认为并不太

**[00:00:30]** machines that it's not really super I
> 机器上，以至于我认为并不太

**[00:00:30]** machines that it's not really super I would say feasible to do the classic I
> 机器上，以至于我认为并不太可行去做经典的，我

**[00:00:30]** would say feasible to do the classic I
> 可行去做经典的，我

**[00:00:30]** would say feasible to do the classic I mean you can but not many people do that
> 可行去做经典的，我的意思是你可以做，但没多少人那样做

**[00:00:30]** mean you can but not many people do that
> 我的意思是你可以做，但没多少人那样做

**[00:00:30]** mean you can but not many people do that um do the classic epoc it's more like um
> 我的意思是你可以做，但没多少人那样做经典的Epoch，更像是

**[00:00:30]** um do the classic epoc it's more like um
> 经典的Epoch，更像是

**[00:00:30]** um do the classic epoc it's more like um drawing random batches from the data set
> 经典的Epoch，更像是从数据集中随机抽取batches

**[00:00:30]** drawing random batches from the data set
> 从数据集中随机抽取batches

**[00:00:30]** drawing random batches from the data set so there might be then overlaps and some
> 从数据集中随机抽取batches，所以可能会有重叠，有些

**[00:00:30]** so there might be then overlaps and some
> 所以可能会有重叠，有些

**[00:00:30]** so there might be then overlaps and some batches are seen twice some not at all
> 所以可能会有重叠，有些batches被看到两次，有些一次也没有

**[00:00:30]** batches are seen twice some not at all
> batches被看到两次，有些一次也没有

**[00:00:30]** batches are seen twice some not at all and so forth but um if we think of the
> batches被看到两次，有些一次也没有，等等，但是如果我们考虑

**[00:00:30]** and so forth but um if we think of the
> 等等，但是如果我们考虑

**[00:00:30]** and so forth but um if we think of the classic Epoch regime where EPO means one
> 等等，但是如果我们考虑经典的Epoch机制，其中Epoch意味着一次

**[00:00:30]** classic Epoch regime where EPO means one
> 经典的Epoch机制，其中Epoch意味着一次

**[00:00:30]** classic Epoch regime where EPO means one pass of the training set usually
> 经典的Epoch机制，其中Epoch意味着一次遍历训练集，通常

**[00:00:30]** pass of the training set usually
> 遍历训练集，通常

**[00:00:30]** pass of the training set usually training for one Epoch is what most
> 遍历训练集，通常训练一个Epoch是大多数

**[00:00:30]** training for one Epoch is what most
> 训练一个Epoch是大多数

**[00:00:30]** training for one Epoch is what most people do you can train a bit longer I
> 训练一个Epoch是大多数人所做的，你可以训练稍长一些，我

**[00:00:30]** people do you can train a bit longer I
> 人所做的，你可以训练稍长一些，我

**[00:00:30]** people do you can train a bit longer I think there was a paper looking into
> 人所做的，你可以训练稍长一些，我记得有一篇论文研究了

**[00:00:30]** think there was a paper looking into
> 记得有一篇论文研究了

**[00:00:30]** think there was a paper looking into that um Pia uh from elua AI where they
> 记得有一篇论文研究了这个问题，嗯，来自elua AI的Pia，他们

**[00:00:30]** that um Pia uh from elua AI where they
> 这个问题，嗯，来自elua AI的Pia，他们

**[00:00:30]** that um Pia uh from elua AI where they trained on duplicate so D duplicated and
> 这个问题，嗯，来自elua AI的Pia，他们在去重后的数据上训练，所以是去重后的

**[00:00:30]** trained on duplicate so D duplicated and
> 在去重后的数据上训练，所以是去重后的

**[00:00:30]** trained on duplicate so D duplicated and duplicated data and they didn't see
> 在去重后的数据上训练，所以是去重后的和未去重的数据，他们没有看到

**[00:00:30]** duplicated data and they didn't see
> 和未去重的数据，他们没有看到

**[00:00:30]** duplicated data and they didn't see really a difference so here on some
> 和未去重的数据，他们没有看到真正的差异，所以在这里一些

**[00:00:30]** really a difference so here on some
> 真正的差异，所以在这里一些

**[00:00:30]** really a difference so here on some smaller scale experiments I saw
> 真正的差异，所以在这里一些较小规模的实验中，我看到

**[00:00:30]** smaller scale experiments I saw
> 较小规模的实验中，我看到

**[00:00:30]** smaller scale experiments I saw basically after two EPO you see
> 较小规模的实验中，我看到基本上在两个Epoch之后，你会看到

**[00:00:31]** basically after two EPO you see
> 基本上在两个Epoch之后，你会看到

**[00:00:31]** basically after two EPO you see overfitting so overfitting means
> 基本上在两个Epoch之后，你会看到过拟合，过拟合意味着

**[00:00:31]** overfitting so overfitting means
> 过拟合，过拟合意味着

**[00:00:31]** overfitting so overfitting means essentially um that you see a larger gap
> 过拟合，过拟合意味着本质上，你会看到验证损失和训练损失之间的差距变大

**[00:00:31]** essentially um that you see a larger gap
> 本质上，你会看到验证损失和训练损失之间的差距变大

**[00:00:31]** essentially um that you see a larger gap between validation and training loss so
> 本质上，你会看到验证损失和训练损失之间的差距变大，所以

**[00:00:31]** between validation and training loss so
> 验证损失和训练损失之间的差距变大，所以

**[00:00:31]** between validation and training loss so the model still improves on the training
> 验证损失和训练损失之间的差距变大，所以模型在训练集上仍在改进

**[00:00:31]** the model still improves on the training
> 模型在训练集上仍在改进

**[00:00:31]** the model still improves on the training set but um it doesn't generalize to the
> 模型在训练集上仍在改进，但它不能泛化到

**[00:00:31]** set but um it doesn't generalize to the
> 但它不能泛化到

**[00:00:31]** set but um it doesn't generalize to the validation set so usually one to two EPO
> 但它不能泛化到验证集，所以通常一到两个Epoch

**[00:00:31]** validation set so usually one to two EPO
> 验证集，所以通常一到两个Epoch

**[00:00:31]** validation set so usually one to two EPO is um Sweet Spot uh you can train more
> 验证集，所以通常一到两个Epoch是最佳点，你可以训练更多

**[00:00:31]** is um Sweet Spot uh you can train more
> 是最佳点，你可以训练更多

**[00:00:31]** is um Sweet Spot uh you can train more and the model will actually still become
> 是最佳点，你可以训练更多，模型实际上仍然会变得

**[00:00:31]** and the model will actually still become
> 模型实际上仍然会变得

**[00:00:31]** and the model will actually still become better at um generating text that looks
> 模型实际上仍然会变得更好，在生成看起来逼真的文本方面

**[00:00:31]** better at um generating text that looks
> 更好，在生成看起来逼真的文本方面

**[00:00:31]** better at um generating text that looks realistic but it's yeah also memorizing
> 更好，在生成看起来逼真的文本方面，但是的，它也在记忆

**[00:00:31]** realistic but it's yeah also memorizing
> 逼真的文本方面，但是的，它也在记忆

**[00:00:31]** realistic but it's yeah also memorizing the training set set a lot at some point
> 这很现实，但在某种程度上也确实是在记忆训练集

**[00:00:31]** the training set set a lot at some point
> 在某种程度上大量记忆训练集

**[00:00:31]** the training set set a lot at some point so it's basically just yeah repeating
> 在某种程度上大量记忆训练集，所以基本上就是在重复

**[00:00:31]** so it's basically just yeah repeating
> 所以基本上就是在重复

**[00:00:31]** so it's basically just yeah repeating the training set which is not a bad idea
> 所以基本上就是在重复训练集，但这并非坏事

**[00:00:31]** the training set which is not a bad idea
> 重复训练集并非坏事

**[00:00:31]** the training set which is not a bad idea um because you can still I mean add some
> 重复训练集并非坏事，因为你可以通过采样增加一些多样性

**[00:00:31]** um because you can still I mean add some
> 因为你可以通过采样增加一些多样性

**[00:00:31]** um because you can still I mean add some variety in the sampling so I'm I don't
> 因为你可以通过采样增加一些多样性，所以我没有相关的幻灯片

**[00:00:31]** variety in the sampling so I'm I don't
> 采样中的多样性，所以我没有

**[00:00:31]** variety in the sampling so I'm I don't have slides on that but you when you
> 采样中的多样性，所以我没有相关的幻灯片，但当你实现生成时

**[00:00:31]** have slides on that but you when you
> 有相关的幻灯片，但当你

**[00:00:31]** have slides on that but you when you implement the generation basically you
> 有相关的幻灯片，但当你实现生成时，基本上会有一些设置

**[00:00:31]** implement the generation basically you
> 实现生成时，基本上

**[00:00:31]** implement the generation basically you have settings like um top K top p and
> 实现生成时，基本上会有一些设置，比如 top K、top p 和 temperature scaling

**[00:00:31]** have settings like um top K top p and
> 有设置，比如 top K、top p 和

**[00:00:31]** have settings like um top K top p and temperature scaling where you can
> 有设置，比如 top K、top p 和 temperature scaling，你可以通过它们

**[00:00:31]** temperature scaling where you can
> temperature scaling，你可以通过

**[00:00:31]** temperature scaling where you can control the amount of yeah Randomness so
> temperature scaling，你可以通过它们控制随机性的程度

**[00:00:31]** control the amount of yeah Randomness so
> 控制随机性的程度

**[00:00:31]** control the amount of yeah Randomness so it doesn't regenerate the training
> 控制随机性的程度，这样它就不会重新生成训练数据

**[00:00:31]** it doesn't regenerate the training
> 它不会重新生成训练

**[00:00:31]** it doesn't regenerate the training data but uh yeah you can also just stop
> 它不会重新生成训练数据，但你也可以提前停止训练

**[00:00:31]** data but uh yeah you can also just stop
> 数据，但你也可以提前停止

**[00:00:31]** data but uh yeah you can also just stop training early especially if you have a
> 数据，但你也可以提前停止训练，特别是当你拥有大数据集和小数据集时

**[00:00:31]** training early especially if you have a
> 提前停止训练，特别是当你拥有

**[00:00:31]** training early especially if you have a large data set and small if you have a
> 提前停止训练，特别是当你拥有大数据集和小数据集时，实际上我会说

**[00:00:32]** large data set and small if you have a
> 大数据集和小数据集，如果你有

**[00:00:32]** large data set and small if you have a small data set actually it I would say
> 大数据集和小数据集，实际上我会说，对于小数据集，延长训练时间几乎是有意义的

**[00:00:32]** small data set actually it I would say
> 小数据集，实际上我会说

**[00:00:32]** small data set actually it I would say it almost makes sense to train longer
> 小数据集，实际上我会说，延长训练时间几乎是有意义的

**[00:00:32]** it almost makes sense to train longer
> 延长训练时间几乎是有意义的

**[00:00:32]** it almost makes sense to train longer until you know your LM generates
> 延长训练时间几乎是有意义的，直到你的 LM 生成连贯的文本

**[00:00:32]** until you know your LM generates
> 直到你的 LM 生成

**[00:00:32]** until you know your LM generates coherent text and then you can control
> 直到你的 LM 生成连贯的文本，然后你可以在采样中更好地控制记忆

**[00:00:32]** coherent text and then you can control
> 连贯的文本，然后你可以控制

**[00:00:32]** coherent text and then you can control the memorization a bit more in the
> 连贯的文本，然后你可以在采样中更好地控制记忆

**[00:00:32]** the memorization a bit more in the
> 在采样中更好地控制记忆

**[00:00:32]** the memorization a bit more in the sampling so one one challenge that LMS
> 在采样中更好地控制记忆，所以 LM 仍然面临的一个挑战是

**[00:00:32]** sampling so one one challenge that LMS
> 采样，所以 LM 仍然面临的一个挑战是

**[00:00:32]** sampling so one one challenge that LMS also still have is um yeah what type of
> 采样，所以 LM 仍然面临的一个挑战是，哪种记忆是好的、期望的

**[00:00:32]** also still have is um yeah what type of
> 仍然面临的一个挑战是，哪种记忆是

**[00:00:32]** also still have is um yeah what type of memorization is good uh desired and
> 仍然面临的一个挑战是，哪种记忆是好的、期望的，哪种是不期望的

**[00:00:32]** memorization is good uh desired and
> 记忆是好的、期望的

**[00:00:32]** memorization is good uh desired and which is not desired so I don't think
> 记忆是好的、期望的，哪种是不期望的，所以我认为

**[00:00:32]** which is not desired so I don't think
> 哪种是不期望的，所以我认为

**[00:00:32]** which is not desired so I don't think there's
> 哪种是不期望的，所以我认为目前还没有好的解决方案

**[00:00:32]** there's
> 目前还没有

**[00:00:32]** there's a uh good solution yet what you almost
> 目前还没有好的解决方案，你几乎想要做的是

**[00:00:32]** a uh good solution yet what you almost
> 好的解决方案，你几乎想要做的是

**[00:00:32]** a uh good solution yet what you almost want to do is kind of like mask um
> 好的解决方案，你几乎想要做的是，像掩码一样

**[00:00:32]** want to do is kind of like mask um
> 想要做的是，像掩码一样

**[00:00:32]** want to do is kind of like mask um certain things to be memorized for
> 想要做的是，像掩码一样，屏蔽某些需要被记忆的内容

**[00:00:32]** certain things to be memorized for
> 某些需要被记忆的内容

**[00:00:32]** certain things to be memorized for example if you think about Chip and A
> 某些需要被记忆的内容，例如，想想 ChatGPT

**[00:00:32]** example if you think about Chip and A
> 例如，想想 ChatGPT

**[00:00:32]** example if you think about Chip and A lot of people use chip for you know like
> 例如，想想 ChatGPT，很多人用它来问历史问题

**[00:00:32]** lot of people use chip for you know like
> 很多人用它来问历史问题

**[00:00:32]** lot of people use chip for you know like uh asking it history questions and of
> 很多人用它来问历史问题，当然你希望 LM 记住历史日期

**[00:00:32]** uh asking it history questions and of
> 问它历史问题，当然

**[00:00:32]** uh asking it history questions and of course you want the LM to memorize
> 问它历史问题，当然你希望 LM 记住历史日期

**[00:00:32]** course you want the LM to memorize
> 你希望 LM 记住

**[00:00:32]** course you want the LM to memorize historic dates if it would make up let's
> 你希望 LM 记住历史日期，如果它编造了独立日的日期

**[00:00:32]** historic dates if it would make up let's
> 历史日期，如果它编造了

**[00:00:32]** historic dates if it would make up let's say the date for Independence Day and
> 历史日期，如果它编造了独立日的日期，告诉你比如是12月12日

**[00:00:32]** say the date for Independence Day and
> 独立日的日期，告诉你

**[00:00:32]** say the date for Independence Day and tell you it's I don't know December 12th
> 独立日的日期，告诉你比如是12月12日，那会非常误导人

**[00:00:32]** tell you it's I don't know December 12th
> 告诉你比如是12月12日

**[00:00:32]** tell you it's I don't know December 12th U that would be pretty misleading so we
> 告诉你比如是12月12日，那会非常误导人，所以我们确实希望 LM 记住

**[00:00:32]** U that would be pretty misleading so we
> 那会非常误导人，所以我们

**[00:00:32]** U that would be pretty misleading so we do want the LM actually to memorize
> 那会非常误导人，所以我们确实希望 LM 记住训练集中的某些内容

**[00:00:32]** do want the LM actually to memorize
> 确实希望 LM 记住

**[00:00:32]** do want the LM actually to memorize certain things from the training set
> 确实希望 LM 记住训练集中的某些内容

**[00:00:32]** certain things from the training set
> 训练集中的某些内容

**[00:00:32]** certain things from the training set it's just that we um also want wanted
> 训练集中的某些内容，只是我们不希望它记住所有内容

**[00:00:32]** it's just that we um also want wanted
> 只是我们不希望

**[00:00:32]** it's just that we um also want wanted not to memorize everything because then
> 只是我们不希望它记住所有内容，因为那样它就无法生成其他内容

**[00:00:33]** not to memorize everything because then
> 不希望它记住所有内容，因为那样

**[00:00:33]** not to memorize everything because then it can't generate um anything else that
> 不希望它记住所有内容，因为那样它就无法生成其他内容

**[00:00:33]** it can't generate um anything else that
> it can't generate um anything else that

**[00:00:33]** it can't generate um anything else that is not in the training set so that's
> 它无法生成训练集中不存在的任何内容，所以这

**[00:00:33]** is not in the training set so that's
> 不在训练集中，所以这

**[00:00:33]** is not in the training set so that's like it's a bit tricky I think it's an
> 不在训练集中，所以这有点棘手，我认为这是一个

**[00:00:33]** like it's a bit tricky I think it's an
> 有点棘手，我认为这是一个

**[00:00:33]** like it's a bit tricky I think it's an unsolved problem where yeah we do want
> 有点棘手，我认为这是一个未解决的问题，我们确实想要

**[00:00:33]** unsolved problem where yeah we do want
> 未解决的问题，我们确实想要

**[00:00:33]** unsolved problem where yeah we do want some memorization but of course uh yeah
> 未解决的问题，我们确实想要一些记忆，但当然

**[00:00:33]** some memorization but of course uh yeah
> 一些记忆，但当然

**[00:00:33]** some memorization but of course uh yeah we don't want to just you know copy
> 一些记忆，但当然我们不想只是复制

**[00:00:33]** we don't want to just you know copy
> 我们不想只是复制

**[00:00:33]** we don't want to just you know copy paste um training
> 我们不想只是复制粘贴训练

**[00:00:33]** paste um training
> 粘贴训练

**[00:00:33]** paste um training data um yeah so by the way if you are
> 粘贴训练数据，顺便说一下，如果你

**[00:00:33]** data um yeah so by the way if you are
> 数据，顺便说一下，如果你

**[00:00:33]** data um yeah so by the way if you are interested in pre-training my colleague
> 数据，顺便说一下，如果你对预训练感兴趣，我的同事

**[00:00:33]** interested in pre-training my colleague
> 对预训练感兴趣，我的同事

**[00:00:33]** interested in pre-training my colleague um Adrian put together a studio um on on
> 对预训练感兴趣，我的同事Adrian整理了一个studio

**[00:00:33]** um Adrian put together a studio um on on
> Adrian整理了一个studio

**[00:00:33]** um Adrian put together a studio um on on lightning on our platform
> Adrian整理了一个studio，关于lightning，在我们的平台上

**[00:00:33]** lightning on our platform
> lightning，在我们的平台上

**[00:00:33]** lightning on our platform where it's basically you don't have to
> lightning，在我们的平台上，基本上你不需要

**[00:00:33]** where it's basically you don't have to
> 基本上你不需要

**[00:00:33]** where it's basically you don't have to worry about installing anything or the
> 基本上你不需要担心安装任何东西或

**[00:00:33]** worry about installing anything or the
> 担心安装任何东西或

**[00:00:33]** worry about installing anything or the machines it's all in in the cloud but
> 担心安装任何东西或机器，一切都在云端，但

**[00:00:33]** machines it's all in in the cloud but
> 机器，一切都在云端，但

**[00:00:33]** machines it's all in in the cloud but yeah it's a training 1.1 billion model
> 机器，一切都在云端，但这是一个训练11亿参数的模型

**[00:00:33]** yeah it's a training 1.1 billion model
> 这是一个训练11亿参数的模型

**[00:00:33]** yeah it's a training 1.1 billion model um on 1 billion tokens takes about 4
> 这是一个训练11亿参数的模型，在10亿tokens上，大约需要4

**[00:00:33]** um on 1 billion tokens takes about 4
> 在10亿tokens上，大约需要4

**[00:00:33]** um on 1 billion tokens takes about 4 weeks on 64 a100 gpus but just as an
> 在10亿tokens上，大约需要4周，使用64个a100 gpu，但只是作为一个

**[00:00:33]** weeks on 64 a100 gpus but just as an
> 周，使用64个a100 gpu，但只是作为一个

**[00:00:33]** weeks on 64 a100 gpus but just as an example um and yeah pre-training takes a
> 周，使用64个a100 gpu，但只是作为一个例子，预训练需要

**[00:00:33]** example um and yeah pre-training takes a
> 例子，预训练需要

**[00:00:33]** example um and yeah pre-training takes a long time it's uh usually not necessary
> 例子，预训练需要很长时间，通常没有必要

**[00:00:33]** long time it's uh usually not necessary
> 很长时间，通常没有必要

**[00:00:33]** long time it's uh usually not necessary if you are interested in adapting uh llm
> 很长时间，通常没有必要，如果你对调整LLM感兴趣

**[00:00:33]** if you are interested in adapting uh llm
> 如果你对调整LLM感兴趣

**[00:00:33]** if you are interested in adapting uh llm for a certain task usually we start with
> 如果你对调整LLM以适应特定任务感兴趣，通常我们从

**[00:00:33]** for a certain task usually we start with
> 适应特定任务感兴趣，通常我们从

**[00:00:33]** for a certain task usually we start with a pre-trained llm but yeah if you're
> 适应特定任务感兴趣，通常我们从预训练的LLM开始，但如果你

**[00:00:34]** a pre-trained llm but yeah if you're
> 预训练的LLM开始，但如果你

**[00:00:34]** a pre-trained llm but yeah if you're interested in pre-training um check out
> 预训练的LLM开始，但如果你对预训练感兴趣，请查看

**[00:00:34]** interested in pre-training um check out
> 对预训练感兴趣，请查看

**[00:00:34]** interested in pre-training um check out the studio by my colleague Adrien um so
> 对预训练感兴趣，请查看我同事Adrien的studio

**[00:00:34]** the studio by my colleague Adrien um so
> 我同事Adrien的studio

**[00:00:34]** the studio by my colleague Adrien um so but that brings us also to the topic um
> 我同事Adrien的studio，但这也引出了我们的主题

**[00:00:34]** but that brings us also to the topic um
> 但这也引出了我们的主题

**[00:00:34]** but that brings us also to the topic um most of the time when we work with LMS
> 但这也引出了我们的主题，大多数时候，当我们使用LMS时

**[00:00:34]** most of the time when we work with LMS
> 大多数时候，当我们使用LMS时

**[00:00:34]** most of the time when we work with LMS we work with pre-trained weights um so
> 大多数时候，当我们使用LMS时，我们使用预训练权重

**[00:00:34]** we work with pre-trained weights um so
> 我们使用预训练权重

**[00:00:34]** we work with pre-trained weights um so there is usually someone who was kind
> 我们使用预训练权重，所以通常有人很友好

**[00:00:34]** there is usually someone who was kind
> 所以通常有人很友好

**[00:00:34]** there is usually someone who was kind enough to share the weights openly there
> 所以通常有人很友好地公开分享权重，有很多公司或研究

**[00:00:34]** enough to share the weights openly there
> 地公开分享权重，有很多公司或研究

**[00:00:34]** enough to share the weights openly there are many companies or research
> 地公开分享权重，有很多公司或研究机构这样做，所以在liity

**[00:00:34]** are many companies or research
> 机构这样做，所以在liity

**[00:00:34]** are many companies or research institutes um who do that so um in liity
> 机构这样做，所以在liity库中，我帮助开发了

**[00:00:34]** institutes um who do that so um in liity
> 库中，我帮助开发了

**[00:00:34]** institutes um who do that so um in liity the library I helped developing with my
> 库中，我帮助开发了，与我的同事Adrien、Carlos和Luca一起

**[00:00:34]** the library I helped developing with my
> 与我的同事Adrien、Carlos和Luca一起

**[00:00:34]** the library I helped developing with my colleagues um Adrien Carlos and Luca we
> 与我的同事Adrien、Carlos和Luca一起，我们支持超过20个LM模型

**[00:00:34]** colleagues um Adrien Carlos and Luca we
> 我们支持超过20个LM模型

**[00:00:34]** colleagues um Adrien Carlos and Luca we uh yeah support more than 20 LM model
> 我们支持超过20个LM模型权重，全部基于相同的模型

**[00:00:34]** uh yeah support more than 20 LM model
> 权重，全部基于相同的模型

**[00:00:34]** uh yeah support more than 20 LM model weights all based on um the same model
> 权重，全部基于相同的模型架构，就像我之前告诉你的GPD

**[00:00:34]** weights all based on um the same model
> 架构，就像我之前告诉你的GPD

**[00:00:34]** weights all based on um the same model architecture like I told you before GPD
> 架构，就像我之前告诉你的GPD，但当然有轻微的架构

**[00:00:34]** architecture like I told you before GPD
> 但当然有轻微的架构

**[00:00:34]** architecture like I told you before GPD but of course with slight architecture
> 但当然有轻微的架构变化，但我们尽量保持

**[00:00:34]** but of course with slight architecture
> 变化，但我们尽量保持

**[00:00:34]** but of course with slight architecture variations but yeah we try to keep the
> 变化，但我们尽量保持代码库可读，以便它

**[00:00:34]** variations but yeah we try to keep the
> 代码库可读，以便它

**[00:00:34]** variations but yeah we try to keep the code base um readable so that it's
> 代码库可读，以便它实际上也是一个不错的库，用于

**[00:00:34]** code base um readable so that it's
> 实际上也是一个不错的库，用于

**[00:00:34]** code base um readable so that it's actually also maybe a nice library to
> 实际上也是一个不错的库，用于研究某些

**[00:00:34]** actually also maybe a nice library to
> 研究某些

**[00:00:34]** actually also maybe a nice library to study the difference between certain
> 研究某些

**[00:00:34]** study the difference between certain
> 研究某些

**[00:00:34]** study the difference between certain llms so in any case if you are
> 研究某些LLM之间的差异，这样在任何情况下，如果你

**[00:00:34]** llms so in any case if you are
> LLM，这样在任何情况下，如果你

**[00:00:34]** llms so in any case if you are interested in using that you can
> LLM，这样在任何情况下，如果你有兴趣使用它们，你可以

**[00:00:34]** interested in using that you can
> 有兴趣使用它们，你可以

**[00:00:34]** interested in using that you can actually download weights by just LGBT
> 有兴趣使用它们，你可以直接通过LGBT下载权重

**[00:00:34]** actually download weights by just LGBT
> 直接通过LGBT下载权重

**[00:00:34]** actually download weights by just LGBT download um the name of the model and
> 直接通过LGBT下载权重，下载模型的名称，然后

**[00:00:34]** download um the name of the model and
> 下载模型的名称，然后

**[00:00:34]** download um the name of the model and then you can chat it chat with it find
> 下载模型的名称，然后你可以与它聊天，与它交互，进行

**[00:00:35]** then you can chat it chat with it find
> 然后你可以与它聊天，与它交互，进行

**[00:00:35]** then you can chat it chat with it find tune it further pre-train it or
> 然后你可以与它聊天，与它交互，进一步fine-tune它，pre-train它，或者

**[00:00:35]** tune it further pre-train it or
> 进一步fine-tune它，pre-train它，或者

**[00:00:35]** tune it further pre-train it or pre-train it even from scratch or or
> 进一步fine-tune它，pre-train它，甚至从头开始pre-train，或者

**[00:00:35]** pre-train it even from scratch or or
> 从头开始pre-train，或者

**[00:00:35]** pre-train it even from scratch or or deploy
> 从头开始pre-train，或者部署

**[00:00:35]** deploy
> 部署

**[00:00:35]** deploy it so yeah like I said most of the time
> 部署它。所以，是的，就像我说的，大多数时候

**[00:00:35]** it so yeah like I said most of the time
> 它。所以，是的，就像我说的，大多数时候

**[00:00:35]** it so yeah like I said most of the time when we work with llms we are not
> 它。所以，是的，就像我说的，大多数时候当我们处理LLM时，我们并不

**[00:00:35]** when we work with llms we are not
> 当我们处理LLM时，我们并不

**[00:00:35]** when we work with llms we are not interested so much in pre-training but
> 当我们处理LLM时，我们并不那么关注pre-training，而是

**[00:00:35]** interested so much in pre-training but
> 那么关注pre-training，而是

**[00:00:35]** interested so much in pre-training but um in adapting it for Downstream tasks
> 那么关注pre-training，而是关注将其适配到下游任务

**[00:00:35]** um in adapting it for Downstream tasks
> 关注将其适配到下游任务

**[00:00:35]** um in adapting it for Downstream tasks for example by um fine-tuning
> 关注将其适配到下游任务，例如通过fine-tuning

**[00:00:35]** for example by um fine-tuning
> 例如通过fine-tuning

**[00:00:35]** for example by um fine-tuning llms so here
> 例如通过fine-tuning LLM。所以这里

**[00:00:35]** llms so here
> LLM。所以这里

**[00:00:35]** llms so here um I have two steps for the fine tuning
> LLM。所以这里，对于fine-tuning，我有两个步骤

**[00:00:35]** um I have two steps for the fine tuning
> 对于fine-tuning，我有两个步骤

**[00:00:35]** um I have two steps for the fine tuning there's one the fine-tuning with class
> 对于fine-tuning，我有两个步骤：一个是使用类别标签进行fine-tuning

**[00:00:35]** there's one the fine-tuning with class
> 一个是使用类别标签进行fine-tuning

**[00:00:35]** there's one the fine-tuning with class labels um it's an example to develop a
> 一个是使用类别标签进行fine-tuning，这是一个开发

**[00:00:35]** labels um it's an example to develop a
> 分类器的例子，然后我们还会讨论

**[00:00:35]** labels um it's an example to develop a classifier and then we will also talk
> 分类器的例子，然后我们还会讨论

**[00:00:35]** classifier and then we will also talk
> 分类器，然后我们还会讨论

**[00:00:35]** classifier and then we will also talk about building a personal
> 分类器，然后我们还会讨论构建一个个人

**[00:00:35]** about building a personal
> 构建一个个人

**[00:00:35]** about building a personal assistant so if you are interested for
> 构建一个个人助手。所以如果你有兴趣，例如

**[00:00:35]** assistant so if you are interested for
> 助手。所以如果你有兴趣，例如

**[00:00:35]** assistant so if you are interested for example in text classification a popular
> 助手。所以如果你有兴趣，例如在文本分类中，一个流行的

**[00:00:35]** example in text classification a popular
> 在文本分类中，一个流行的

**[00:00:35]** example in text classification a popular example would be classifying if a text
> 在文本分类中，一个流行的例子是判断一段文本

**[00:00:35]** example would be classifying if a text
> 例子是判断一段文本

**[00:00:35]** example would be classifying if a text message or email is um ham or spam so
> 例子是判断一段文本消息或电子邮件是正常邮件还是垃圾邮件。所以

**[00:00:35]** message or email is um ham or spam so
> 消息或电子邮件是正常邮件还是垃圾邮件。所以

**[00:00:35]** message or email is um ham or spam so for that I mean it's almost like a
> 消息或电子邮件是正常邮件还是垃圾邮件。所以，对于这个，我的意思是这几乎就像一个

**[00:00:35]** for that I mean it's almost like a
> 对于这个，我的意思是这几乎就像一个

**[00:00:35]** for that I mean it's almost like a classic machine learning problem you
> 对于这个，我的意思是这几乎就像一个经典的机器学习问题，你

**[00:00:35]** classic machine learning problem you
> 经典的机器学习问题，你

**[00:00:35]** classic machine learning problem you have a label ham or spam and the text
> 经典的机器学习问题，你有一个标签（正常或垃圾）和文本

**[00:00:35]** have a label ham or spam and the text
> 有一个标签（正常或垃圾）和文本

**[00:00:35]** have a label ham or spam and the text you want to classify and um all you have
> 有一个标签（正常或垃圾）和文本，你想要进行分类。并且，所有你需要

**[00:00:35]** you want to classify and um all you have
> 进行分类。并且，所有你需要

**[00:00:35]** you want to classify and um all you have to do if you want to adapt an existing
> 进行分类。并且，所有你需要做的，如果你想适配一个现有的

**[00:00:36]** to do if you want to adapt an existing
> 做的，如果你想适配一个现有的

**[00:00:36]** to do if you want to adapt an existing model so what all you have to do is you
> 做的，如果你想适配一个现有的模型，那么你需要做的就是

**[00:00:36]** model so what all you have to do is you
> 模型，那么你需要做的就是

**[00:00:36]** model so what all you have to do is you have to replace the output layer so
> 模型，那么你需要做的就是替换输出层。所以

**[00:00:36]** have to replace the output layer so
> 替换输出层。所以

**[00:00:36]** have to replace the output layer so originally the output layer in this case
> 替换输出层。所以，最初，在这种情况下，输出层

**[00:00:36]** originally the output layer in this case
> 最初，在这种情况下，输出层

**[00:00:36]** originally the output layer in this case of the smallest GPD model has 768 hidden
> 最初，在这种情况下，最小的GPT模型的输出层有768个隐藏

**[00:00:36]** of the smallest GPD model has 768 hidden
> 节点，然后它映射到50,000个输出

**[00:00:36]** of the smallest GPD model has 768 hidden noes and then it maps to 50,000 output
> 节点，然后它映射到50,000个输出

**[00:00:36]** noes and then it maps to 50,000 output
> 节点，所以这里的输出节点

**[00:00:36]** noes and then it maps to 50,000 output nodes so the output noes here um
> 节点，所以这里的输出节点

**[00:00:36]** nodes so the output noes here um
> 节点，所以这里的输出节点代表了词汇表的大小，所以

**[00:00:36]** nodes so the output noes here um represent the size of the vocabulary so
> 代表了词汇表的大小，所以

**[00:00:36]** represent the size of the vocabulary so
> 代表了词汇表的大小，所以

**[00:00:36]** represent the size of the vocabulary so how many um yeah words are in this
> 代表了词汇表的大小，所以这个模型所训练的tokenizer中有多少个词

**[00:00:36]** how many um yeah words are in this
> 这个模型所训练的tokenizer中有多少个词

**[00:00:36]** how many um yeah words are in this tokenizer that the model has been
> 这个模型所训练的tokenizer中有多少个词。通常发生的情况是

**[00:00:36]** tokenizer that the model has been
> 通常发生的情况是

**[00:00:36]** tokenizer that the model has been trained on and so what happens usually
> 通常发生的情况是，它从这个映射回

**[00:00:36]** trained on and so what happens usually
> 它从这个映射回

**[00:00:36]** trained on and so what happens usually is that it maps from this back into the
> 它从这个映射回单词，但在这种情况下，如果我们只

**[00:00:36]** is that it maps from this back into the
> 单词，但在这种情况下，如果我们只

**[00:00:36]** is that it maps from this back into the words but in this case if we are only
> is that it maps from this back into the words but in this case if we are only

**[00:00:36]** words but in this case if we are only
> words but in this case if we are only

**[00:00:36]** words but in this case if we are only interested in classifying ham and spam
> 单词，但在这个案例中，如果我们只对分类垃圾邮件和正常邮件感兴趣

**[00:00:36]** interested in classifying ham and spam
> 只对分类垃圾邮件和正常邮件感兴趣

**[00:00:36]** interested in classifying ham and spam we don't need 50,000 words here we only
> 只对分类垃圾邮件和正常邮件感兴趣，我们这里不需要50,000个单词，只需要

**[00:00:36]** we don't need 50,000 words here we only
> 我们这里不需要50,000个单词，只需要

**[00:00:36]** we don't need 50,000 words here we only need um two so um we can actually
> 我们这里不需要50,000个单词，只需要两个，所以实际上我们可以

**[00:00:36]** need um two so um we can actually
> 只需要两个，所以实际上我们可以

**[00:00:36]** need um two so um we can actually replace this um output layer by a
> 只需要两个，所以实际上我们可以将这个输出层替换为一个

**[00:00:36]** replace this um output layer by a
> 将这个输出层替换为一个

**[00:00:36]** replace this um output layer by a smaller layer so this one Maps now from
> 将这个输出层替换为一个更小的层，所以这个层现在从

**[00:00:36]** smaller layer so this one Maps now from
> 更小的层，所以这个层现在从

**[00:00:36]** smaller layer so this one Maps now from 768 to yeah ham and spam instead of 5
> 更小的层，所以这个层现在从768映射到正常邮件和垃圾邮件，而不是5

**[00:00:36]** 768 to yeah ham and spam instead of 5
> 768映射到正常邮件和垃圾邮件，而不是5

**[00:00:36]** 768 to yeah ham and spam instead of 5 50,000 wasn't basically so it's just
> 768映射到正常邮件和垃圾邮件，而不是5万，基本上就是这样，这只是

**[00:00:36]** 50,000 wasn't basically so it's just
> 5万，基本上就是这样，这只是

**[00:00:36]** 50,000 wasn't basically so it's just also for efficiency really it helps
> 5万，基本上就是这样，这只是为了效率，它确实有助于

**[00:00:37]** also for efficiency really it helps
> 为了效率，它确实有助于

**[00:00:37]** also for efficiency really it helps really getting better performance out of
> 为了效率，它确实有助于获得更好的性能

**[00:00:37]** really getting better performance out of
> 获得更好的性能

**[00:00:37]** really getting better performance out of it helps with efficiency and so forth so
> 获得更好的性能，有助于提高效率等等，所以

**[00:00:37]** it helps with efficiency and so forth so
> 有助于提高效率等等，所以

**[00:00:37]** it helps with efficiency and so forth so it's really simple it's just like one
> 有助于提高效率等等，所以这真的很简单，就像一行

**[00:00:37]** it's really simple it's just like one
> 这真的很简单，就像一行

**[00:00:37]** it's really simple it's just like one line of code I have it in my GitHub
> 这真的很简单，就像一行代码，我在我的GitHub仓库里有

**[00:00:37]** line of code I have it in my GitHub
> 一行代码，我在我的GitHub仓库里有

**[00:00:37]** line of code I have it in my GitHub repository one line of code you change
> 一行代码，我在我的GitHub仓库里有一行代码，你改变

**[00:00:37]** repository one line of code you change
> 一行代码，你改变

**[00:00:37]** repository one line of code you change this um layer and then you find un it
> 一行代码，你改变这个层，然后你微调它

**[00:00:37]** this um layer and then you find un it
> 这个层，然后你微调它

**[00:00:37]** this um layer and then you find un it and um yeah so basically we evaluate it
> 这个层，然后你微调它，嗯，是的，所以基本上我们评估它

**[00:00:37]** and um yeah so basically we evaluate it
> 嗯，是的，所以基本上我们评估它

**[00:00:37]** and um yeah so basically we evaluate it the same way as before we track the loss
> 嗯，是的，所以基本上我们以同样的方式评估它，我们跟踪loss

**[00:00:37]** the same way as before we track the loss
> 以同样的方式评估它，我们跟踪loss

**[00:00:37]** the same way as before we track the loss um during the training this actually
> 以同样的方式评估它，我们在训练期间跟踪loss，这实际上

**[00:00:37]** um during the training this actually
> 在训练期间，这实际上

**[00:00:37]** um during the training this actually looks pretty good no overfitting but in
> 在训练期间，这实际上看起来不错，没有过拟合，但

**[00:00:37]** looks pretty good no overfitting but in
> 看起来不错，没有过拟合，但

**[00:00:37]** looks pretty good no overfitting but in addition now we have a Target task so
> 看起来不错，没有过拟合，但除此之外，现在我们有一个目标任务，所以

**[00:00:37]** addition now we have a Target task so
> 除此之外，现在我们有一个目标任务，所以

**[00:00:37]** addition now we have a Target task so instead of just you know looking at the
> 除此之外，现在我们有一个目标任务，所以不是仅仅查看

**[00:00:37]** instead of just you know looking at the
> 不是仅仅查看

**[00:00:37]** instead of just you know looking at the next word
> 不是仅仅查看下一个单词

**[00:00:37]** next word
> 下一个单词

**[00:00:37]** next word what we can also do is we can take a
> 下一个单词，我们还可以做的是，我们可以

**[00:00:37]** what we can also do is we can take a
> 我们还可以做的是，我们可以

**[00:00:37]** what we can also do is we can take a look at the classification accuracy so
> 我们还可以做的是，我们可以查看分类准确率，所以

**[00:00:37]** look at the classification accuracy so
> 查看分类准确率，所以

**[00:00:37]** look at the classification accuracy so classification accuracy here means um
> 查看分类准确率，所以这里的分类准确率意味着

**[00:00:37]** classification accuracy here means um
> 这里的分类准确率意味着

**[00:00:37]** classification accuracy here means um how many examples it classifies
> 这里的分类准确率意味着它正确分类了多少个样本

**[00:00:37]** how many examples it classifies
> 它正确分类了多少个样本

**[00:00:37]** how many examples it classifies correctly so how many if I have 100 um
> 它正确分类了多少个样本，所以如果我有100条

**[00:00:37]** correctly so how many if I have 100 um
> 正确，所以如果我有100条

**[00:00:37]** correctly so how many if I have 100 um messages and uh I classify 80 correctly
> 正确，所以如果我有100条消息，并且我正确分类了80条

**[00:00:37]** messages and uh I classify 80 correctly
> 消息，并且我正确分类了80条

**[00:00:37]** messages and uh I classify 80 correctly I would have a 80% accuracy for example
> 消息，并且我正确分类了80条，那么我的准确率就是80%，例如

**[00:00:37]** I would have a 80% accuracy for example
> 那么我的准确率就是80%，例如

**[00:00:37]** I would have a 80% accuracy for example so it's it's more like helps us um to
> 那么我的准确率就是80%，例如，这更像是有助于我们

**[00:00:37]** so it's it's more like helps us um to
> 这更像是有助于我们

**[00:00:37]** so it's it's more like helps us um to evaluate the as a human the performance
> 这更像是有助于我们作为人类评估模型在目标任务上的性能

**[00:00:37]** evaluate the as a human the performance
> 评估模型在目标任务上的性能

**[00:00:37]** evaluate the as a human the performance of the model on the target task now one
> 评估模型在目标任务上的性能，现在有人

**[00:00:37]** of the model on the target task now one
> 模型在目标任务上的性能，现在有人

**[00:00:37]** of the model on the target task now one might ask why don't we just train the
> 模型在目标任务上的性能，现在有人可能会问，为什么我们不直接训练

**[00:00:37]** might ask why don't we just train the
> 可能会问，为什么我们不直接训练

**[00:00:37]** might ask why don't we just train the model to um to yeah optimize or maximize
> 可能会问，为什么我们不直接训练模型来优化或最大化

**[00:00:38]** model to um to yeah optimize or maximize
> 模型来优化或最大化

**[00:00:38]** model to um to yeah optimize or maximize the accuracy so that's unfortunately not
> 模型来优化或最大化准确率呢？不幸的是，这

**[00:00:38]** the accuracy so that's unfortunately not
> 准确率呢？不幸的是，这

**[00:00:38]** the accuracy so that's unfortunately not possible because accuracy is not
> 准确率呢？不幸的是，这是不可能的，因为准确率不是

**[00:00:38]** possible because accuracy is not
> 不可能的，因为准确率不是

**[00:00:38]** possible because accuracy is not differentiable so we do still need the
> 不可能的，因为准确率是不可微的，所以我们仍然需要

**[00:00:38]** differentiable so we do still need the
> 不可微的，所以我们仍然需要

**[00:00:38]** differentiable so we do still need the loss as the loss function to minimize
> 不可微的，所以我们仍然需要loss作为损失函数来最小化

**[00:00:38]** loss as the loss function to minimize
> loss作为损失函数来最小化

**[00:00:38]** loss as the loss function to minimize during training because yeah we can
> loss作为损失函数在训练期间最小化，因为是的，我们可以

**[00:00:38]** during training because yeah we can
> 在训练期间，因为是的，我们可以

**[00:00:38]** during training because yeah we can calculate a loss derivative or gradient
> 在训练期间，因为是的，我们可以计算loss的导数或gradient

**[00:00:38]** calculate a loss derivative or gradient
> 计算loss的导数或gradient

**[00:00:38]** calculate a loss derivative or gradient with respect to the model weights where
> 计算loss的导数或gradient，相对于模型权重

**[00:00:38]** with respect to the model weights where
> 相对于模型权重

**[00:00:38]** with respect to the model weights where accuracy is not differentiable so we we
> 关于模型权重，由于准确率不可微，所以我们

**[00:00:38]** accuracy is not differentiable so we we
> 准确率不可微，所以我们

**[00:00:38]** accuracy is not differentiable so we we can't use that to train the model but we
> 准确率不可微，所以我们不能用它来训练模型，但我们可以

**[00:00:38]** can't use that to train the model but we
> 不能用它来训练模型，但我们可以

**[00:00:38]** can't use that to train the model but we can take a look at that during training
> 不能用它来训练模型，但我们可以训练期间观察它

**[00:00:38]** can take a look at that during training
> 训练期间观察它

**[00:00:38]** can take a look at that during training and then we can get an idea of how good
> 训练期间观察它，然后就能了解模型

**[00:00:38]** and then we can get an idea of how good
> 然后就能了解模型

**[00:00:38]** and then we can get an idea of how good our model is so here it's almost 100%
> 然后就能了解模型有多好，这里训练准确率几乎达到100%

**[00:00:38]** our model is so here it's almost 100%
> 模型有多好，这里训练准确率几乎达到100%

**[00:00:38]** our model is so here it's almost 100% training accuracy maybe I would say
> 模型有多好，这里训练准确率几乎达到100%，我可能会说

**[00:00:38]** training accuracy maybe I would say
> 训练准确率，我可能会说

**[00:00:38]** training accuracy maybe I would say 97% validation accuracy and once you're
> 训练准确率，我可能会说97%的验证准确率，一旦你

**[00:00:38]** 97% validation accuracy and once you're
> 97%的验证准确率，一旦你

**[00:00:38]** 97% validation accuracy and once you're done with that you would also take a
> 97%的验证准确率，一旦你完成这个，你还会查看

**[00:00:38]** done with that you would also take a
> 完成这个，你还会查看

**[00:00:38]** done with that you would also take a look at the test
> 完成这个，你还会查看测试

**[00:00:38]** look at the test
> 查看测试

**[00:00:38]** look at the test accuracy um but by the way also yeah we
> 查看测试准确率，不过顺便说一下，是的，我们

**[00:00:38]** accuracy um but by the way also yeah we
> 准确率，不过顺便说一下，是的，我们

**[00:00:38]** accuracy um but by the way also yeah we don't need to fine-tune all the layers I
> 准确率，不过顺便说一下，是的，我们不需要微调所有层，我

**[00:00:38]** don't need to fine-tune all the layers I
> 不需要微调所有层，我

**[00:00:38]** don't need to fine-tune all the layers I I told you yeah you can replace that
> 不需要微调所有层，我告诉过你，你可以替换那个

**[00:00:38]** I told you yeah you can replace that
> 我告诉过你，你可以替换那个

**[00:00:38]** I told you yeah you can replace that output layer and then um find to the
> 我告诉过你，你可以替换那个输出层，然后微调

**[00:00:38]** output layer and then um find to the
> 输出层，然后微调

**[00:00:38]** output layer and then um find to the whole model um actually you don't need
> 输出层，然后微调整个模型，实际上你不需要

**[00:00:38]** whole model um actually you don't need
> 整个模型，实际上你不需要

**[00:00:38]** whole model um actually you don't need to update all the layers during training
> 整个模型，实际上你不需要在训练期间更新所有层

**[00:00:38]** to update all the layers during training
> 在训练期间更新所有层

**[00:00:38]** to update all the layers during training so here I did some experiments and um
> 在训练期间更新所有层，所以我做了一些实验，并且

**[00:00:38]** so here I did some experiments and um
> 所以我做了一些实验，并且

**[00:00:38]** so here I did some experiments and um for example on the left hand side I'm
> 所以我做了一些实验，例如在左侧，我

**[00:00:38]** for example on the left hand side I'm
> 例如在左侧，我

**[00:00:38]** for example on the left hand side I'm only updating the uh last layer it's a
> 例如在左侧，我只更新最后一层，这是一个

**[00:00:38]** only updating the uh last layer it's a
> 只更新最后一层，这是一个

**[00:00:38]** only updating the uh last layer it's a slightly different data set um so that's
> 只更新最后一层，这是一个略有不同的数据集，所以

**[00:00:39]** slightly different data set um so that's
> 略有不同的数据集，所以

**[00:00:39]** slightly different data set um so that's why the accuracies are look a bit
> 略有不同的数据集，所以准确率看起来有点

**[00:00:39]** why the accuracies are look a bit
> 准确率看起来有点

**[00:00:39]** why the accuracies are look a bit different than on the previous slide but
> 准确率看起来有点不同于上一张幻灯片，但

**[00:00:39]** different than on the previous slide but
> 不同于上一张幻灯片，但

**[00:00:39]** different than on the previous slide but um so for reference here you get about
> 不同于上一张幻灯片，但作为参考，这里你得到大约

**[00:00:39]** um so for reference here you get about
> 作为参考，这里你得到大约

**[00:00:39]** um so for reference here you get about 75% if you only update the last layer
> 作为参考，这里你得到大约75%，如果只更新最后一层

**[00:00:39]** 75% if you only update the last layer
> 75%，如果只更新最后一层

**[00:00:39]** 75% if you only update the last layer you get about I would say yeah 90
> 75%，如果只更新最后一层，你得到大约，我会说，是的，90

**[00:00:39]** you get about I would say yeah 90
> 你得到大约，我会说，是的，90

**[00:00:39]** you get about I would say yeah 90 something percent if you update or
> 你得到大约，我会说，是的，90%左右，如果你更新或

**[00:00:39]** something percent if you update or
> 左右，如果你更新或

**[00:00:39]** something percent if you update or fine-tune all the layers but you can see
> 左右，如果你更新或微调所有层，但你可以看到

**[00:00:39]** fine-tune all the layers but you can see
> 微调所有层，但你可以看到

**[00:00:39]** fine-tune all the layers but you can see it's not necessary here you I'm only
> 微调所有层，但你可以看到这并非必要，这里我只

**[00:00:39]** it's not necessary here you I'm only
> 这并非必要，这里我只

**[00:00:39]** it's not necessary here you I'm only updating the last two layers plus the
> 这并非必要，这里我只更新最后两层加上

**[00:00:39]** updating the last two layers plus the
> 更新最后两层加上

**[00:00:39]** updating the last two layers plus the last two Transformer blocks and you can
> 更新最后两层加上最后两个Transformer块，你可以

**[00:00:39]** last two Transformer blocks and you can
> 最后两个Transformer块，你可以

**[00:00:39]** last two Transformer blocks and you can say okay it doesn't really get better
> 最后两个Transformer块，你可以说，好吧，它并没有真正变得更好

**[00:00:39]** say okay it doesn't really get better
> 说，好吧，它并没有真正变得更好

**[00:00:39]** say okay it doesn't really get better after this point so it really doesn't
> 说，好吧，它并没有真正变得更好超过这个点，所以确实不需要

**[00:00:39]** after this point so it really doesn't
> 超过这个点，所以确实不需要

**[00:00:39]** after this point so it really doesn't require updating all the layers it
> 超过这个点，所以确实不需要更新所有层，它

**[00:00:39]** require updating all the layers it
> 更新所有层，它

**[00:00:39]** require updating all the layers it really yeah is enough to update the last
> 更新所有层，它确实，是的，更新最后

**[00:00:39]** really yeah is enough to update the last
> 确实，是的，更新最后

**[00:00:39]** really yeah is enough to update the last few layers here and it's also actually
> 确实，是的，更新最后几层就足够了，而且实际上

**[00:00:39]** few layers here and it's also actually
> 几层就足够了，而且实际上

**[00:00:39]** few layers here and it's also actually faster so compared to fine tuning all
> 几层就足够了，而且实际上更快，所以与微调所有

**[00:00:39]** faster so compared to fine tuning all
> 更快，所以与微调所有

**[00:00:39]** faster so compared to fine tuning all the layers where it takes the most time
> 更快，所以与微调所有层相比，这需要最多时间

**[00:00:39]** the layers where it takes the most time
> 层相比，这需要最多时间

**[00:00:39]** the layers where it takes the most time um updating only a few layers yeah is
> 层相比，这需要最多时间，只更新少数几层，是的，

**[00:00:39]** um updating only a few layers yeah is
> 只更新少数几层，是的，

**[00:00:39]** um updating only a few layers yeah is twice as fast in this case for
> 只更新少数几层，是的，在这种情况下速度快两倍，对于

**[00:00:39]** twice as fast in this case for
> 速度快两倍，对于

**[00:00:39]** twice as fast in this case for example okay so um this was
> 速度快两倍，例如，好吧，所以这是

**[00:00:39]** example okay so um this was
> 例如，好吧，所以这是

**[00:00:39]** example okay so um this was classification fine-tuning but I think
> 例如，好吧，所以这是分类微调，但我认为

**[00:00:39]** classification fine-tuning but I think
> 分类微调，但我认为

**[00:00:39]** classification fine-tuning but I think most people I would say are more excited
> 分类微调，但我认为大多数人会更兴奋

**[00:00:39]** most people I would say are more excited
> 大多数人会更兴奋

**[00:00:39]** most people I would say are more excited or interested in probably instruction
> 大多数人会更兴奋或对指令微调感兴趣

**[00:00:39]** or interested in probably instruction
> 或对指令微调感兴趣

**[00:00:39]** or interested in probably instruction fine-tuning to create uh personal
> 或对指令微调感兴趣，以创建个人

**[00:00:40]** fine-tuning to create uh personal
> 以创建个人

**[00:00:40]** fine-tuning to create uh personal assistance and chat parts and so forth
> 以创建个人助手和聊天部件等

**[00:00:40]** assistance and chat parts and so forth
> 助手和聊天部件等

**[00:00:40]** assistance and chat parts and so forth and that's basically what um chat GPT is
> 助手和聊天部件等，这基本上就是Chat GPT的功能

**[00:00:40]** and that's basically what um chat GPT is
> 这基本上就是Chat GPT的功能

**[00:00:40]** and that's basically what um chat GPT is personally I would say uh yeah don't
> 这基本上就是Chat GPT的功能，我个人认为不要

**[00:00:40]** personally I would say uh yeah don't
> 我个人认为不要

**[00:00:40]** personally I would say uh yeah don't underestimate um classification fine
> 我个人认为不要低估分类微调

**[00:00:40]** underestimate um classification fine
> 低估分类微调

**[00:00:40]** underestimate um classification fine tuning I would say personally um most
> 低估分类微调，我个人认为大多数

**[00:00:40]** tuning I would say personally um most
> 微调，我个人认为大多数

**[00:00:40]** tuning I would say personally um most tasks today I I would say maybe not most
> 微调，我个人认为今天的任务，也许不是大多数

**[00:00:40]** tasks today I I would say maybe not most
> 今天的任务，也许不是大多数

**[00:00:40]** tasks today I I would say maybe not most I mean this is just me saying that but I
> 今天的任务，也许不是大多数，这只是我个人的看法

**[00:00:40]** I mean this is just me saying that but I
> 这只是我个人的看法

**[00:00:40]** I mean this is just me saying that but I think many business tasks are really
> 这只是我个人的看法，但我认为许多商业任务实际上是

**[00:00:40]** think many business tasks are really
> 许多商业任务实际上是

**[00:00:40]** think many business tasks are really classification tasks if you think about
> 许多商业任务实际上是分类任务，如果你考虑一下

**[00:00:40]** classification tasks if you think about
> 分类任务，如果你考虑一下

**[00:00:40]** classification tasks if you think about you know sorting your documents um
> 分类任务，比如整理文档

**[00:00:40]** you know sorting your documents um
> 整理文档

**[00:00:40]** you know sorting your documents um category izing input emails um
> 整理文档，对输入邮件进行分类

**[00:00:40]** category izing input emails um
> 对输入邮件进行分类

**[00:00:40]** category izing input emails um classifying customer sentiment
> 对输入邮件进行分类，对客户情绪进行分类

**[00:00:40]** classifying customer sentiment
> 对客户情绪进行分类

**[00:00:40]** classifying customer sentiment predicting whether a customer turns or
> 对客户情绪进行分类，预测客户是否会流失

**[00:00:40]** predicting whether a customer turns or
> 预测客户是否会流失

**[00:00:40]** predicting whether a customer turns or not based on some yeah text
> 预测客户是否会流失，基于某些文本

**[00:00:40]** not based on some yeah text
> 基于某些文本

**[00:00:40]** not based on some yeah text communication and so forth I do think
> 基于某些文本交流等，我确实认为

**[00:00:40]** communication and so forth I do think
> 交流等，我确实认为

**[00:00:40]** communication and so forth I do think there are lots of practical business
> 交流等，我确实认为有很多实际的商业

**[00:00:40]** there are lots of practical business
> 有很多实际的商业

**[00:00:40]** there are lots of practical business cases where it's actually a
> 有很多实际的商业案例，实际上是一个

**[00:00:40]** cases where it's actually a
> 案例，实际上是一个

**[00:00:40]** cases where it's actually a classification problem rather than let's
> 案例，实际上是一个分类问题，而不是

**[00:00:40]** classification problem rather than let's
> 分类问题，而不是

**[00:00:40]** classification problem rather than let's say a chatbot problem of course chatbots
> 分类问题，而不是聊天机器人问题，当然聊天机器人

**[00:00:40]** say a chatbot problem of course chatbots
> 聊天机器人问题，当然聊天机器人

**[00:00:40]** say a chatbot problem of course chatbots are also super useful for certain tasks
> 聊天机器人问题，当然聊天机器人在某些任务上也非常有用

**[00:00:40]** are also super useful for certain tasks
> 在某些任务上也非常有用

**[00:00:40]** are also super useful for certain tasks but they are more like general purpose
> 在某些任务上也非常有用，但它们更像是通用型的

**[00:00:40]** but they are more like general purpose
> 但它们更像是通用型的

**[00:00:40]** but they are more like general purpose where I think a lot of business cases um
> 但它们更像是通用型的，而我认为许多商业案例

**[00:00:40]** where I think a lot of business cases um
> 而我认为许多商业案例

**[00:00:40]** where I think a lot of business cases um they would be solved or can be addressed
> 而我认为许多商业案例可以通过分类微调来解决或处理

**[00:00:40]** they would be solved or can be addressed
> 可以通过分类微调来解决或处理

**[00:00:40]** they would be solved or can be addressed with classification fine tuning which is
> 可以通过分类微调来解决或处理，这

**[00:00:40]** with classification fine tuning which is
> 这

**[00:00:40]** with classification fine tuning which is much um cheaper and simpler to implement
> 这更便宜且更容易实现

**[00:00:41]** much um cheaper and simpler to implement
> 更便宜且更容易实现

**[00:00:41]** much um cheaper and simpler to implement but anyways so let's talk about building
> 更便宜且更容易实现，但无论如何，我们来谈谈构建

**[00:00:41]** but anyways so let's talk about building
> 但无论如何，我们来谈谈构建

**[00:00:41]** but anyways so let's talk about building a personal assistant uh using an
> 但无论如何，我们来谈谈使用指令数据集构建个人助手

**[00:00:41]** a personal assistant uh using an
> 使用指令数据集构建个人助手

**[00:00:41]** a personal assistant uh using an instruction data set so for that usually
> 使用指令数据集构建个人助手，通常

**[00:00:41]** instruction data set so for that usually
> 指令数据集，通常

**[00:00:41]** instruction data set so for that usually when we are um interested in building a
> 指令数据集，通常当我们有兴趣构建一个

**[00:00:41]** when we are um interested in building a
> 当我们有兴趣构建一个

**[00:00:41]** when we are um interested in building a personal assistant like chat GPT like
> 当我们有兴趣构建一个像Chat GPT这样的个人助手

**[00:00:41]** personal assistant like chat GPT like
> 像Chat GPT这样的个人助手

**[00:00:41]** personal assistant like chat GPT like chatbot on the data set looks like as
> 像Chat GPT这样的聊天机器人时，数据集看起来像

**[00:00:41]** chatbot on the data set looks like as
> 聊天机器人时，数据集看起来像

**[00:00:41]** chatbot on the data set looks like as follows where we have an instruction an
> 聊天机器人时，数据集如下所示：包含指令、

**[00:00:41]** follows where we have an instruction an
> 如下所示：包含指令、

**[00:00:41]** follows where we have an instruction an input and an output and you know this
> 如下所示：包含指令、输入和输出，这个

**[00:00:41]** input and an output and you know this
> 输入和输出，这个

**[00:00:41]** input and an output and you know this input is also optional you can actually
> 输入和输出，这个输入也是可选的，实际上你可以

**[00:00:41]** input is also optional you can actually
> 输入也是可选的，实际上你可以

**[00:00:41]** input is also optional you can actually also append that to the instruct it's
> 输入也是可选的，实际上你也可以将其附加到指令中

**[00:00:41]** also append that to the instruct it's
> 也可以将其附加到指令中

**[00:00:41]** also append that to the instruct it's really um yeah it's it's just like a
> 也可以将其附加到指令中，这实际上只是格式问题

**[00:00:41]** really um yeah it's it's just like a
> 实际上只是格式问题

**[00:00:41]** really um yeah it's it's just like a matter of format both both ways Works
> 实际上只是格式问题，两种方式都有效

**[00:00:41]** matter of format both both ways Works
> 两种方式都有效

**[00:00:41]** matter of format both both ways Works work but this is yeah just an example
> 格式问题，两种方式都可行，但这只是一个例子

**[00:00:41]** work but this is yeah just an example
> 可行，但这只是一个例子

**[00:00:41]** work but this is yeah just an example the most common example so what we would
> 可行，但这只是一个例子，最常见的例子，所以我们接下来会

**[00:00:41]** the most common example so what we would
> 最常见的例子，所以我们接下来会

**[00:00:41]** the most common example so what we would do then is we would take this um data
> 最常见的例子，所以我们接下来会做的是，我们会拿这个数据

**[00:00:41]** do then is we would take this um data
> 做的是，我们会拿这个数据

**[00:00:41]** do then is we would take this um data set and then we usually apply a prompt
> 做的是，我们会拿这个数据集，然后通常我们会应用一个prompt

**[00:00:41]** set and then we usually apply a prompt
> 然后通常我们会应用一个prompt

**[00:00:41]** set and then we usually apply a prompt um template and in this case I'm using
> 然后通常我们会应用一个prompt模板，在这个例子中，我使用的是

**[00:00:41]** um template and in this case I'm using
> 模板，在这个例子中，我使用的是

**[00:00:41]** um template and in this case I'm using the classic alpaka style prompt template
> 模板，在这个例子中，我使用的是经典的alpaka风格prompt模板

**[00:00:41]** the classic alpaka style prompt template
> 经典的alpaka风格prompt模板

**[00:00:41]** the classic alpaka style prompt template so what it is doing is it's adding this
> 经典的alpaka风格prompt模板，它的作用是添加这段

**[00:00:41]** so what it is doing is it's adding this
> 它的作用是添加这段

**[00:00:41]** so what it is doing is it's adding this text below is an instruction that
> 它的作用是添加这段文本：“下面是一条指令，它

**[00:00:41]** text below is an instruction that
> 下面是一条指令，它

**[00:00:41]** text below is an instruction that describes a task Write a response that
> 下面是一条指令，它描述了一个任务。写一个响应，它

**[00:00:41]** describes a task Write a response that
> 描述了一个任务。写一个响应，它

**[00:00:41]** describes a task Write a response that appropriately completes the request and
> 描述了一个任务。写一个响应，它恰当地完成请求”，然后

**[00:00:41]** appropriately completes the request and
> 恰当地完成请求”，然后

**[00:00:41]** appropriately completes the request and then it's it's basically from up here
> 恰当地完成请求”，然后它基本上是从上面这里

**[00:00:41]** then it's it's basically from up here
> 它基本上是从上面这里

**[00:00:41]** then it's it's basically from up here the instruction um nicely formatted and
> 它基本上是从上面这里，指令被很好地格式化，并且

**[00:00:42]** the instruction um nicely formatted and
> 指令被很好地格式化，并且

**[00:00:42]** the instruction um nicely formatted and that is um what we feed to the
> 指令被很好地格式化，并且这就是我们输入到

**[00:00:42]** that is um what we feed to the
> 这就是我们输入到

**[00:00:42]** that is um what we feed to the llm so we we have this input that goes
> 这就是我们输入到LLM的内容。所以我们有这个输入，它进入

**[00:00:42]** llm so we we have this input that goes
> LLM。所以我们有这个输入，它进入

**[00:00:42]** llm so we we have this input that goes into the llm and um the llm then is
> LLM。所以我们有这个输入，它进入LLM，然后LLM

**[00:00:42]** into the llm and um the llm then is
> 进入LLM，然后LLM

**[00:00:42]** into the llm and um the llm then is supposed to generate the response here
> 进入LLM，然后LLM应该在这里生成响应

**[00:00:42]** supposed to generate the response here
> 应该在这里生成响应

**[00:00:42]** supposed to generate the response here so here the response would be great
> 应该在这里生成响应，所以这里的响应会是“团队取得了很好的成果”

**[00:00:42]** so here the response would be great
> 所以这里的响应会是“团队取得了很好的成果”

**[00:00:42]** so here the response would be great results were achieved by the team so the
> 所以这里的响应会是“团队取得了很好的成果”，所以

**[00:00:42]** results were achieved by the team so the
> 团队取得了很好的成果”，所以

**[00:00:42]** results were achieved by the team so the the instruction is rewrite the following
> 团队取得了很好的成果”，所以指令是“用被动语态重写以下

**[00:00:42]** the instruction is rewrite the following
> 指令是“用被动语态重写以下

**[00:00:42]** the instruction is rewrite the following sentence using passive voice for
> 指令是“用被动语态重写以下句子”，例如

**[00:00:42]** sentence using passive voice for
> 句子”，例如

**[00:00:42]** sentence using passive voice for example um yeah and so this is how an
> 句子”，例如，嗯，是的，这就是一个

**[00:00:42]** example um yeah and so this is how an
> 嗯，是的，这就是一个

**[00:00:42]** example um yeah and so this is how an instruction data set looks like and
> 嗯，是的，这就是一个instruction数据集的样子，并且

**[00:00:42]** instruction data set looks like and
> instruction数据集的样子，并且

**[00:00:42]** instruction data set looks like and usually um the data set sizes I would
> instruction数据集的样子，并且通常数据集的大小，我

**[00:00:42]** usually um the data set sizes I would
> 通常数据集的大小，我

**[00:00:42]** usually um the data set sizes I would say they range yeah I would say between,
> 通常数据集的大小，我会说范围在，嗯，我会说在

**[00:00:42]** say they range yeah I would say between,
> 我会说范围在，嗯，我会说在

**[00:00:42]** say they range yeah I would say between, to 50,000 100,000 examples so the Paka
> 我会说范围在，嗯，我会说在5万到10万个例子之间。所以alpaka

**[00:00:42]** to 50,000 100,000 examples so the Paka
> 5万到10万个例子之间。所以alpaka

**[00:00:42]** to 50,000 100,000 examples so the Paka data set back then it was the first I
> 5万到10万个例子之间。所以alpaka数据集，当时它是第一个，我

**[00:00:42]** data set back then it was the first I
> 数据集，当时它是第一个，我

**[00:00:42]** data set back then it was the first I would say publicly available instruction
> 数据集，当时它是第一个，我会说公开可用的instruction

**[00:00:42]** would say publicly available instruction
> 我会说公开可用的instruction

**[00:00:42]** would say publicly available instruction data set was about 51,000 if I remember
> 我会说公开可用的instruction数据集，大约有5.1万个，如果我没记错的话

**[00:00:42]** data set was about 51,000 if I remember
> 数据集，大约有5.1万个，如果我没记错的话

**[00:00:42]** data set was about 51,000 if I remember correctly and but also people showed um
> 数据集，大约有5.1万个，如果我没记错的话。但是人们也展示了，嗯

**[00:00:42]** correctly and but also people showed um
> 如果我没记错的话。但是人们也展示了，嗯

**[00:00:42]** correctly and but also people showed um I think the data set was called Lima
> 如果我没记错的话。但是人们也展示了，嗯，我认为那个数据集叫做Lima

**[00:00:42]** I think the data set was called Lima
> 我认为那个数据集叫做Lima

**[00:00:42]** I think the data set was called Lima they only had 1,000 examples and got I
> 我认为那个数据集叫做Lima，他们只有1000个例子，并且得到了，我

**[00:00:42]** they only had 1,000 examples and got I
> 他们只有1000个例子，并且得到了，我

**[00:00:42]** they only had 1,000 examples and got I think even better results it was more
> 他们只有1000个例子，并且得到了，我认为甚至更好的结果。这更

**[00:00:42]** think even better results it was more
> 认为甚至更好的结果。这更

**[00:00:42]** think even better results it was more about again the quality versus quality
> 认为甚至更好的结果。这更关乎质量与质量

**[00:00:42]** about again the quality versus quality
> 关乎质量与质量

**[00:00:42]** about again the quality versus quality uh quantity so it really depends I think
> 关乎质量与质量，呃，数量。所以这真的取决于，我认为

**[00:00:42]** uh quantity so it really depends I think
> 数量。所以这真的取决于，我认为

**[00:00:42]** uh quantity so it really depends I think the more the better usually of course
> 数量。所以这真的取决于，我认为通常越多越好，当然

**[00:00:43]** the more the better usually of course
> 通常越多越好，当然

**[00:00:43]** the more the better usually of course but you know you can get away with um
> 通常越多越好，当然，但是你知道，你可以用，嗯

**[00:00:43]** but you know you can get away with um
> 但是你知道，你可以用，嗯

**[00:00:43]** but you know you can get away with um thousand um thousand
> 但是你知道，你可以用，嗯，一千个，嗯，一千个

**[00:00:43]** thousand um thousand
> 一千个，嗯，一千个

**[00:00:43]** thousand um thousand examples um there is an additional bonus
> 一千个，嗯，一千个例子。还有一个额外的步骤

**[00:00:43]** examples um there is an additional bonus
> 例子。还有一个额外的步骤

**[00:00:43]** examples um there is an additional bonus um I would say step um it's called
> 例子。还有一个额外的步骤，嗯，我会说，叫做

**[00:00:43]** um I would say step um it's called
> 嗯，我会说，叫做

**[00:00:43]** um I would say step um it's called preference tuning and this usually
> 嗯，我会说，叫做preference tuning，这通常

**[00:00:43]** preference tuning and this usually
> preference tuning and this usually

**[00:00:43]** preference tuning and this usually follows the instruction finding because
> 偏好调优通常跟在指令微调之后，因为

**[00:00:43]** follows the instruction finding because
> 跟在指令微调之后，因为

**[00:00:43]** follows the instruction finding because this is already a long talk I won't go
> 跟在指令微调之后，因为这次演讲已经很长了，我就不

**[00:00:43]** this is already a long talk I won't go
> 这次演讲已经很长了，我就不

**[00:00:43]** this is already a long talk I won't go into too much detail about preference
> 这次演讲已经很长了，我就不深入讲解偏好调优的细节了

**[00:00:43]** into too much detail about preference
> 深入讲解偏好调优的细节了

**[00:00:43]** into too much detail about preference tuning because it's a whole can of worms
> 深入讲解偏好调优的细节了，因为这是个复杂的问题

**[00:00:43]** tuning because it's a whole can of worms
> 因为这是个复杂的问题

**[00:00:43]** tuning because it's a whole can of worms there are lots of um yeah lots of
> 因为这是个复杂的问题，有很多，嗯，很多

**[00:00:43]** there are lots of um yeah lots of
> 有很多，嗯，很多

**[00:00:43]** there are lots of um yeah lots of techniques but in a nutshell what
> 有很多，嗯，很多技术方法，但简而言之，偏好调优的核心是

**[00:00:43]** techniques but in a nutshell what
> 技术方法，但简而言之，偏好调优的核心是

**[00:00:43]** techniques but in a nutshell what preference tuning is all about is kind
> 技术方法，但简而言之，偏好调优的核心是

**[00:00:43]** preference tuning is all about is kind
> 偏好调优的核心是

**[00:00:43]** preference tuning is all about is kind of to refine the responses by the llm so
> 偏好调优的核心是优化LLM生成的回复，所以

**[00:00:43]** of to refine the responses by the llm so
> 优化LLM生成的回复，所以

**[00:00:43]** of to refine the responses by the llm so when we have the input here what are the
> 优化LLM生成的回复，所以当我们有输入时，比如“购买新笔记本电脑时

**[00:00:43]** when we have the input here what are the
> 当我们有输入时，比如“购买新笔记本电脑时

**[00:00:43]** when we have the input here what are the key features to look for when purchasing
> 当我们有输入时，比如“购买新笔记本电脑时，需要关注哪些关键特性？”

**[00:00:43]** key features to look for when purchasing
> 需要关注哪些关键特性？”

**[00:00:43]** key features to look for when purchasing a new laptop um especially uh when we
> 需要关注哪些关键特性？”嗯，特别是当我们

**[00:00:43]** a new laptop um especially uh when we
> 嗯，特别是当我们

**[00:00:43]** a new laptop um especially uh when we control the random seat for example in
> 嗯，特别是当我们控制随机种子时，例如在

**[00:00:43]** control the random seat for example in
> 控制随机种子时，例如在

**[00:00:43]** control the random seat for example in llm it can generate different responses
> 控制随机种子时，例如在LLM中，它可以生成不同的回复

**[00:00:43]** llm it can generate different responses
> LLM中，它可以生成不同的回复

**[00:00:43]** llm it can generate different responses or we can also provide as a human
> LLM中，它可以生成不同的回复，或者我们也可以提供人类

**[00:00:43]** or we can also provide as a human
> 或者我们也可以提供人类

**[00:00:43]** or we can also provide as a human preferred response for example but uh
> 或者我们也可以提供人类偏好的回复作为示例，但是

**[00:00:43]** preferred response for example but uh
> 偏好的回复作为示例，但是

**[00:00:43]** preferred response for example but uh long story short there can be different
> 偏好的回复作为示例，但是长话短说，回答这个问题可以有不同

**[00:00:43]** long story short there can be different
> 长话短说，回答这个问题可以有不同

**[00:00:43]** long story short there can be different ways you can answer this answer right so
> 长话短说，回答这个问题可以有不同方式，对吧？所以

**[00:00:43]** ways you can answer this answer right so
> 方式，对吧？所以

**[00:00:43]** ways you can answer this answer right so there could be a more technical response
> 方式，对吧？所以可能有一个更技术性的回复

**[00:00:43]** there could be a more technical response
> 可能有一个更技术性的回复

**[00:00:43]** there could be a more technical response um where it says when purchasing a new
> 可能有一个更技术性的回复，嗯，比如“购买新笔记本电脑时

**[00:00:43]** um where it says when purchasing a new
> 嗯，比如“购买新笔记本电脑时

**[00:00:43]** um where it says when purchasing a new laptop the focuses on key specifications
> 嗯，比如“购买新笔记本电脑时，重点在于关键规格”

**[00:00:44]** laptop the focuses on key specifications
> 重点在于关键规格”

**[00:00:44]** laptop the focuses on key specifications it would describe the Ram size the
> 重点在于关键规格”，它会描述RAM大小、

**[00:00:44]** it would describe the Ram size the
> 它会描述RAM大小、

**[00:00:44]** it would describe the Ram size the storage type and so forth and there's
> 它会描述RAM大小、存储类型等等，还有

**[00:00:44]** storage type and so forth and there's
> 存储类型等等，还有

**[00:00:44]** storage type and so forth and there's let's say a more user friendly response
> 存储类型等等，还有，比如说，一个更用户友好的回复

**[00:00:44]** let's say a more user friendly response
> 比如说，一个更用户友好的回复

**[00:00:44]** let's say a more user friendly response um depends on the audience of course but
> 比如说，一个更用户友好的回复，嗯，当然取决于受众，但是

**[00:00:44]** um depends on the audience of course but
> 嗯，当然取决于受众，但是

**[00:00:44]** um depends on the audience of course but so um it would then be more like think
> 嗯，当然取决于受众，但是，嗯，它会更像“想想它如何融入你的日常生活”

**[00:00:44]** so um it would then be more like think
> 它会更像“想想它如何融入你的日常生活”

**[00:00:44]** so um it would then be more like think about how it fits into your daily life
> 它会更像“想想它如何融入你的日常生活，选择轻便的型号，如果你

**[00:00:44]** about how it fits into your daily life
> 选择轻便的型号，如果你

**[00:00:44]** about how it fits into your daily life choose the lightweight model that you if
> 选择轻便的型号，如果你经常出差的话。顺便说一句，

**[00:00:44]** choose the lightweight model that you if
> 经常出差的话。顺便说一句，

**[00:00:44]** choose the lightweight model that you if you travel frequently by the way
> 经常出差的话。顺便说一句，我个人用的是MacBook Air，因为

**[00:00:44]** you travel frequently by the way
> 我个人用的是MacBook Air，因为

**[00:00:44]** you travel frequently by the way personally I have a MacBook Air because
> 我个人用的是MacBook Air，因为我确实认为它是一台非常好的

**[00:00:44]** personally I have a MacBook Air because
> 我确实认为它是一台非常好的

**[00:00:44]** personally I have a MacBook Air because I do think it's actually a really good
> 我确实认为它是一台非常好的机器，而且我经常出差，所以

**[00:00:44]** I do think it's actually a really good
> 机器，而且我经常出差，所以

**[00:00:44]** I do think it's actually a really good machine and I do have to travel a lot so
> 机器，而且我经常出差，所以，嗯，是的，我确实喜欢轻便的笔记本电脑

**[00:00:44]** machine and I do have to travel a lot so
> 嗯，是的，我确实喜欢轻便的笔记本电脑

**[00:00:44]** machine and I do have to travel a lot so um yeah I I do like a lightweight laptop
> 嗯，是的，我确实喜欢轻便的笔记本电脑，用于我的工作。我有一台

**[00:00:44]** um yeah I I do like a lightweight laptop
> 用于我的工作。我有一台

**[00:00:44]** um yeah I I do like a lightweight laptop for my for my work for my job I have a
> 用于我的工作。我有一台MacBook Pro，因为它，你知道，更

**[00:00:44]** for my for my work for my job I have a
> MacBook Pro，因为它，你知道，更

**[00:00:44]** for my for my work for my job I have a MacBook Pro because it's you know needed
> MacBook Pro，因为它，你知道，更适用于开发工作，所以

**[00:00:44]** MacBook Pro because it's you know needed
> 更适用于开发工作，所以

**[00:00:44]** MacBook Pro because it's you know needed more for the development work so you
> 更适用于开发工作，所以你知道，这是不同的

**[00:00:44]** more for the development work so you
> 你知道，这是不同的

**[00:00:44]** more for the development work so you know it's different different
> 你知道，这是不同的，嗯，嗯，是的，这里有不同的考量

**[00:00:44]** know it's different different
> 嗯，嗯，是的，这里有不同的考量

**[00:00:44]** know it's different different uh uh yeah different considerations here
> 嗯，嗯，是的，这里有不同的考量。这两个，比如说，是模型可能给出的不同回复

**[00:00:44]** uh uh yeah different considerations here
> 这两个，比如说，是模型可能给出的不同回复

**[00:00:44]** uh uh yeah different considerations here both let's say are different responses a
> 这两个，比如说，是模型可能给出的不同回复。也许一个模型可以

**[00:00:44]** both let's say are different responses a
> 也许一个模型可以

**[00:00:44]** both let's say are different responses a model might give maybe a model could
> 也许一个模型可以也有一个包含两者的回复

**[00:00:44]** model might give maybe a model could
> 也有一个包含两者的回复

**[00:00:44]** model might give maybe a model could have also a response that includes both
> 也有一个包含两者的回复

**[00:00:44]** have also a response that includes both
> have also a response that includes both

**[00:00:44]** have also a response that includes both but that could maybe be confusing for a
> 也有一个包含两者的响应，但这可能会让用户感到困惑

**[00:00:44]** but that could maybe be confusing for a
> 但这可能会让用户感到困惑

**[00:00:44]** but that could maybe be confusing for a user so you basically in preference
> 但这可能会让用户感到困惑，所以基本上在偏好调整中

**[00:00:44]** user so you basically in preference
> 用户所以基本上在偏好调整中

**[00:00:44]** user so you basically in preference tuning you would choose one or the other
> 用户所以基本上在偏好调整中，你会选择其中一个

**[00:00:44]** tuning you would choose one or the other
> 调整中你会选择其中一个

**[00:00:44]** tuning you would choose one or the other and do that for a large number of
> 调整中你会选择其中一个，并对大量示例执行此操作，以某种方式引导模型

**[00:00:45]** and do that for a large number of
> 并对大量示例执行此操作

**[00:00:45]** and do that for a large number of examples to kind of like steer the model
> 并对大量示例执行此操作，以某种方式引导模型

**[00:00:45]** examples to kind of like steer the model
> 示例以某种方式引导模型

**[00:00:45]** examples to kind of like steer the model more into whats the
> 示例以某种方式引导模型，使其更符合你期望的行为

**[00:00:45]** more into whats the
> 更符合你期望的

**[00:00:45]** more into whats the behavior um that you want it to have so
> 更符合你期望的行为，嗯，你希望它拥有的行为

**[00:00:45]** behavior um that you want it to have so
> 行为，嗯，你希望它拥有的行为

**[00:00:45]** behavior um that you want it to have so really just refinement of whether it
> 行为，嗯，你希望它拥有的行为，所以实际上只是细化它是否

**[00:00:45]** really just refinement of whether it
> 实际上只是细化它是否

**[00:00:45]** really just refinement of whether it would should be more technical or more
> 实际上只是细化它应该更技术性还是更

**[00:00:45]** would should be more technical or more
> 应该更技术性还是更

**[00:00:45]** would should be more technical or more user friendly and actually in practice
> 应该更技术性还是更用户友好，实际上在实践中

**[00:00:45]** user friendly and actually in practice
> 用户友好，实际上在实践中

**[00:00:45]** user friendly and actually in practice it's uh mostly applied to safety um so
> 用户友好，实际上实践中它主要用于安全性，嗯，所以

**[00:00:45]** it's uh mostly applied to safety um so
> 它主要用于安全性，嗯，所以

**[00:00:45]** it's uh mostly applied to safety um so to improve the safety of a model so for
> 它主要用于安全性，嗯，所以为了提高模型的安全性，例如

**[00:00:45]** to improve the safety of a model so for
> 为了提高模型的安全性，例如

**[00:00:45]** to improve the safety of a model so for example not to give instructions on
> 为了提高模型的安全性，例如不提供关于如何制造炸弹的指令

**[00:00:45]** example not to give instructions on
> 例如不提供关于如何制造炸弹的指令

**[00:00:45]** example not to give instructions on let's say how to build a bump or
> 例如不提供关于如何制造炸弹或类似东西的指令，或者不使用

**[00:00:45]** let's say how to build a bump or
> 比如说如何制造炸弹或类似东西

**[00:00:45]** let's say how to build a bump or something like that that or not to use
> 比如说如何制造炸弹或类似东西，或者不使用

**[00:00:45]** something like that that or not to use
> 或者不使用

**[00:00:45]** something like that that or not to use um spare words and so forth but then
> 或者不使用嗯，脏话等等，但同时

**[00:00:45]** um spare words and so forth but then
> 嗯，脏话等等，但同时

**[00:00:45]** um spare words and so forth but then also yeah to to improve the helpfulness
> 嗯，脏话等等，但同时也要提高有用性

**[00:00:45]** also yeah to to improve the helpfulness
> 也要提高有用性

**[00:00:45]** also yeah to to improve the helpfulness as well to give complete answers and not
> 也要提高有用性，提供完整的答案，而不是

**[00:00:45]** as well to give complete answers and not
> 提供完整的答案，而不是

**[00:00:45]** as well to give complete answers and not abbreviate the answers and so forth so
> 提供完整的答案，而不是缩写答案等等，所以

**[00:00:45]** abbreviate the answers and so forth so
> 缩写答案等等，所以

**[00:00:45]** abbreviate the answers and so forth so really it's yeah I would say a
> 缩写答案等等，所以实际上，是的，我会说这是

**[00:00:45]** really it's yeah I would say a
> 实际上，是的，我会说这是

**[00:00:45]** really it's yeah I would say a fine-tuning of the fine-tuning a
> 实际上，是的，我会说这是对微调的微调

**[00:00:45]** fine-tuning of the fine-tuning a
> 对微调的微调

**[00:00:45]** fine-tuning of the fine-tuning a fine-tuning of the instruction
> 对微调的微调，对指令微调的微调

**[00:00:45]** fine-tuning of the instruction
> 对指令微调的微调

**[00:00:45]** fine-tuning of the instruction fine-tuning essentially like a
> 对指令微调的微调，本质上就像

**[00:00:45]** fine-tuning essentially like a
> 本质上就像

**[00:00:45]** fine-tuning essentially like a preference fine tuning so um I have more
> 本质上就像偏好微调，所以嗯，我有更多

**[00:00:45]** preference fine tuning so um I have more
> 偏好微调，所以嗯，我有更多

**[00:00:45]** preference fine tuning so um I have more um I've written articles about that that
> 偏好微调，所以嗯，我写过关于这个的文章

**[00:00:45]** um I've written articles about that that
> 嗯，我写过关于这个的文章

**[00:00:45]** um I've written articles about that that cover some of the techniques behind it I
> 嗯，我写过关于这个的文章，涵盖了背后的一些技术

**[00:00:45]** cover some of the techniques behind it I
> 涵盖了背后的一些技术

**[00:00:45]** cover some of the techniques behind it I don't want to go too much into detail
> 涵盖了背后的一些技术，我不想在这次演讲中深入细节

**[00:00:45]** don't want to go too much into detail
> 不想在这次演讲中深入细节

**[00:00:45]** don't want to go too much into detail here in this talk because it's a really
> 不想在这次演讲中深入细节，因为这是一个非常长的演讲，我认为这是一个非常

**[00:00:45]** here in this talk because it's a really
> 因为这是一个非常长的演讲，我认为这是一个非常

**[00:00:45]** here in this talk because it's a really long talk and I think this is a really
> 因为这是一个非常长的演讲，我认为这是一个非常有趣的话题，值得嗯，嗯

**[00:00:45]** long talk and I think this is a really
> 有趣的话题，值得嗯，嗯

**[00:00:45]** long talk and I think this is a really interesting topic that deserves yeah uh
> 有趣的话题，值得嗯，嗯更多的时间本身，但这里有一些资源，如果你

**[00:00:45]** interesting topic that deserves yeah uh
> 更多的时间本身，但这里有一些资源，如果你

**[00:00:45]** interesting topic that deserves yeah uh more time itself so but here are if
> 更多的时间本身，但这里有一些资源，如果你感兴趣的话

**[00:00:46]** more time itself so but here are if
> 感兴趣的话

**[00:00:46]** more time itself so but here are if you're interested some resources if you
> 感兴趣的话，如果你想了解更多关于这个的内容

**[00:00:46]** you're interested some resources if you
> 如果你想了解更多关于这个的内容

**[00:00:46]** you're interested some resources if you want to read more about
> 如果你想了解更多关于这个的内容，那么嗯，一个有趣且非常重要的

**[00:00:46]** want to read more about
> 那么嗯，一个有趣且非常重要的

**[00:00:46]** want to read more about that so um one interesting and very
> 那么嗯，一个有趣且非常重要的主题当然是评估

**[00:00:46]** that so um one interesting and very
> 主题当然是评估

**[00:00:46]** that so um one interesting and very important topic of course is evaluating
> 主题当然是评估LLM，我之前展示过，嗯，在

**[00:00:46]** important topic of course is evaluating
> LLM，我之前展示过，嗯，在

**[00:00:46]** important topic of course is evaluating Elms I show I showed you earlier uh in
> LLM，我之前展示过，嗯，在分类微调的情况下

**[00:00:46]** Elms I show I showed you earlier uh in
> 分类微调的情况下

**[00:00:46]** Elms I show I showed you earlier uh in the case of classification fine-tuning
> 分类微调的情况下，我们如何评估一个分类器，我们可以

**[00:00:46]** the case of classification fine-tuning
> 我们如何评估一个分类器，我们可以

**[00:00:46]** the case of classification fine-tuning how we can evaluate a classifier we can
> 我们如何评估一个分类器，我们可以使用分类准确率作为

**[00:00:46]** how we can evaluate a classifier we can
> 使用分类准确率作为

**[00:00:46]** how we can evaluate a classifier we can use the classification accuracy as a
> 使用分类准确率作为指标，但实际上我们如何做到这一点呢

**[00:00:46]** use the classification accuracy as a
> 指标，但实际上我们如何做到这一点呢

**[00:00:46]** use the classification accuracy as a metric but how do we do that actually
> use the classification accuracy as a metric but how do we do that actually

**[00:00:46]** metric but how do we do that actually
> metric but how do we do that actually

**[00:00:46]** metric but how do we do that actually with um instruction ftuned and
> 指标，但具体如何通过指令微调来实现呢？

**[00:00:46]** with um instruction ftuned and
> 通过指令微调

**[00:00:46]** with um instruction ftuned and preference fine tuned models so one
> 通过指令微调和偏好微调模型，所以一个

**[00:00:46]** preference fine tuned models so one
> 偏好微调模型，所以一个

**[00:00:46]** preference fine tuned models so one thing you may have read about is um MML
> 偏好微调模型，所以你可能读到过的一个东西是MML

**[00:00:46]** thing you may have read about is um MML
> 你可能读到过的一个东西是MML

**[00:00:46]** thing you may have read about is um MML um so this is a number usually between
> 你可能读到过的一个东西是MML，这是一个通常在

**[00:00:46]** um so this is a number usually between
> 这是一个通常在

**[00:00:46]** um so this is a number usually between zero and 100 and when you know when op
> 这是一个通常在0到100之间的数值，当你知道当op

**[00:00:46]** zero and 100 and when you know when op
> 0到100之间，当你知道当op

**[00:00:46]** zero and 100 and when you know when op gives
> 0到100之间，当你知道当op给出

**[00:00:46]** gives
> 给出

**[00:00:46]** gives a web webinar webinar I mean a web uh
> 给出一个网络研讨会，我是说一个网络

**[00:00:46]** a web webinar webinar I mean a web uh
> 一个网络研讨会，我是说一个网络

**[00:00:46]** a web webinar webinar I mean a web uh introduction to the new uh model or
> 一个网络研讨会，我是说一个网络介绍新模型或

**[00:00:46]** introduction to the new uh model or
> 介绍新模型或

**[00:00:46]** introduction to the new uh model or Gemini the new version is revealed or
> 介绍新模型或Gemini，新版本发布或

**[00:00:46]** Gemini the new version is revealed or
> Gemini，新版本发布或

**[00:00:46]** Gemini the new version is revealed or people share the new model they usually
> Gemini，新版本发布或人们分享新模型时，他们通常

**[00:00:46]** people share the new model they usually
> 人们分享新模型时，他们通常

**[00:00:46]** people share the new model they usually give you an mmu score so it's usually
> 人们分享新模型时，他们通常会给你一个MMU分数，所以通常是

**[00:00:46]** give you an mmu score so it's usually
> 给你一个MMU分数，所以通常是

**[00:00:46]** give you an mmu score so it's usually yeah score between 0 and 100 and people
> 给你一个MMU分数，所以通常是0到100之间的分数，人们

**[00:00:46]** yeah score between 0 and 100 and people
> 0到100之间的分数，人们

**[00:00:46]** yeah score between 0 and 100 and people use that to rank um llms and this is I
> 0到100之间的分数，人们用这个来排名LLM，我认为

**[00:00:47]** use that to rank um llms and this is I
> 用这个来排名LLM，我认为

**[00:00:47]** use that to rank um llms and this is I would say for some reason nowadays one
> 用这个来排名LLM，我认为出于某种原因，如今这是

**[00:00:47]** would say for some reason nowadays one
> 出于某种原因，如今这是

**[00:00:47]** would say for some reason nowadays one of the most popular numbers to evaluate
> 出于某种原因，如今这是评估LLM最流行的指标之一

**[00:00:47]** of the most popular numbers to evaluate
> 评估LLM最流行的指标之一

**[00:00:47]** of the most popular numbers to evaluate um LMS so what that number really means
> 评估LLM最流行的指标之一，所以这个数字真正意味着

**[00:00:47]** um LMS so what that number really means
> 所以这个数字真正意味着

**[00:00:47]** um LMS so what that number really means is so basically it goes back to a paper
> 所以这个数字真正意味着，基本上它源于一篇论文

**[00:00:47]** is so basically it goes back to a paper
> 基本上它源于一篇论文

**[00:00:47]** is so basically it goes back to a paper it's called measuring massive multitask
> 基本上它源于一篇论文，叫做测量大规模多任务

**[00:00:47]** it's called measuring massive multitask
> 叫做测量大规模多任务

**[00:00:47]** it's called measuring massive multitask language understanding and what it
> 叫做测量大规模多任务语言理解，它

**[00:00:47]** language understanding and what it
> 语言理解，它

**[00:00:47]** language understanding and what it basically means is how good is your llm
> 语言理解，它基本上意味着你的LLM有多好

**[00:00:47]** basically means is how good is your llm
> 基本上意味着你的LLM有多好

**[00:00:47]** basically means is how good is your llm at um answering multiple choice
> 基本上意味着你的LLM在回答多项选择

**[00:00:47]** at um answering multiple choice
> 在回答多项选择

**[00:00:47]** at um answering multiple choice questions so um basically the input a
> 在回答多项选择题方面的能力，所以基本上MMU中的典型输入可能看起来像

**[00:00:47]** questions so um basically the input a
> 问题，所以基本上MMU中的典型输入可能看起来像

**[00:00:47]** questions so um basically the input a typical input in mmu might look like uh
> 问题，所以基本上MMU中的典型输入可能看起来像，哪个角色以说“生存还是毁灭”而闻名？问题是什么？

**[00:00:47]** typical input in mmu might look like uh
> 哪个角色以说“生存还是毁灭”而闻名？问题是什么？

**[00:00:47]** typical input in mmu might look like uh which character is known for saying to
> 哪个角色以说“生存还是毁灭”而闻名？问题是什么？然后有四个答案，模型

**[00:00:47]** which character is known for saying to
> 然后有四个答案，模型

**[00:00:47]** which character is known for saying to be or not to be what is the question and
> 然后有四个答案，模型应该回答其中一个，例如这里的情况是哈姆雷特

**[00:00:47]** be or not to be what is the question and
> 应该回答其中一个，例如这里的情况是哈姆雷特

**[00:00:47]** be or not to be what is the question and then are four answers and the model
> 应该回答其中一个，例如这里的情况是哈姆雷特，然后你

**[00:00:47]** then are four answers and the model
> 然后你

**[00:00:47]** then are four answers and the model should respond with one of those for
> 然后你基本上计算，所以你计算

**[00:00:47]** should respond with one of those for
> 基本上计算，所以你计算

**[00:00:47]** should respond with one of those for example here in this case Hamlet and um
> 基本上计算，所以你根据正确答案的数量除以总答案数来计算分数

**[00:00:47]** example here in this case Hamlet and um
> 根据正确答案的数量除以总答案数来计算分数

**[00:00:47]** example here in this case Hamlet and um yeah and then you
> 根据正确答案的数量除以总答案数来计算分数，从而得到一个准确率分数

**[00:00:47]** yeah and then you
> 从而得到一个准确率分数

**[00:00:47]** yeah and then you basically uh calc so you calculate the
> 从而得到一个准确率分数，所以这就是为什么它

**[00:00:47]** basically uh calc so you calculate the
> 所以这就是为什么它

**[00:00:47]** basically uh calc so you calculate the score based on the number of correct
> 所以这就是为什么它在0到100之间，它

**[00:00:47]** score based on the number of correct
> 在0到100之间，它

**[00:00:47]** score based on the number of correct answers divided by the total number of
> 在0到100之间，它基本上是说

**[00:00:47]** answers divided by the total number of
> 基本上是说

**[00:00:47]** answers divided by the total number of answers to get kind of like an accuracy
> 基本上是说，100%的分数意味着模型正确回答了MMLU中的所有多项选择题

**[00:00:47]** answers to get kind of like an accuracy
> 100%的分数意味着模型正确回答了MMLU中的所有多项选择题

**[00:00:47]** answers to get kind of like an accuracy score from that so that's why it's
> 100%的分数意味着模型正确回答了MMLU中的所有多项选择题，所以它

**[00:00:47]** score from that so that's why it's
> 所以它

**[00:00:47]** score from that so that's why it's between zero and 100 and it's
> 所以它基本上就是多项选择

**[00:00:47]** between zero and 100 and it's
> 基本上就是多项选择

**[00:00:47]** between zero and 100 and it's essentially saying so
> 基本上就是多项选择

**[00:00:47]** essentially saying so
> essentially saying so

**[00:00:47]** essentially saying so 100% score would mean that yeah the
> essentially saying so 100% score would mean that yeah the

**[00:00:48]** 100% score would mean that yeah the
> 100% score would mean that yeah the

**[00:00:48]** 100% score would mean that yeah the model answers all the multiple choice
> 100% score would mean that yeah the model answers all the multiple choice

**[00:00:48]** model answers all the multiple choice
> model answers all the multiple choice

**[00:00:48]** model answers all the multiple choice questions in mlu correctly so it's
> model answers all the multiple choice questions in mlu correctly so it's

**[00:00:48]** questions in mlu correctly so it's
> questions in mlu correctly so it's

**[00:00:48]** questions in mlu correctly so it's basically just multiple choice
> questions in mlu correctly so it's basically just multiple choice

**[00:00:48]** basically just multiple choice
> basically just multiple choice

**[00:00:48]** basically just multiple choice performance I wouldn't say it's terrible
> 基本上只是多项选择的表现，我不会说它很糟糕

**[00:00:48]** performance I wouldn't say it's terrible
> 表现我不会说它很糟糕

**[00:00:48]** performance I wouldn't say it's terrible or something like that but it's also not
> 表现我不会说它很糟糕或类似的话，但它也不是

**[00:00:48]** or something like that but it's also not
> 或类似的话，但它也不是

**[00:00:48]** or something like that but it's also not the whole story it's really just yeah
> 或类似的话，但它也不是全部真相，真的只是，嗯

**[00:00:48]** the whole story it's really just yeah
> 全部真相，真的只是，嗯

**[00:00:48]** the whole story it's really just yeah multiple choice questions and you might
> 全部真相，真的只是，嗯多项选择题，你可能

**[00:00:48]** multiple choice questions and you might
> 多项选择题，你可能

**[00:00:48]** multiple choice questions and you might remember from college how much we all um
> 多项选择题，你可能记得大学时我们有多

**[00:00:48]** remember from college how much we all um
> 记得大学时我们有多

**[00:00:48]** remember from college how much we all um L um multiple choice questions and how
> 记得大学时我们有多，嗯，多项选择题，以及

**[00:00:48]** L um multiple choice questions and how
> 嗯，多项选择题，以及

**[00:00:48]** L um multiple choice questions and how you know how good they are in
> 嗯，多项选择题，以及你知道它们在

**[00:00:48]** you know how good they are in
> 你知道它们在

**[00:00:48]** you know how good they are in determining how smart we are really it's
> 你知道它们在衡量我们有多聪明方面有多好，真的

**[00:00:48]** determining how smart we are really it's
> 衡量我们有多聪明方面有多好，真的

**[00:00:48]** determining how smart we are really it's really just memorization in my opinion
> 衡量我们有多聪明方面有多好，真的在我看来只是死记硬背

**[00:00:48]** really just memorization in my opinion
> 在我看来只是死记硬背

**[00:00:48]** really just memorization in my opinion anyways um if you're interested in tasks
> 在我看来只是死记硬背，总之，嗯，如果你对任务

**[00:00:48]** anyways um if you're interested in tasks
> 总之，嗯，如果你对任务

**[00:00:48]** anyways um if you're interested in tasks or evaluting models like that I mean it
> 总之，嗯，如果你对任务或像那样评估模型感兴趣，我的意思是

**[00:00:48]** or evaluting models like that I mean it
> 或像那样评估模型感兴趣，我的意思是

**[00:00:48]** or evaluting models like that I mean it I wouldn't say it's useless and I I
> 或像那样评估模型感兴趣，我的意思是，我不会说它没用，我我

**[00:00:48]** I wouldn't say it's useless and I I
> 我不会说它没用，我我

**[00:00:48]** I wouldn't say it's useless and I I honestly so I want to be clear I really
> 我不会说它没用，我我老实说，所以我想说清楚，我真的

**[00:00:48]** honestly so I want to be clear I really
> 老实说，所以我想说清楚，我真的

**[00:00:48]** honestly so I want to be clear I really think it's useful as a metric because we
> 老实说，所以我想说清楚，我真的认为它作为一个指标很有用，因为我们

**[00:00:48]** think it's useful as a metric because we
> 认为它作为一个指标很有用，因为我们

**[00:00:48]** think it's useful as a metric because we can use that to really um measure
> 认为它作为一个指标很有用，因为我们可以用它来真正，嗯，衡量

**[00:00:48]** can use that to really um measure
> 可以用它来真正，嗯，衡量

**[00:00:48]** can use that to really um measure performance in terms of training I'm
> 可以用它来真正，嗯，衡量训练方面的表现，我

**[00:00:48]** performance in terms of training I'm
> 训练方面的表现，我

**[00:00:48]** performance in terms of training I'm just saying it's not sufficient to only
> 训练方面的表现，我只是说仅仅

**[00:00:48]** just saying it's not sufficient to only
> 我只是说仅仅

**[00:00:48]** just saying it's not sufficient to only measure mmu usually you need a bit more
> 我只是说仅仅衡量MMU是不够的，通常你需要更多

**[00:00:48]** measure mmu usually you need a bit more
> 衡量MMU是不够的，通常你需要更多

**[00:00:48]** measure mmu usually you need a bit more than that so um yeah but if you're
> 衡量MMU是不够的，通常你需要更多，所以，嗯，是的，但如果你

**[00:00:48]** than that so um yeah but if you're
> 所以，嗯，是的，但如果你

**[00:00:48]** than that so um yeah but if you're interested um so we have also in leg for
> 所以，嗯，是的，但如果你感兴趣，嗯，我们也在leg中，例如

**[00:00:49]** interested um so we have also in leg for
> 感兴趣，嗯，我们也在leg中，例如

**[00:00:49]** interested um so we have also in leg for example um support for the evaluation
> 感兴趣，嗯，我们也在leg中，例如，嗯，支持评估

**[00:00:49]** example um support for the evaluation
> 例如，嗯，支持评估

**[00:00:49]** example um support for the evaluation harness with one line of code really you
> 例如，嗯，支持评估harness，只需一行代码，真的你

**[00:00:49]** harness with one line of code really you
> harness，只需一行代码，真的你

**[00:00:49]** harness with one line of code really you can evaluate mlu and other benchmarks um
> harness，只需一行代码，真的你可以评估MLU和其他基准，嗯

**[00:00:49]** can evaluate mlu and other benchmarks um
> 可以评估MLU和其他基准，嗯

**[00:00:49]** can evaluate mlu and other benchmarks um for a given model and so it would
> 可以评估MLU和其他基准，嗯，针对给定模型，所以它会

**[00:00:49]** for a given model and so it would
> 针对给定模型，所以它会

**[00:00:49]** for a given model and so it would basically give you the score so for for
> 针对给定模型，所以它基本上会给你分数，所以对于

**[00:00:49]** basically give you the score so for for
> 基本上会给你分数，所以对于

**[00:00:49]** basically give you the score so for for example 5 2 only gets 24% on mlu
> 基本上会给你分数，所以例如，5 2在MLU上只得到24%

**[00:00:49]** example 5 2 only gets 24% on mlu
> 例如，5 2在MLU上只得到24%

**[00:00:49]** example 5 2 only gets 24% on mlu compared to let's say these here so you
> 例如，5 2在MLU上只得到24%，相比之下，比如说这些，所以你可以

**[00:00:49]** compared to let's say these here so you
> 相比之下，比如说这些，所以你可以

**[00:00:49]** compared to let's say these here so you can say okay these models I mean it
> 相比之下，比如说这些，所以你可以说，好吧，这些模型，我的意思是

**[00:00:49]** can say okay these models I mean it
> 说，好吧，这些模型，我的意思是

**[00:00:49]** can say okay these models I mean it makes sense GPT 4 is much much better
> 说，好吧，这些模型，我的意思是，有道理，GPT 4比一个，嗯，20亿参数的模型如5 2好得多

**[00:00:49]** makes sense GPT 4 is much much better
> 有道理，GPT 4比一个，嗯，20亿参数的模型如5 2好得多

**[00:00:49]** makes sense GPT 4 is much much better than a um two billion model like 5 2 um
> 有道理，GPT 4比一个，嗯，20亿参数的模型如5 2好得多，嗯

**[00:00:49]** than a um two billion model like 5 2 um
> 比一个，嗯，20亿参数的模型如5 2好得多，嗯

**[00:00:49]** than a um two billion model like 5 2 um but yeah it's not the whole story so if
> 比一个，嗯，20亿参数的模型如5 2好得多，嗯，但这不是全部真相，所以如果

**[00:00:49]** but yeah it's not the whole story so if
> 但这不是全部真相，所以如果

**[00:00:49]** but yeah it's not the whole story so if we um want to evaluate how good an LM is
> 但这不是全部真相，所以如果我们，嗯，想评估一个LM有多好

**[00:00:49]** we um want to evaluate how good an LM is
> 我们，嗯，想评估一个LM有多好

**[00:00:49]** we um want to evaluate how good an LM is it's it needs to be doing more than just
> 我们，嗯，想评估一个LM有多好，它需要做的不仅仅是

**[00:00:49]** it's it needs to be doing more than just
> 它需要做的不仅仅是

**[00:00:49]** it's it needs to be doing more than just answering multiple choice questions
> 它需要做的不仅仅是回答多项选择题

**[00:00:49]** answering multiple choice questions
> 回答多项选择题

**[00:00:49]** answering multiple choice questions right so if we use cgbd it can do
> 回答多项选择题，对吧？所以如果我们使用CGBD，它可以做

**[00:00:49]** right so if we use cgbd it can do
> 对吧？所以如果我们使用CGBD，它可以做

**[00:00:49]** right so if we use cgbd it can do grammar correction it can rewrite your
> 对吧？所以如果我们使用CGBD，它可以做语法纠正，它可以重写你的

**[00:00:49]** grammar correction it can rewrite your
> 语法纠正，它可以重写你的

**[00:00:49]** grammar correction it can rewrite your text it can up make up news stories and
> 语法纠正，它可以重写你的文本，它可以编造新闻故事和

**[00:00:49]** text it can up make up news stories and
> 文本，它可以编造新闻故事和

**[00:00:49]** text it can up make up news stories and so
> 文本，它可以编造新闻故事和等等

**[00:00:49]** so
> 等等

**[00:00:49]** so forth so um one other metric people use
> 等等，所以，嗯，人们使用的另一个指标

**[00:00:49]** forth so um one other metric people use
> 所以，嗯，人们使用的另一个指标

**[00:00:49]** forth so um one other metric people use um well not metric more like a platform
> 所以，嗯，人们使用的另一个指标，嗯，不算是指标，更像是一个平台

**[00:00:49]** um well not metric more like a platform
> um well not metric more like a platform

**[00:00:49]** um well not metric more like a platform form or Benchmark is alpaka eval and
> 嗯，不完全是指标，更像是一种平台形式或基准，是Alpaca Eval

**[00:00:50]** form or Benchmark is alpaka eval and
> 形式或基准是Alpaca Eval

**[00:00:50]** form or Benchmark is alpaka eval and that is more like a way to measure the
> 形式或基准是Alpaca Eval，它更像是一种衡量

**[00:00:50]** that is more like a way to measure the
> 它更像是一种衡量

**[00:00:50]** that is more like a way to measure the conversational performance of an llm so
> 它更像是一种衡量LLM对话性能的方式

**[00:00:50]** conversational performance of an llm so
> 对话性能的方式

**[00:00:50]** conversational performance of an llm so here um how that works is they
> 对话性能的方式，它的工作原理是

**[00:00:50]** here um how that works is they
> 它的工作原理是

**[00:00:50]** here um how that works is they compare uh I think yeah see I need an LM
> 它的工作原理是比较，嗯，我觉得，是的，我需要一个LM

**[00:00:50]** compare uh I think yeah see I need an LM
> 比较，嗯，我觉得，是的，我需要一个LM

**[00:00:50]** compare uh I think yeah see I need an LM to fix the sentence here actually should
> 比较，嗯，我觉得，是的，我需要一个LM来修正这里的句子，实际上应该

**[00:00:50]** to fix the sentence here actually should
> 来修正这里的句子，实际上应该

**[00:00:50]** to fix the sentence here actually should be compare the response by gbd4 preview
> 是比较GPT-4 Preview的响应

**[00:00:50]** be compare the response by gbd4 preview
> 是比较GPT-4 Preview的响应

**[00:00:50]** be compare the response by gbd4 preview using um oh no sorry it's actually
> 是比较GPT-4 Preview的响应，使用，嗯，哦不，抱歉，实际上

**[00:00:50]** using um oh no sorry it's actually
> 使用，嗯，哦不，抱歉，实际上

**[00:00:50]** using um oh no sorry it's actually correct so what I'm trying to say is
> 使用，嗯，哦不，抱歉，实际上是正确的，所以我想说的是

**[00:00:50]** correct so what I'm trying to say is
> 正确的，所以我想说的是

**[00:00:50]** correct so what I'm trying to say is this method works by comparing a given
> 正确的，所以我想说的是，这种方法通过比较给定的

**[00:00:50]** this method works by comparing a given
> 这种方法通过比较给定的

**[00:00:50]** this method works by comparing a given llm to the performance of gp4 preview
> 这种方法通过比较给定的LLM与GPT-4 Preview的性能

**[00:00:50]** llm to the performance of gp4 preview
> LLM与GPT-4 Preview的性能

**[00:00:50]** llm to the performance of gp4 preview and then it uses um GPD 4 based Auto
> LLM与GPT-4 Preview的性能，然后使用基于GPT-4的自动

**[00:00:50]** and then it uses um GPD 4 based Auto
> 然后使用基于GPT-4的自动

**[00:00:50]** and then it uses um GPD 4 based Auto annotator to kind of so it's basically
> 然后使用基于GPT-4的自动标注器，基本上就是

**[00:00:50]** annotator to kind of so it's basically
> 标注器，基本上就是

**[00:00:50]** annotator to kind of so it's basically saying asks gp4 hey how does my model
> 标注器，基本上就是问GPT-4：“嘿，我的模型

**[00:00:50]** saying asks gp4 hey how does my model
> 问GPT-4：“嘿，我的模型

**[00:00:50]** saying asks gp4 hey how does my model compare to GPT 4 preview and um it's
> 问GPT-4：“嘿，我的模型与GPT-4 Preview相比如何？”然后

**[00:00:50]** compare to GPT 4 preview and um it's
> 与GPT-4 Preview相比如何？”然后

**[00:00:50]** compare to GPT 4 preview and um it's basically doing that to calculate a win
> 与GPT-4 Preview相比如何？”然后基本上通过计算胜率

**[00:00:50]** basically doing that to calculate a win
> 基本上通过计算胜率

**[00:00:50]** basically doing that to calculate a win rate I think it's like based on a um how
> 基本上通过计算胜率，我认为它是基于，嗯，你的给定模型

**[00:00:50]** rate I think it's like based on a um how
> 率，我认为它是基于，嗯，你的给定模型

**[00:00:50]** rate I think it's like based on a um how often is your given model better than
> 率，我认为它是基于，嗯，你的给定模型比GPT-4更好的频率

**[00:00:50]** often is your given model better than
> 比GPT-4更好的频率

**[00:00:50]** often is your given model better than gp4 so in this case if we look at gp4
> 比GPT-4更好的频率，所以在这种情况下，如果我们看GPT-4

**[00:00:50]** gp4 so in this case if we look at gp4
> GPT-4，所以在这种情况下，如果我们看GPT-4

**[00:00:50]** gp4 so in this case if we look at gp4 Omni it's 57% of the time um better than
> GPT-4，所以在这种情况下，如果我们看GPT-4 Omni，它在57%的情况下比

**[00:00:51]** Omni it's 57% of the time um better than
> Omni，它在57%的情况下比

**[00:00:51]** Omni it's 57% of the time um better than gp4 preview so in this case is a win
> Omni，它在57%的情况下比GPT-4 Preview更好，所以在这种情况下，胜率

**[00:00:51]** gp4 preview so in this case is a win
> GPT-4 Preview更好，所以在这种情况下，胜率

**[00:00:51]** gp4 preview so in this case is a win rate of 51 so this is a length corrected
> GPT-4 Preview更好，所以在这种情况下，胜率是51，这是一个长度校正

**[00:00:51]** rate of 51 so this is a length corrected
> 率是51，这是一个长度校正

**[00:00:51]** rate of 51 so this is a length corrected version so you can you know use use
> 率是51，这是一个长度校正版本，所以你可以使用

**[00:00:51]** version so you can you know use use
> 版本，所以你可以使用

**[00:00:51]** version so you can you know use use either this or this version but um the
> 版本，所以你可以使用这个或这个版本，但是

**[00:00:51]** either this or this version but um the
> 这个或这个版本，但是

**[00:00:51]** either this or this version but um the Bott B bottom line is that GPT Omni
> 这个或这个版本，但是底线是GPT Omni

**[00:00:51]** Bott B bottom line is that GPT Omni
> 底线是GPT Omni

**[00:00:51]** Bott B bottom line is that GPT Omni according to this gp4 based Auto
> 底线是GPT Omni，根据这个基于GPT-4的自动

**[00:00:51]** according to this gp4 based Auto
> 根据这个基于GPT-4的自动

**[00:00:51]** according to this gp4 based Auto annotator is actually better than gbd4
> 根据这个基于GPT-4的自动标注器，实际上比GPT-4 Preview更好

**[00:00:51]** annotator is actually better than gbd4
> 标注器，实际上比GPT-4 Preview更好

**[00:00:51]** annotator is actually better than gbd4 preview question is what does it mean to
> 标注器，实际上比GPT-4 Preview更好，问题是“更好”意味着什么？

**[00:00:51]** preview question is what does it mean to
> Preview更好，问题是“更好”意味着什么？

**[00:00:51]** preview question is what does it mean to be better better in what is it more
> Preview更好，问题是“更好”意味着什么？在哪些方面更好？是更

**[00:00:51]** be better better in what is it more
> 在哪些方面更好？是更

**[00:00:51]** be better better in what is it more correct or does the answer look more I
> 在哪些方面更好？是更正确，还是答案看起来更，我不知道，更有吸引力之类的？

**[00:00:51]** correct or does the answer look more I
> 正确，还是答案看起来更，我不知道，更有吸引力之类的？

**[00:00:51]** correct or does the answer look more I don't know attractive or something like
> 正确，还是答案看起来更，我不知道，更有吸引力之类的？所以这也不是非常

**[00:00:51]** don't know attractive or something like
> 所以这也不是非常

**[00:00:51]** don't know attractive or something like that so that's also not super I would
> 所以这也不是非常科学，但它至少是一个

**[00:00:51]** that so that's also not super I would
> 科学，但它至少是一个

**[00:00:51]** that so that's also not super I would say scientific but it is at least a
> 科学，但它至少是一个有用的东西，你知道，这是另一个

**[00:00:51]** say scientific but it is at least a
> 有用的东西，你知道，这是另一个

**[00:00:51]** say scientific but it is at least a useful thing you know it's another thing
> 有用的东西，你知道，这是另一个可以添加到MMLU分数和其他基准之上的东西

**[00:00:51]** useful thing you know it's another thing
> 可以添加到MMLU分数和其他基准之上的东西

**[00:00:51]** useful thing you know it's another thing that you can add on top of um of the mmu
> 可以添加到MMLU分数和其他基准之上的东西，有趣的是

**[00:00:51]** that you can add on top of um of the mmu
> 分数和其他基准之上的东西，有趣的是

**[00:00:51]** that you can add on top of um of the mmu score and the other benchmarks what's
> 分数和其他基准之上的东西，有趣的是，我在这里看到

**[00:00:51]** score and the other benchmarks what's
> 我在这里看到

**[00:00:51]** score and the other benchmarks what's interesting I'm just seeing here is that
> 我在这里看到，GPT-4本身只得了38%，所以也许这意味着

**[00:00:51]** interesting I'm just seeing here is that
> GPT-4本身只得了38%，所以也许这意味着

**[00:00:51]** interesting I'm just seeing here is that gp4 itself only got 38% so maybe that
> GPT-4本身只得了38%，所以也许这意味着GPT-4 Preview比GPT-4更好

**[00:00:51]** gp4 itself only got 38% so maybe that
> GPT-4 Preview比GPT-4更好

**[00:00:51]** gp4 itself only got 38% so maybe that means that um gp4 preview was better
> GPT-4 Preview比GPT-4更好，所以GPT-4随着时间的推移变差了

**[00:00:51]** means that um gp4 preview was better
> 所以GPT-4随着时间的推移变差了

**[00:00:51]** means that um gp4 preview was better than GPT 4 so GPT 4 got worse over time
> 所以GPT-4随着时间的推移变差了

**[00:00:51]** than GPT 4 so GPT 4 got worse over time
> than GPT 4 so GPT 4 got worse over time

**[00:00:51]** than GPT 4 so GPT 4 got worse over time or something like that so it's kind of
> 比 GPT 4 更差，所以 GPT 4 随着时间推移变差了或类似情况，所以这有点

**[00:00:52]** or something like that so it's kind of
> 或类似情况，所以这有点

**[00:00:52]** or something like that so it's kind of funny okay um moving on there's another
> 或类似情况，所以这有点好笑，好吧，我们继续，还有另一个

**[00:00:52]** funny okay um moving on there's another
> 好笑，好吧，我们继续，还有另一个

**[00:00:52]** funny okay um moving on there's another tool that people often refer to when
> 好笑，好吧，我们继续，还有另一个工具，人们经常用它来

**[00:00:52]** tool that people often refer to when
> 工具，人们经常用它来

**[00:00:52]** tool that people often refer to when talking about llm performance and that
> 工具，人们经常用它来讨论 LLM 性能，那就是

**[00:00:52]** talking about llm performance and that
> 讨论 LLM 性能，那就是

**[00:00:52]** talking about llm performance and that is the LM sis chatbot Arena so here it's
> 讨论 LLM 性能，那就是 LM sis Chatbot Arena，所以这里

**[00:00:52]** is the LM sis chatbot Arena so here it's
> 是 LM sis Chatbot Arena，所以这里

**[00:00:52]** is the LM sis chatbot Arena so here it's really more like a pairwise comparison
> 是 LM sis Chatbot Arena，所以这里更像是一个成对比较

**[00:00:52]** really more like a pairwise comparison
> 更像是一个成对比较

**[00:00:52]** really more like a pairwise comparison so um here it's basically a crowdsourced
> 更像是一个成对比较，所以这里基本上是一个众包

**[00:00:52]** so um here it's basically a crowdsourced
> 所以这里基本上是一个众包

**[00:00:52]** so um here it's basically a crowdsourced evaluation where there are two models
> 所以这里基本上是一个众包评估，其中有两个模型

**[00:00:52]** evaluation where there are two models
> 评估，其中有两个模型

**[00:00:52]** evaluation where there are two models left and right and there's also an
> 评估，其中有两个模型，左边和右边，还有一个

**[00:00:52]** left and right and there's also an
> 左边和右边，还有一个

**[00:00:52]** left and right and there's also an anonymous version where um you don't
> 左边和右边，还有一个匿名版本，其中你不知道

**[00:00:52]** anonymous version where um you don't
> 匿名版本，其中你不知道

**[00:00:52]** anonymous version where um you don't know what the models are so you use that
> 匿名版本，其中你不知道模型是什么，所以你使用那个

**[00:00:52]** know what the models are so you use that
> 模型是什么，所以你使用那个

**[00:00:52]** know what the models are so you use that thing and you get two answers and then
> 模型是什么，所以你使用那个东西，得到两个答案，然后

**[00:00:52]** thing and you get two answers and then
> 东西，得到两个答案，然后

**[00:00:52]** thing and you get two answers and then you get um two rates like you can say a
> 东西，得到两个答案，然后你得到两个评分，比如你可以说 A

**[00:00:52]** you get um two rates like you can say a
> 你得到两个评分，比如你可以说 A

**[00:00:52]** you get um two rates like you can say a is better B is better it's a tie or both
> 你得到两个评分，比如你可以说 A 更好，B 更好，平局，或者两者

**[00:00:52]** is better B is better it's a tie or both
> 更好，B 更好，平局，或者两者

**[00:00:52]** is better B is better it's a tie or both are bad and then based on that it's
> 更好，B 更好，平局，或者两者都不好，然后基于此，它

**[00:00:52]** are bad and then based on that it's
> 都不好，然后基于此，它

**[00:00:52]** are bad and then based on that it's Computing a pair wise ranking here so
> 都不好，然后基于此，它计算一个成对排名，所以

**[00:00:52]** Computing a pair wise ranking here so
> 计算一个成对排名，所以

**[00:00:52]** Computing a pair wise ranking here so according to that gp40 is also the best
> 计算一个成对排名，所以根据那个，GPT-4 也是最好的

**[00:00:52]** according to that gp40 is also the best
> 根据那个，GPT-4 也是最好的

**[00:00:52]** according to that gp40 is also the best model so basically if we take these
> 根据那个，GPT-4 也是最好的模型，所以基本上如果我们把这些

**[00:00:52]** model so basically if we take these
> 模型，所以基本上如果我们把这些

**[00:00:52]** model so basically if we take these things together if I go back a bit uh
> 模型，所以基本上如果我们把这些东西放在一起，如果我稍微回退一下

**[00:00:52]** things together if I go back a bit uh
> 东西放在一起，如果我稍微回退一下

**[00:00:52]** things together if I go back a bit uh what was it here so we can see okay in
> 东西放在一起，如果我稍微回退一下，这里是什么，我们可以看到，好的

**[00:00:52]** what was it here so we can see okay in
> 这里是什么，我们可以看到，好的

**[00:00:52]** what was it here so we can see okay in this case maybe Gemini Ultra is slightly
> 这里是什么，我们可以看到，好的，在这种情况下，Gemini Ultra 可能稍微

**[00:00:52]** this case maybe Gemini Ultra is slightly
> 在这种情况下，Gemini Ultra 可能稍微

**[00:00:52]** this case maybe Gemini Ultra is slightly better than GPD 4 but I mean there is a
> 在这种情况下，Gemini Ultra 可能稍微优于 GPT-4，但我的意思是，有一个

**[00:00:52]** better than GPD 4 but I mean there is a
> 优于 GPT-4，但我的意思是，有一个

**[00:00:52]** better than GPD 4 but I mean there is a signal there because according to this
> 优于 GPT-4，但我的意思是，有一个信号，因为根据这个

**[00:00:52]** signal there because according to this
> 信号，因为根据这个

**[00:00:52]** signal there because according to this one and um this one GPD 40 seems to be a
> 信号，因为根据这个和这个，GPT-4 似乎是一个

**[00:00:53]** one and um this one GPD 40 seems to be a
> 和这个，GPT-4 似乎是一个

**[00:00:53]** one and um this one GPD 40 seems to be a pretty good model so I mean of course
> 和这个，GPT-4 似乎是一个相当不错的模型，所以我的意思是，当然

**[00:00:53]** pretty good model so I mean of course
> 相当不错的模型，所以我的意思是，当然

**[00:00:53]** pretty good model so I mean of course your mileage may vary based on what
> 相当不错的模型，所以我的意思是，当然你的体验可能因你在模型中寻找什么而异，但是的

**[00:00:53]** your mileage may vary based on what
> 你的体验可能因你在模型中寻找什么而异，但是的

**[00:00:53]** your mileage may vary based on what you're looking for in a model but yeah
> 你的体验可能因你在模型中寻找什么而异，但是的，这通常是 LM 性能

**[00:00:53]** you're looking for in a model but yeah
> 这通常是 LM 性能

**[00:00:53]** you're looking for in a model but yeah this is usually how um LM performance
> 这通常是 LM 性能评估的工作方式，我也有自己的

**[00:00:53]** this is usually how um LM performance
> 评估的工作方式，我也有自己的

**[00:00:53]** this is usually how um LM performance evaluations work like I have also my own
> 评估的工作方式，我也有自己的方法，我认为其他

**[00:00:53]** evaluations work like I have also my own
> 方法，我认为其他

**[00:00:53]** evaluations work like I have also my own approach um I I think the other
> 方法，我认为其他方法很好，但除此之外

**[00:00:53]** approach um I I think the other
> 方法很好，但除此之外

**[00:00:53]** approach um I I think the other approaches are great but in addition
> 方法很好，但除此之外，我可以在电脑上快速做的一件事

**[00:00:53]** approaches are great but in addition
> 我可以在电脑上快速做的一件事

**[00:00:53]** approaches are great but in addition something quick I can do on my computer
> 我可以在电脑上快速做的一件事是，但我的意思是，它也不完美

**[00:00:53]** something quick I can do on my computer
> 是，但我的意思是，它也不完美

**[00:00:53]** something quick I can do on my computer is but I I mean it's also not perfect
> 是，但我的意思是，它也不完美，因为它也可能很随意

**[00:00:53]** is but I I mean it's also not perfect
> 因为它也可能很随意

**[00:00:53]** is but I I mean it's also not perfect because it's also it can be arbitrary
> 因为它也可能很随意，但我喜欢做的是，我只用

**[00:00:53]** because it's also it can be arbitrary
> 但我喜欢做的是，我只用

**[00:00:53]** because it's also it can be arbitrary but what I like to do is I use just the
> 但我喜欢做的是，我只用 GPT-4 AI 来评分我的答案，所以

**[00:00:53]** but what I like to do is I use just the
> GPT-4 AI 来评分我的答案，所以

**[00:00:53]** but what I like to do is I use just the gp4 AI to um score my answer so
> GPT-4 AI 来评分我的答案，所以基本上我有一个给定的输入，我有一个

**[00:00:53]** gp4 AI to um score my answer so
> 基本上我有一个给定的输入，我有一个

**[00:00:53]** gp4 AI to um score my answer so basically I have a given input I have a
> 基本上我有一个给定的输入，我有一个来自数据集的正确输出，以及

**[00:00:53]** basically I have a given input I have a
> 来自数据集的正确输出，以及

**[00:00:53]** basically I have a given input I have a correct um output from the data set and
> 来自数据集的正确输出，以及我的模型响应，然后我

**[00:00:53]** correct um output from the data set and
> 我的模型响应，然后我

**[00:00:53]** correct um output from the data set and then I have my model response and then I
> 我的模型响应，然后我让 GPT-4 对响应进行评分，基于一个

**[00:00:53]** then I have my model response and then I
> 让 GPT-4 对响应进行评分，基于一个

**[00:00:53]** then I have my model response and then I asked gbd4 to score the response on a
> 让 GPT-4 对响应进行评分，基于一个

**[00:00:53]** asked gbd4 to score the response on a
> asked gbd4 to score the response on a

**[00:00:53]** asked gbd4 to score the response on a scale from 0 to 100 and I get a score
> 让 gbd4 对回答进行 0 到 100 分的评分，我得到了一个分数

**[00:00:53]** scale from 0 to 100 and I get a score
> 0 到 100 分的评分，我得到了一个分数

**[00:00:53]** scale from 0 to 100 and I get a score between 0 and 100 and usually I mean
> 0 到 100 分的评分，我得到了一个 0 到 100 之间的分数，通常我的意思是

**[00:00:53]** between 0 and 100 and usually I mean
> 0 到 100 之间，通常我的意思是

**[00:00:53]** between 0 and 100 and usually I mean it's also pretty reliable in terms of
> 0 到 100 之间，通常我的意思是，它在判断哪个模型更好方面也相当可靠

**[00:00:53]** it's also pretty reliable in terms of
> 它在判断哪个模型更好方面也相当可靠

**[00:00:53]** it's also pretty reliable in terms of saying which model is better if I would
> 它在判断哪个模型更好方面也相当可靠，如果我要看结果的话

**[00:00:53]** saying which model is better if I would
> 判断哪个模型更好，如果我要看结果的话

**[00:00:53]** saying which model is better if I would look at the results so it's just yet
> 判断哪个模型更好，如果我要看结果的话，这只是另一种方式，这只是我个人的

**[00:00:53]** look at the results so it's just yet
> 看结果，所以这只是

**[00:00:53]** look at the results so it's just yet another way that's just my my personal
> 看结果，所以这只是另一种方式，这只是我个人的

**[00:00:53]** another way that's just my my personal
> 另一种方式，这只是我个人的

**[00:00:53]** another way that's just my my personal way in addition to the other ones um
> 另一种方式，这只是我个人的方式，除了其他那些之外

**[00:00:54]** way in addition to the other ones um
> 方式，除了其他那些之外

**[00:00:54]** way in addition to the other ones um yeah so with that I hope I gave you an
> 方式，除了其他那些之外，嗯，是的，所以这样我希望我给了你一个

**[00:00:54]** yeah so with that I hope I gave you an
> 是的，所以这样我希望我给了你一个

**[00:00:54]** yeah so with that I hope I gave you an interesting overview of what goes into
> 是的，所以这样我希望我给了你一个关于构建 LLM 所涉及内容的精彩概述

**[00:00:54]** interesting overview of what goes into
> 关于构建 LLM 所涉及内容的精彩概述

**[00:00:54]** interesting overview of what goes into building an llm just yeah to end this
> 关于构建 LLM 所涉及内容的精彩概述，只是为了结束这次演讲

**[00:00:54]** building an llm just yeah to end this
> 构建 LLM，只是为了结束这次演讲

**[00:00:54]** building an llm just yeah to end this talk I wanted to also give you some
> 构建 LLM，只是为了结束这次演讲，我还想给你一些经验法则

**[00:00:54]** talk I wanted to also give you some
> 演讲，我还想给你一些

**[00:00:54]** talk I wanted to also give you some rules of thumb so with that I mean
> 演讲，我还想给你一些经验法则，所以我的意思是

**[00:00:54]** rules of thumb so with that I mean
> 经验法则，所以我的意思是

**[00:00:54]** rules of thumb so with that I mean um how do we I mean there are so many
> 经验法则，所以我的意思是，嗯，我们如何，我的意思是，我讲了很多东西

**[00:00:54]** um how do we I mean there are so many
> 嗯，我们如何，我的意思是，有很多

**[00:00:54]** um how do we I mean there are so many things I covered how do we make sense of
> 嗯，我们如何，我的意思是，我讲了很多东西，我们如何理解它们

**[00:00:54]** things I covered how do we make sense of
> 我讲了很多东西，我们如何理解

**[00:00:54]** things I covered how do we make sense of them what should you use for your given
> 我讲了很多东西，我们如何理解它们，对于你的特定项目应该使用什么

**[00:00:54]** them what should you use for your given
> 它们，对于你的特定项目应该使用什么

**[00:00:54]** them what should you use for your given pro project for example I would say so
> 它们，对于你的特定项目应该使用什么，例如，我会说

**[00:00:54]** pro project for example I would say so
> 项目，例如，我会说

**[00:00:54]** pro project for example I would say so we covered pre-training from scratch I
> 项目，例如，我会说，我们讨论了从头开始预训练

**[00:00:54]** we covered pre-training from scratch I
> 我们讨论了从头开始预训练

**[00:00:54]** we covered pre-training from scratch I would say pre-training from scratch
> 我们讨论了从头开始预训练，我会说从头开始预训练

**[00:00:54]** would say pre-training from scratch
> 会说从头开始预训练

**[00:00:54]** would say pre-training from scratch um that's the most expensive thing and I
> 会说从头开始预训练，嗯，那是最昂贵的事情

**[00:00:54]** um that's the most expensive thing and I
> 那是最昂贵的事情

**[00:00:54]** um that's the most expensive thing and I would say that's almost never necessary
> 那是最昂贵的事情，我会说那几乎从来都不是必要的

**[00:00:54]** would say that's almost never necessary
> 会说那几乎从来都不是必要的

**[00:00:54]** would say that's almost never necessary unless you're trying to develop a new
> 会说那几乎从来都不是必要的，除非你试图开发一种新的架构之类的东西

**[00:00:54]** unless you're trying to develop a new
> 除非你试图开发一种新的

**[00:00:54]** unless you're trying to develop a new architecture or something um or would
> 除非你试图开发一种新的架构之类的东西，嗯，或者想要完全控制你的 LLM

**[00:00:54]** architecture or something um or would
> 架构之类的东西，嗯，或者想要

**[00:00:54]** architecture or something um or would have full control over your llm or
> 架构之类的东西，嗯，或者想要完全控制你的 LLM 或类似的东西

**[00:00:54]** have full control over your llm or
> 完全控制你的 LLM 或

**[00:00:54]** have full control over your llm or something like that I would say that's
> 完全控制你的 LLM 或类似的东西，我会说那是不必要的

**[00:00:54]** something like that I would say that's
> 类似的东西，我会说那是不必要的

**[00:00:54]** something like that I would say that's not um necessary I think this is really
> 类似的东西，我会说那是不必要的，我认为这真的

**[00:00:54]** not um necessary I think this is really
> 不必要的，我认为这真的

**[00:00:54]** not um necessary I think this is really yeah more like for research purposes or
> 不必要的，我认为这真的，嗯，更像是用于研究目的

**[00:00:54]** yeah more like for research purposes or
> 更像是用于研究目的

**[00:00:54]** yeah more like for research purposes or really if you're a big company and want
> 更像是用于研究目的，或者如果你是一家大公司，想要建立自己的基础模型

**[00:00:54]** really if you're a big company and want
> 如果你是一家大公司，想要

**[00:00:54]** really if you're a big company and want to establish Your Own Foundation model
> 如果你是一家大公司，想要建立自己的基础模型

**[00:00:54]** to establish Your Own Foundation model
> 建立自己的基础模型

**[00:00:54]** to establish Your Own Foundation model for example um continued pre-training is
> 建立自己的基础模型，例如，持续预训练是一个过程

**[00:00:54]** for example um continued pre-training is
> 例如，嗯，持续预训练是

**[00:00:54]** for example um continued pre-training is a process where you do actually do the
> 例如，嗯，持续预训练是一个过程，你实际上做的是和预训练一样的事情

**[00:00:54]** a process where you do actually do the
> 一个过程，你实际上做的是

**[00:00:54]** a process where you do actually do the same as pre-training but you take an
> 一个过程，你实际上做的是和预训练一样的事情，但你拿一个现有的模型

**[00:00:54]** same as pre-training but you take an
> 和预训练一样的事情，但你拿一个

**[00:00:54]** same as pre-training but you take an existing model and continue doing that
> 和预训练一样的事情，但你拿一个现有的模型，继续在更小的数据集上做这件事

**[00:00:54]** existing model and continue doing that
> 现有的模型，继续做这件事

**[00:00:54]** existing model and continue doing that on a smaller data set to instill new
> 现有的模型，继续在更小的数据集上做这件事，以灌输新知识

**[00:00:55]** on a smaller data set to instill new
> 在更小的数据集上，以灌输新

**[00:00:55]** on a smaller data set to instill new knowledge so this is um the pre-training
> 在更小的数据集上，以灌输新知识，所以这是从基础模型开始的预训练

**[00:00:55]** knowledge so this is um the pre-training
> 知识，所以这是，嗯，预训练

**[00:00:55]** knowledge so this is um the pre-training starting with a foundation model is
> 知识，所以这是，嗯，从基础模型开始的预训练，实际上在我看来是最有效的方法之一

**[00:00:55]** starting with a foundation model is
> 从基础模型开始是

**[00:00:55]** starting with a foundation model is actually in my opinion one of the most
> 从基础模型开始，实际上在我看来是最有效的方法之一

**[00:00:55]** actually in my opinion one of the most
> 实际上在我看来是最有效的方法之一

**[00:00:55]** actually in my opinion one of the most effective ways to instill new knowledge
> 实际上在我看来是最有效的方法之一，用于向 LLM 灌输新知识

**[00:00:55]** effective ways to instill new knowledge
> 向 LLM 灌输新知识的最有效方法

**[00:00:55]** effective ways to instill new knowledge into an llm so if you have a model that
> 向 LLM 灌输新知识的最有效方法，所以如果你有一个模型

**[00:00:55]** into an llm so if you have a model that
> 向 LLM 灌输新知识，所以如果你有一个模型

**[00:00:55]** into an llm so if you have a model that um you know can certain thing do certain
> 向 LLM 灌输新知识，所以如果你有一个模型，嗯，你知道，它能做某些事情

**[00:00:55]** um you know can certain thing do certain
> 嗯，你知道，它能做某些事情

**[00:00:55]** um you know can certain thing do certain things but it has no knowledge about
> 嗯，你知道，它能做某些事情，但它对某些领域没有知识

**[00:00:55]** things but it has no knowledge about
> 事情，但它没有关于

**[00:00:55]** things but it has no knowledge about things from 2024 because it has been
> 但它对2024年之后的事情一无所知，因为它是在

**[00:00:55]** things from 2024 because it has been
> 2024年之后的事情，因为它是在

**[00:00:55]** things from 2024 because it has been trained in 23 so in that case instead of
> 2024年之后的事情，因为它是在23年训练的，所以在这种情况下，与其

**[00:00:55]** trained in 23 so in that case instead of
> 23年训练的，所以在这种情况下，与其

**[00:00:55]** trained in 23 so in that case instead of training the whole model from scratch
> 23年训练的，所以在这种情况下，与其从头训练整个模型

**[00:00:55]** training the whole model from scratch
> 从头训练整个模型

**[00:00:55]** training the whole model from scratch you can just train it on additional data
> 从头训练整个模型，你可以只用额外数据训练它

**[00:00:55]** you can just train it on additional data
> 你可以只用额外数据训练它

**[00:00:55]** you can just train it on additional data from 2024 for example so basically
> 你可以只用额外数据训练它，比如2024年的数据，所以基本上

**[00:00:55]** from 2024 for example so basically
> 比如2024年的数据，所以基本上

**[00:00:55]** from 2024 for example so basically updating your llm um fine-tuning yeah we
> 比如2024年的数据，所以基本上就是更新你的LLM，嗯，微调，是的，我们

**[00:00:55]** updating your llm um fine-tuning yeah we
> 更新你的LLM，嗯，微调，是的，我们

**[00:00:55]** updating your llm um fine-tuning yeah we talked about special use cases for
> 更新你的LLM，嗯，微调，是的，我们讨论过特殊用例，比如

**[00:00:55]** talked about special use cases for
> 讨论过特殊用例，比如

**[00:00:55]** talked about special use cases for example spam classification that's
> 讨论过特殊用例，比如垃圾邮件分类，这

**[00:00:55]** example spam classification that's
> 垃圾邮件分类，这

**[00:00:55]** example spam classification that's useful for that or I mean in general
> 垃圾邮件分类，对此很有用，或者我的意思是，一般来说

**[00:00:55]** useful for that or I mean in general
> 对此很有用，或者我的意思是，一般来说

**[00:00:55]** useful for that or I mean in general text classification tasks but then also
> 对此很有用，或者我的意思是，一般来说，文本分类任务，但还有

**[00:00:55]** text classification tasks but then also
> 文本分类任务，但还有

**[00:00:55]** text classification tasks but then also um to follow instructions to build a
> 文本分类任务，但还有，嗯，遵循指令来构建

**[00:00:55]** um to follow instructions to build a
> 嗯，遵循指令来构建

**[00:00:55]** um to follow instructions to build a chatbot and so forth and then preference
> 嗯，遵循指令来构建聊天机器人等等，然后偏好

**[00:00:55]** chatbot and so forth and then preference
> 聊天机器人等等，然后偏好

**[00:00:55]** chatbot and so forth and then preference tuning is really to improve the
> 聊天机器人等等，然后偏好调优实际上是为了提高

**[00:00:55]** tuning is really to improve the
> 调优实际上是为了提高

**[00:00:55]** tuning is really to improve the helpfulness safety of um models that are
> 调优实际上是为了提高模型的帮助性和安全性，这些模型

**[00:00:55]** helpfulness safety of um models that are
> 模型的帮助性和安全性，这些模型

**[00:00:55]** helpfulness safety of um models that are intended to be for example a
> 模型的帮助性和安全性，这些模型旨在用于例如

**[00:00:55]** intended to be for example a
> 旨在用于例如

**[00:00:55]** intended to be for example a chatbot um yeah so just give you an
> 旨在用于例如聊天机器人，嗯，是的，所以给你一个

**[00:00:55]** chatbot um yeah so just give you an
> 聊天机器人，嗯，是的，所以给你一个

**[00:00:55]** chatbot um yeah so just give you an example where all these things are
> 聊天机器人，嗯，是的，所以给你一个例子，说明所有这些方法是如何

**[00:00:55]** example where all these things are
> 例子，说明所有这些方法是如何

**[00:00:55]** example where all these things are applied so there was this um code llama
> 例子，说明所有这些方法是如何应用的。有一个叫Code Llama的

**[00:00:56]** applied so there was this um code llama
> 应用的。有一个叫Code Llama的

**[00:00:56]** applied so there was this um code llama model by meta AI where they developed a
> 应用的。有一个叫Code Llama的模型，由Meta AI开发，他们构建了一个

**[00:00:56]** model by meta AI where they developed a
> 模型，由Meta AI开发，他们构建了一个

**[00:00:56]** model by meta AI where they developed a model specifically for coding so
> 模型，由Meta AI开发，他们构建了一个专门用于编码的模型，所以

**[00:00:56]** model specifically for coding so
> 专门用于编码的模型，所以

**[00:00:56]** model specifically for coding so basically here they pre-trained a model
> 专门用于编码的模型，所以基本上他们预训练了一个模型

**[00:00:56]** basically here they pre-trained a model
> 基本上他们预训练了一个模型

**[00:00:56]** basically here they pre-trained a model I mean it's the same company so they for
> 基本上他们预训练了一个模型，我的意思是，是同一家公司，所以他们对于

**[00:00:56]** I mean it's the same company so they for
> 我的意思是，是同一家公司，所以他们对于

**[00:00:56]** I mean it's the same company so they for this project they started with Lama 2
> 我的意思是，是同一家公司，所以他们对于这个项目，从Llama 2开始

**[00:00:56]** this project they started with Lama 2
> 这个项目，从Llama 2开始

**[00:00:56]** this project they started with Lama 2 but practically they also pre-trained
> 这个项目，从Llama 2开始，但实际上他们也预训练了

**[00:00:56]** but practically they also pre-trained
> 但实际上他们也预训练了

**[00:00:56]** but practically they also pre-trained Lama 2 because it's done all in been
> 但实际上他们也预训练了Llama 2，因为这一切都是在内部完成的

**[00:00:56]** Lama 2 because it's done all in been
> Llama 2，因为这一切都是在内部完成的

**[00:00:56]** Lama 2 because it's done all in been done in house so the pre-training was
> Llama 2，因为这一切都是在内部完成的，所以预训练就是

**[00:00:56]** done in house so the pre-training was
> 所以预训练就是

**[00:00:56]** done in house so the pre-training was the Lama 2 creation um then they had
> 所以预训练就是创建Llama 2，嗯，然后他们进行了

**[00:00:56]** the Lama 2 creation um then they had
> 创建Llama 2，嗯，然后他们进行了

**[00:00:56]** the Lama 2 creation um then they had some more continued pre-training where
> 创建Llama 2，嗯，然后他们进行了一些持续的预训练，其中

**[00:00:56]** some more continued pre-training where
> 一些持续的预训练，其中

**[00:00:56]** some more continued pre-training where this was more like trained on language
> 一些持续的预训练，这更像是用语言训练，然后他们，嗯，继续预训练

**[00:00:56]** this was more like trained on language
> 这更像是用语言训练，然后他们，嗯，继续预训练

**[00:00:56]** this was more like trained on language and then they um continued pre-training
> 这更像是用语言训练，然后他们，嗯，继续预训练它，专门针对代码，嗯，然后他们

**[00:00:56]** and then they um continued pre-training
> 它，专门针对代码，嗯，然后他们

**[00:00:56]** and then they um continued pre-training it on code specifically um and then they
> 它，专门针对代码，嗯，然后他们进行了一些更多的持续预训练，例如

**[00:00:56]** it on code specifically um and then they
> 进行了一些更多的持续预训练，例如

**[00:00:56]** it on code specifically um and then they had some more continued pre-training for
> 进行了一些更多的持续预训练，例如在这个项目中，他们开发了

**[00:00:56]** had some more continued pre-training for
> 例如在这个项目中，他们开发了

**[00:00:56]** had some more continued pre-training for example in this so they developed
> 例如在这个项目中，他们开发了多个模型，其中一个他们专门训练了

**[00:00:56]** example in this so they developed
> 多个模型，其中一个他们专门训练了

**[00:00:56]** example in this so they developed multiple models in this one they trained
> 多个模型，其中一个他们专门训练了更多Python代码，嗯，并且

**[00:00:56]** multiple models in this one they trained
> 更多Python代码，嗯，并且

**[00:00:56]** multiple models in this one they trained specifically more on python code um and
> 更多Python代码，嗯，并且然后他们有一个阶段，他们训练了

**[00:00:56]** specifically more on python code um and
> 然后他们有一个阶段，他们训练了

**[00:00:56]** specifically more on python code um and then they had a stage where they trained
> 然后他们有一个阶段，他们训练了更长的上下文，嗯，他们称之为

**[00:00:56]** then they had a stage where they trained
> 更长的上下文，嗯，他们称之为

**[00:00:56]** then they had a stage where they trained on longer contexts and um they called it
> 更长的上下文，嗯，他们称之为微调，但这本质上也是一个

**[00:00:56]** on longer contexts and um they called it
> 微调，但这本质上也是一个

**[00:00:56]** on longer contexts and um they called it fine-tuning but it's essentially also a
> 微调，但这本质上也是一个持续的预训练任务，然后，呃

**[00:00:56]** fine-tuning but it's essentially also a
> 持续的预训练任务，然后，呃

**[00:00:56]** fine-tuning but it's essentially also a continued pre-training task and then uh
> 持续的预训练任务，然后，呃，在最后一步，他们进行了指令

**[00:00:56]** continued pre-training task and then uh
> 在最后一步，他们进行了指令

**[00:00:56]** continued pre-training task and then uh at the last step they had instruction
> continued pre-training task and then uh at the last step they had instruction

**[00:00:56]** at the last step they had instruction
> at the last step they had instruction

**[00:00:56]** at the last step they had instruction fine-tuning to also create an
> 在最后一步，他们进行了指令微调，以创建一个

**[00:00:56]** fine-tuning to also create an
> 微调以创建一个

**[00:00:56]** fine-tuning to also create an instruction variant
> 微调以创建一个指令变体

**[00:00:56]** instruction variant
> 指令变体

**[00:00:56]** instruction variant here um yeah and with that I think we
> 指令变体，嗯，是的，这样我认为我们

**[00:00:56]** here um yeah and with that I think we
> 嗯，是的，这样我认为我们

**[00:00:57]** here um yeah and with that I think we covered pretty much all of the stages of
> 嗯，是的，这样我认为我们涵盖了几乎所有阶段

**[00:00:57]** covered pretty much all of the stages of
> 涵盖了几乎所有阶段

**[00:00:57]** covered pretty much all of the stages of building an llm from um coding the
> 涵盖了从编码开始构建LLM的几乎所有阶段

**[00:00:57]** building an llm from um coding the
> 从编码开始构建LLM

**[00:00:57]** building an llm from um coding the architecture of course without showing
> 从编码架构开始构建LLM，当然没有

**[00:00:57]** architecture of course without showing
> 架构，当然没有

**[00:00:57]** architecture of course without showing you too much code here or not any code
> 向你展示太多代码，或者没有任何代码

**[00:00:57]** you too much code here or not any code
> 向你展示太多代码，或者没有任何代码

**[00:00:57]** you too much code here or not any code because it would be a very long talk
> 向你展示太多代码，或者没有任何代码，因为这会是一个很长的演讲

**[00:00:57]** because it would be a very long talk
> 因为这会是一个很长的演讲

**[00:00:57]** because it would be a very long talk otherwise uh talking about the
> 否则会是一个很长的演讲，嗯，讨论

**[00:00:57]** otherwise uh talking about the
> 否则，嗯，讨论

**[00:00:57]** otherwise uh talking about the pre-training and the fine-tuning so if
> 否则，嗯，讨论预训练和微调，所以如果

**[00:00:57]** pre-training and the fine-tuning so if
> 预训练和微调，所以如果

**[00:00:57]** pre-training and the fine-tuning so if you're interested I have more you know
> 预训练和微调，所以如果你感兴趣，我有更多你知道的

**[00:00:57]** you're interested I have more you know
> 你感兴趣，我有更多你知道的

**[00:00:57]** you're interested I have more you know concrete examples in my um build a large
> 你感兴趣，我有更多具体例子，在我的嗯《从零构建大语言模型》书中

**[00:00:57]** concrete examples in my um build a large
> 具体例子，在我的嗯《从零构建大语言模型》书中

**[00:00:57]** concrete examples in my um build a large language model from scratchbook where
> 具体例子，在我的嗯《从零构建大语言模型》书中，是的，它应用了所有这些阶段

**[00:00:57]** language model from scratchbook where
> 是的，它应用了所有这些阶段

**[00:00:57]** language model from scratchbook where yeah it's applying all these stages in
> 是的，它应用了所有这些阶段在代码中，嗯，所以是的，构建你自己的小型

**[00:00:57]** yeah it's applying all these stages in
> 代码中，嗯，所以是的，构建你自己的小型

**[00:00:57]** yeah it's applying all these stages in code um so yeah to build your own small
> 代码中，嗯，所以是的，构建你自己的小型个人助手，如果你感兴趣

**[00:00:57]** code um so yeah to build your own small
> 个人助手，如果你感兴趣

**[00:00:57]** code um so yeah to build your own small personal assistant if you're interested
> 个人助手，如果你感兴趣的话，嗯，如果你在寻找GPU

**[00:00:57]** personal assistant if you're interested
> 的话，嗯，如果你在寻找GPU

**[00:00:57]** personal assistant if you're interested in that um if you looking for GPU
> 解决方案，我们在Lightning AI，我们

**[00:00:57]** in that um if you looking for GPU
> 解决方案，我们在Lightning AI，我们

**[00:00:57]** in that um if you looking for GPU Solutions so we are at lightning AI we
> 解决方案，我们在Lightning AI，我们正在构建嗯Lightning AI Studio

**[00:00:57]** Solutions so we are at lightning AI we
> 正在构建嗯Lightning AI Studio

**[00:00:57]** Solutions so we are at lightning AI we are building um the lightning AI Studio
> 正在构建嗯Lightning AI Studio，这是一个Studio环境，你可以

**[00:00:57]** are building um the lightning AI Studio
> 这是一个Studio环境，你可以

**[00:00:57]** are building um the lightning AI Studio which is the studio environment you can
> 这是一个Studio环境，你可以使用Visual Studio Code、Jupyter Notebook

**[00:00:57]** which is the studio environment you can
> 使用Visual Studio Code、Jupyter Notebook

**[00:00:57]** which is the studio environment you can use Visual Studio code jupyter notebook
> 使用Visual Studio Code、Jupyter Notebook，并在多个GPU上训练模型

**[00:00:57]** use Visual Studio code jupyter notebook
> 并在多个GPU上训练模型

**[00:00:57]** use Visual Studio code jupyter notebook and train models on uh multiple gpus and
> 并在多个GPU上训练模型，嗯，还有一个好处是你可以灵活地

**[00:00:57]** and train models on uh multiple gpus and
> 嗯，还有一个好处是你可以灵活地

**[00:00:57]** and train models on uh multiple gpus and um what's nice also is you can flexibly
> 嗯，还有一个好处是你可以灵活地在CPU和GPU之间切换，嗯，为了

**[00:00:57]** um what's nice also is you can flexibly
> 在CPU和GPU之间切换，嗯，为了

**[00:00:57]** um what's nice also is you can flexibly switch between CPUs and gpus um for
> 在CPU和GPU之间切换，嗯，例如，这实际上就是我使用的

**[00:00:57]** switch between CPUs and gpus um for
> 例如，这实际上就是我使用的

**[00:00:57]** switch between CPUs and gpus um for example so that's actually what I use
> 例如，这实际上就是我用于所有开发工作的

**[00:00:57]** example so that's actually what I use
> 用于所有开发工作的

**[00:00:57]** example so that's actually what I use for all my development work as
> 用于所有开发工作的，嗯，我们还有很多例子

**[00:00:57]** for all my development work as
> 嗯，我们还有很多例子

**[00:00:57]** for all my development work as well um we also have a lot of examples
> 嗯，我们还有很多例子在Lightning AI上，所以我们有很多

**[00:00:58]** well um we also have a lot of examples
> 在Lightning AI上，所以我们有很多

**[00:00:58]** well um we also have a lot of examples on Lightning AI so we have a lot of
> 在Lightning AI上，所以我们有很多Studio可以开始，所以基本上

**[00:00:58]** on Lightning AI so we have a lot of
> Studio可以开始，所以基本上

**[00:00:58]** on Lightning AI so we have a lot of Studios to um start so it's basically
> Studio可以开始，所以基本上你可以把它看作嗯几乎像

**[00:00:58]** Studios to um start so it's basically
> 你可以把它看作嗯几乎像

**[00:00:58]** Studios to um start so it's basically you can think of it as um almost like
> 你可以把它看作嗯几乎像GitHub仓库，但它们已经是

**[00:00:58]** you can think of it as um almost like
> GitHub仓库，但它们已经是

**[00:00:58]** you can think of it as um almost like GitHub repositories but they are already
> GitHub仓库，但它们已经是嗯可运行的，你不需要安装

**[00:00:58]** GitHub repositories but they are already
> 嗯可运行的，你不需要安装

**[00:00:58]** GitHub repositories but they are already um working you don't have to install
> 嗯可运行的，你不需要安装任何东西，比如嗯包依赖

**[00:00:58]** um working you don't have to install
> 任何东西，比如嗯包依赖

**[00:00:58]** um working you don't have to install anything like um package dependencies
> 任何东西，比如嗯包依赖之类的，如果你启动一个Studio，它是一个

**[00:00:58]** anything like um package dependencies
> 之类的，如果你启动一个Studio，它是一个

**[00:00:58]** anything like um package dependencies and stuff if you start a studio it's a
> 之类的，如果你启动一个Studio，它是一个模板，已经可以运行

**[00:00:58]** and stuff if you start a studio it's a
> 一个模板，已经可以运行

**[00:00:58]** and stuff if you start a studio it's a it's a template that already runs
> 一个模板，已经可以运行，无需安装任何东西，是的

**[00:00:58]** it's a template that already runs
> 无需安装任何东西，是的

**[00:00:58]** it's a template that already runs without having to install anything yeah
> 无需安装任何东西，是的，所以我们到了这个

**[00:00:58]** without having to install anything yeah
> 所以我们到了这个

**[00:00:58]** without having to install anything yeah and so we are at the end of this
> 所以我们到了这个演示的结尾，所以如果你想联系

**[00:00:58]** and so we are at the end of this
> 演示的结尾，所以如果你想联系

**[00:00:58]** and so we are at the end of this presentation so if you want to contact
> 演示的结尾，所以如果你想联系我，你可以在这里找到我

**[00:00:58]** presentation so if you want to contact
> 我，你可以在这里找到我

**[00:00:58]** presentation so if you want to contact me you can find me here
> 我，你可以在这里找到我，并且如果你想要访问

**[00:00:58]** me you can find me here
> 并且如果你想要访问

**[00:00:58]** me you can find me here and also if you would like access to the
> 并且如果你想要访问

**[00:00:58]** and also if you would like access to the
> and also if you would like access to the

**[00:00:58]** and also if you would like access to the slides they are here available on my
> 另外，如果您想获取幻灯片，它们可以在我的

**[00:00:58]** slides they are here available on my
> 幻灯片可以在我的

**[00:00:58]** slides they are here available on my website so I hope uh this was useful I
> 幻灯片可以在我的网站上找到，希望这有用，我

**[00:00:58]** website so I hope uh this was useful I
> 网站上找到，希望这有用，我

**[00:00:58]** website so I hope uh this was useful I hope it was not too long I have not
> 网站上找到，希望这有用，我希望没有太长，我还没

**[00:00:58]** hope it was not too long I have not
> 希望没有太长，我还没

**[00:00:58]** hope it was not too long I have not checked my watch but I hope it is um
> 希望没有太长，我还没看表，但希望它

**[00:00:58]** checked my watch but I hope it is um
> 看表，但希望它

**[00:00:58]** checked my watch but I hope it is um below let's say 1 hour and I hope this
> 看表，但希望它在一小时以内，希望这

**[00:00:58]** below let's say 1 hour and I hope this
> 在一小时以内，希望这

**[00:00:58]** below let's say 1 hour and I hope this was an informative video thanks for
> 在一小时以内，希望这是一个有教育意义的视频，感谢

**[00:00:58]** was an informative video thanks for
> 是一个有教育意义的视频，感谢

**[00:00:58]** was an informative video thanks for watching
> 是一个有教育意义的视频，感谢观看

---

## 🎯 关键要点

- [ ] 要点 1
- [ ] 要点 2
- [ ] 要点 3
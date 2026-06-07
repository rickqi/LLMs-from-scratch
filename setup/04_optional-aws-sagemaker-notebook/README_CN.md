# AWS CloudFormation 模板：带有 LLMs-from-scratch 仓库的 Jupyter Notebook

此 CloudFormation 模板在 Amazon SageMaker 中创建一个启用了 GPU 的 Jupyter notebook，并带有执行角色和 LLMs-from-scratch GitHub 仓库。

## 它的作用：

1. 创建具有 SageMaker notebook 实例必要权限的 IAM 角色。
2. 创建一个用于加密 notebook 实例的 KMS 密钥和别名。
3. 配置 notebook 实例生命周期配置脚本，该脚本：
   - 在用户的主目录中安装单独的 Miniconda 安装。
   - 创建一个带有 TensorFlow 2.15.0 和 PyTorch 2.1.0（均支持 CUDA）的自定义 Python 环境。
   - 安装其他有用的包，如 Jupyter Lab、Matplotlib 等。
   - 将自定义环境注册为 Jupyter 内核。
4. 创建具有指定配置的 SageMaker notebook 实例，包括启用了 GPU 的实例类型、执行角色和默认代码仓库。

## 如何使用：

1. 下载 CloudFormation 模板文件 (`cloudformation-template.yml`)。
2. 在 AWS 管理控制台中，导航到 CloudFormation 服务。
3. 创建一个新堆栈并上传模板文件。
4. 为 notebook 实例提供名称（例如，"LLMsFromScratchNotebook"）（默认为 LLMs-from-scratch GitHub 仓库）。
5. 检查并接受模板的参数，然后创建堆栈。
6. 堆栈创建完成后，SageMaker notebook 实例将在 SageMaker 控制台中可用。
7. 打开 notebook 实例并使用预配置的环境开始处理您的 LLMs-from-scratch 项目。

## 关键点：

- 该模板创建一个启用了 GPU 的 (`ml.g4dn.xlarge`) notebook 实例，具有 50GB 的存储空间。
- 它设置一个带有 TensorFlow 2.15.0 和 PyTorch 2.1.0（均支持 CUDA）的自定义 Miniconda 环境。
- 自定义环境被注册为 Jupyter 内核，使其在 notebook 中可用。
- 该模板还创建一个用于加密 notebook 实例的 KMS 密码和具有必要权限的 IAM 角色。
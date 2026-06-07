# Docker 环境设置指南

如果您更喜欢隔离项目依赖项和配置的开发设置，使用 Docker 是一个非常有效的解决方案。这种方法消除了手动安装软件包和库的需要，并确保了一致的开发环境。

本指南将引导您设置本书的可选 docker 环境，如果您更喜欢使用在 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 和 [../02_installing-python-libraries](../02_installing-python-libraries) 中解释的 conda 方法，则可以跳过。

<br>

## 下载并安装 Docker

使用 Docker 最简单的方法是为您的相关平台安装 [Docker Desktop](https://docs.docker.com/desktop/)。

Linux (Ubuntu) 用户可能更喜欢安装 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) 并遵循 [安装后](https://docs.docker.com/engine/install/linux-postinstall/) 步骤。

<br>

## 在 Visual Studio Code 中使用 Docker DevContainer

Docker DevContainer 或开发容器是一种允许开发人员使用 Docker 容器作为完整开发环境的工具。这种方法确保用户无论其本地机器设置如何，都能快速获得一致的开发环境并运行起来。

虽然 DevContainer 也与其他 IDE 一起工作，但用于处理 DevContainer 的常用 IDE/编辑器是 Visual Studio Code (VS Code)。下面的指南解释了如何在 VS Code 上下文中使用本书的 DevContainer，但类似的过程也应该适用于 PyCharm。如果您没有它并且想使用它，请 [安装](https://code.visualstudio.com/download) 它。

1. 克隆此 GitHub 存储库并 `cd` 到项目根目录。

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

2. 将 `.devcontainer` 文件夹从 `setup/03_optional-docker-environment/` 移动到当前目录（项目根目录）。

```bash
mv setup/03_optional-docker-environment/.devcontainer ./
```

3. 在 Docker Desktop 中，确保 **_desktop-linux_ 构建器** 正在运行并将用于构建 Docker 容器（请参阅 _Docker Desktop_ -> _更改设置_ -> _构建器_ -> _desktop-linux_ -> _..._ -> _使用_）

4. 如果您有 [支持 CUDA 的 GPU](https://developer.nvidia.com/cuda-gpus)，您可以加速训练和推理：

    4.1 按照描述 [这里](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) 安装 **NVIDIA Container Toolkit**。NVIDIA Container Toolkit 如 [这里](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2) 所述支持。

    4.2 在 Docker Engine 守护进程配置中将 _nvidia_ 添加为运行时（请参阅 _Docker Desktop_ -> _更改设置_ -> _Docker Engine_）。将这些行添加到您的配置中：

    ```json
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
    ```
    
    例如，完整的 Docker Engine 守护进程配置 json 代码应该如下所示：

    ```json
    {
      "builder": {
        "gc": {
          "defaultKeepStorage": "20GB",
          "enabled": true
        }
      },
      "experimental": false,
      "runtimes": {
        "nvidia": {
          "path": "nvidia-container-runtime",
          "runtimeArgs": []
        }
      }
    }
    ```

    并重新启动 Docker Desktop。

5. 在终端中输入 `code .` 以在 VS Code 中打开项目。或者，您可以启动 VS Code 并从 UI 选择要打开的项目。

6. 从左侧的 VS Code _扩展_ 菜单安装 **Remote Development** 扩展。

7. 打开 DevContainer。

由于 `.devcontainer` 文件存在于主 `LLMs-from-scratch` 目录中（以 `.` 开头的文件夹在您的操作系统中可能不可见，具体取决于您的设置），VS Code 应该自动检测它并询问您是否要在 devcontainer 中打开项目。如果没有，只需按 `Ctrl + Shift + P` 打开命令面板并开始键入 `dev containers` 以查看所有 DevContainer 特定选项。


&nbsp;
> ⚠️ **关于以 root 身份运行的注意事项**
>
> 默认情况下，DevContainer 以 *root 用户* 身份运行。出于安全考虑，这通常不推荐，但为了简化本书的设置，使用了 root 配置，以便所有必需的包都能在容器内干净地安装。
>
> 如果您尝试在容器内手动启动 Jupyter Lab，您可能会看到此错误：
>
>   ```bash
>   Running as root is not recommended. Use --allow-root to bypass.
>   ```
>
>   在这种情况下，您可以运行：
>
>   ```bash
>   uv run jupyter lab --allow-root
>   ```
>
> - 使用带有 Jupyter 扩展的 VS Code 时，通常不需要手动启动 Jupyter Lab。通过扩展打开笔记本应该可以直接工作。
> - 偏好更严格安全性的高级用户可以修改 `.devcontainer.json` 来设置非 root 用户，但这需要额外的配置，并且对于大多数用例来说不是必需的。


8. 选择 **在容器中重新打开**。

Docker 现在将开始构建 `.devcontainer` 配置中指定的 Docker 映像的过程（如果以前没有构建过），或者如果映像可从注册表中获取，则提取该映像。

整个过程是自动化的，可能需要几分钟，具体取决于您的系统和互联网速度。或者，单击 VS Code 右下角的"启动 Dev Container (显示日志)"以查看当前构建进度。

完成后，VS Code 将自动连接到容器并在新创建的 Docker 开发环境中重新打开项目。您将能够编写、执行和调试代码，就像它们在您的本地机器上运行一样，但具有 Docker 的隔离性和一致性带来的额外好处。

&nbsp;

> **警告：**
> 如果您在构建过程中遇到错误，这可能是因为您的机器不支持 NVIDIA 容器工具包，因为您的机器没有兼容的 GPU。在这种情况下，编辑 `devcontainer.json` 文件以删除 `"runArgs": ["--runtime=nvidia", "--gpus=all"],` 行并再次运行"重新打开 Dev Container"过程。

9. 完成。

映像被提取和构建后，您应该在容器内拥有安装了所有包、准备就绪的项目，用于开发。

<br>

## 卸载 Docker 映像

以下是卸载或删除 Docker 容器和映像的说明（如果您不再打算使用它）。此过程不会从系统中删除 Docker 本身，而是清理项目特定的 Docker 工件。

1. 列出所有 Docker 映像以找到与您的 DevContainer 关联的映像：

```bash
docker image ls
```

2. 使用映像 ID 或名称删除 Docker 映像：

```bash
docker image rm [IMAGE_ID_OR_NAME]
```

<br>

## 卸载 Docker

如果您决定 Docker 不适合您并希望卸载它，请参阅 [此处](https://docs.docker.com/desktop/uninstall/) 的官方文档，其中概述了适用于您的特定操作系统的步骤。
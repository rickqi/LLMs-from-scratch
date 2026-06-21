# 医疗模型 COS 备份恢复指南

> 从 COS 云备份恢复完整训练环境，继续训练或推理。适配源环境与目标环境差异。

## 一、备份内容

| 类别 | 内容 | 大小 |
|------|------|------|
| LoRA 权重 | V1-V7 指令模型 + DPO 模型 (18个) | ~230 MB |
| 训练数据 | data_full/train.txt + val.txt (29K样本) | ~30 MB |
| 指令QA | 973 条训练 QA + 15批 doc-search 增强 | ~2 MB |
| DPO数据 | 4个阶段的偏好数据 | ~1.3 MB |
| 训练日志 | 全部训练日志 + training_log.json | ~2 MB |
| 脚本 | train_qwen_lora.py, train_dpo.py, data_prep.py 等 | ~100 KB |
| 评估 | 9题评测结果 + test_questions.py | ~50 KB |

**COS 路径**: `cos://ins-kq6zz7wo-1313469539/LLMs-from-scratch/projects/chinese-medical-text-generation/`

## 二、恢复步骤

### 2.1 前置条件

```bash
# 目标环境需要
Python >= 3.10
CUDA GPU (推荐 >= 12GB VRAM 用于 1.7B)
40GB 磁盘空间
```

### 2.2 安装依赖

```bash
# 安装 COS SDK
pip install cos-python-sdk-v5

# 安装训练依赖
pip install torch transformers peft datasets accelerate

# 可选：QA 生成
pip install openai  # DeepSeek API
```

### 2.3 从 COS 下载备份

```bash
export COS_SECRET_ID="your_id"
export COS_SECRET_KEY="your_key"

# 下载整个项目备份
cd /path/to/LLMs-from-scratch/projects/
mkdir -p chinese-medical-text-generation
cd chinese-medical-text-generation

# 下载脚本中的 list-and-download 功能
python3 << 'PYEOF'
import subprocess, os
from pathlib import Path
from qcloud_cos import CosConfig, CosS3Client

bucket = "ins-kq6zz7wo-1313469539"
region = "ap-guangzhou"
prefix = "LLMs-from-scratch/projects/chinese-medical-text-generation/"

secret_id = os.environ["COS_SECRET_ID"]
secret_key = os.environ["COS_SECRET_KEY"]
config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
client = CosS3Client(config)

# List all files
all_keys = []
marker = ""
while True:
    resp = client.list_objects(Bucket=bucket, Prefix=prefix, Marker=marker, MaxKeys=1000)
    for obj in resp.get("Contents", []):
        all_keys.append(obj["Key"])
    if resp.get("IsTruncated") != "true":
        break
    marker = resp.get("NextMarker", all_keys[-1])

print(f"Total: {len(all_keys)} files")

# Download all
for key in all_keys:
    local_path = key[len(prefix):]  # strip prefix
    if not local_path:
        continue
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client.download_file(Bucket=bucket, Key=key, DestFilePath=local_path)
    print(f"  ✅ {local_path}")

print(f"\nDownloaded {len(all_keys)} files")
PYEOF
```

### 2.4 处理环境差异

恢复后必须修改以下配置以适应新环境：

#### a) 基座模型路径

每个 `output_*/best_model/adapter_config.json` 中包含：
```json
"base_model_name_or_path": "/home/models/ms_cache/Qwen/Qwen3-1___7B"
```

**如果新环境模型路径不同**，需要：

```bash
# 方案1: 在新环境下载模型到相同路径
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-1.7B --local-dir /home/models/ms_cache/Qwen/Qwen3-1___7B

# 方案2: 运行时指定 --model_name 参数覆盖
python generate.py --model_dir ./output_17b_inst_v7/best_model \
    --base_model /new/path/to/Qwen3-1.7B \
    --instruct --prompt "问题"
```

#### b) HF 镜像

所有脚本在 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` 已内置。

#### c) doc-search 路径（可选）

如果需要继续生成 doc-search 检索增强 QA：
```python
# 修改 scripts/gen_qa_docsearch.py 中的路径
MEDICA_INDEXES = "新索引路径"
MEDICA_RAW_DIRS = ["新raw目录路径"]
```

### 2.5 验证恢复

```bash
# 1. 验证模型可加载
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
m = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained('/home/models/ms_cache/Qwen/Qwen3-1___7B',
        torch_dtype='auto', device_map='auto', trust_remote_code=True),
    'output_17b_inst_v7/best_model'
)
print('✅ V7 loaded')
"

# 2. 推理测试
python generate.py --model_dir ./output_17b_inst_v7/best_model \
    --base_model /home/models/ms_cache/Qwen/Qwen3-1___7B \
    --instruct --prompt "胃癌的典型临床表现有哪些？"

# 3. 继续训练验证
python train_qwen_lora.py \
    --model_name /home/models/ms_cache/Qwen/Qwen3-1___7B \
    --resume_from ./output_17b_inst_v7/best_model \
    --data_dir ./data_full \
    --instruction_data ./docs/med_instruction_train_chatml.json \
    --output_dir ./output_17b_inst_v8 --epochs 1 --lr 1e-5
```

## 三、增量备份（日常）

```bash
# 每次训练完成后
python scripts/cos_backup.py backup --incremental

# 仅备份最新模型
python scripts/cos_backup.py backup --lora-only
```

## 四、环境差异对照表

| 配置项 | 源环境 | 目标环境需修改 |
|--------|--------|:--:|
| 基座模型路径 | `/home/models/ms_cache/Qwen/Qwen3-1___7B` | ✅ 必须 |
| HF 镜像 | `hf-mirror.com` (脚本内置) | 🟢 不需要 |
| GPU 类型 | NVIDIA RTX 5080 (16GB) | 🟡 batch_size 可能需调整 |
| doc-search 索引 | `D:\docs\raw\medica\` (Windows) | 🔴 仅 QA 生成时需要 |
| Python 版本 | 3.10+ | 🟢 |
| COS 凭证 | 环境变量 | ✅ 必须 |
| DeepSeek API Key | 环境变量 | 🟡 仅 QA 生成时需要 |

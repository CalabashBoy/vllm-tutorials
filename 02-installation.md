# vLLM 完全入门教程（二）：安装与环境配置

> 目标：从零开始搭建 vLLM 运行环境
> 预计时间：30 分钟

## 1. 安装方式选择

vLLM 提供多种安装方式，根据你的需求选择：

| 方式 | 适用场景 | 难度 | 推荐度 |
|------|---------|------|--------|
| pip 安装 | 快速上手、本地开发 | 简单 | ⭐⭐⭐⭐⭐ |
| Docker | 生产部署、环境隔离 | 中等 | ⭐⭐⭐⭐⭐ |
| 源码安装 | 二次开发、调试 | 复杂 | ⭐⭐⭐ |
| conda 安装 | 多 Python 版本管理 | 中等 | ⭐⭐⭐⭐ |

---

## 2. 方式一：pip 安装（推荐新手）

### 2.1 创建虚拟环境

```bash
# 创建项目目录
mkdir ~/vllm-learning
cd ~/vllm-learning

# 创建虚拟环境（使用 Python 3.10）
python3.10 -m venv venv

# 激活环境
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 升级 pip
pip install --upgrade pip
```

### 2.2 安装 vLLM

```bash
# 基础安装（推荐）
pip install vllm

# 安装指定 CUDA 版本（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# 安装最新开发版（想要最新功能）
pip install git+https://github.com/vllm-project/vllm.git
```

**⚠️ 国内用户注意：**
```bash
# 使用清华镜像加速
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云
pip install vllm -i https://mirrors.aliyun.com/pypi/simple/
```

### 2.3 验证安装

```bash
# 检查 vLLM 版本
python -c "import vllm; print(vllm.__version__)"

# 预期输出类似：0.5.4

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# 预期输出：
# PyTorch: 2.3.0+cu121
# CUDA: 12.1
```

### 2.4 安装常见问题

#### 问题 1：CUDA 版本不匹配

**错误信息：**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案：**
```bash
# 查看当前 CUDA 版本
nvidia-smi  # 看右上角的 CUDA Version

# 安装对应版本的 PyTorch
# CUDA 11.8
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  
pip install torch==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

#### 问题 2：内存不足

**错误信息：**
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

**解决方案：**
```bash
# 使用临时目录（需要 20GB+ 空间）
export TMPDIR=/path/to/large/disk/tmp
pip install vllm --no-cache-dir
```

#### 问题 3：GCC 版本过低

**错误信息：**
```
error: #error "-- unsupported GNU version! gcc versions later than 11 are not supported!"
```

**解决方案：**
```bash
# Ubuntu/Debian
sudo apt-get install gcc-11 g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install vllm
```

---

## 3. 方式二：Docker 安装（推荐生产环境）

### 3.1 安装 Docker

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install docker.io

# Mac
brew install --cask docker

# 检查安装
docker --version
```

### 3.2 拉取 vLLM 镜像

```bash
# 拉取最新镜像
docker pull vllm/vllm-openai:latest

# 拉取指定版本（推荐，更稳定）
docker pull vllm/vllm-openai:v0.5.4

# 查看镜像
docker images | grep vllm
```

### 3.3 运行容器

```bash
# 基础运行
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model facebook/opt-125m

# 详细参数说明
docker run \
  --runtime nvidia \          # 使用 NVIDIA Container Runtime
  --gpus all \                # 使用所有 GPU
  -p 8000:8000 \              # 端口映射
  -v ~/.cache/huggingface:/root/.cache/huggingface \  # 挂载模型缓存
  --ipc=host \                # 共享内存（多卡必需）
  vllm/vllm-openai:latest \
  --model facebook/opt-125m   # 模型名称
```

### 3.4 Docker Compose（推荐）

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./models:/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      --model facebook/opt-125m
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.9
```

运行：
```bash
docker-compose up -d
docker-compose logs -f
```

---

## 4. 方式三：源码安装（开发者）

```bash
# 克隆仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 创建环境
conda create -n vllm python=3.10
conda activate vllm

# 安装依赖
pip install -r requirements-build.txt
pip install -e .

# 编译（可选，修改 C++ 代码后需要）
python setup.py build_ext --inplace
```

---

## 5. 模型下载配置

### 5.1 HuggingFace 镜像（国内必需）

```bash
# 设置镜像（推荐添加到 ~/.bashrc）
export HF_ENDPOINT=https://hf-mirror.com

# 或者使用环境变量
HF_ENDPOINT=https://hf-mirror.com python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
```

### 5.2 手动下载模型

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
huggingface-cli download --resume-download facebook/opt-125m --local-dir ./models/opt-125m

# 使用本地模型
python -m vllm.entrypoints.openai.api_server --model ./models/opt-125m
```

### 5.3 使用 ModelScope（阿里）

```bash
# 安装
pip install modelscope

# 下载
python -c "from modelscope import snapshot_download; snapshot_download('modelscope/Llama-2-7b-chat-ms', cache_dir='./models')"
```

---

## 6. 环境验证测试

### 6.1 启动测试服务器

```bash
# 使用小模型测试（适合 6GB 显存）
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --port 8000

# 预期输出：
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6.2 API 测试

```bash
# 另开一个终端
curl http://localhost:8000/v1/models

# 预期输出：
# {"object":"list","data":[{"id":"facebook/opt-125m","object":"model"}]}

# 发送聊天请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 6.3 Python 客户端测试

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# 简单对话
response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "你好，请介绍一下自己"}]
)
print(response.choices[0].message.content)
```

---

## 7. 开发工具配置

### 7.1 VS Code 配置

`.vscode/settings.json`：
```json
{
  "python.defaultInterpreterPath": "~/vllm-learning/venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "black"
}
```

### 7.2 调试配置

`.vscode/launch.json`：
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "vLLM Server",
      "type": "python",
      "request": "launch",
      "module": "vllm.entrypoints.openai.api_server",
      "args": ["--model", "facebook/opt-125m", "--port", "8000"],
      "console": "integratedTerminal"
    }
  ]
}
```

---

## 8. 本章小结

**安装方式选择：**
- 新手学习 → pip 安装
- 生产部署 → Docker
- 二次开发 → 源码安装

**关键配置：**
- ✅ CUDA 版本匹配（11.8 或 12.1）
- ✅ HuggingFace 镜像（国内必需）
- ✅ 模型缓存目录

**验证成功标准：**
- ✅ `vllm --version` 正常输出
- ✅ 服务器启动无报错
- ✅ API 请求返回 200

---

## 9. 课后练习

1. 使用 pip 安装 vLLM，并验证版本
2. 下载一个中文模型（如 ChatGLM3-6B）
3. 成功启动服务器并通过 API 对话
4. 记录安装过程中遇到的问题和解决方法

---

*上一篇：[01-什么是 vLLM](01-introduction.md)*
*下一篇：[03-快速上手：运行你的第一个模型](03-quickstart.md)*

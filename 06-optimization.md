# vLLM 完全入门教程（六）：性能优化与模型量化

> 目标：掌握 vLLM 的性能优化技巧和量化方法
> 难度：⭐⭐⭐
> 预计时间：35 分钟

## 1. 为什么需要优化？

### 1.1 成本计算

假设你部署了一个 7B 模型：

```
场景：日均 10 万请求，平均每个请求 500 tokens

无优化（FP32）：
- 需要：A100 40GB × 2 台
- 成本：$3/小时 × 24 × 30 = $2160/月

优化后（INT8 量化）：
- 需要：A100 40GB × 1 台
- 成本：$1080/月

节省：50%
```

### 1.2 优化方向

| 方向 | 收益 | 难度 |
|------|------|------|
| **量化** | 模型大小减半，速度提升 | 低 |
| **批处理** | 吞吐量 5-10 倍提升 | 中 |
| **缓存优化** | 降低延迟 | 中 |
| **多卡并行** | 支持大模型 | 中 |

---

## 2. 模型量化详解

### 2.1 什么是量化？

```
FP16（半精度浮点）：
- 每个参数：2 字节
- 7B 模型：14 GB
- 精度高，速度慢

INT8（8位整数）：
- 每个参数：1 字节
- 7B 模型：7 GB
- 精度略有损失，速度快

INT4（4位整数）：
- 每个参数：0.5 字节
- 7B 模型：3.5 GB
- 精度损失明显，速度最快
```

### 2.2 vLLM 支持的量化方法

| 方法 | 精度 | 速度 | 适用场景 |
|------|------|------|---------|
| AWQ | INT4 | ⭐⭐⭐⭐⭐ | 极致压缩 |
| GPTQ | INT4 | ⭐⭐⭐⭐ | 平衡选择 |
| SqueezeLLM | INT4 | ⭐⭐⭐ | 特定模型 |
| FP8 | FP8 | ⭐⭐⭐⭐ | H100 专用 |

### 2.3 AWQ 量化实践

**AWQ（Activation-aware Weight Quantization）** 是目前效果最好的 4bit 量化方法。

#### 安装依赖

```bash
pip install autoawq
```

#### 量化模型

```python
# quantize_awq.py
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "Llama-2-7b-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 准备校准数据（用样本数据确定量化参数）
examples = [
    tokenizer("auto-gptq is an easy-to-use model quantization library", return_tensors="pt"),
    tokenizer("The capital of France is Paris", return_tensors="pt"),
]

# 量化
model.quantize(tokenizer, quant_config=quant_config, calib_data=examples)

# 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

#### vLLM 加载 AWQ 模型

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --quantization awq \
  --dtype float16
```

### 2.4 GPTQ 量化实践

**GPTQ** 是另一种流行的 4bit 量化方法。

```bash
# 使用 AutoGPTQ 量化
pip install auto-gptq

# 量化脚本
python -c "
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_id = 'meta-llama/Llama-2-7b-hf'
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)

model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

# 准备校准数据
calib_data = ['auto-gptq is an easy-to-use model', 'example 2', 'example 3']

# 量化并保存
model.quantize(calib_data)
model.save_quantized('Llama-2-7b-gptq')
"
```

vLLM 中使用：
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Llama-2-7b-gptq \
  --quantization gptq
```

### 2.5 量化效果对比

```
模型：Llama-2-7B

FP16（原版）：
- 模型大小：14 GB
- 显存需求：16 GB+
- 速度：100 tokens/s
- 困惑度：5.12

INT8 量化：
- 模型大小：7 GB
- 显存需求：10 GB
- 速度：120 tokens/s（+20%）
- 困惑度：5.15（几乎无损）

INT4 (AWQ)：
- 模型大小：4 GB
- 显存需求：6 GB
- 速度：150 tokens/s（+50%）
- 困惑度：5.45（轻微损失）
```

---

## 3. 批处理优化

### 3.1 动态批处理参数

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --max-num-batched-tokens 4096 \      # 最大批处理 token 数
  --max-num-seqs 256 \                  # 最大并发序列数
  --max-model-len 4096                  # 最大序列长度
```

**参数调优建议：**

```
场景 A：高并发短对话（客服场景）
- max-num-batched-tokens: 2048
- max-num-seqs: 512
- max-model-len: 1024

场景 B：长文档生成（写作助手）
- max-num-batched-tokens: 8192
- max-num-seqs: 64
- max-model-len: 8192
```

### 3.2 前缀缓存（Prefix Caching）

当多个请求有相同前缀时，可以复用 KV Cache：

```bash
# 启用前缀缓存
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --enable-prefix-caching
```

**效果示例：**
```
请求1: "请翻译以下英文到中文: Hello" → 正常生成
请求2: "请翻译以下英文到中文: World" → 前缀复用，速度提升 50%
请求3: "请翻译以下英文到中文: Apple" → 前缀复用，速度提升 50%
```

### 3.3 异步输出处理

```bash
# 开启异步处理（减少生成延迟）
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --async-output-processing
```

---

## 4. 内存优化

### 4.1 GPU 内存利用率调优

```bash
# 根据你的 GPU 和模型调整
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --gpu-memory-utilization 0.95        # 使用 95% 显存
```

**常见配置：**
| GPU | 模型 | 推荐利用率 |
|-----|------|-----------|
| RTX 4090 24GB | 7B FP16 | 0.90 |
| A100 40GB | 13B FP16 | 0.95 |
| A100 80GB | 70B INT4 | 0.95 |

### 4.2 交换空间（CPU Offloading）

当 GPU 显存不够时，可以将部分 KV Cache 换出到 CPU 内存：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --swap-space 4                        # 4 GB CPU swap
```

**注意：** 换出到 CPU 会显著增加延迟，仅作为兜底方案。

---

## 5. 多卡优化

### 5.1 张量并行 vs 流水线并行

```
张量并行（Tensor Parallelism）：
- 每张卡计算模型的一部分层
- 通信量大，但延迟低
- 适合单节点多卡

流水线并行（Pipeline Parallelism）：
- 每张卡计算不同的层组
- 通信量小，但有气泡
- 适合多机多卡
```

### 5.2 配置示例

```bash
# 单节点 4 卡张量并行
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1

# 双节点 8 卡（每节点 4 卡）
# Node 1
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
```

---

## 6. 性能测试与调优

### 6.1 基准测试脚本

```python
# benchmark_throughput.py
import time
import requests
from concurrent.futures import ThreadPoolExecutor

def send_request(prompt):
    """发送单个请求"""
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "meta-llama/Llama-2-7b-hf",
            "prompt": prompt,
            "max_tokens": 100
        }
    )
    return response.json()

def benchmark(concurrency=10, total=100):
    """性能测试"""
    prompts = ["Hello, how are you?"] * total
    
    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(send_request, prompts))
    elapsed = time.time() - start
    
    throughput = total / elapsed
    latency = elapsed / total
    
    print(f"并发数: {concurrency}")
    print(f"总请求: {total}")
    print(f"总时间: {elapsed:.2f}s")
    print(f"吞吐量: {throughput:.2f} req/s")
    print(f"平均延迟: {latency*1000:.2f}ms")

if __name__ == "__main__":
    benchmark(concurrency=50, total=500)
```

### 6.2 调优流程

```
1. 基线测试（记录当前性能）
   ↓
2. 分析瓶颈（监控 GPU 利用率、内存使用）
   ↓
3. 选择优化方向（量化/批处理/缓存）
   ↓
4. 实施优化
   ↓
5. 对比测试
   ↓
6. 迭代优化
```

---

## 7. 优化效果总结

| 优化手段 | 吞吐量提升 | 延迟降低 | 成本节省 |
|---------|-----------|---------|---------|
| INT8 量化 | 20% | 10% | 50% |
| INT4 量化 | 50% | 30% | 75% |
| 批处理优化 | 500% | - | 80% |
| 前缀缓存 | 50% | 50% | - |
| 多卡并行 | 300% | - | - |

**综合优化案例：**
```
原始：
- 7B 模型 FP16
- 吞吐量：20 req/s
- 成本：$2000/月

优化后：
- 7B 模型 INT4
- 批处理优化
- 吞吐量：150 req/s（7.5 倍）
- 成本：$500/月（75% 节省）
```

---

## 8. 课后练习

1. 对一个 7B 模型进行 AWQ 量化，对比原始模型
2. 测试不同 batch size 对性能的影响
3. 启用前缀缓存，测试相同前缀的请求速度
4. 编写性能监控脚本，持续跟踪优化效果

---

*上一篇：[05-生产环境部署](05-deployment.md)*
*下一篇：进阶专题（持续更新）*

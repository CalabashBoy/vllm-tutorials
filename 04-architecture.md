# vLLM 完全入门教程（四）：深入理解 PagedAttention

> 目标：深入理解 vLLM 的核心技术原理
> 难度：⭐⭐⭐⭐（需要一定计算机基础）
> 预计时间：40 分钟

## 1. 为什么要深入理解 PagedAttention？

**实际场景：**
你部署了一个 LLM 服务，发现：
- GPU 显存总是不够用
- 并发一高就 OOM
- 不知道该如何优化

**理解 PagedAttention 后：**
- 你能合理设置 `--gpu-memory-utilization`
- 你知道如何计算最大并发数
- 你能针对性地优化性能

---

## 2. Transformer 回顾

### 2.1 Attention 机制是什么？

```
输入: "我爱机器学习"

Step 1: 每个词变成向量（Embedding）
我 → [0.1, 0.2, ...]
爱 → [0.3, 0.4, ...]
机 → [0.5, 0.6, ...]
器 → [0.7, 0.8, ...]
学 → [0.9, 1.0, ...]
习 → [1.1, 1.2, ...]

Step 2: Attention 计算
每个词都要 "看" 其他所有词，计算相关性

Step 3: 生成下一个词的概率分布
P(下一个词 | 我爱机器学习) = ?
```

### 2.2 KV Cache 是什么？

**观察发现：** 生成第 N 个词时，前 N-1 个词的 Key 和 Value 都是不变的。

```python
# 不优化的方式（重复计算）
for i in range(max_length):
    # 每次都要重新计算所有位置的 attention
    output = transformer(full_sequence)

# 优化的方式（KV Cache）
k_cache, v_cache = [], []
for i in range(max_length):
    # 只计算新的位置
    k, v = compute_kv(new_token)
    k_cache.append(k)
    v_cache.append(v)
    output = attention_with_cache(q, k_cache, v_cache)
```

**KV Cache 的大小：**
```
对于 batch_size=1, seq_len=4096:
- 每层有 K 和 V 两个 cache
- 每个 cache 大小: seq_len × head_dim
- 假设: 32 层, 32 个头, 每个头 128 维

每层 KV Cache = 2 × 4096 × 128 × 32 = 33.5 MB
总 KV Cache = 33.5 MB × 32 层 = 1.07 GB
```

### 2.3 传统 KV Cache 的问题

```
预分配策略的问题：

请求 1: "你好" (2 tokens)
预分配: 4096 tokens × 1.07 GB = 4.3 GB 显存
实际使用: 2 tokens × 1.07 GB = 2.1 MB
浪费率: 99.95%

请求 2: "今天天气怎么样" (7 tokens)  
同样浪费 99.9%

10 个并发请求：
预分配: 43 GB 显存
实际需要: 21 MB 显存
OOM！
```

---

## 3. PagedAttention 详解

### 3.1 核心思想：操作系统虚拟内存

```
操作系统虚拟内存：
┌─────────────────────────────────────┐
│ 程序A需要 100MB，程序B需要 200MB     │
│ 物理内存只有 256MB                   │
│ 但每个程序都认为自己有 4GB 地址空间   │
└─────────────────────────────────────┘

解决方案：
1. 内存分页（4KB 一页）
2. 按需分配
3. 不用的页换出到磁盘
4. 物理上不连续，逻辑上连续
```

**PagedAttention 把这整套机制搬到了 GPU 上！**

### 3.2 PagedAttention 的内存管理

```
传统方式（连续内存）：
┌──────────────────────────────────────────────────────┐
│  请求1 (4096 tokens)  │  请求2 (4096 tokens)  │ ...  │
│  ←──── 预分配 ────→   │  ←──── 预分配 ────→   │      │
│       4.3 GB          │       4.3 GB          │      │
└──────────────────────────────────────────────────────┘
实际用了: 100 tokens → 还是占用 4.3 GB

PagedAttention 方式（分页）：
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ B0  │ B1  │ B2  │ B3  │ B4  │ B5  │ B6  │ B7  │  物理块池
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

请求1的逻辑视图（实际用了 3 个块）:
Block Table: [B0, B3, B7]  ← 通过映射表找到物理位置
逻辑上连续: [token0-15] → [token16-31] → [token32-47]

请求2的逻辑视图（实际用了 2 个块）:
Block Table: [B1, B5]
逻辑上连续: [token0-15] → [token16-31]

未使用的块: B2, B4, B6（可以分配给其他请求）
```

### 3.3 块大小（Block Size）的选择

```python
# vLLM 默认块大小: 16 tokens

为什么选 16？
- 太小：Block Table 太大，管理开销高
- 太大：内部碎片多（一个块内用不完）
- 16 是经验值，在开销和碎片间平衡

计算公式：
num_gpu_blocks = gpu_memory // (block_size × head_size × num_heads × num_layers × 2)

示例（RTX 4090 24GB）：
- block_size = 16
- head_size = 128
- num_heads = 32
- num_layers = 32
- 每个块大小 = 16 × 128 × 32 × 32 × 2 = 4 MB
- num_gpu_blocks ≈ 24GB / 4MB ≈ 6000 个块
```

### 3.4 内存共享：Copy-on-Write

```
场景：多个请求有相同的前缀

请求1: "请翻译: 你好" → 翻译为 Hello
请求2: "请翻译: 世界" → 翻译为 World

前缀 "请翻译: " 可以共享！

初始状态：
请求1: [请翻译: ][你][好]
       ↓ 共享 ↓
请求2: [请翻译: ][世][界]
       [Block0][Block1][Block2]

共享块 Block0 被两个请求引用（引用计数=2）

当请求1继续生成：
请求1: [请翻译: ][你][好][是][中][国][人]
       [Block0][Block1][Block2][Block3][Block4][Block5]
       ↑ 共享 ↑
请求2: [请翻译: ][世][界]
       [Block0][Block6][Block7]
       
注意：请求2 从 "世界" 开始不同了，所以 Block1 和 Block2 不再共享
使用 Copy-on-Write 机制复制
```

### 3.5 连续批处理的实现

```
时间线展示：

时刻 0ms:
  Batch: [ReqA(token=10), ReqB(token=8)]
  一起 forward，生成下一个 token
  
时刻 20ms:
  ReqA 完成（达到 max_tokens）
  ReqC 新请求进来
  Batch: [ReqB(token=9), ReqC(token=1)]
  
时刻 40ms:
  ReqB 完成
  ReqD 进来
  Batch: [ReqC(token=2), ReqD(token=1)]

关键：GPU 一直在满负荷工作！
```

---

## 4. 源码层面的理解

### 4.1 核心数据结构

```python
# vllm/core/block_manager.py

class BlockAllocator:
    """块分配器"""
    
    def __init__(self, block_size: int, num_gpu_blocks: int, num_cpu_blocks: int):
        self.block_size = block_size  # 默认 16
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        
        # 空闲块列表
        self.gpu_blocks: List[PhysicalTokenBlock] = [
            PhysicalTokenBlock(block_number=i) 
            for i in range(num_gpu_blocks)
        ]
        self.cpu_blocks: List[PhysicalTokenBlock] = [
            PhysicalTokenBlock(block_number=i)
            for i in range(num_cpu_blocks)
        ]
    
    def allocate(self, num_blocks: int, device: Device) -> List[PhysicalTokenBlock]:
        """分配 num_blocks 个块"""
        blocks = self._get_free_blocks(num_blocks, device)
        for block in blocks:
            block.ref_count = 1
        return blocks
    
    def free(self, block: PhysicalTokenBlock) -> None:
        """释放块（引用计数-1，为0时回收）"""
        block.ref_count -= 1
        if block.ref_count == 0:
            self._add_to_free_list(block)
```

### 4.2 Sequence 状态管理

```python
# vllm/sequence.py

class Sequence:
    """表示一个推理序列"""
    
    def __init__(self, seq_id: int, prompt: str):
        self.seq_id = seq_id
        self.prompt = prompt
        self.output_text = ""
        
        # Block Table：逻辑块到物理块的映射
        self.block_table: List[PhysicalTokenBlock] = []
        
        # 状态
        self.status = SequenceStatus.WAITING  # WAITING/RUNNING/DONE
        
    def append_token(self, token: int):
        """添加生成的 token"""
        # 检查当前块是否已满
        if self._is_last_block_full():
            # 分配新块
            new_block = block_allocator.allocate(1, Device.GPU)[0]
            self.block_table.append(new_block)
```

### 4.3 Scheduler 调度器

```python
# vllm/core/scheduler.py

class Scheduler:
    """决定哪些 sequence 应该被处理"""
    
    def schedule(self) -> SchedulerOutputs:
        # 1. 处理已完成的序列
        for seq in self.running:
            if seq.is_finished():
                self.running.remove(seq)
                self._free_blocks(seq)
        
        # 2. 尝试加入 waiting 队列中的新请求
        for seq in self.waiting:
            if self._can_allocate(seq):
                self._allocate_blocks(seq)
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                break  # 内存不足，停止添加
        
        # 3. 返回本次要处理的 batch
        return SchedulerOutputs(self.running)
```

---

## 5. 性能优化实战

### 5.1 计算你的最大并发数

```python
def calculate_max_concurrency(
    gpu_memory_gb: float,
    model_size_gb: float,
    seq_length: int,
    block_size: int = 16
):
    """
    计算给定配置下的最大并发数
    """
    # 可用显存（留 10% 缓冲）
    available_memory = gpu_memory_gb * 0.9 - model_size_gb
    
    # 每个序列需要的 KV Cache
    # 假设: 32层, 32头, 128维, 2 (K+V)
    bytes_per_token = 32 * 32 * 128 * 2 * 4  # 4 bytes per float32
    
    # 考虑块对齐的浪费（平均浪费半个块）
    tokens_per_seq = seq_length + block_size // 2
    memory_per_seq = tokens_per_seq * bytes_per_token / (1024**3)
    
    max_concurrency = int(available_memory / memory_per_seq)
    return max_concurrency

# 示例：RTX 4090 跑 LLaMA-7B
print(calculate_max_concurrency(
    gpu_memory_gb=24,
    model_size_gb=14,  # FP16
    seq_length=2048
))
# 输出: 约 15-20 并发
```

### 5.2 调优参数建议

```bash
# 根据你的场景调整

# 场景1: 高并发，短对话
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b \
  --gpu-memory-utilization 0.95 \      # 尽量用满显存
  --max-model-len 1024 \                # 限制序列长度
  --max-num-seqs 100                    # 最多100个并发

# 场景2: 低并发，长文档
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \                # 支持长文档
  --max-num-seqs 10                     # 少并发，长序列
```

### 5.3 监控指标解读

```bash
# vLLM 提供 Prometheus 指标
curl http://localhost:8000/metrics

关键指标：
- vllm:gpu_cache_usage_perc      # GPU KV Cache 使用率
  目标: 70-90%，太低浪费，太高易OOM
  
- vllm:num_requests_running      # 当前运行请求数
  目标: 接近你的 max-num-seqs
  
- vllm:time_to_first_token_seconds  # 首token延迟
  目标: < 500ms
  
- vllm:time_per_output_token_seconds  # 生成速度
  目标: 每个token < 50ms
```

---

## 6. 本章小结

**核心概念：**
1. ✅ **KV Cache** - 避免重复计算，但传统方式浪费严重
2. ✅ **PagedAttention** - 分页管理，按需分配，支持共享
3. ✅ **Block Table** - 逻辑到物理的映射表
4. ✅ **Copy-on-Write** - 安全共享前缀，修改时复制

**性能公式：**
```
最大并发数 ≈ (GPU显存 × 利用率 - 模型大小) / (序列长度 × 每token内存)
```

**调优方向：**
- 增加 `--gpu-memory-utilization`（如果不会OOM）
- 调整 `--max-model-len` 匹配实际需求
- 监控 `gpu_cache_usage_perc` 找到最佳平衡点

---

## 7. 课后练习

1. 画出 PagedAttention 的内存分配示意图
2. 计算你的 GPU 能支持的最大并发数
3. 实验不同 `--gpu-memory-utilization` 值的效果
4. 观察 `vllm:gpu_cache_usage_perc` 指标，找到最佳配置

---

*上一篇：[03-快速上手](03-quickstart.md)*
*下一篇：[05-生产环境部署](05-deployment.md)*

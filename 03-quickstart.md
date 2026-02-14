# vLLM 完全入门教程（三）：快速上手 - 运行你的第一个模型

> 目标：从零开始运行 vLLM 服务器，并完成第一次对话
> 预计时间：20 分钟

## 1. 启动 vLLM 服务器

### 1.1 最简启动方式

```bash
# 进入你的虚拟环境
cd ~/vllm-learning
source venv/bin/activate

# 启动服务器
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --port 8000
```

**输出解析：**
```
INFO 06-15 10:30:12 llm_engine.py:79] Initializing an LLM engine with config: model='facebook/opt-125m', tokenizer='facebook/opt-125m', ...
INFO 06-15 10:30:15 llm_engine.py:230] # GPU blocks: 1536, # CPU blocks: 512  ← 关键信息：GPU 内存分配
INFO 06-15 10:30:16 api_server.py:219] Started server process [12345]
INFO 06-15 10:30:16 api_server.py:221] Uvicorn running on http://0.0.0.0:8000  ← 服务地址
```

### 1.2 常用启动参数详解

```bash
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \           # 模型名称或路径
  --tokenizer facebook/opt-125m \       # Tokenizer（通常和模型相同）
  --port 8000 \                         # 服务端口
  --host 0.0.0.0 \                      # 绑定地址（0.0.0.0 允许外部访问）
  --tensor-parallel-size 1 \            # GPU 数量（单卡=1）
  --gpu-memory-utilization 0.9 \        # GPU 内存使用率（默认 0.9）
  --max-model-len 2048 \                # 最大序列长度
  --max-num-batched-tokens 4096 \       # 最大批处理 token 数
  --dtype float16                       # 数据类型（float16/bfloat16）
```

---

## 2. 使用 OpenAI 客户端调用

### 2.1 安装客户端

```bash
pip install openai
```

### 2.2 基础对话示例

创建 `test_chat.py`：

```python
from openai import OpenAI

# 配置客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM 服务地址
    api_key="dummy"                        # vLLM 不检查 API key，但必须提供
)

def chat(message):
    """单次对话"""
    response = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0.7,      # 创造性（0-2，越大越随机）
        max_tokens=100        # 最大生成 token 数
    )
    return response.choices[0].message.content

# 测试
if __name__ == "__main__":
    user_input = input("你: ")
    reply = chat(user_input)
    print(f"AI: {reply}")
```

运行：
```bash
python test_chat.py
```

### 2.3 多轮对话示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

class ChatSession:
    def __init__(self):
        self.messages = []
    
    def send(self, message):
        """发送消息并获取回复"""
        self.messages.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model="facebook/opt-125m",
            messages=self.messages,
            temperature=0.7,
            max_tokens=150
        )
        
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply
    
    def clear(self):
        """清空对话历史"""
        self.messages = []

# 使用示例
session = ChatSession()

print("开始对话（输入 'quit' 退出，'clear' 清空历史）")
while True:
    user_input = input("\n你: ").strip()
    
    if user_input.lower() == 'quit':
        break
    elif user_input.lower() == 'clear':
        session.clear()
        print("[历史已清空]")
        continue
    
    reply = session.send(user_input)
    print(f"AI: {reply}")
```

---

## 3. 流式输出（Streaming）

### 3.1 什么是流式输出？

```
普通输出：等待整个回复生成完成 → 一次性显示（慢）
流式输出：生成一个 token 显示一个（快，体验好）

流式效果：
AI: 这 → 这是 → 这是一 → 这是一个 → 这是一个测 → 这是一个测试
```

### 3.2 流式输出示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

def chat_stream(message):
    """流式对话"""
    stream = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[{"role": "user", "content": message}],
        stream=True  # ← 启用流式
    )
    
    print("AI: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

# 使用
chat_stream("请讲一个短故事")
```

---

## 4. 采样参数详解

### 4.1 Temperature（温度）

```python
# temperature 控制输出的随机性

# 0.0 - 确定性输出（每次结果一样）
# 0.7 - 平衡（推荐）
# 1.5 - 很有创造性
# 2.0 - 非常随机，可能不通顺

response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "写一句诗"}],
    temperature=0.7  # 调整这个值看效果
)
```

**实验：** 同一个问题用不同 temperature 问 3 次，观察差异。

### 4.2 Top-p（核采样）

```python
# top_p 控制候选词的范围
# 0.1 - 只考虑最可能的 10% 的词（保守）
# 0.9 - 考虑 90% 的词（多样）

response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "写一句诗"}],
    temperature=0.7,
    top_p=0.9
)
```

### 4.3 常用参数组合

| 场景 | temperature | top_p | max_tokens |
|------|-------------|-------|------------|
| 问答/事实 | 0.1-0.3 | 0.5 | 256 |
| 聊天对话 | 0.7-0.9 | 0.9 | 512 |
| 创意写作 | 1.0-1.3 | 0.95 | 1024 |
| 代码生成 | 0.2-0.5 | 0.8 | 512 |

---

## 5. 其他 API 功能

### 5.1 获取模型列表

```python
models = client.models.list()
for model in models.data:
    print(f"模型ID: {model.id}")
    print(f"所有者: {model.owned_by}")
```

### 5.2 文本补全（Completion API）

```python
# 非聊天模型使用 completion
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="今天天气很好，",
    max_tokens=50
)
print(response.choices[0].text)
```

### 5.3 嵌入（Embeddings）

```python
# 获取文本向量（用于语义搜索等）
response = client.embeddings.create(
    model="facebook/opt-125m",
    input="这是一段文本"
)
embedding = response.data[0].embedding
print(f"向量维度: {len(embedding)}")
```

---

## 6. 实用工具函数

### 6.1 对话保存与加载

```python
import json

class PersistentChat:
    def __init__(self, session_file="chat_history.json"):
        self.session_file = session_file
        self.messages = self.load()
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )
    
    def load(self):
        try:
            with open(self.session_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save(self):
        with open(self.session_file, 'w') as f:
            json.dump(self.messages, f, indent=2)
    
    def send(self, message, stream=True):
        self.messages.append({"role": "user", "content": message})
        
        response = self.client.chat.completions.create(
            model="facebook/opt-125m",
            messages=self.messages,
            stream=stream
        )
        
        if stream:
            reply = ""
            print("AI: ", end="", flush=True)
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    reply += content
            print()
        else:
            reply = response.choices[0].message.content
            print(f"AI: {reply}")
        
        self.messages.append({"role": "assistant", "content": reply})
        self.save()
        return reply

# 使用
chat = PersistentChat()
chat.send("你好，请介绍一下机器学习")
```

---

## 7. 本章小结

**核心技能：**
- ✅ 启动 vLLM 服务器
- ✅ 使用 OpenAI 客户端调用
- ✅ 多轮对话维护
- ✅ 流式输出实现
- ✅ 采样参数调整

**关键代码模板：**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 基础调用
response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
    max_tokens=100
)

# 流式
response = client.chat.completions.create(..., stream=True)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

---

## 8. 课后练习

1. 启动服务器，成功运行示例代码
2. 尝试不同的 temperature 值，观察输出变化
3. 实现一个带历史记录的聊天机器人
4. 测试流式输出，感受响应速度差异
5. 尝试使用其他模型（如 ChatGLM3-6B）

---

*上一篇：[02-安装与环境配置](02-installation.md)*
*下一篇：[04-深入理解 PagedAttention](04-architecture.md)*

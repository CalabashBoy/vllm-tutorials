# vLLM 完全入门教程（五）：生产环境部署

> 目标：将 vLLM 部署到生产环境
> 难度：⭐⭐⭐⭐
> 预计时间：45 分钟

## 1. 生产环境 vs 开发环境

| 方面 | 开发环境 | 生产环境 |
|------|---------|---------|
| 稳定性 | 可以重启 | 必须高可用 |
| 监控 | 偶尔看日志 | 完整监控告警 |
| 安全 | 本地访问 | 认证、限流、审计 |
| 扩展 | 单机 | 多机负载均衡 |
| 容错 | 简单处理 | 优雅降级 |

---

## 2. Docker 生产部署

### 2.1 单卡部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:v0.5.4
    container_name: vllm-server
    ports:
      - "8000:8000"
    volumes:
      - model-cache:/root/.cache/huggingface
      - ./logs:/var/log/vllm
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_HOME=/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      --model meta-llama/Llama-2-7b-hf
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.9
      --max-model-len 4096
      --max-num-seqs 256
      --dtype float16
      --port 8000
      --host 0.0.0.0
      --ssl-keyfile /etc/ssl/private/key.pem
      --ssl-certfile /etc/ssl/certs/cert.pem
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

volumes:
  model-cache:
```

启动：
```bash
docker-compose up -d
docker-compose logs -f
```

### 2.2 多卡并行部署

```yaml
version: '3.8'

services:
  vllm-70b:
    image: vllm/vllm-openai:v0.5.4
    container_name: vllm-llama70b
    ports:
      - "8000:8000"
    volumes:
      - model-cache:/root/.cache/huggingface
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用 4 张卡
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]
    shm_size: '16gb'  # 共享内存，多卡必需
    command: >
      --model meta-llama/Llama-2-70b-hf
      --tensor-parallel-size 4          # 4卡张量并行
      --pipeline-parallel-size 1        # 不使用流水线并行
      --gpu-memory-utilization 0.9
      --max-model-len 4096
    restart: unless-stopped
```

### 2.3 多机部署（Ray Cluster）

当单台机器 GPU 不够时：

```bash
# Node 1 (Head)
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Node 2,3,4 (Workers)
ray start --address='head-node-ip:6379'

# 验证集群
ray status
```

```yaml
# docker-compose.yml
services:
  vllm-distributed:
    image: vllm/vllm-openai:v0.5.4
    environment:
      - RAY_ADDRESS=ray://head-node:10001
    command: >
      --model meta-llama/Llama-2-70b-hf
      --tensor-parallel-size 8          # 8 卡跨 2 台机器
```

---

## 3. API 网关与负载均衡

### 3.1 Nginx 反向代理

```nginx
# /etc/nginx/sites-available/vllm
upstream vllm_backend {
    least_conn;  # 最少连接负载均衡
    
    server 192.168.1.10:8000 max_fails=3 fail_timeout=30s;
    server 192.168.1.11:8000 max_fails=3 fail_timeout=30s;
    server 192.168.1.12:8000 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # 限流
    limit_req_zone $binary_remote_addr zone=vllm:10m rate=10r/s;
    limit_req zone=vllm burst=20 nodelay;
    
    # 连接限制
    limit_conn_zone $binary_remote_addr zone=addr:10m;
    limit_conn addr 10;
    
    location / {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;  # 生成长文本需要更长时间
        
        # 缓冲
        proxy_buffering off;
        
        # 错误处理
        proxy_intercept_errors on;
        error_page 502 503 504 = @fallback;
    }
    
    location @fallback {
        return 503 "{\"error\": \"Service temporarily unavailable\"}";
    }
}
```

### 3.2 Kong API 网关

```yaml
# docker-compose.yml
services:
  kong:
    image: kong:3.5
    environment:
      KONG_DATABASE: "off"
      KONG_DECLARATIVE_CONFIG: /kong/declarative/kong.yml
      KONG_PROXY_ACCESS_LOG: /dev/stdout
      KONG_ADMIN_ACCESS_LOG: /dev/stdout
      KONG_PROXY_ERROR_LOG: /dev/stderr
      KONG_ADMIN_ERROR_LOG: /dev/stderr
      KONG_PLUGINS: rate-limiting,key-auth
    volumes:
      - ./kong.yml:/kong/declarative/kong.yml
    ports:
      - "8000:8000"
      - "8443:8443"
```

```yaml
# kong.yml
_format_version: "3.0"
services:
  - name: vllm-service
    url: http://vllm:8000
    routes:
      - name: vllm-route
        paths:
          - /
    plugins:
      - name: rate-limiting
        config:
          minute: 60
          policy: redis
          redis_host: redis
      - name: key-auth
        config:
          key_names:
            - api-key
consumers:
  - username: client1
    keyauth_credentials:
      - key: sk-abc123
```

---

## 4. 监控与日志

### 4.1 Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"

volumes:
  prometheus_data:
  grafana_data:
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:8000']
    metrics_path: /metrics
```

### 4.2 关键监控指标

| 指标 | 告警阈值 | 含义 |
|------|---------|------|
| vllm:gpu_cache_usage_perc | > 95% | GPU 缓存快满了 |
| vllm:num_requests_waiting | > 10 | 队列堆积 |
| vllm:time_to_first_token_seconds | > 2s | 首 token 太慢 |
| vllm:prompt_tokens_total | - | 总输入 token 数 |
| vllm:generation_tokens_total | - | 总生成 token 数 |

### 4.3 日志收集

```python
# logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = record.created
        log_record['level'] = record.levelname

# 配置 vLLM 日志
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('/var/log/vllm/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('vllm')
logHandler = logging.StreamHandler()
formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
```

---

## 5. 高可用与故障恢复

### 5.1 健康检查脚本

```python
# health_check.py
import requests
import sys
import time

def check_health(endpoint="http://localhost:8000/health"):
    try:
        response = requests.get(endpoint, timeout=5)
        if response.status_code == 200:
            print(f"✓ {time.strftime('%Y-%m-%d %H:%M:%S')} - Healthy")
            return True
        else:
            print(f"✗ {time.strftime('%Y-%m-%d %H:%M:%S')} - Unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ {time.strftime('%Y-%m-%d %H:%M:%S')} - Error: {e}")
        return False

def check_metrics(endpoint="http://localhost:8000/metrics"):
    """检查关键指标"""
    try:
        response = requests.get(endpoint, timeout=10)
        metrics = response.text
        
        # 解析 GPU 缓存使用率
        for line in metrics.split('\n'):
            if 'vllm:gpu_cache_usage_perc' in line and '#' not in line:
                usage = float(line.split()[-1])
                if usage > 0.95:
                    print(f"⚠️ GPU cache usage high: {usage:.1%}")
                    return False
        return True
    except Exception as e:
        print(f"✗ Metrics check failed: {e}")
        return False

if __name__ == "__main__":
    healthy = check_health() and check_metrics()
    sys.exit(0 if healthy else 1)
```

### 5.2 自动重启配置

```bash
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM Inference Server
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/vllm
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
ExecReload=/usr/local/bin/docker-compose restart
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl enable vllm
sudo systemctl start vllm
sudo systemctl status vllm
```

---

## 6. 安全加固

### 6.1 API 认证

```python
# api_key_middleware.py
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
security = HTTPBearer()

VALID_API_KEYS = {"sk-abc123", "sk-def456"}

@app.middleware("http")
async def verify_api_key(request, call_next):
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    
    api_key = auth[7:]  # Remove "Bearer "
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return await call_next(request)
```

### 6.2 输入过滤

```python
# input_filter.py
import re

FORBIDDEN_PATTERNS = [
    r"\b(password|secret|token)\s*[=:]\s*\S+",  # 密码泄露
    r"\b\d{16,19}\b",  # 信用卡号
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
]

def sanitize_input(text: str) -> tuple[bool, str]:
    """返回 (是否安全, 原因)"""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Detected sensitive pattern"
    
    # 长度限制
    if len(text) > 10000:
        return False, "Input too long"
    
    return True, ""
```

---

## 7. 性能基准测试

### 7.1 使用 wrk/benchmark

```python
# benchmark.py
import asyncio
import aiohttp
import time
import statistics

async def send_request(session, url, payload):
    start = time.time()
    async with session.post(url, json=payload) as response:
        await response.json()
    return time.time() - start

async def benchmark(concurrency=10, total_requests=100):
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "meta-llama/Llama-2-7b-hf",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100
    }
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(total_requests):
            task = send_request(session, url, payload)
            tasks.append(task)
            if len(tasks) >= concurrency:
                await asyncio.gather(*tasks)
                tasks = []
        
        if tasks:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(benchmark(concurrency=50, total_requests=500))
```

---

## 8. 本章小结

**生产部署 checklist：**
- [ ] Docker 容器化
- [ ] Nginx/Kong 反向代理
- [ ] API 认证与限流
- [ ] Prometheus + Grafana 监控
- [ ] 健康检查与自动重启
- [ ] 日志收集与分析
- [ ] 输入过滤与安全加固
- [ ] 负载测试验证

**扩展方向：**
- 多机分布式部署（Ray）
- 模型热更新（不停机切换）
- A/B 测试（多模型灰度）
- 成本优化（Spot 实例、自动扩缩容）

---

*上一篇：[04-深入理解 PagedAttention](04-architecture.md)*
*下一篇：[06-性能优化与量化](06-optimization.md)*

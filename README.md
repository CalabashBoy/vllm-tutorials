# vLLM 完全入门教程

> 从零开始掌握高性能大模型推理框架

## 📚 教程目录

本教程系列专为初学者设计，从基础概念到生产部署，循序渐进掌握 vLLM。

| 章节 | 标题 | 难度 | 预计时间 | 内容概述 |
|-----|------|------|---------|---------|
| 01 | [什么是 vLLM](01-introduction.md) | ⭐⭐ | 15分钟 | 核心概念、PagedAttention 原理、与传统方案对比 |
| 02 | [安装与环境配置](02-installation.md) | ⭐⭐ | 30分钟 | 多种安装方式、环境验证、常见问题解决 |
| 03 | [快速上手](03-quickstart.md) | ⭐⭐ | 20分钟 | 启动服务、API 调用、流式输出、参数调优 |
| 04 | [深入理解 PagedAttention](04-architecture.md) | ⭐⭐⭐⭐ | 40分钟 | 源码解析、内存管理、性能优化原理 |
| 05 | [生产环境部署](05-deployment.md) | ⭐⭐⭐⭐ | 45分钟 | Docker、监控、负载均衡、高可用 |
| 06 | [性能优化与量化](06-optimization.md) | ⭐⭐⭐ | 35分钟 | AWQ/GPTQ 量化、批处理优化、多卡并行 |

## 🎯 学习路径建议

### 路径 A：快速上手（2小时）
适合想快速体验 vLLM 的同学：
1. 阅读第 1 章（了解是什么）
2. 跟随第 2 章安装
3. 完成第 3 章示例
4. 即可开始使用

### 路径 B：系统学习（1天）
适合想深入掌握的同学：
1. 完整阅读所有章节
2. 每章完成课后练习
3. 实际部署一个生产环境
4. 尝试优化自己的模型

### 路径 C：问题驱动
适合有具体需求的同学：
- **OOM/显存不够** → 第 6 章（量化）
- **响应太慢** → 第 4 章（PagedAttention）+ 第 6 章（优化）
- **要上线服务** → 第 5 章（生产部署）

## 📖 前置知识

- **Python 基础**：会写简单脚本
- **命令行基础**：会使用 cd、ls、pip 等命令
- **深度学习基础（可选）**：了解 Transformer、Attention 概念更好

## 🛠️ 环境要求

**最低配置**（本地学习）：
- GPU：NVIDIA GTX 1060 6GB+
- 内存：16GB+
- 系统：Linux/macOS/Windows (WSL2)

**推荐配置**（生产部署）：
- GPU：NVIDIA RTX 4090 / A100
- 内存：32GB+
- 显存：24GB+

## 🚀 快速开始

```bash
# 1. 克隆本教程
git clone https://github.com/CalabashBoy/vllm-tutorials.git
cd vllm-tutorials

# 2. 安装 vLLM
pip install vllm

# 3. 启动服务
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --port 8000

# 4. 测试
python 03-quickstart.py
```

## 📊 学习进度追踪

| 章节 | 阅读完成 | 代码实践 | 课后练习 |
|-----|---------|---------|---------|
| 01 | ⬜ | ⬜ | ⬜ |
| 02 | ⬜ | ⬜ | ⬜ |
| 03 | ⬜ | ⬜ | ⬜ |
| 04 | ⬜ | ⬜ | ⬜ |
| 05 | ⬜ | ⬜ | ⬜ |
| 06 | ⬜ | ⬜ | ⬜ |

## 💡 常见问题

**Q: 我是小白，能学会吗？**
A: 本教程专为初学者设计，只要会基础 Python，跟着做就能掌握。

**Q: 没有 GPU 能学吗？**
A: 部分章节可以 CPU 运行，但建议使用云服务器（AutoDL、阿里云等）。

**Q: 学完能做什么？**
A: 可以部署自己的 ChatGPT 服务、搭建 AI 应用后端、优化现有模型性能。

## 🔗 相关资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)

## 🤝 贡献

如果发现错误或想补充内容，欢迎提交 Issue 或 PR。

## 📄 License

本教程采用 [MIT License](LICENSE) 开源。

---

**作者**: CalabashBoy  
**创建时间**: 2026-02-15  
**最后更新**: 2026-02-15

祝你学习愉快！🎉

# Nexus

> 🤖 从零开始构建的多智能体框架 - 轻量级、原生、教学友好

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green.svg)](https://platform.openai.com/docs/api-reference)

Nexus，基于OpenAI原生API构建，提供了从简单对话到复杂推理的完整Agent范式实现。

## 🚀 快速开始

### 系统要求

- **Python 3.10+** （必需）
- 支持的操作系统：Windows、macOS、Linux

### 安装

#### 🔧 环境配置

创建 `.env` 文件：
```bash
# 模型名称
LLM_MODEL_ID=your-model-name

# API密钥
LLM_API_KEY=your-api-key-here

# 服务地址
LLM_BASE_URL=your-api-base-url
```


## 🤖 Agent范式详解

### 1. ReActAgent - 推理与行动结合

适用场景：需要外部信息、工具调用的任务

```python
from nexus import ReActAgent, ToolRegistry, search, calculate

# 创建工具注册表
tool_registry = ToolRegistry()
tool_registry.register_function("search", "网页搜索工具", search)
tool_registry.register_function("calculate", "数学计算工具", calculate)

# 创建ReAct Agent
react_agent = ReActAgent(
    name="研究助手",
    llm=llm,
    tool_registry=tool_registry,
    max_steps=5
)

# 执行需要工具的任务
result = react_agent.run("搜索最新的GPT-4发展情况，并计算其参数量相比GPT-3的增长倍数")
```

### 2. ReflectionAgent - 自我反思与迭代优化

适用场景：代码生成、文档写作等需要迭代优化的任务

```python
from nexus import ReflectionAgent

# 创建Reflection Agent
reflection_agent = ReflectionAgent(
    name="代码专家",
    llm=llm,
    max_iterations=3
)

# 生成并优化代码
code = reflection_agent.run("编写一个高效的素数筛选算法，要求时间复杂度尽可能低")
print(f"最终代码:\n{code}")
```

### 3. PlanAndSolveAgent - 分解规划与逐步执行

适用场景：复杂多步骤问题、数学应用题、逻辑推理

```python
from nexus import PlanAndSolveAgent

# 创建Plan and Solve Agent
plan_agent = PlanAndSolveAgent(name="问题解决专家", llm=llm)

# 解决复杂问题
problem = """
一家公司第一年营收100万，第二年增长20%，第三年增长15%。
如果每年的成本是营收的70%，请计算三年的总利润。
"""
answer = plan_agent.run(problem)
```

## 🛠️ 工具系统

Nexus提供了完整的工具生态系统：

### 内置工具

```python
from nexus import ToolRegistry, SearchTool, CalculatorTool

# 方式1：使用Tool对象（推荐）
registry = ToolRegistry()
registry.register_tool(SearchTool())
registry.register_tool(CalculatorTool())

# 方式2：直接注册函数（简便）
def my_tool(input_text: str) -> str:
    return f"处理结果: {input_text}"

registry.register_function("my_tool", "自定义工具描述", my_tool)
```

### 目前支持的工具

- **🔍 SearchTool**: 网页搜索（支持Tavily、SerpApi、模拟搜索）
- **🧮 CalculatorTool**: 数学计算（支持复杂表达式和数学函数）
- **🔧 自定义工具**: 支持任意Python函数注册为工具

## ⚙️ 配置详解

Nexus支持灵活的配置方式，**参数优先，环境变量兜底**：

### 🎯 统一配置格式（推荐）

编辑 `.env` 文件，配置你的API密钥。

只需配置4个环境变量，框架自动检测provider：

```env
LLM_MODEL_ID=your-model-id
LLM_API_KEY=ms-your_api_key_here
LLM_BASE_URL=your-api-base-url
LLM_TIMEOUT=60
```

### 支持的LLM提供商

| 提供商 | 自动检测 | 专用环境变量 | 统一配置示例 |
|--------|----------|-------------|-------------|
| **ModelScope** | ✅ | `MODELSCOPE_API_KEY` | `LLM_API_KEY=ms-xxx...` |
| **OpenAI** | ✅ | `OPENAI_API_KEY` | `LLM_API_KEY=sk-xxx...` |
| **DeepSeek** | ✅ | `DEEPSEEK_API_KEY` | `LLM_BASE_URL=api.deepseek.com` |
| **通义千问** | ✅ | `DASHSCOPE_API_KEY` | `LLM_BASE_URL=dashscope.aliyuncs.com` |
| **月之暗面 Kimi** | ✅ | `KIMI_API_KEY` | `LLM_BASE_URL=api.moonshot.cn` |
| **智谱AI GLM** | ✅ | `ZHIPU_API_KEY` | `LLM_BASE_URL=open.bigmodel.cn` |
| **Ollama** | ✅ | `OLLAMA_API_KEY` | `LLM_BASE_URL=localhost:11434` |
| **vLLM** | ✅ | `VLLM_API_KEY` | `LLM_BASE_URL=localhost:8000` |
| **其他本地部署** | ✅ | - | `LLM_BASE_URL=localhost:PORT` |


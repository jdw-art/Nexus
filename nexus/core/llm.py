"""Nexus统一LLM接口 - 基于OpenAI原生API"""

from typing import Literal

# 支持的LLM提供商
SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen",
    "modelscope",
    "kimi",
    "zhipu",
    "ollama",
    "vllm",
    "local",
    "auto",
    "custom",
]

class NexusLLM:
    """
    为Nexus定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。

    设计理念：
    - 参数优先，环境变量兜底
    - 流式响应为默认，提供更好的用户体验
    - 支持多种LLM提供商
    - 统一的调用接口
    """
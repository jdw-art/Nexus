"""LLM服务模块"""
from agent_projects.nexus_trip_planner.backend.app.config import get_settings
from nexus import NexusLLM

# 全局LLM实例
_llm_instance = None

def get_llm() -> NexusLLM:
    """
    获取LLM实例（单例模式）
    :return: NexusLLM实例
    """
    global _llm_instance

    if _llm_instance is None:
        settings = get_settings()

        # 初始化LLM实例
        _llm_instance = NexusLLM()

        print(f"✅ LLM服务初始化成功")
        print(f"   提供商: {_llm_instance.provider}")
        print(f"   模型: {_llm_instance.model}")

    return _llm_instance

def reset_llm():
    """重置LLM实例，用于测试或重新配置"""
    global _llm_instance
    _llm_instance = None
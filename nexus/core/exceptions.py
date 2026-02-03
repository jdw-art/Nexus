"""异常体系"""

class NexusException(Exception):
    """HelloAgents基础异常类"""
    pass

class LLMException(NexusException):
    """LLM相关异常"""
    pass

class AgentException(NexusException):
    """Agent相关异常"""
    pass

class ConfigException(NexusException):
    """配置相关异常"""
    pass

class ToolException(NexusException):
    """工具相关异常"""
    pass

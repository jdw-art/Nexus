"""
记忆系统基础类和配置

按照第8章架构设计的基础组件：
- MemoryItem: 记忆项数据结构
- MemoryConfig: 记忆系统配置
- BaseMemory: 记忆基类
"""
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List

from pydantic import BaseModel


class MemoryItem(BaseModel):
    """记忆项数据结构"""
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

class MemoryConfig(BaseModel):
    """记忆系统配置"""

    # 存储路径
    storage_path: str = "./memory_data"

    # 统计显示用的基础配置（仅用于展示）
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # 感知记忆特定配置
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]

class BaseMemory(ABC):
    """
    记忆基类

    定义所有记忆类型的通用接口和行为
    """
    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """
        添加记忆项
        :param memory_item: 记忆项对象
        :return: 记忆ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """
        检索相关记忆
        :param query: 查询内容
        :param limit: 返回数量限制
        :param kwargs: 其他检索参数
        :return: 相关记忆列表
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, content: str = None,
               importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        """
        更新记忆
        :param memory_id: 记忆ID
        :param content: 新内容
        :param importance: 新重要性
        :param metadata: 新元数据
        :return: 是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """
        删除记忆
        :param memory_id: 删除记忆ID
        :return: 是否删除成功
        """

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """
        检查记忆是否存在
        :param memory_id: 记忆ID
        :return: 是否存在
        """
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        :return: 统计信息字典
        """
        pass

    def _generate_id(self) -> str:
        """
        生成记忆ID
        :return: 记忆ID
        """
        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """
        计算记忆重要性
        :param content: 记忆内容
        :param base_importance: 基础重要性
        :return: 计算后的记忆重要性
        """
        importance = base_importance

        # 基于内容长度
        if len(content) > 100:
            importance += 0.1

        # 基于关键词
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()
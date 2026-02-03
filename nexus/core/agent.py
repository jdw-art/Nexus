"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Optional

from nexus.core import NexusLLM, Config, Message


class Agent(ABC):
    """
    Agent基类
    所有的Agent都必须继承自该基类，并重写对应的方法
    """

    def __init__(
            self,
            name: str,
            llm: NexusLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent，强制子类重写该方法"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        return self.__str__()
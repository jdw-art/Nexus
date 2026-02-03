"""消息系统"""
from datetime import datetime
from typing import Literal, Optional, Dict, Any

from pydantic import BaseModel

# 根据OpenAI规范，制定消息角色范畴
MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    """
    消息类
    定义了框架内统一的消息格式，规范管理会话历史
    确保了智能体与模型之间信息传递的标准化
    """

    content: str
    role: MessageRole
    timestamp: datetime = None
    # metadata为后续提供扩展
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
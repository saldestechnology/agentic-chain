from agentic.message.base_message import BaseMessage
from typing import Literal

class UserMessage(BaseMessage):
    role: Literal["user"] = "user"
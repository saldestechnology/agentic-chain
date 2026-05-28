from agentic.message.base_message import BaseMessage
from typing import Literal

class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"
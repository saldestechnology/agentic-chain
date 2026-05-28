from agentic.message.base_message import BaseMessage
from typing import Literal

class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
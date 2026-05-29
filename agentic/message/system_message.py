from agentic.message.base_message import BaseMessage, UserRole
from pydantic import Field


class SystemMessage(BaseMessage):
    role: UserRole = Field(default="system")


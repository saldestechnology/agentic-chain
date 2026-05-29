from pydantic import Field
from agentic.memory.base_memory import BaseMemory
from agentic.message.base_message import BaseMessage


class SessionMemory(BaseMemory):
    """
    A volatile in memory based chat message log
    """

    storage: list[BaseMessage] = Field(default_factory=list)

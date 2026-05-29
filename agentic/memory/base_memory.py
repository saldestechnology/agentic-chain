from pydantic import BaseModel, Field
from typing import Optional

from agentic.message.base_message import BaseMessage
from agentic.message.system_message import SystemMessage


class BaseMemory(BaseModel):
    _system_prompt: Optional[str] = None
    storage: list[BaseMessage] = Field(default_factory=list)

    def add_system_prompt(self, system_prompt: str) -> None:
        if not system_prompt:
            return

        self._system_prompt = system_prompt

        if len(self.storage) > 0:
            if self.storage[0].role == "system":
                return

        self.storage.insert(0, SystemMessage(content=system_prompt))

    def get_memory(self) -> list[BaseMessage]:
        return self.storage

    def add_memory(self, message: BaseMessage) -> None:
        self.storage.append(message)

    def clear(self) -> None:
        self.storage.clear()

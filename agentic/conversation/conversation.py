from pydantic import BaseModel, Field
from typing import Optional, TypedDict, TypeAlias, overload

from agentic.memory.base_memory import BaseMemory
from agentic.memory.session_memory import SessionMemory
from agentic.message.system_message import SystemMessage
from agentic.message.user_message import UserMessage
from agentic.message.assistant_message import AssistantMessage


class UserParam(TypedDict):
    user: str


class AssistantParam(TypedDict):
    assistant: str


MessageParam: TypeAlias = UserParam | AssistantParam


class Conversation(BaseModel):
    system_prompt: Optional[SystemMessage] = None
    memory: BaseMemory = Field(default_factory=lambda: SessionMemory())

    @overload
    def add_message(self, *, user: str) -> "Conversation": ...

    @overload
    def add_message(self, *, assistant: str) -> "Conversation": ...

    def add_message(self, **message: str) -> "Conversation":
        """Append a message to the conversation ledger"""
        match message:
            case {"user": str() as content}:
                self.memory.add_memory(UserMessage(content=content))
            case {"assistant": str() as content}:
                self.memory.add_memory(AssistantMessage(content=content))
            case _:
                raise TypeError(
                    "Must be called with either user or assistant parameter"
                )
        return self

    def update_system_prompt(self, content: str) -> "Conversation":
        """Update the system message without shifting ledger history indexes"""
        if not content:
            self.system_prompt = None
        else:
            self.system_prompt = SystemMessage(content=content)
        return self

    def clear_history(self) -> None:
        """Clear conversation ledger"""
        self.memory.clear()

    def truncate_history(self, keep: int) -> None:
        """Safely trunace conversation ledger without changing the system message"""
        current_mem = self.memory.get_memory()
        self.memory.storage = current_mem[-keep:]

    def summarize_history(self) -> None:
        raise NotImplementedError("summarize_history is not implemented!")

    def compile(self) -> list[dict[str, str]]:
        """Seralize the entire conversation ledger into raw API primitives"""
        payload = [self.system_prompt.dump()] if self.system_prompt else []
        payload.extend([message.dump() for message in self.memory.storage])
        return payload

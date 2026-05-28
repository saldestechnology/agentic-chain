from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Optional
from abc import ABC, abstractmethod

from agentic.message.base_message import BaseMessage
from agentic.message.system_message import SystemMessage
import json

S = TypeVar("S", bound=BaseMessage)

class BaseMemory(BaseModel, ABC, Generic[S]):
    _system_prompt: Optional[str] = None
    storage: list[S] = Field(default_factory=list)
    
    def add_system_prompt(self, system_prompt: str) -> None:
        if not system_prompt:
            return
        
        self._system_prompt = system_prompt
        
        if len(self.storage) > 0:
            first_msg = self.storage[0]
            first_role = first_msg.role if hasattr(first_msg, "role") else first_msg.get("role")
            if first_role == "system":
                return
            
        self.storage.insert(0, SystemMessage(content=system_prompt))
        
    def model_dump(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for message in self.storage:
            result.append(message.model_dump())
        return result
    
    @abstractmethod
    def get_memory(self) -> list[S]:
        return self.storage
    
    @abstractmethod
    def add_memory(self, message: S) -> None:
        self.storage.append(message)
    
    @abstractmethod
    def clear(self) -> None:
        self.storage.clear()
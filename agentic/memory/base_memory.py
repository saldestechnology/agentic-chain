from agentic.message.base_message import BaseMessage
from agentic.message.system_message import SystemMessage
from pydantic import BaseModel
from typing import Generic, TypeVar, Optional
from abc import ABC, abstractmethod
import json

M = TypeVar("M", bound=BaseMessage)

class BaseMemory(BaseModel, ABC, Generic[M]):
    _system_prompt: Optional[str] = None
    memory: list[M]
    
    def add_system_prompt(self, system_prompt: str) -> None:
        if not system_prompt:
            return
        
        self._system_prompt = system_prompt
        
        if len(self.memory) > 0:
            first_msg = self.memory[0]
            first_role = first_msg.role if hasattr(first_msg, "role") else first_msg.get("role")
            if first_role == "system":
                return
            
        self.memory.insert(0, SystemMessage(content=system_prompt))
        
    def model_dump(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for message in self.memory:
            result.append(message.model_dump())
        return result
    
    @abstractmethod
    def get_memory(self) -> list[M]:
        return self.memory
    
    @abstractmethod
    def add_memory(self, message: M) -> None:
        self.memory.append(message)
    
    @abstractmethod
    def clear(self) -> None:
        self.memory.clear()
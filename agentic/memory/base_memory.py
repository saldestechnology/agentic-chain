from pydantic import BaseModel
from abc import ABC, abstractmethod

class BaseMemory(BaseModel, ABC):
    memory: list[dict[str, str]] = []
    
    @abstractmethod
    def get_memory(self) -> list[dict[str, str]]:
        return self.memory
    
    @abstractmethod
    def add_memory(self, message: dict[str, str]) -> None:
        self.memory.append(message)
    
    @abstractmethod
    def clear(self) -> None:
        self.memory.clear()
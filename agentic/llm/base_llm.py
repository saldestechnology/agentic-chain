from agentic.runnable.runnable import Runnable
from agentic.memory.base_memory import BaseMemory
from agentic.memory.memory import remember
from typing import Generic, TypeVar, Optional
from abc import ABC, abstractmethod

M = TypeVar("M") # LLM Model

class BaseLLM(Runnable[str, str], ABC, Generic[M]):
    name: str
    model_name: str
    # Optional: System prompt
    system_prompt: Optional[str] = ""
    # Optional: chat history memory
    memory: Optional[BaseMemory]
    
    _client: Optional[M]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Automatically wrap 'invoke' if the subclass defined it
        if "invoke" in cls.__dict__:
            cls.invoke = remember(cls.invoke)
    
    @abstractmethod
    def invoke(self, prompt: str, stream: bool = False) -> str:
        pass
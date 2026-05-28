from agentic.runnable.runnable import Runnable
from agentic.memory.base_memory import BaseMemory
from typing import Generic, TypeVar

M = TypeVar("M") # LLM Model

class BaseLLM(Runnable[str, str], Generic[M]):
    name: str
    model_name: str
    # Optional: System prompt
    system_prompt: str | None = None
    # Optional: chat history memory
    memory: BaseMemory | None = None
    
    _client: M | None = None
    
    def invoke(self, prompt: str, stream: bool = False) -> str:
        return NotImplementedError("LLM providers must implement their own invoke function")
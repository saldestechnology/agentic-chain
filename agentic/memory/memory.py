from agentic.memory.base_memory import BaseMemory
from agentic.message.user_message import UserMessage
from agentic.message.assistant_message import AssistantMessage
from agentic.utils.logging import log
from typing import Callable 
from functools import wraps

def remember(func: Callable[..., str]):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        content: str = args[0] if args else kwargs.get("prompt")
        memory: BaseMemory | None = getattr(self, "memory", None)
        if memory:
            memory.add_memory(UserMessage(content=content))
        content_response = func(self, *args, **kwargs)
        log(memory.model_dump(), log_level="DEBUG")
        if memory:
            memory.add_memory(AssistantMessage(content=content_response))
        return content_response
    return wrapper
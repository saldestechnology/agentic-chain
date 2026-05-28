from agentic.conversation.conversation import Conversation
from agentic.message.user_message import UserMessage
from agentic.message.assistant_message import AssistantMessage
from agentic.utils.logging import log
from typing import Callable 
from functools import wraps

def remember(func: Callable[..., str]):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        content: str = args[0] if args else kwargs.get("prompt")
        conversation: Conversation | None = getattr(self, "conversation", None)
        if not conversation:
            return func(self, *args, **kwargs)
        conversation.add_message(user=content)
        content_response = func(self, *args, **kwargs)
        conversation.add_message(assistant=content_response)
        log(conversation.compile(), log_level="DEBUG")
        return content_response
    return wrapper
from agentic.conversation.conversation import Conversation
from agentic.utils.logging import log
from typing import Callable
from functools import wraps


def remember(func: Callable[..., str]):
    @wraps(func)
    def wrapper(self, data: str):
        conversation: Conversation | None = getattr(self, "conversation", None)
        if not conversation:
            return func(self, data)
        conversation.add_message(user=data)
        content_response = func(self, data)
        conversation.add_message(assistant=data)
        log(conversation.compile(), log_level="DEBUG")
        return content_response

    return wrapper

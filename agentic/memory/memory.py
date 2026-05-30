from agentic.conversation.conversation import Conversation
from agentic.utils.logging import log
from typing import Callable, TypeVar
from functools import wraps

T = TypeVar("T")


def remember[T](func: Callable[..., str]) -> Callable[..., str]:
    @wraps(func)
    def wrapper(self: T, data: str) -> str:
        conversation: Conversation | None = getattr(self, "conversation", None)
        if not conversation:
            return func(self, data)
        conversation.add_message(user=data)
        content_response = func(self, data)
        conversation.add_message(assistant=content_response)
        log(conversation.compile(), log_level="DEBUG")
        return content_response

    return wrapper

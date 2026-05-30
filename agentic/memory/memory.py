from functools import wraps
from typing import Callable

from agentic.utils.logging import log


def remember[T](func: Callable[[T, str], str]) -> Callable[[T, str], str]:
    @wraps(func)
    def wrapper(self: T, data: str) -> str:
        if not (conversation := getattr(self, "conversation", None)):
            return func(self, data)
        conversation.add_message(user=data)
        response = func(self, data)
        conversation.add_message(assistant=response)
        log(conversation.compile(), log_level="DEBUG")
        return response

    return wrapper

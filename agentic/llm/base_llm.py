from typing import Any, Generic, TypeVar, Optional
from abc import ABC

from pydantic import PrivateAttr

from agentic.runnable.runnable import Runnable
from agentic.conversation.conversation import Conversation
from agentic.memory.memory import remember

M = TypeVar("M")


class BaseLLM(Runnable[str, str], ABC, Generic[M]):
    model_name: str
    # Optional: System prompt
    system_prompt: str = ""
    # Optional: chat history memory
    conversation: Optional[Conversation] = None

    _client: M = PrivateAttr()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Meta-programming so developers don't have add @remember decorator to the
        invoke method. Instead adding Conversation to a BaseLLM object just works.
        """
        super().__init_subclass__(**kwargs)

        if "invoke" in cls.__dict__:
            # NOTE: This dynamic method reassignment is a short-term pattern.
            # We should look into a more sustainable solution in the future,
            # either by refactoring BaseLLM to use a concrete invoke() + abstract _invoke(),
            # or by finding a decorator pattern that mypy can verify statically.
            cls.invoke = remember(cls.invoke)  # type: ignore[assignment, method-assign]

    def get_conversation(self, data: str) -> Conversation:
        """If conversation is initialised it gets returned otherwise one shot"""
        if not (payload := self.conversation):
            payload = (
                Conversation()
                .update_system_prompt(self.system_prompt)
                .add_message(user=data)
            )
        return payload

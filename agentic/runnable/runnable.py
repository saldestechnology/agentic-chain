from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from typing import Any, Callable, Generic, TypeVar
from abc import ABC, abstractmethod

Input = TypeVar("Input")
Output = TypeVar("Output")
Middle = TypeVar("Middle")


class Runnable(BaseModel, ABC, Generic[Input, Output]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="")

    @abstractmethod
    def invoke(self, data: Input) -> Output:
        """Invoke the runnable class or callable"""
        pass

    def __or__(self, other: Any) -> "RunnableSequence[Any, Any, Any]":
        if isinstance(other, Runnable):
            return RunnableSequence.model_construct(
                first=self,
                second=other,
            )
        if callable(other):
            return RunnableSequence.model_construct(
                first=self,
                second=RunnableLambda.model_construct(func=other, name=other.__name__),
                name=other.__name__,
            )
        return NotImplemented

    def __ror__(self, other: Any) -> "RunnableSequence[Any, Any, Any]":
        if callable(other):
            return RunnableSequence.model_construct(
                first=RunnableLambda.model_construct(func=other),
                second=self,
                name=other.__name__,
            )
        return NotImplemented


class RunnableLambda(Runnable[Input, Output]):
    func: Callable[[Input], Output]

    def invoke(self, data: Input) -> Output:
        return self.func(data)


class RunnableSequence(Runnable[Input, Output], Generic[Input, Middle, Output]):
    first: SerializeAsAny[Runnable[Input, Middle]]
    second: SerializeAsAny[Runnable[Middle, Output]]

    def invoke(self, data: Input) -> Output:
        return self.second.invoke(self.first.invoke(data))

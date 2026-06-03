from abc import ABC, abstractmethod
from agentic.runnable.runnable import Runnable
from typing import Type, Any
from pydantic import BaseModel, Field


class BaseTool(Runnable[dict[str, Any], str], ABC):
    name: str = Field(default="")
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def func(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool with validated arguments"""

    def invoke(self, data: dict[str, Any]) -> str:
        validate_args = self.args_schema(**data)
        return self.func(**validate_args.model_dump())

    def tool_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }

from abc import ABC, abstractmethod
from agentic.runnable.runnable import Runnable
from typing import Type, Any
from pydantic import BaseModel, Field
import time


class BaseTool(Runnable[dict[str, Any], str], ABC):
    name: str = Field(default="")
    description: str
    args_schema: Type[BaseModel]
    max_retries: int = Field(default=3, ge=0,
                              description="Number of internal retries on failure")

    @abstractmethod
    def func(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool with validated arguments"""

    def invoke(self, data: dict[str, Any]) -> str:
        validate_args = self.args_schema(**data)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.func(**validate_args.model_dump())
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.0)
        return f"Tool '{self.name}' failed after {self.max_retries + 1} attempts: {last_error}"

    def tool_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }
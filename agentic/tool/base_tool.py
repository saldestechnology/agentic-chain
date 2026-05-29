from agentic.runnable.runnable import Runnable
from typing import Type, Callable, Any
from pydantic import BaseModel, Field


class BaseTool(Runnable):
    name: str = Field(default="")
    description: str
    args_schema: Type[BaseModel]
    func: Callable[..., Any]

    def invoke(self, data: dict) -> Any:
        validate_args = self.args_schema(**data)
        return self.func(**validate_args.model_dump())

    def tool_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


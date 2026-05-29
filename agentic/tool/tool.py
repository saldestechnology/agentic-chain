from agentic.tool.base_tool import BaseTool
from pydantic import BaseModel
from typing import Callable, Any, Type


def tool(name: str, description: str, args_schema: Type[BaseModel]):
    def decorator(func: Callable[..., Any]) -> BaseTool:
        return BaseTool.model_construct(
            name=name, description=description, args_schema=args_schema, func=func
        )

    return decorator


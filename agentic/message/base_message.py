from pydantic import BaseModel
from typing import Literal

class BaseMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
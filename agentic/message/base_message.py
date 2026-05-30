from pydantic import BaseModel
from typing import Literal

UserRole = Literal["user", "assistant", "system"]


class BaseMessage(BaseModel):
    role: UserRole
    content: str

    def dump(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

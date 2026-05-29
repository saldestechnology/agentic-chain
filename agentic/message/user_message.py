from agentic.message.base_message import BaseMessage, UserRole


class UserMessage(BaseMessage):
    role: UserRole = "user"


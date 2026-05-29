from agentic.message.base_message import BaseMessage, UserRole


class AssistantMessage(BaseMessage):
    role: UserRole = "assistant"


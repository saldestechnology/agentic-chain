from agentic.memory.base_memory import BaseMemory
from agentic.message.base_message import BaseMessage

class SessionMemory(BaseMemory):
    """
    A volatile in memory based chat message log
    """
    
    memory: list[BaseMessage] = []
    
    def get_memory(self) -> list[BaseMessage]:
        return self.memory
    
    def add_memory(self, message: BaseMessage):
        self.memory.append(message)
        
    def clear(self):
        self.memory.clear()
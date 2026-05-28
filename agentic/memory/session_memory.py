from agentic.memory.base_memory import BaseMemory

class SessionMemory(BaseMemory):
    """
    A volatile in memory based chat message log
    """
    
    memory: list[dict[str, str]] = []
    
    def get_memory(self) -> list[dict[str, str]]:
        return self.memory
    
    def add_memory(self, message: dict[str, str]):
        self.memory.append(message)
        
    def clear(self):
        self.memory.clear()
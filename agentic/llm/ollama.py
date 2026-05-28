from ollama import Client
from agentic.llm.base_llm import BaseLLM
from agentic.memory.base_memory import BaseMemory
from agentic.utils.logging import log

class Ollama(BaseLLM[Client]):
    name: str = "ollama"
    model_name: str = "gemma4:31b-cloud"

    def model_post_init(self, __context):
        log("Loading ollama into memory (this may take a while)...")
        self._client = Client()
        if self.memory:
            self.memory.add_system_prompt(self.system_prompt)
        log("Model loaded successfully.")

    def invoke(self, prompt: str) -> str:
        log(f"Called with {prompt}")
        if not self.memory:
            messages= [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{ "role": msg.role, "content": msg.content } for msg in self.memory.get_memory()]
            self.memory.add_system_prompt(self.system_prompt)

        return self._client.chat(
            model=self.model_name,
            messages=messages,
        ).message.content.strip()
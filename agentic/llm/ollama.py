from typing import Any
from ollama import ChatResponse, Client

from agentic.llm.base_llm import BaseLLM
from agentic.utils.logging import log


class Ollama(BaseLLM[Client]):
    name: str = "ollama"
    model_name: str = "gemma4:31b-cloud"

    def model_post_init(self, __context: Any) -> None:
        log(f"Loading {self.model_name} into memory (this may take a while)...")
        self._client = Client()
        if self.conversation:
            self.conversation.update_system_prompt(self.system_prompt)
        log("Model loaded successfully.")

    def invoke(self, data: str) -> str:
        log(data)
        payload = self.get_conversation(data)
        response: ChatResponse = self._client.chat(
            model=self.model_name,
            messages=payload.compile(),
        )
        return str(response.message.content).strip()

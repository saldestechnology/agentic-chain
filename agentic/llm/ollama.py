from ollama import Client

from agentic.llm.base_llm import BaseLLM
from agentic.conversation.conversation import Conversation
from agentic.utils.logging import log

class Ollama(BaseLLM[Client]):
    name: str = "ollama"
    model_name: str = "gemma4:31b-cloud"

    def model_post_init(self, __context):
        log("Loading ollama into memory (this may take a while)...")
        self._client = Client()
        if self.conversation:
            self.conversation.update_system_prompt(self.system_prompt)
        log("Model loaded successfully.")

    def invoke(self, prompt: str) -> str:
        log(f"Called with {prompt}")
        if not self.conversation:
            payload = (
                Conversation()
                .update_system_prompt(self.system_prompt)
                .add_message(user=prompt)
            )
        else:
            payload = self.conversation

        return self._client.chat(
            model=self.model_name,
            messages=payload.compile(),
        ).message.content.strip()
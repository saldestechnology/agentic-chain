from ollama import ChatResponse, Client

from agentic.llm.base_llm import BaseLLM
from agentic.conversation.conversation import Conversation
from agentic.utils.logging import log


class Ollama(BaseLLM[Client]):
    name: str = "ollama"
    model_name: str = "gemma4:31b-cloud"

    def model_post_init(self, __context):
        self._client = Client()
        if self.conversation:
            self.conversation.update_system_prompt(str(self.system_prompt))

    def invoke(self, data: str) -> str:
        log(f"Called with {data}")
        if not isinstance(self._client, Client):
            return ""
        if not self.conversation:
            payload = (
                Conversation()
                .update_system_prompt(str(self.system_prompt))
                .add_message(user=data)
            )
        else:
            payload = self.conversation

        response: ChatResponse = self._client.chat(
            model=self.model_name,
            messages=payload.compile(),
        )
        return str(response.message.content).strip()

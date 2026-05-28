from ollama import Client
from agentic.llm.base_llm import BaseLLM
from agentic.utils.logging import log

class Ollama(BaseLLM[Client]):
    name: str = "ollama"
    model_name: str = "gemma4:31b-cloud"
    system_prompt: str = ""

    def model_post_init(self, __context):
        log("Loading ollama into memory (this may take a while)...")
        self._client = Client()
        log("Model loaded successfully.")

    def invoke(self, prompt: str, stream=False) -> str:
        log(f"Called with {prompt}")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        output = self._client.chat(
            model=self.model_name,
            messages=messages,
            stream=stream
        )
        return output.message.content.strip()
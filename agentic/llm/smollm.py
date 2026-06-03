from typing import Any
from transformers import pipeline, logging, TextGenerationPipeline
from agentic.llm.base_llm import BaseLLM
from agentic.utils.logging import log


class SmolLM(BaseLLM[TextGenerationPipeline]):
    name: str = "smol_llm"
    model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    def model_post_init(self, __context: Any) -> None:
        log(f"Loading {self.model_name} into memory (this may take a while)...")
        logging.set_verbosity_error()  # type: ignore[no-untyped-call]  # Only show errors from transformers
        self._client = pipeline(
            "text-generation",
            model=self.model_name,
        )
        if self.conversation:
            self.conversation.update_system_prompt(self.system_prompt)
        log("Model loaded successfully.")

    def invoke(self, data: str) -> str:
        log(data)
        payload = self.get_conversation(data)
        response = self._client(payload.compile())
        return str(response[0]["generated_text"][-1]["content"]).strip()

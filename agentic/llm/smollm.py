from transformers import pipeline, logging, TextGenerationPipeline
from agentic.llm.base_llm import BaseLLM
from agentic.utils.logging import log


class SmolLM(BaseLLM[TextGenerationPipeline]):
    name: str = "smol_llm"
    model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    def model_post_init(self, __context):
        log(f"Loading {self.model_name} into memory (this may take a while)...")
        logging.set_verbosity_error()  # Only show errors from transformers
        self._client = pipeline(
            "text-generation",
            model=self.model_name,
        )
        log("Model loaded successfully.")

    def invoke(self, data: str) -> str:
        log(f"Called with {data}")
        output = self._client([{"role": "user", "content": data}])
        return output[0]["generated_text"][-1]["content"].strip()


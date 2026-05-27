from transformers import pipeline
from runnable.runnable import Runnable

class SmolLM(Runnable):
    name: str = "smol_llm"
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    # Private propety
    _pipe = None
    
    def model_post_init(self, __context):
        print(f"Loading {self.model_name} into memory (this may take a while)...")
        self._pipe = pipeline("text-generation", model=self.model_name)
        print(f"Model loaded successfully.")
        
    def invoke(self, prompt: str) -> str:
        output = self._pipe([{"role": "user", "content": prompt}])
        return output[0]['generated_text'][-1]['content'].strip()
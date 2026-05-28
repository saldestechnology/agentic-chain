from agentic.runnable.runnable import Runnable
from agentic.template.prompt_template import PromptTemplate
from agentic.memory.base_memory import BaseMemory

class BaseAgent(Runnable):
    # The LLM used on this agent
    llm: Runnable[str, str]
    # Prompt template
    prompt: PromptTemplate
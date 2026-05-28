from agentic.runnable.runnable import Runnable
from agentic.template.prompt_template import PromptTemplate
from agentic.llm.base_llm import BaseLLM

class BaseAgent(Runnable):
    # The LLM used on this agent
    llm: BaseLLM[str]
    # Prompt template
    prompt: PromptTemplate
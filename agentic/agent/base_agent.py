from typing import Any

from agentic.runnable.runnable import Runnable
from agentic.template.prompt_template import PromptTemplate
from agentic.llm.base_llm import BaseLLM


class BaseAgent(Runnable[Any, Any]):
    # The LLM used on this agent
    llm: BaseLLM  # type: ignore[type-arg]
    # Prompt template
    prompt: PromptTemplate


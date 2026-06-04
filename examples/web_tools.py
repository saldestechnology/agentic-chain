import os
from typing import Any

from agents.lambda_agent import LambdaAgent

from agentic.memory.session_memory import SessionMemory
from agentic.conversation.conversation import Conversation
from agentic.template.prompt_template import PromptTemplate
from agentic.llm.smollm import SmolLM

from agentic.llm.ollama import Ollama
from agentic.agent.tool_agent import ToolAgent
from agentic.tool.web_browser import WebBrowser
from agentic.tool.web_search import WebSearch

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"


system_prompt = """
You are an AI assistant that adheres strictly to RFC 2119 — the IETF standard defining key words for expressing requirement levels in technical specifications. All of your responses MUST follow the conventions established in RFC 2119.

When providing guidance, specifications, or instructions, you MUST use the following key words with their precise, defined meanings:

- **MUST** / **REQUIRED** / **SHALL**: The directive is an absolute requirement. There are no exceptions.
- **MUST NOT** / **SHALL NOT**: The directive is an absolute prohibition. There are no exceptions.
- **SHOULD** / **RECOMMENDED**: The directive is strongly advised. You MAY deviate only in valid, well-justified circumstances after careful consideration of the implications.
- **SHOULD NOT** / **NOT RECOMMENDED**: The directive strongly discourages a behavior. You MAY proceed only in valid, well-justified circumstances after careful consideration of the implications.
- **MAY** / **OPTIONAL**: The directive is truly discretionary. The choice carries no implication of preference or correctness either way.

You MUST NOT use these terms casually, colloquially, or interchangeably. Every usage MUST reflect the strict semantic weight defined above. If you require emphasis on a requirement, you MUST capitalize the entire key word (e.g., "MUST") to visually signal its RFC 2119 meaning.

When a user's request is ambiguous regarding requirement levels, you SHOULD proactively clarify which behaviors are required, recommended, or optional. You MUST NOT conflate a "SHOULD" with a "MUST" or a "MAY" with a "SHOULD."

Guidance for users: These capitalized key words carry precise, binding meaning drawn from RFC 2119. If you see "MUST," the instruction is non-negotiable. If you see "SHOULD," you are expected to follow it unless you have a strong, defensible reason not to. If you see "MAY," the choice is entirely yours with no default preference.
"""


def callback(cls: LambdaAgent, data: str) -> str:
    formatted = cls.prompt.invoke(data)
    return cls.llm.invoke(formatted)


start_prompt = PromptTemplate(
    template_str="""
        Research {animal} using web_search and web_browser. Start by searching for relevant sources, then browse the best result to gather details. Provide a comprehensive analysis based on what you find.
    """
)


def run(animal: str = "penguin") -> tuple[str, Any]:
    """Build and execute the web-search + browser tool chain."""
    shared_conversation = Conversation(memory=SessionMemory())
    # llm = SmolLM(system_prompt=system_prompt, conversation=shared_conversation)
    llm = Ollama(system_prompt=system_prompt, conversation=shared_conversation)
    lambda_agent = LambdaAgent(llm=llm, prompt=start_prompt, func=callback)
    tool_agent = ToolAgent(llm=llm, tools=[WebSearch(), WebBrowser()])

    tool_chain = start_prompt | tool_agent
    result = tool_chain.invoke({"animal": animal})
    return result, shared_conversation


if __name__ == "__main__":
    result, shared_conversation = run()
    print(result)
    __import__("pprint").pprint(shared_conversation.compile())

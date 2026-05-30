import os
from typing_extensions import TypedDict
import json

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

from agentic.memory.session_memory import SessionMemory
from agentic.conversation.conversation import Conversation
from agentic.template.prompt_template import PromptTemplate
from agentic.parsers.parser import json_to_dict
from agentic.llm.ollama import Ollama
from agentic.utils.logging import log
from agents.critique_agent import CritiqueAgent
from agents.creative_agent import CreativeAgent
from agents.prompt_optimizer_agent import PromptOptimizerAgent


class Haiku(TypedDict):
    title: str
    lines: list[str]


def clean_output(text: Haiku | str) -> str:
    log({"content": text, "type": type(text).__name__})

    if isinstance(text, dict):
        haiku: Haiku = text
    else:
        try:
            haiku = json.loads(text)
        except json.JSONDecodeError:
            haiku = {"title": "Generated Poem", "lines": text.split("\n")}

    log(haiku)

    title = haiku.get("title", "Untitled")
    lines_list = haiku.get("lines", [])

    if isinstance(lines_list, str):
        lines_str = lines_list
    else:
        lines_str = "\n".join(lines_list)

    return f"\n=== THE HAIKU: {title} ===\n{lines_str}\n"


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

author_prompt = PromptTemplate(
    template_str="""
    Write a haiku poem on {subject}.
    A haiku is a short, unrhymed Japanese poetic form that traditionally captures a fleeting moment in nature. 
    It is composed of exactly 3 lines and a 5-7-5 syllable. 
    You MUST answer in the exact JSON format and structure as the examples.
    You MUST NOT add or remove any fields.
    YOU MUST NOT write anything but JSON, no commentary, no notes.
    
    Example 1: {{"title": "Green Summer", "lines": ["Green leaves catch the light",
               "Soft breeze whispers through the trees",
               "Summer comes to life."]}}
    Example 2: {{"title": "Autum dance", "lines": ["Cold wind shakes the branch",
               "Autumn leaves dance to the ground",
               "Winter starts to wake."]}}
    Output: """
)

shared_conversation = Conversation(memory=SessionMemory())
llm = Ollama(system_prompt=system_prompt, conversation=shared_conversation)
critique_agent = CritiqueAgent(llm=llm)
creative_agent = CreativeAgent(llm=llm)
prompt_optimizer = PromptOptimizerAgent(llm=llm)

# PromptTemplate -> LLM -> output func
haiku_chain = (
    author_prompt  # dict -> str
    | prompt_optimizer  # str - str
    | creative_agent  # str -> str (LLM JSON string)
    | json_to_dict  # str -> dict (Parsed object!)
    | critique_agent  # dict -> str (LLM JSON string)
    | json_to_dict  # str -> dict
    | clean_output  # dict -> str (Final formatted printout)
)

result = haiku_chain.invoke({"subject": "software engineering"})

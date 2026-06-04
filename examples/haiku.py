import os
import json
from typing import Any
from typing_extensions import TypedDict

from agents.lambda_agent import LambdaAgent
from agents.critique_agent import CritiqueAgent
from agents.creative_agent import CreativeAgent
from agents.prompt_optimizer_agent import PromptOptimizerAgent

from agentic.memory.session_memory import SessionMemory
from agentic.conversation.conversation import Conversation
from agentic.template.prompt_template import PromptTemplate
from agentic.parsers.parser import json_to_dict
from agentic.llm.smollm import SmolLM

from agentic.llm.ollama import Ollama
from agentic.utils.logging import log

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"


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
You are an AI assistant that adheres strictly to RFC 2119.
All of your responses MUST follow the conventions established in RFC 2119.
"""

start_prompt = PromptTemplate(
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


def callback(cls: LambdaAgent, data: str) -> str:
    formatted = cls.prompt.invoke(data)
    return cls.llm.invoke(formatted)


def run(subject: str = "software engineering") -> tuple[str, Any]:
    shared_conversation = Conversation(memory=SessionMemory())
    # llm = SmolLM(system_prompt=system_prompt, conversation=shared_conversation)
    llm = Ollama(system_prompt=system_prompt, conversation=shared_conversation)
    lambda_agent = LambdaAgent(llm=llm, prompt=start_prompt, func=callback)

    creative_agent = CreativeAgent(llm=llm)
    prompt_optimizer = PromptOptimizerAgent(llm=llm)
    critique_agent = CritiqueAgent(llm=llm)

    haiku_chain = (
        start_prompt
        | prompt_optimizer
        | creative_agent
        | json_to_dict
        | critique_agent
        | json_to_dict
        | clean_output
    )

    result = haiku_chain.invoke({"subject": subject})
    return result, shared_conversation


if __name__ == "__main__":
    result, shared_conversation = run()
    print(result)
    __import__("pprint").pprint(shared_conversation.compile())

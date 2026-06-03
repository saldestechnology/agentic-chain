from typing import Any

from agentic.agent.base_agent import BaseAgent
from agentic.template.prompt_template import PromptTemplate
from agentic.utils.logging import log


class CritiqueAgent(BaseAgent):
    name: str = "critique_agent"
    prompt: PromptTemplate = PromptTemplate(
        template_str="""
        Critique and improve this haiku:
        Title: {title}
        Lines:
        {lines}
        
        A haiku is a short, unrhymed Japanese poetic form that traditionally captures a fleeting moment in nature. 
        It is composed of exactly 3 lines and a 5-7-5 syllable. 
        
        You MUST answer in the exact JSON format and structure as the examples.
        You MUST NOT add or remove any fields.
        YOU MUST NOT write anything but JSON, no commentary, no notes.

        You MUST answer in the exact JSON format and structure as the examples.
        You MUST NOT add or remove any fields.
        YOU MUST NOT write anything but JSON, no commentary, no notes.
        You MUST NOT wrap the output in markdown code blocks.

        Output: """
    )

    def invoke(self, data: dict[str, Any]) -> str:
        log(f'Processing structured data: "{data}"')
        prompt_string = self.prompt.invoke(
            {
                "title": data.get("title", "Untitled"),
                "lines": "\n".join(data.get("lines", [])),
            }
        )
        return self.llm.invoke(prompt_string)


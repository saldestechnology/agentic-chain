from agentic.agent.base_agent import BaseAgent
from agentic.template.prompt_template import PromptTemplate
from agentic.utils.logging import log

class CreativeAgent(BaseAgent):
    name: str = "creative_agent"
    prompt: PromptTemplate = PromptTemplate(
        template_str="""
        You're a creative writer specialising in haiku. This is your assigment: {assignment}
        """
    )

    def invoke(self, data: str) -> str:
        creative_prompt = self.prompt.invoke({"assignment": data})
        log(f"{creative_prompt}")
        return self.llm.invoke(creative_prompt)
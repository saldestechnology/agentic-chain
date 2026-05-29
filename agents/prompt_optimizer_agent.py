from agentic.agent.base_agent import BaseAgent
from agentic.template.prompt_template import PromptTemplate
from agentic.utils.logging import log


class PromptOptimizerAgent(BaseAgent):
    name: str = "critique_agent"
    prompt: PromptTemplate = PromptTemplate(
        template_str="""
       <original-prompt>
       {prompt}
       </original-prompt> 
        based on this prompt, write an optimized prompt. You MUST just output the optimized prompt. The JSON MUST NOT be wrapped in a markdown code block.

        OUTPUT: """
    )

    def invoke(self, data: str) -> str:
        log(f'Optimizing prompt: "{data}"')
        prompt_string = self.prompt.invoke({"prompt": data})
        return self.llm.invoke(prompt_string)


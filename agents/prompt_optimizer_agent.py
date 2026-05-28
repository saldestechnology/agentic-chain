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

    def invoke(self, prompt: str) -> str:
        log(f"Optimizing prompt: \"{prompt}\"")
        prompt_string = self.prompt.invoke({
            "prompt": prompt
        })
        return self.llm.invoke(prompt_string)
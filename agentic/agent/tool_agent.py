import json
from pydantic import BaseModel

from agentic.agent.base_agent import BaseAgent
from agentic.llm.base_llm import BaseLLM
from agentic.template.prompt_template import PromptTemplate
from agentic.tool.base_tool import BaseTool
from agentic.tool.state_machine import AgentCtx, AgentState, AgentEvent, agent_state


class ToolAgent(BaseAgent, BaseModel):
    llm: BaseLLM  # type: ignore[type-arg]
    tools: list[BaseTool]
    max_iterations: int = 5
    prompt: PromptTemplate = PromptTemplate(template_str="")

    class Config:
        arbitrary_types_allowed = True

    def _get_tools_system_prompt(self) -> str:
        specs = [tool.tool_spec() for tool in self.tools]
        return PromptTemplate(
            template_str="""
            You are an analytical assistant with tools:\n{specs}\n\n
            To trigger a tool choice, output a structural JSON using this schema format:\n
            {{"type": "tool_call", "tool_name": "NAME", "arguments": {{...}}}}\n
            You MUST use a tool to answer questions. Do not output conversational text.
            """
        ).invoke({"specs": json.dumps(specs, indent=2)})

    def invoke(self, data: str) -> str:
        tool_prompt = self._get_tools_system_prompt()
        if self.llm.conversation:
            self.llm.conversation.update_system_prompt(tool_prompt)
        else:
            self.llm.system_prompt = tool_prompt

        ctx = AgentCtx(
            llm=self.llm,
            tools=self.tools,
            current_input=data,
            original_input=data,
            max_iterations=self.max_iterations,
        )

        state = AgentState.PLANNING

        while state != AgentState.FINISHED:
            match state:
                case AgentState.PLANNING:
                    if ctx.iterations >= ctx.max_iterations:
                        state = agent_state.handle(ctx, state, AgentEvent.MAX_REACHED)
                    else:
                        state = agent_state.handle(ctx, state, AgentEvent.CALL_LLM)

                case AgentState.PARSING:
                    res = ctx.last_response.strip()
                    if res.startswith("{") and "tool_call" in res:
                        state = agent_state.handle(ctx, state, AgentEvent.DETECTED_TOOL)
                    else:
                        state = agent_state.handle(
                            ctx, state, AgentEvent.DETECTED_ANSWER
                        )

                case AgentState.EXECUTING_TOOL:
                    state = agent_state.handle(ctx, state, AgentEvent.TOOL_COMPLETED)

        return ctx.final_output

from enum import Enum, auto
from typing import TypeVar, Generic, Dict, Tuple, List, Callable, Union, cast, Any
import json

from pydantic import BaseModel

from agentic.llm.base_llm import BaseLLM
from agentic.tool.base_tool import BaseTool


class AgentState(Enum):
    PLANNING = auto()
    PARSING = auto()
    EXECUTING_TOOL = auto()
    FINISHED = auto()


class AgentEvent(Enum):
    CALL_LLM = auto()
    DETECTED_TOOL = auto()
    DETECTED_ANSWER = auto()
    TOOL_COMPLETED = auto()
    MAX_REACHED = auto()


class AgentCtx(BaseModel):
    llm: BaseLLM  # type: ignore[type-arg]
    tools: List[BaseTool]
    current_input: str
    last_response: str = ""
    iterations: int = 0
    max_iterations: int = 5
    final_output: str = ""

    class Config:
        arbitrary_types_allowed = True


StateT = TypeVar("StateT")
EventT = TypeVar("EventT")
CtxT = TypeVar("CtxT")


class StateMachine(Generic[StateT, EventT, CtxT]):
    def __init__(self) -> None:
        self._transitions: Dict[
            Tuple[StateT, EventT], Tuple[StateT, Callable[[CtxT], None]]
        ] = {}

    def transition(
        self,
        from_states: Union[StateT, Tuple[StateT, ...]],
        event: EventT,
        to_state: StateT,
    ) -> Callable[[Callable[[CtxT], None]], Callable[[CtxT], None]]:
        def decorator(func: Callable[[CtxT], None]) -> Callable[[CtxT], None]:
            raw_states = (
                (from_states,) if not isinstance(from_states, tuple) else from_states
            )
            states: Tuple[StateT, ...] = cast(Tuple[StateT, ...], raw_states)
            for state in states:
                self._transitions[(state, event)] = (to_state, func)
            return func

        return decorator

    def handle(self, ctx: CtxT, current_state: StateT, event: EventT) -> StateT:
        lookup = (current_state, event)
        if lookup not in self._transitions:
            raise ValueError(
                f"Illegal state transition from {current_state} via event {event}"
            )
        next_state, callback = self._transitions[lookup]
        callback(ctx)
        return next_state


agent_state: StateMachine[AgentState, AgentEvent, AgentCtx] = StateMachine()


@agent_state.transition(AgentState.PLANNING, AgentEvent.CALL_LLM, AgentState.PARSING)
def plan_next_step(ctx: AgentCtx) -> None:
    ctx.iterations += 1
    ctx.last_response = ctx.llm.invoke(ctx.current_input)


@agent_state.transition(
    AgentState.PARSING, AgentEvent.DETECTED_TOOL, AgentState.EXECUTING_TOOL
)
def prepare_tool_run(ctx: AgentCtx) -> None:
    pass


@agent_state.transition(
    AgentState.PARSING, AgentEvent.DETECTED_ANSWER, AgentState.FINISHED
)
def extract_final_answer(ctx: AgentCtx) -> None:
    ctx.final_output = ctx.last_response


@agent_state.transition(
    AgentState.EXECUTING_TOOL, AgentEvent.TOOL_COMPLETED, AgentState.PLANNING
)
def execute_tool_logic(ctx: AgentCtx) -> None:
    try:
        # Mini inline JSON cleaner step if the LLM wraps the output in code block
        cleaned = ctx.last_response.replace("```json", "").replace("```", "").strip()
        parsed_call = json.loads(cleaned)
        tool_name = parsed_call.get("tool_name")
        args = parsed_call.get("arguments", {})

        tool = next((t for t in ctx.tools if t.name == tool_name), None)
        if tool:
            result = tool.invoke(args)
            observation = f"Tool '{tool_name}' returned: {result}"
        else:
            observation = f"Error: Tool '{tool_name}' not found."
    except Exception as e:
        observation = f"Error processing tool execution step: {str(e)}"

    if ctx.llm.conversation:
        ctx.llm.conversation.add_message(user=observation)
        ctx.current_input = f"A tool execution completed. Here is the observation data:\n{observation}\nAnalyze it and answer."


@agent_state.transition(
    AgentState.PLANNING, AgentEvent.MAX_REACHED, AgentState.FINISHED
)
def halt_loop(ctx: AgentCtx) -> None:
    ctx.final_output = "Error: Agent reached safety iteration limit."

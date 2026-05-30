from typing import Callable, Self

from agentic.agent.base_agent import BaseAgent


class LambdaAgent(BaseAgent):
    name: str = "creative_agent"
    func: Callable[[Self, str], str]

    def invoke(self, data: str) -> str:
        return self.func(self, data)

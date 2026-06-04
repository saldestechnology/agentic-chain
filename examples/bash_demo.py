import os
from typing import Any

from agents.lambda_agent import LambdaAgent

from agentic.memory.session_memory import SessionMemory
from agentic.conversation.conversation import Conversation
from agentic.template.prompt_template import PromptTemplate
from agentic.llm.ollama import Ollama
from agentic.llm.smollm import SmolLM
from agentic.agent.tool_agent import ToolAgent
from agentic.tool.bash_tool import FileTool, RgrepTool, CurlTool

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"


system_prompt = """
You are an AI assistant that adheres strictly to RFC 2119.
All of your responses MUST follow the conventions established in RFC 2119.
"""


def callback(cls: LambdaAgent, data: str) -> str:
    formatted = cls.prompt.invoke(data)
    return cls.llm.invoke(formatted)


start_prompt = PromptTemplate(
    template_str="""
        Explore the codebase at {path}. Use the available bash tools to investigate:
        - file_tool: list and read files
        - rgrep_tool: search file contents for patterns
        - curl_tool: fetch web resources if needed
        Provide a concise summary of the project structure and any interesting findings.
    """
)


def run(path: str = ".") -> tuple[str, Any]:
    """Build and execute the bash-tool agent chain."""
    shared_conversation = Conversation(memory=SessionMemory())
    llm = Ollama(system_prompt=system_prompt, conversation=shared_conversation)
    # llm = SmolLM(system_prompt=system_prompt, conversation=shared_conversation)
    lambda_agent = LambdaAgent(llm=llm, prompt=start_prompt, func=callback)
    tool_agent = ToolAgent(
        llm=llm, tools=[FileTool(), RgrepTool(), CurlTool()]
    )

    tool_chain = start_prompt | tool_agent
    result = tool_chain.invoke({"path": path})
    return result, shared_conversation


# ---------------------------------------------------------------------------
# Direct tool smoke tests (no LLM)
# ---------------------------------------------------------------------------
def demo_file_tool() -> None:
    ft = FileTool()
    print("=== FileTool: ls -la ===")
    print(ft.invoke({"command": "ls", "path": ".", "max_output_lines": 10}))
    print()

    print("=== FileTool: cat README ===")
    print(ft.invoke({"command": "cat", "path": "README.md", "max_output_lines": 20}))
    print()


def demo_rgrep_tool() -> None:
    rg = RgrepTool()
    print("=== RgrepTool: search 'class' in agentic/ ===")
    print(rg.invoke({"pattern": "^class ", "path": "agentic", "max_output_lines": 15}))
    print()


def demo_curl_tool() -> None:
    ct = CurlTool()
    print("=== CurlTool: GET example.com ===")
    print(ct.invoke({"url": "https://example.com", "max_output_lines": 10}))
    print()


def demo_security() -> None:
    ft = FileTool()
    print("=== Security: forbidden command ===")
    print(ft.invoke({"command": "rm", "path": "/", "max_output_lines": 5}))
    print()


if __name__ == "__main__":
    # Agent pipeline (uses LLM)
    result, shared_conversation = run()
    print(result)
    __import__("pprint").pprint(shared_conversation.compile())

    # Uncomment below to run direct tool smoke tests without LLM:
    # demo_file_tool()
    # demo_rgrep_tool()
    # demo_curl_tool()
    # demo_security()

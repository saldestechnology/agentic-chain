#!/usr/bin/env python3
"""
SmolLLM – Entry point.

Run one of the example chains below by uncommenting the import you want.
"""

# Default: web search + browser tool chain
from examples.web_tools import run as web_tools_run

# Alternative: structured haiku generation pipeline
# from examples.haiku import run as haiku_run

if __name__ == "__main__":
    result, shared_conversation = web_tools_run(animal="penguin")
    __import__("pprint").pprint(shared_conversation.compile())

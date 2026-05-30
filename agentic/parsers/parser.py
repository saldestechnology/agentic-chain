import json
from typing import Any, cast
from agentic.utils.logging import log


def json_to_dict(text: str) -> dict[Any, Any]:
    """Safely parse LLM output into a dictionary mid-chain."""
    try:
        return cast(dict, json.loads(text.strip()))
    except json.JSONDecodeError:
        log("Failed to parse JSON, routing raw text.")
        return {"title": "Generated Poem", "lines": [text]}

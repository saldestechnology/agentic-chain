import json
from agentic.utils.logging import log

def json_to_dict(text: str) -> dict:
    """Safely parse LLM output into a dictionary mid-chain."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Fallback if the LLM didn't output pristine JSON
        log(f"Failed to parse JSON, routing raw text.")
        return {"title": "Generated Poem", "lines": [text]}
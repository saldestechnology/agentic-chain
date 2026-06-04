from ddgs import DDGS
from pydantic import BaseModel, Field
from typing import Type

from agentic.tool.base_tool import BaseTool


class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query string to look up")
    num_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of search results to return (1-10)"
    )


class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = (
        "Perform a web search for a given query. Returns a numbered list of "
        "result titles, URLs, and short summaries. Use this to discover relevant "
        "links, then use web_browser to fetch full content from a chosen URL."
    )
    args_schema: Type[BaseModel] = WebSearchArgs
    max_retries: int = 3

    def func(self, query: str, num_results: int = 3) -> str:
        raw = DDGS().text(query, max_results=num_results)
        if not raw:
            return f"No results found for query: {query}"

        chunks = []
        for i, r in enumerate(raw[:num_results], 1):
            title = r.get("title", "No Title")
            href = r.get("href", "No URL")
            body = r.get("body", "").replace("\n", " ")[:200]
            chunks.append(f"{i}. {title}\n   URL: {href}\n   Summary: {body}\n")

        return "\n".join(chunks)

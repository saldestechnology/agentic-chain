from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import Optional, Type
from agentic.tool.base_tool import BaseTool
import httpx


class WebBrowserArgs(BaseModel):
    url: str = Field(
        description="The exact HTTP/HTTPS URL to scrape or fetch data from"
    )


class WebBrowser(BaseTool):
    name: str = "web_browser"
    description: str = "A tool that can scrape raw content from a specific website URL"
    args_schema: Type[BaseModel] = WebBrowserArgs

    def clean_html_response(
        self, content: str, line_limit: Optional[int] = None
    ) -> str:
        soup = BeautifulSoup(content, features="html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        text = soup.get_text(separator="\n").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return " ".join(lines if not line_limit else lines[:line_limit])

    def func(self, url: str) -> str:
        headers = {"User-Agent": "Chainreact-Agent/0.1"}
        with httpx.Client(headers=headers) as client:
            r = client.get(url)
            match r.status_code:
                case 200:
                    return self.clean_html_response(r.text)
                case _:
                    return f"Unable to fetch resource: ({r.status_code})"

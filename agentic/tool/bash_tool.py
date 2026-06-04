from abc import ABC, abstractmethod
import json
import subprocess
from typing import Any, Type

from pydantic import BaseModel, Field

from agentic.tool.base_tool import BaseTool


class BaseBashArgs(BaseModel):
    max_output_lines: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of output lines before truncation (1-500)",
    )


class BaseBashTool(BaseTool, ABC):
    """Abstract base for sandboxed shell-command tools.

    Subclasses define ``_build_command`` to turn validated kwargs into an
    ``argv`` list and set ``allowed_commands`` to whitelist the base command(s).
    """

    timeout_seconds: int = 120
    default_max_output_lines: int = 50
    allowed_commands: list[str] = []

    def func(self, *args: Any, **kwargs: Any) -> str:
        max_lines: int = kwargs.pop("max_output_lines", self.default_max_output_lines)
        argv = self._build_command(**kwargs)

        if not argv:
            return json.dumps(
                {
                    "ret_code": -1,
                    "stdout": "",
                    "stderr": "No command generated.",
                    "truncated": False,
                }
            )

        base_cmd = argv[0]
        if self.allowed_commands and base_cmd not in self.allowed_commands:
            return json.dumps(
                {
                    "ret_code": -1,
                    "stdout": "",
                    "stderr": (
                        f"Command '{base_cmd}' is not in the allowed whitelist. "
                        f"Allowed: {', '.join(self.allowed_commands)}"
                    ),
                    "truncated": False,
                }
            )

        try:
            proc = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _stdout: str = ""
            _stderr: str = ""
            if exc.stdout is not None:
                _stdout = (
                    exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
                )
            if exc.stderr is not None:
                _stderr = (
                    exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
                )
            stdout = _stdout[:4096]
            stderr = _stderr[:4096]
            return json.dumps(
                {
                    "ret_code": -2,
                    "stdout": stdout,
                    "stderr": (
                        f"Command timed out after {self.timeout_seconds}s.\n{stderr}"
                    ),
                    "truncated": False,
                }
            )
        except Exception as exc:
            return json.dumps(
                {
                    "ret_code": -3,
                    "stdout": "",
                    "stderr": f"Execution error: {exc}",
                    "truncated": False,
                }
            )

        stdout, stderr, truncated = self._truncate(proc.stdout, proc.stderr, max_lines)
        return json.dumps(
            {
                "ret_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
            }
        )

    @abstractmethod
    def _build_command(self, **kwargs: Any) -> list[str]:
        """Return the argv list for ``subprocess.run``.

        Must **not** return a shell string; each element is passed directly
        to the executable to avoid injection attacks.
        """

    def _truncate(
        self, stdout: str, stderr: str, max_lines: int
    ) -> tuple[str, str, bool]:
        out_lines = stdout.splitlines()
        err_lines = stderr.splitlines()
        total = len(out_lines) + len(err_lines)
        truncated = total > max_lines

        if not truncated:
            return stdout, stderr, False

        # Allocate proportional budget
        out_budget = max(1, int(max_lines * (len(out_lines) / total)))
        err_budget = max(1, max_lines - out_budget)

        out_result = "\n".join(out_lines[:out_budget])
        if len(out_lines) > out_budget:
            out_result += f"\n[truncated: {len(out_lines) - out_budget} more lines]"

        err_result = "\n".join(err_lines[:err_budget])
        if len(err_lines) > err_budget:
            err_result += f"\n[truncated: {len(err_lines) - err_budget} more lines]"

        return out_result, err_result, True


# ---------------------------------------------------------------------------
# Concrete tools
# ---------------------------------------------------------------------------


class FileToolArgs(BaseBashArgs):
    command: str = Field(
        description="The file command to run. One of: ls, cat, find, head, tail, wc",
    )
    path: str = Field(
        default=".",
        description="Path or glob to operate on",
    )


class FileTool(BaseBashTool):
    name: str = "file_tool"
    description: str = (
        "Read or list files and directories. Safe commands only: ls, cat, find, "
        "head, tail, wc. Returns structured JSON with ret_code, stdout, stderr."
    )
    args_schema: Type[BaseModel] = FileToolArgs
    allowed_commands: list[str] = ["ls", "cat", "find", "head", "tail", "wc"]

    def _build_command(self, **kwargs: Any) -> list[str]:
        cmd = kwargs.get("command", "ls")
        path = kwargs.get("path", ".")
        match cmd:
            case "ls":
                return ["ls", "-la", path]
            case "cat":
                return ["cat", path]
            case "find":
                return ["find", path, "-maxdepth", "2"]
            case "head":
                return ["head", "-n", "50", path]
            case "tail":
                return ["tail", "-n", "50", path]
            case "wc":
                return ["wc", "-l", path]
            case _:
                return [cmd, path]


class RgrepToolArgs(BaseBashArgs):
    pattern: str = Field(description="Regex pattern to search for")
    path: str = Field(default=".", description="Directory or file to search in")
    recursive: bool = Field(default=True, description="Search recursively")


class RgrepTool(BaseBashTool):
    name: str = "rgrep_tool"
    description: str = (
        "Search file contents for a pattern using ripgrep or grep. "
        "Returns structured JSON with ret_code, stdout, stderr."
    )
    args_schema: Type[BaseModel] = RgrepToolArgs
    allowed_commands: list[str] = ["rg", "grep"]

    def _build_command(self, **kwargs: Any) -> list[str]:
        pattern = kwargs.get("pattern", "")
        path = kwargs.get("path", ".")
        recursive = kwargs.get("recursive", True)

        # Prefer ripgrep if available, fall back to grep
        import shutil

        if shutil.which("rg"):
            argv = ["rg", "--color", "never", "--line-number", pattern, path]
            if not recursive:
                argv.insert(1, "--max-depth")
                argv.insert(2, "1")
            return argv

        argv = ["grep", "--line-number", pattern, path]
        if recursive:
            argv.insert(1, "-r")
        return argv


class CurlToolArgs(BaseBashArgs):
    url: str = Field(description="HTTP/HTTPS URL to fetch")
    method: str = Field(
        default="GET", description="HTTP method: GET, POST, PUT, DELETE"
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Optional request headers as key-value pairs"
    )


class CurlTool(BaseBashTool):
    name: str = "curl_tool"
    description: str = (
        "Make HTTP requests using curl. Returns structured JSON with "
        "ret_code, stdout (response body), stderr (curl diagnostics)."
    )
    args_schema: Type[BaseModel] = CurlToolArgs
    allowed_commands: list[str] = ["curl"]

    def _build_command(self, **kwargs: Any) -> list[str]:
        url = kwargs.get("url", "")
        method = kwargs.get("method", "GET").upper()
        headers: dict[str, str] = kwargs.get("headers", {})

        argv = [
            "curl",
            "-s",  # silent
            "-w",
            "\\nHTTP_CODE:%{http_code}",
            "-L",  # follow redirects
        ]

        if method != "GET":
            argv.extend(["-X", method])

        for key, val in headers.items():
            argv.extend(["-H", f"{key}: {val}"])

        argv.append(url)
        return argv

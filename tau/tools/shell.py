"""Shell tool — run bash commands with safety controls."""

from __future__ import annotations

import subprocess
import shlex
import logging
from typing import Any

from tau.core.types import ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)

# Module-level config (set by CLI at startup from TauConfig)
_shell_config: dict[str, Any] = {
    "require_confirmation": True,
    "timeout": 30,
    "allowed_commands": [],
}


def configure_shell(require_confirmation: bool, timeout: int, allowed_commands: list[str]) -> None:
    _shell_config["require_confirmation"] = require_confirmation
    _shell_config["timeout"] = timeout
    _shell_config["allowed_commands"] = allowed_commands


def _is_allowed(command: str) -> bool:
    allowed: list[str] = _shell_config["allowed_commands"]
    if not allowed:
        return True
    first_token = shlex.split(command)[0] if command.strip() else ""
    return any(first_token.startswith(a) for a in allowed)


def _ask_confirmation(command: str) -> bool:
    import sys
    sys.stderr.write(f"\n  ⚠  tau wants to run:\n\n    {command}\n\n  Allow? [y/N] ")
    sys.stderr.flush()
    answer = sys.stdin.readline().strip().lower()
    return answer in ("y", "yes")


def run_bash(command: str, workdir: str = ".") -> str:
    if not _is_allowed(command):
        return f"Error: command not in allowlist — {command!r}"

    if _shell_config["require_confirmation"]:
        if not _ask_confirmation(command):
            return "Cancelled by user."

    logger.debug("run_bash: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=_shell_config["timeout"],
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {_shell_config['timeout']}s"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr.rstrip()}")
    parts.append(f"[exit {result.returncode}]")
    return "\n".join(parts)


SHELL_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="run_bash",
        description=(
            "Run a shell (bash) command and return stdout, stderr, and exit code. "
            "Use for running tests, installing packages, compiling, etc."
        ),
        parameters={
            "command": ToolParameter(
                type="string",
                description="The bash command to execute.",
            ),
            "workdir": ToolParameter(
                type="string",
                description="Working directory for the command. Defaults to '.'.",
                required=False,
            ),
        },
        handler=run_bash,
    ),
]

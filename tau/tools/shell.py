"""Shell tool — run bash commands with safety controls."""

from __future__ import annotations

import subprocess
import shlex
import logging
from typing import Any, Callable

from tau.core.types import ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)


def _compact_python_tracebacks(text: str) -> str:
    """Collapse Python tracebacks to a short, user-facing summary.

    Keeps non-traceback output intact and replaces each traceback block with
    the exception line plus a short hint.
    """
    if not text:
        return text

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() != "Traceback (most recent call last):":
            out.append(line)
            i += 1
            continue

        # Consume traceback block lines until we hit the exception summary,
        # usually "ValueError: ..." (or another *Error/*Exception type).
        exc_line = ""
        j = i + 1
        while j < len(lines):
            cur = lines[j].rstrip()
            stripped = cur.strip()
            if stripped and ("Error:" in stripped or stripped.endswith("Exception")):
                exc_line = stripped
                j += 1
                break
            j += 1

        if not exc_line:
            # Fallback to the last non-empty traceback line we scanned.
            for k in range(min(j, len(lines)) - 1, i, -1):
                if lines[k].strip():
                    exc_line = lines[k].strip()
                    break

        if exc_line:
            out.append(f"Python exception: {exc_line}")
        else:
            out.append("Python exception occurred.")
        out.append("(traceback hidden; rerun manually for full details)")
        i = max(j, i + 1)

    return "\n".join(out).rstrip()

# Module-level config (set by CLI at startup from TauConfig)
_shell_config: dict[str, Any] = {
    "require_confirmation": True,
    "timeout": 30,
    "allowed_commands": [],
    "use_persistent_shell": False,
    "workspace_root": ".",
}

_persistent_shell: PersistentShell | None = None


class PersistentShell:
    """Manages a long-running bash process across turn boundaries."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        import uuid
        import os
        self._sentinel = f"__TAU_CMD_DONE_{uuid.uuid4()}__"
        self._initial_cwd = os.getcwd()
        # Strip macOS memory-debugging env vars so they don't leak into
        # child processes (e.g. javac) and produce MallocStackLogging noise.
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("Malloc")}
        self._process = subprocess.Popen(
            ["bash", "--noprofile", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=clean_env,
        )
        # Flush the initial interactive bash prompt
        self._execute("echo started")

    def execute(self, command: str, workdir: str | None = None) -> str:
        if self._process.poll() is not None:
            # Shell died — restart it
            self.__init__(self.timeout)

        full_cmd = command
        if workdir:
            import os
            # Resolve relative workdir against initial CWD so the path is
            # stable regardless of where previous commands left the shell.
            abs_workdir = os.path.abspath(os.path.join(self._initial_cwd, workdir))
            full_cmd = f"cd {shlex.quote(abs_workdir)} && {command}"

        # We wrap the command to emit the sentinel even if it fails
        # and to capture the exit code.
        wrapped = f" {full_cmd}; echo \"\\n[exit $?]\\n{self._sentinel}\"\n"
        
        try:
            self._process.stdin.write(wrapped)
            self._process.stdin.flush()
        except BrokenPipeError:
            self.__init__(self.timeout)
            self._process.stdin.write(wrapped)
            self._process.stdin.flush()

        output_lines: list[str] = []
        found_command_echo = False
        while True:
            line = self._process.stdout.readline()
            if not line:
                break
            if self._sentinel in line:
                break
            
            # Skip the echoed command if it shows up in the stream
            if not found_command_echo and full_cmd in line:
                found_command_echo = True
                continue
            
            output_lines.append(line)

        # Cleanup: remove leading bash prompts and trailing whitespace
        res = "".join(output_lines).strip()
        # Remove any leading "bash-X.X$ " prompt if it leaked
        import re
        res = re.sub(r'^bash-[0-9.]+[$#] ', '', res)
        return _compact_python_tracebacks(res.strip())

    def _execute(self, command: str) -> None:
        """Internal low-level execute without exit-code wrapping."""
        self._process.stdin.write(f"{command}; echo {self._sentinel}\n")
        self._process.stdin.flush()
        while True:
            line = self._process.stdout.readline()
            if not line or self._sentinel in line:
                break

    def __del__(self) -> None:
        if hasattr(self, "_process") and self._process.poll() is None:
            self._process.terminate()

# Pluggable confirmation hook — CLI replaces this so it can flush the
# in-progress stream to stdout before showing the prompt.
_confirm_hook: Callable[[str], bool] | None = None


def configure_shell(
    require_confirmation: bool,
    timeout: int,
    allowed_commands: list[str],
    use_persistent_shell: bool = False,
    confirm_hook: Callable[[str], bool] | None = None,
    workspace_root: str = ".",
) -> None:
    _shell_config["require_confirmation"] = require_confirmation
    _shell_config["timeout"] = timeout
    _shell_config["allowed_commands"] = allowed_commands
    _shell_config["use_persistent_shell"] = use_persistent_shell
    _shell_config["workspace_root"] = workspace_root
    global _confirm_hook
    _confirm_hook = confirm_hook


def _default_confirm(command: str) -> bool:
    import sys
    sys.stderr.write(f"\n  ⚠  tau wants to run:\n\n    {command}\n\n  Allow? [y/N] ")
    sys.stderr.flush()
    return sys.stdin.readline().strip().lower() in ("y", "yes")


def _is_allowed(command: str) -> bool:
    allowed: list[str] = _shell_config["allowed_commands"]
    if not allowed:
        return True
    first_token = shlex.split(command)[0] if command.strip() else ""
    return any(first_token.startswith(a) for a in allowed)


def run_bash(command: str, workdir: str = "") -> str:
    if not _is_allowed(command):
        return f"Error: command not in allowlist — {command!r}"

    if _shell_config["require_confirmation"]:
        confirm = _confirm_hook if _confirm_hook is not None else _default_confirm
        if not confirm(command):
            return "Cancelled by user."

    effective_workdir = workdir or _shell_config["workspace_root"]

    if _shell_config.get("use_persistent_shell"):
        global _persistent_shell
        if _persistent_shell is None:
            _persistent_shell = PersistentShell(timeout=_shell_config["timeout"])
        return _persistent_shell.execute(command, workdir=effective_workdir)

    logger.debug("run_bash (ephemeral): %s", command)
    import os
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("Malloc")}
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=effective_workdir,
            capture_output=True,
            text=True,
            timeout=_shell_config["timeout"],
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {_shell_config['timeout']}s"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"

    parts: list[str] = []
    if result.stdout:
        parts.append(_compact_python_tracebacks(result.stdout.rstrip()))
    if result.stderr:
        compact_stderr = _compact_python_tracebacks(result.stderr.rstrip())
        parts.append(f"[stderr]\n{compact_stderr}")
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
                description="Working directory for the command. Defaults to the workspace root.",
                required=False,
            ),
        },
        handler=run_bash,
    ),
]

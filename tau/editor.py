"""Editor richness — @file references, tab completion, image paste, inline shell."""

from __future__ import annotations

import base64
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# @file reference expansion
# ---------------------------------------------------------------------------

# Matches @path where path is a relative or absolute file path.
# Stops at whitespace or common punctuation that wouldn't be in a path.
_AT_FILE_RE = re.compile(r"@((?:[A-Za-z]:)?[^\s@,;\"'`]+)")

# Max file size we'll inline (to avoid blowing up the context).
_MAX_INLINE_BYTES = 256 * 1024  # 256 KB


def expand_at_files(text: str, workspace_root: str) -> tuple[str, list[str]]:
    """Expand ``@path/to/file`` references in *text*.

    Each ``@ref`` is replaced by a fenced code block with the file contents.
    Returns ``(expanded_text, list_of_resolved_paths)`` — paths that were
    successfully inlined.  References to non-existent files are left as-is.
    """
    root = Path(workspace_root).resolve()
    inlined: list[str] = []

    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        # Security: must be inside workspace
        if not str(p).startswith(str(root)):
            return m.group(0)
        if not p.is_file():
            return m.group(0)
        if p.stat().st_size > _MAX_INLINE_BYTES:
            return m.group(0) + " (file too large to inline)"
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return m.group(0)
        rel = p.relative_to(root)
        inlined.append(str(p))
        return f"<file path=\"{rel}\">\n{content}\n</file>"

    expanded = _AT_FILE_RE.sub(_replace, text)
    return expanded, inlined


# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------

def complete_path(prefix: str, workspace_root: str) -> list[str]:
    """Return file/directory completions for *prefix* under *workspace_root*.

    If prefix is empty, lists workspace root entries.
    Directories are returned with a trailing ``/``.
    """
    root = Path(workspace_root).resolve()
    p = Path(prefix) if prefix else Path(".")
    if not p.is_absolute():
        p = root / p
    p = p.resolve()

    # Security: stay within workspace
    if not str(p).startswith(str(root)):
        return []

    if p.is_dir():
        parent = p
        partial = ""
    else:
        parent = p.parent
        partial = p.name

    # Special case: if the raw prefix ends with "/" it's a directory listing,
    # but if it's just a filename fragment in the root we need the parent dir.
    # Also handle: a lone "." should list dot-prefixed entries in root.
    if prefix == ".":
        parent = root
        partial = "."
    elif prefix.endswith("/"):
        parent = p
        partial = ""

    if not parent.is_dir():
        return []

    results: list[str] = []
    try:
        for entry in sorted(parent.iterdir(), key=lambda e: e.name):
            if entry.name.startswith(".") and not partial.startswith("."):
                continue  # skip hidden by default
            if partial and not entry.name.lower().startswith(partial.lower()):
                continue
            try:
                rel = entry.relative_to(root)
            except ValueError:
                continue
            suffix = "/" if entry.is_dir() else ""
            results.append(str(rel) + suffix)
    except OSError:
        pass
    return results


def complete_slash_commands(
    prefix: str,
    builtin_commands: Iterable[str],
    extension_commands: Iterable[str] = (),
) -> list[str]:
    """Return matching ``/command`` completions for *prefix*."""
    all_cmds = list(builtin_commands) + list(extension_commands)
    if not prefix.startswith("/"):
        return []
    typed = prefix[1:].lower()
    return [f"/{c}" for c in all_cmds if c.lower().startswith(typed)]


# Builtin slash command names (matches _SLASH_HELP in cli.py)
BUILTIN_SLASH_COMMANDS = [
    "help", "queue", "steer", "clear", "compact",
    "model", "think", "tokens", "tree", "fork", "bookmark", "image",
    "voice", "voice-retry", "doctor", "copy", "export", "share", "import", "reload", "prompt", "prompts",
    "theme", "themes",
]




# ---------------------------------------------------------------------------
# Image detection from clipboard / file path
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}


def is_image_path(path: str) -> bool:
    """Return True if *path* looks like an image file that exists."""
    p = Path(path)
    return p.suffix.lower() in _IMAGE_EXTENSIONS and p.is_file()


def detect_pasted_image_macos() -> str | None:
    """Try to grab an image from the macOS clipboard and save it to a temp file.

    Returns the temp file path if an image was found, else ``None``.
    On non-macOS or if no image is in the clipboard, returns ``None``.
    """
    import sys
    if sys.platform != "darwin":
        return None
    try:
        # Check if clipboard has image data
        result = subprocess.run(
            ["osascript", "-e", 'clipboard info for (clipboard info)'],
            capture_output=True, text=True, timeout=3,
        )
        # A more reliable check: use a tiny AppleScript that writes clipboard
        # image to a temp file.
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        script = (
            f'set fp to POSIX file "{tmp.name}"\n'
            'try\n'
            '  set img to (the clipboard as «class PNGf»)\n'
            '  set fh to open for access fp with write permission\n'
            '  write img to fh\n'
            '  close access fh\n'
            '  return "ok"\n'
            'on error\n'
            '  return "no_image"\n'
            'end try\n'
        )
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        if r.stdout.strip() == "ok" and Path(tmp.name).stat().st_size > 0:
            return tmp.name
        else:
            # Clean up empty temp file
            Path(tmp.name).unlink(missing_ok=True)
            return None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Inline shell execution
# ---------------------------------------------------------------------------

_SHELL_PREFIX = "!"


def is_shell_command(text: str) -> bool:
    """Return True if *text* starts with ``!`` (inline shell escape)."""
    return text.startswith(_SHELL_PREFIX) and len(text) > 1


def run_inline_shell(command: str, workspace_root: str, timeout: int = 30) -> str:
    """Execute a shell command and return its combined stdout+stderr.

    Only for **user-initiated** inline commands (``!ls``, ``!git status``).
    These bypass the agent and run directly.
    """
    import platform
    env = os.environ.copy()
    if platform.system() == "Darwin":
        # Suppress macOS MallocStackLogging noise injected by the debugger
        # shim into every spawned subprocess.
        env.pop("MallocStackLogging", None)
        env.pop("MallocStackLoggingDirectory", None)
        env["MallocStackLogging"] = "0"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace_root,
            env=env,
        )
        # Filter residual MallocStackLogging lines that slip through via the
        # dynamic linker before env changes take effect.
        def _filter_malloc_noise(text: str) -> str:
            lines = [
                l for l in text.splitlines()
                if "MallocStackLogging" not in l
            ]
            return "\n".join(lines) + ("\n" if lines else "")

        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout)
        stderr_clean = _filter_malloc_noise(result.stderr) if result.stderr else ""
        if stderr_clean.strip():
            parts.append(stderr_clean)
        output = "".join(parts)
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"(command timed out after {timeout}s)"
    except Exception as exc:  # noqa: BLE001
        return f"(error: {exc})"

"""Context file discovery and loading for tau.

Discovers and concatenates ``AGENTS.md`` / ``CLAUDE.md`` files from:
  1. ``~/.tau/AGENTS.md``           — global instructions
  2. Parent directories walking up  — org / monorepo instructions
  3. ``<workspace>/AGENTS.md``      — project-specific instructions

All matching files are concatenated in that order and injected into the
system prompt via ``ContextManager.inject_prompt_fragment()``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# File names we recognise (case-sensitive on Unix, but we check both forms)
_CONTEXT_FILENAMES = ("AGENTS.md", "CLAUDE.md")

TAU_HOME = Path.home() / ".tau"


def _find_context_files(workspace_root: str) -> list[Path]:
    """
    Return an ordered list of context file paths to load.

    Order: global → ancestors (root → workspace parent) → workspace itself.
    Duplicates are removed (preserving first occurrence).
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            found.append(resolved)

    # 1. Global: ~/.tau/AGENTS.md (or CLAUDE.md)
    for name in _CONTEXT_FILENAMES:
        _add(TAU_HOME / name)

    # 2. Walk UP from workspace to filesystem root
    workspace = Path(workspace_root).resolve()
    ancestors: list[Path] = []
    current = workspace.parent
    while current != current.parent:
        ancestors.append(current)
        current = current.parent
    ancestors.append(current)  # filesystem root

    # Reverse so root comes first → nearest parent last
    for ancestor in reversed(ancestors):
        for name in _CONTEXT_FILENAMES:
            _add(ancestor / name)

    # 3. Workspace directory itself
    for name in _CONTEXT_FILENAMES:
        _add(workspace / name)

    return found


def load_context_files(workspace_root: str) -> str:
    """
    Discover and concatenate all context files.

    Returns the combined text (may be empty if no files found).
    """
    files = _find_context_files(workspace_root)
    if not files:
        return ""

    parts: list[str] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"# Context: {path}\n\n{content}")
                logger.debug("Loaded context file: %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read context file %s: %s", path, exc)

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Per-project system prompt override: .tau/SYSTEM.md
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_FILE = ".tau/SYSTEM.md"


def load_system_prompt_override(workspace_root: str) -> str | None:
    """Return the contents of `<workspace>/.tau/SYSTEM.md` if it exists.

    When present this file **replaces** the default system prompt entirely,
    giving projects full control over the agent's persona and instructions.
    Returns ``None`` when no override file is found.
    """
    path = Path(workspace_root).resolve() / _SYSTEM_PROMPT_FILE
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        if content:
            logger.debug("Loaded system prompt override: %s", path)
            return content
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read system prompt file %s: %s", path, exc)
    return None

"""Prompt templates — reusable Markdown files with ``{{variable}}`` substitution.

Template discovery locations (higher priority wins on name collision):
  1. ``<workspace>/.tau/prompts/``   — project-specific templates
  2. ``~/.tau/prompts/``             — global (user) templates

A template is any ``.md`` file in one of the directories above.  The stem
of the filename becomes the template name (e.g. ``fix-bug.md`` → ``fix-bug``).

Variables use ``{{name}}`` syntax.  Undefined variables are left as-is so
the LLM can see them; extra variables that don't appear in the template are
silently ignored.

Example template (``~/.tau/prompts/review.md``)::

    Review the file **{{file}}** and look for:
    - bugs or logic errors
    - missing edge-case handling
    - style issues

    Provide fixes as a unified diff.

Usage::

    tau run --template review --var file=src/main.py
    # or in the REPL:
    /prompt review file=src/main.py
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

TAU_HOME = Path.home() / ".tau"
_PROMPTS_DIR = "prompts"

# Matches {{variable_name}} — alphanumerics, underscores, hyphens
_VAR_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_-]*)\}\}")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _template_dirs(workspace_root: str) -> list[Path]:
    """Return template directories in priority order (project first)."""
    dirs: list[Path] = []
    # Project-local
    project = Path(workspace_root).resolve() / ".tau" / _PROMPTS_DIR
    if project.is_dir():
        dirs.append(project)
    # Global
    global_dir = TAU_HOME / _PROMPTS_DIR
    if global_dir.is_dir():
        dirs.append(global_dir)
    return dirs


def list_templates(workspace_root: str) -> dict[str, Path]:
    """Return ``{name: path}`` for all discovered templates.

    Project templates shadow global ones with the same name.
    """
    templates: dict[str, Path] = {}
    # Scan in reverse priority so that project-local overwrites global
    for d in reversed(_template_dirs(workspace_root)):
        for p in sorted(d.glob("*.md")):
            if p.is_file():
                templates[p.stem] = p
    return templates


# ---------------------------------------------------------------------------
# Variable extraction & rendering
# ---------------------------------------------------------------------------

def extract_variables(template_text: str) -> list[str]:
    """Return the list of unique variable names found in the template."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _VAR_RE.finditer(template_text):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def render(template_text: str, variables: dict[str, str]) -> str:
    """Substitute ``{{name}}`` placeholders with provided values.

    Variables that appear in the template but are missing from *variables*
    are left untouched (``{{name}}`` stays), so the LLM can see them.
    """
    def _replace(m: re.Match) -> str:
        name = m.group(1)
        return variables.get(name, m.group(0))

    return _VAR_RE.sub(_replace, template_text)


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def load_template(name: str, workspace_root: str) -> str | None:
    """Load template content by name.  Returns ``None`` if not found."""
    templates = list_templates(workspace_root)
    path = templates.get(name)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read template %s (%s): %s", name, path, exc)
        return None


def resolve_template(
    name: str,
    workspace_root: str,
    variables: dict[str, str],
) -> str | None:
    """Load a template by name, substitute variables, and return the result.

    Returns ``None`` if the template is not found.
    """
    text = load_template(name, workspace_root)
    if text is None:
        return None
    return render(text, variables)


def parse_var_args(var_list: tuple[str, ...] | list[str]) -> dict[str, str]:
    """Parse ``("key=value", ...)`` into a dict.

    Raises ``click.BadParameter`` (via ``ValueError``) for malformed entries.
    """
    result: dict[str, str] = {}
    for item in var_list:
        if "=" not in item:
            raise ValueError(
                f"Invalid --var format: {item!r}. Expected key=value."
            )
        key, _, value = item.partition("=")
        result[key.strip()] = value.strip()
    return result

"""Skill loader — discovers and loads skill bundles into registry + context."""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from tau.core.types import ToolDefinition

if TYPE_CHECKING:
    from tau.core.context import ContextManager
    from tau.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

TAU_HOME = Path.home() / ".tau"
BUILTIN_SKILLS_DIR = Path(__file__).parent


@dataclass
class SkillMeta:
    name: str
    version: str = "0.1.0"
    description: str = ""
    system_prompt_fragment: str = ""


@dataclass
class Skill:
    meta: SkillMeta
    tools: list[ToolDefinition] = field(default_factory=list)
    path: Path = field(default_factory=Path)


# ---------------------------------------------------------------------------
# SkillLoader
# ---------------------------------------------------------------------------

class SkillLoader:
    def __init__(self, extra_paths: list[str] | None = None, disabled: list[str] | None = None) -> None:
        self._search_paths: list[Path] = [
            BUILTIN_SKILLS_DIR,
            TAU_HOME / "skills",
        ]
        for p in (extra_paths or []):
            self._search_paths.append(Path(p).expanduser())
        self._disabled = set(disabled or [])

    def discover(self) -> list[Skill]:
        skills: list[Skill] = []
        seen: set[str] = set()
        for base in self._search_paths:
            if not base.is_dir():
                continue
            for skill_dir in sorted(base.iterdir()):
                if not skill_dir.is_dir():
                    continue
                yaml_path = skill_dir / "skill.yaml"
                if not yaml_path.exists():
                    continue
                skill = self._load_skill(skill_dir)
                if skill is None:
                    continue
                if skill.meta.name in self._disabled:
                    logger.debug("Skill %r is disabled — skipping.", skill.meta.name)
                    continue
                if skill.meta.name in seen:
                    logger.debug("Skill %r already loaded — skipping duplicate.", skill.meta.name)
                    continue
                seen.add(skill.meta.name)
                skills.append(skill)
                logger.debug("Discovered skill: %s (%s)", skill.meta.name, skill.path)
        return skills

    def _load_skill(self, skill_dir: Path) -> Skill | None:
        yaml_path = skill_dir / "skill.yaml"
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse %s: %s", yaml_path, exc)
            return None

        meta = SkillMeta(
            name=raw.get("name", skill_dir.name),
            version=raw.get("version", "0.1.0"),
            description=raw.get("description", ""),
            system_prompt_fragment=raw.get("system_prompt_fragment", ""),
        )

        tools: list[ToolDefinition] = []
        tools_py = skill_dir / "tools.py"
        if tools_py.exists():
            tools = _import_tools(tools_py, meta.name)

        return Skill(meta=meta, tools=tools, path=skill_dir)

    def load_into(self, registry: "ToolRegistry", context: "ContextManager") -> None:
        for skill in self.discover():
            if skill.tools:
                registry.register_many(skill.tools)
            if skill.meta.system_prompt_fragment:
                context.inject_prompt_fragment(skill.meta.system_prompt_fragment)
            logger.debug(
                "Loaded skill %r — %d tools, prompt_fragment=%s",
                skill.meta.name,
                len(skill.tools),
                bool(skill.meta.system_prompt_fragment),
            )


def _import_tools(path: Path, skill_name: str) -> list[ToolDefinition]:
    """Dynamically import a tools.py and return its TOOLS list."""
    module_name = f"tau_skill_{skill_name.replace('-', '_')}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return []
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        tools = getattr(mod, "TOOLS", [])
        if not isinstance(tools, list):
            logger.warning("Skill %r: tools.py TOOLS is not a list.", skill_name)
            return []
        return tools
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not import tools from %s: %s", path, exc)
        return []

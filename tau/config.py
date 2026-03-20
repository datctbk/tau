"""Configuration loading for tau.

Priority (highest → lowest):
  1. CLI flags
  2. Environment variables  (TAU_PROVIDER, TAU_MODEL, OPENAI_API_KEY, …)
  3. ~/.tau/config.toml
  4. Built-in defaults
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

TAU_HOME = Path.home() / ".tau"
CONFIG_PATH = TAU_HOME / "config.toml"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class OpenAIProviderConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"


class GoogleProviderConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_")
    api_key: str = ""


class OllamaProviderConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")
    base_url: str = "http://localhost:11434"


class ShellToolConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_SHELL_")
    require_confirmation: bool = True
    timeout: int = 30
    allowed_commands: list[str] = []   # empty = allow all


class SkillsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_SKILLS_")
    paths: list[str] = []
    disabled: list[str] = []


class ExtensionsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_EXTENSIONS_")
    paths: list[str] = []
    disabled: list[str] = []


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class TauConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_")

    provider: str = "openai"
    model: str = "gpt-4o"
    max_tokens: int = 8192
    max_turns: int = 20
    trim_strategy: str = "sliding_window"
    system_prompt: str = (
        "You are tau, a minimal CLI coding agent. "
        "Use the available tools to help the user with coding tasks. "
        "Be concise and precise."
    )

    # provider sub-configs
    openai: OpenAIProviderConfig = OpenAIProviderConfig()
    google: GoogleProviderConfig = GoogleProviderConfig()
    ollama: OllamaProviderConfig = OllamaProviderConfig()

    # tool / skill sub-configs
    shell: ShellToolConfig = ShellToolConfig()
    skills: SkillsConfig = SkillsConfig()
    extensions: ExtensionsConfig = ExtensionsConfig()

    @field_validator("trim_strategy")
    @classmethod
    def validate_trim(cls, v: str) -> str:
        allowed = {"sliding_window", "summarise"}
        if v not in allowed:
            raise ValueError(f"trim_strategy must be one of {allowed}")
        return v


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse config file %s: %s", path, exc)
        return {}


def load_config(config_path: Path = CONFIG_PATH) -> TauConfig:
    """Load config from TOML, then overlay env vars."""
    raw = _load_toml(config_path)
    init: dict[str, Any] = {}
    defaults = raw.get("defaults", {})
    init.update(defaults)
    for section in ("openai", "google", "ollama"):
        if section in raw.get("providers", {}):
            init[section] = raw["providers"][section]
    if "shell" in raw.get("tools", {}):
        init["shell"] = raw["tools"]["shell"]
    if "skills" in raw:
        init["skills"] = raw["skills"]
    if "extensions" in raw:
        init["extensions"] = raw["extensions"]
    return TauConfig(**init)


def ensure_tau_home() -> None:
    """Create ~/.tau and its sub-directories if they don't exist."""
    for subdir in ("", "sessions", "skills", "extensions"):
        (TAU_HOME / subdir).mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            "# tau configuration\n"
            "# See: https://github.com/you/tau\n\n"
            "[defaults]\n"
            'provider = "openai"\n'
            'model    = "gpt-4o"\n',
            encoding="utf-8",
        )

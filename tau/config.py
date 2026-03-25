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
    use_persistent_shell: bool = False


class SkillsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_SKILLS_")
    paths: list[str] = []
    disabled: list[str] = []


class ThinkingBudgetsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_THINKING_BUDGETS_")
    minimal: int = 1024
    low: int = 2048
    medium: int = 8192
    high: int = 16384
    xhigh: int = 32768


class ExtensionsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_EXTENSIONS_")
    paths: list[str] = []
    disabled: list[str] = []


class ThemeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_THEME_")
    # Role colours (Rich markup names)
    user_color: str = "cyan"
    assistant_color: str = "green"
    tool_color: str = "yellow"
    system_color: str = "dim"
    error_color: str = "red"
    # UI accents
    accent_color: str = "cyan"
    success_color: str = "green"
    warning_color: str = "yellow"
    border_style: str = "dim"


class ToolsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_TOOLS_")
    disabled: list[str] = []      # tool names to disable, e.g. ["run_bash"]
    enabled_only: list[str] = []  # if non-empty, ONLY these tools are registered


class PricingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_PRICING_")
    # Cost per 1M tokens: input, output, cache_read, cache_write
    models: dict[str, dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00, "cache_read": 1.25},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600, "cache_read": 0.075},
        "o1-mini": {"input": 3.00, "output": 12.00, "cache_read": 1.50},
        "o1-preview": {"input": 15.00, "output": 60.00, "cache_read": 7.50},
        "o3-mini": {"input": 1.10, "output": 4.40, "cache_read": 0.55},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
        "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25, "cache_read": 0.03, "cache_write": 0.30},
        "gemini-2.5-pro": {"input": 2.00, "output": 8.00},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash-thinking-exp": {"input": 0.0, "output": 0.0},
    }

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
    parallel_tools: bool = True
    parallel_tools_max_workers: int = 8

    # provider sub-configs
    openai: OpenAIProviderConfig = OpenAIProviderConfig()
    google: GoogleProviderConfig = GoogleProviderConfig()
    ollama: OllamaProviderConfig = OllamaProviderConfig()

    # tool / skill sub-configs
    shell: ShellToolConfig = ShellToolConfig()
    skills: SkillsConfig = SkillsConfig()
    extensions: ExtensionsConfig = ExtensionsConfig()
    pricing: PricingConfig = PricingConfig()
    thinking_budgets: ThinkingBudgetsConfig = ThinkingBudgetsConfig()
    theme: ThemeConfig = ThemeConfig()
    tools: ToolsConfig = ToolsConfig()

    @field_validator("trim_strategy")
    @classmethod
    def validate_trim(cls, v: str) -> str:
        allowed = {"sliding_window", "summarise"}
        if v not in allowed:
            raise ValueError(f"trim_strategy must be one of {allowed}")
        return v

    def calculate_cost(self, model: str, usage_in: Any) -> float:
        """Calculate the USD cost of a TokenUsage object for the given model."""
        rates = self.pricing.models.get(model)
        if not rates:
            # Sort by longest prefix first to match more specific variants
            for prefix, pre_rates in sorted(self.pricing.models.items(), key=lambda x: len(x[0]), reverse=True):
                if model.startswith(prefix):
                    rates = pre_rates
                    break
        if not rates:
            return 0.0
        
        # We type usage_in as Any to avoid circular import with TokenUsage, but we duck-type it here
        return (
            (getattr(usage_in, "input_tokens", 0) / 1_000_000) * rates.get("input", 0.0) +
            (getattr(usage_in, "output_tokens", 0) / 1_000_000) * rates.get("output", 0.0) +
            (getattr(usage_in, "cache_read_tokens", 0) / 1_000_000) * rates.get("cache_read", 0.0) +
            (getattr(usage_in, "cache_write_tokens", 0) / 1_000_000) * rates.get("cache_write", 0.0)
        )

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
    if "pricing" in raw:
        init["pricing"] = raw["pricing"]
    if "thinking_budgets" in raw:
        init["thinking_budgets"] = raw["thinking_budgets"]
    if "theme" in raw:
        init["theme"] = raw["theme"]
    # [tools] section (note: [tools.shell] already handled above via raw["tools"]["shell"])
    tools_section = raw.get("tools", {})
    tool_disabled = tools_section.get("disabled", [])
    tool_enabled_only = tools_section.get("enabled_only", [])
    if tool_disabled or tool_enabled_only:
        init["tools"] = {"disabled": tool_disabled, "enabled_only": tool_enabled_only}
    if "parallel_tools" in raw:
        init["parallel_tools"] = raw["parallel_tools"]
    if "parallel_tools_max_workers" in raw:
        init["parallel_tools_max_workers"] = raw["parallel_tools_max_workers"]
    return TauConfig(**init)


def ensure_tau_home() -> None:
    """Create ~/.tau and its sub-directories if they don't exist."""
    for subdir in ("", "sessions", "skills", "extensions", "packages"):
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

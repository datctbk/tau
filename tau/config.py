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
THEME_PATH = TAU_HOME / "theme.toml"   # optional dedicated theme file


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
    timeout_seconds: float = 2*60.0


class UnslothProviderConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UNSLOTH_")
    base_url: str = "http://localhost:8001/v1"
    timeout_seconds: float = 2*60.0
    # Streaming read timeout in seconds. <= 0 disables read timeout
    # so long prefill phases can complete before first token.
    stream_read_timeout_seconds: float = 0.0
    # Client-side stream throttling to reduce local UI starvation during generation.
    stream_yield_every_chunks: int = 0
    stream_yield_ms: float = 0.0


class MLXProviderConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLX_")
    device: str = "gpu"  # gpu|cpu
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    memory_limit_gb: float | None = 16.0
    wired_limit_gb: float | None = 12.0
    cache_limit_mb: int = 64
    prefill_step_size: int = 128
    max_kv_size: int | None = 1024
    kv_bits: int | None = 4
    quantized_kv_start: int = 0
    gpu_yield_every: int = 1       # yield GPU every token for fairness
    gpu_yield_ms: float = 8.0      # short pause to let compositor submit frames


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


# ---------------------------------------------------------------------------
# Built-in theme presets
# ---------------------------------------------------------------------------

#: Map of preset name → dict of ThemeConfig field overrides.
THEME_PRESETS: dict[str, dict[str, str]] = {
    "dark": {
        "user_color": "cyan",
        "assistant_color": "green",
        "tool_color": "yellow",
        "system_color": "dim",
        "error_color": "red",
        "accent_color": "cyan",
        "success_color": "green",
        "warning_color": "yellow",
        "border_style": "dim",
    },
    "light": {
        "user_color": "blue",
        "assistant_color": "dark_green",
        "tool_color": "dark_orange",
        "system_color": "grey50",
        "error_color": "dark_red",
        "accent_color": "blue",
        "success_color": "dark_green",
        "warning_color": "dark_orange",
        "border_style": "grey50",
    },
    "solarized-dark": {
        "user_color": "#268bd2",
        "assistant_color": "#859900",
        "tool_color": "#b58900",
        "system_color": "#586e75",
        "error_color": "#dc322f",
        "accent_color": "#2aa198",
        "success_color": "#859900",
        "warning_color": "#cb4b16",
        "border_style": "#073642",
    },
    "solarized-light": {
        "user_color": "#268bd2",
        "assistant_color": "#859900",
        "tool_color": "#b58900",
        "system_color": "#93a1a1",
        "error_color": "#dc322f",
        "accent_color": "#2aa198",
        "success_color": "#859900",
        "warning_color": "#cb4b16",
        "border_style": "#eee8d5",
    },
    "monokai": {
        "user_color": "#66d9e8",
        "assistant_color": "#a9dc76",
        "tool_color": "#ffd866",
        "system_color": "#727072",
        "error_color": "#ff6188",
        "accent_color": "#ab9df2",
        "success_color": "#a9dc76",
        "warning_color": "#fc9867",
        "border_style": "#403e41",
    },
    "dracula": {
        "user_color": "#8be9fd",
        "assistant_color": "#50fa7b",
        "tool_color": "#f1fa8c",
        "system_color": "#6272a4",
        "error_color": "#ff5555",
        "accent_color": "#bd93f9",
        "success_color": "#50fa7b",
        "warning_color": "#ffb86c",
        "border_style": "#44475a",
    },
    "nord": {
        "user_color": "#88c0d0",
        "assistant_color": "#a3be8c",
        "tool_color": "#ebcb8b",
        "system_color": "#4c566a",
        "error_color": "#bf616a",
        "accent_color": "#81a1c1",
        "success_color": "#a3be8c",
        "warning_color": "#d08770",
        "border_style": "#3b4252",
    },
}


class ThemeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAU_THEME_")
    # Optional preset name — applied before individual color overrides.
    preset: str = ""
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
    max_tokens: int = 6144 * 18
    max_turns: int = 20
    trim_strategy: str = "sliding_window"
    compaction_enabled: bool = True
    compaction_threshold: float = 0.60
    system_prompt: str = (
        "You are tau, a minimal CLI coding agent. "
        "Use the available tools to help the user with coding tasks. "
        "Be concise and precise."
    )
    parallel_tools: bool = True
    parallel_tools_max_workers: int = 8
    max_cost: float = 0.0  # USD budget ceiling; 0 = unlimited
    policy_enabled: bool = True
    policy_profile: str = "balanced"
    # --- optional prompt budget controls (off by default) ---
    prompt_budget_enabled: bool = False
    prompt_budget_max_input_tokens: int = 3200
    prompt_budget_output_reserve: int = 1000
    prompt_budget_max_tools_total: int = 12

    # provider sub-configs
    openai: OpenAIProviderConfig = OpenAIProviderConfig()
    google: GoogleProviderConfig = GoogleProviderConfig()
    ollama: OllamaProviderConfig = OllamaProviderConfig()
    mlx: MLXProviderConfig = MLXProviderConfig()
    unsloth: UnslothProviderConfig = UnslothProviderConfig()

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

    @field_validator("compaction_threshold")
    @classmethod
    def validate_compaction_threshold(cls, v: float) -> float:
        if not 0.40 <= v <= 0.90:
            raise ValueError("compaction_threshold must be between 0.40 and 0.90")
        return v

    @field_validator("policy_profile")
    @classmethod
    def validate_policy_profile(cls, v: str) -> str:
        allowed = {"strict", "balanced", "dev"}
        if v not in allowed:
            raise ValueError(f"policy_profile must be one of {allowed}")
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
    for section in ("openai", "google", "ollama", "unsloth"):
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
    # Dedicated ~/.tau/theme.toml overlays the [theme] section in config.toml.
    # Useful for hot-reloading themes without touching the main config.
    theme_raw = _load_toml(THEME_PATH)
    if theme_raw:
        init["theme"] = {**init.get("theme", {}), **theme_raw}
    # Apply preset: preset values are the base; explicit keys override them.
    theme_dict = init.get("theme", {})
    preset_name = theme_dict.get("preset", "")
    if preset_name:
        if preset_name not in THEME_PRESETS:
            logger.warning("Unknown theme preset %r — ignoring. Available: %s",
                           preset_name, ", ".join(THEME_PRESETS))
        else:
            # Preset supplies defaults; any explicit key in theme_dict wins.
            init["theme"] = {**THEME_PRESETS[preset_name], **theme_dict}
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


def get_theme_file_paths() -> list[Path]:
    """Return paths that affect the active theme (for hot-reload watchers)."""
    paths = [CONFIG_PATH]
    if THEME_PATH.exists():
        paths.append(THEME_PATH)
    return paths


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

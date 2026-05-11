"""LLM provider adapters."""

from __future__ import annotations

from tau.core.types import AgentConfig
from tau.config import TauConfig


def get_provider(config: TauConfig, agent_config: AgentConfig):
    """Factory — return the correct provider instance based on config."""
    name = agent_config.provider.lower()

    if name == "openai":
        from tau.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(config, agent_config)

    if name == "google":
        from tau.providers.google_provider import GoogleProvider
        return GoogleProvider(config, agent_config)

    if name == "anthropic":
        from tau.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(config, agent_config)

    if name == "ollama":
        from tau.providers.ollama_provider import OllamaProvider
        return OllamaProvider(config, agent_config)

    if name == "mlx":
        from tau.providers.mlx_provider import MLXProvider
        return MLXProvider(config, agent_config)

    if name == "unsloth":
        from tau.providers.unsloth_provider import UnslothProvider
        return UnslothProvider(config, agent_config)

    raise ValueError(
        f"Unknown provider {name!r}. Available: openai, google, anthropic, ollama, mlx, unsloth"
    )

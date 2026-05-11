from __future__ import annotations

import types

from tau.config import TauConfig
from tau.config import load_config
from tau.core.types import AgentConfig
from tau.providers import get_provider


def test_get_provider_dispatches_anthropic(monkeypatch):
    fake_mod = types.ModuleType("tau.providers.anthropic_provider")

    class FakeAnthropicProvider:
        def __init__(self, config, agent_config):
            self.config = config
            self.agent_config = agent_config

    fake_mod.AnthropicProvider = FakeAnthropicProvider
    import sys
    monkeypatch.setitem(sys.modules, "tau.providers.anthropic_provider", fake_mod)

    cfg = TauConfig()
    agent_cfg = AgentConfig(provider="anthropic", model="claude-sonnet-4-20250514")
    provider = get_provider(cfg, agent_cfg)
    assert isinstance(provider, FakeAnthropicProvider)


def test_get_provider_unknown_lists_anthropic():
    cfg = TauConfig()
    agent_cfg = AgentConfig(provider="unknown-x", model="x")
    try:
        get_provider(cfg, agent_cfg)
    except ValueError as exc:
        msg = str(exc)
        assert "anthropic" in msg
    else:
        raise AssertionError("Expected ValueError")


def test_load_config_reads_anthropic_provider_section(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[defaults]\nprovider = "anthropic"\nmodel = "claude-sonnet-4-20250514"\n\n'
        "[providers.anthropic]\n"
        'api_key = "test-key"\n'
        'base_url = "https://api.anthropic.com"\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.provider == "anthropic"
    assert cfg.anthropic.api_key == "test-key"

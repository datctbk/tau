"""Tests for theme configuration and configurable tool set."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tau.config import TauConfig, ThemeConfig, ToolsConfig, load_config
from tau.core.tool_registry import ToolRegistry
from tau.core.types import ToolDefinition, ToolParameter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        parameters={"x": ToolParameter(type="string", description="input")},
        handler=lambda x: f"ok:{x}",
    )


# ===========================================================================
# ThemeConfig
# ===========================================================================

class TestThemeConfig:
    def test_defaults(self):
        t = ThemeConfig()
        assert t.user_color == "cyan"
        assert t.assistant_color == "green"
        assert t.tool_color == "yellow"
        assert t.system_color == "dim"
        assert t.error_color == "red"
        assert t.accent_color == "cyan"
        assert t.success_color == "green"
        assert t.warning_color == "yellow"
        assert t.border_style == "dim"

    def test_custom_values(self):
        t = ThemeConfig(user_color="blue", accent_color="magenta")
        assert t.user_color == "blue"
        assert t.accent_color == "magenta"

    def test_tau_config_includes_theme(self):
        cfg = TauConfig()
        assert isinstance(cfg.theme, ThemeConfig)
        assert cfg.theme.user_color == "cyan"

    def test_theme_from_toml(self, tmp_path: Path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[defaults]\nprovider = "openai"\n\n'
            "[theme]\n"
            'user_color = "blue"\n'
            'accent_color = "magenta"\n'
            'border_style = "bold cyan"\n',
            encoding="utf-8",
        )
        cfg = load_config(toml_path)
        assert cfg.theme.user_color == "blue"
        assert cfg.theme.accent_color == "magenta"
        assert cfg.theme.border_style == "bold cyan"
        # Unchanged defaults
        assert cfg.theme.assistant_color == "green"


class TestThemeObject:
    def test_theme_load_from_config(self):
        from tau.cli import _Theme
        cfg = TauConfig(theme=ThemeConfig(user_color="red", tool_color="magenta"))
        t = _Theme()
        _Theme._loaded = False
        t.load(cfg)
        assert t.user_color == "red"
        assert t.tool_color == "magenta"
        assert t.assistant_color == "green"  # default

    def test_theme_load_once(self):
        from tau.cli import _Theme
        t = _Theme()
        _Theme._loaded = False
        cfg1 = TauConfig(theme=ThemeConfig(user_color="red"))
        cfg2 = TauConfig(theme=ThemeConfig(user_color="blue"))
        t.load(cfg1)
        t.load(cfg2)  # should be no-op
        assert t.user_color == "red"


# ===========================================================================
# ToolsConfig
# ===========================================================================

class TestToolsConfig:
    def test_defaults(self):
        t = ToolsConfig()
        assert t.disabled == []
        assert t.enabled_only == []

    def test_tau_config_includes_tools(self):
        cfg = TauConfig()
        assert isinstance(cfg.tools, ToolsConfig)

    def test_disabled_tools(self):
        reg = ToolRegistry()
        reg.register(_make_tool("read_file"))
        reg.register(_make_tool("write_file"))
        reg.register(_make_tool("run_bash"))

        cfg = ToolsConfig(disabled=["run_bash"])

        # Simulate what _build_agent does
        if cfg.enabled_only:
            allowed = set(cfg.enabled_only)
            for name in list(reg.names()):
                if name not in allowed:
                    reg.unregister(name)
        elif cfg.disabled:
            for name in cfg.disabled:
                reg.unregister(name)

        assert set(reg.names()) == {"read_file", "write_file"}

    def test_enabled_only_tools(self):
        reg = ToolRegistry()
        reg.register(_make_tool("read_file"))
        reg.register(_make_tool("write_file"))
        reg.register(_make_tool("run_bash"))
        reg.register(_make_tool("edit_file"))

        cfg = ToolsConfig(enabled_only=["read_file", "run_bash"])

        if cfg.enabled_only:
            allowed = set(cfg.enabled_only)
            for name in list(reg.names()):
                if name not in allowed:
                    reg.unregister(name)

        assert set(reg.names()) == {"read_file", "run_bash"}

    def test_enabled_only_takes_precedence(self):
        """When enabled_only is set, disabled is ignored."""
        reg = ToolRegistry()
        reg.register(_make_tool("read_file"))
        reg.register(_make_tool("write_file"))
        reg.register(_make_tool("run_bash"))

        cfg = ToolsConfig(disabled=["run_bash"], enabled_only=["read_file"])

        if cfg.enabled_only:
            allowed = set(cfg.enabled_only)
            for name in list(reg.names()):
                if name not in allowed:
                    reg.unregister(name)
        elif cfg.disabled:
            for name in cfg.disabled:
                reg.unregister(name)

        assert set(reg.names()) == {"read_file"}

    def test_tools_config_from_toml(self, tmp_path: Path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[defaults]\nprovider = "openai"\n\n'
            "[tools]\n"
            'disabled = ["run_bash"]\n',
            encoding="utf-8",
        )
        cfg = load_config(toml_path)
        assert cfg.tools.disabled == ["run_bash"]

    def test_tools_enabled_only_from_toml(self, tmp_path: Path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[defaults]\nprovider = "openai"\n\n'
            "[tools]\n"
            'enabled_only = ["read_file", "list_dir"]\n',
            encoding="utf-8",
        )
        cfg = load_config(toml_path)
        assert cfg.tools.enabled_only == ["read_file", "list_dir"]

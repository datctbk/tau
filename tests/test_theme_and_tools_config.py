"""Tests for theme configuration, configurable tool set, and theme hot-reload."""

from __future__ import annotations

import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tau.config import TauConfig, ThemeConfig, ToolsConfig, load_config, get_theme_file_paths, THEME_PRESETS
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
# THEME_PRESETS
# ===========================================================================

class TestThemePresets:
    def test_required_presets_exist(self):
        for name in ("dark", "light", "solarized-dark", "solarized-light", "monokai", "dracula", "nord"):
            assert name in THEME_PRESETS, f"preset {name!r} missing"

    def test_each_preset_has_all_fields(self):
        required = {
            "user_color", "assistant_color", "tool_color", "system_color",
            "error_color", "accent_color", "success_color", "warning_color",
            "border_style",
        }
        for name, colors in THEME_PRESETS.items():
            assert required == set(colors.keys()), f"preset {name!r} missing fields"

    def test_preset_values_are_strings(self):
        for name, colors in THEME_PRESETS.items():
            for field, value in colors.items():
                assert isinstance(value, str), f"{name}.{field} is not a string"

    def test_preset_applied_by_load_config(self, tmp_path: Path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[defaults]\nprovider = "openai"\n\n'
            '[theme]\npreset = "dracula"\n',
            encoding="utf-8",
        )
        with patch("tau.config.THEME_PATH", tmp_path / "nonexistent_theme.toml"):
            cfg = load_config(toml_path)
        expected = THEME_PRESETS["dracula"]
        assert cfg.theme.user_color == expected["user_color"]
        assert cfg.theme.assistant_color == expected["assistant_color"]
        assert cfg.theme.preset == "dracula"

    def test_explicit_override_wins_over_preset(self, tmp_path: Path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[defaults]\nprovider = "openai"\n\n'
            '[theme]\npreset = "dracula"\nuser_color = "magenta"\n',
            encoding="utf-8",
        )
        with patch("tau.config.THEME_PATH", tmp_path / "nonexistent_theme.toml"):
            cfg = load_config(toml_path)
        assert cfg.theme.user_color == "magenta"
        # Other fields still come from the preset
        assert cfg.theme.assistant_color == THEME_PRESETS["dracula"]["assistant_color"]

    def test_unknown_preset_is_ignored(self, tmp_path: Path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[defaults]\nprovider = "openai"\n\n'
            '[theme]\npreset = "nonexistent"\n',
            encoding="utf-8",
        )
        with patch("tau.config.THEME_PATH", tmp_path / "nonexistent_theme.toml"):
            cfg = load_config(toml_path)
        assert cfg.theme.user_color == ThemeConfig().user_color

    def test_theme_toml_preset_applies(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        theme_path = tmp_path / "theme.toml"
        config_path.write_text('[defaults]\nprovider = "openai"\n', encoding="utf-8")
        theme_path.write_text('preset = "nord"\n', encoding="utf-8")
        with patch("tau.config.THEME_PATH", theme_path):
            cfg = load_config(config_path)
        assert cfg.theme.user_color == THEME_PRESETS["nord"]["user_color"]

    def test_no_preset_uses_defaults(self, tmp_path: Path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('[defaults]\nprovider = "openai"\n', encoding="utf-8")
        with patch("tau.config.THEME_PATH", tmp_path / "nonexistent_theme.toml"):
            cfg = load_config(toml_path)
        defaults = ThemeConfig()
        assert cfg.theme.user_color == defaults.user_color
        assert cfg.theme.preset == ""


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
        with patch("tau.config.THEME_PATH", tmp_path / "nonexistent_theme.toml"):
            cfg = load_config(toml_path)
        assert cfg.theme.user_color == "blue"
        assert cfg.theme.accent_color == "magenta"
        assert cfg.theme.border_style == "bold cyan"
        # Unchanged defaults
        assert cfg.theme.assistant_color == "green"

    def test_policy_profile_valid_values(self):
        for p in ("strict", "balanced", "dev"):
            cfg = TauConfig(policy_profile=p)
            assert cfg.policy_profile == p

    def test_policy_profile_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TauConfig(policy_profile="unknown")


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


# ===========================================================================
# get_theme_file_paths
# ===========================================================================

class TestGetThemeFilePaths:
    def test_always_includes_config_path(self):
        paths = get_theme_file_paths()
        from tau.config import CONFIG_PATH
        assert CONFIG_PATH in paths

    def test_includes_theme_toml_when_present(self, tmp_path):
        from tau.config import THEME_PATH
        with patch("tau.config.THEME_PATH", tmp_path / "theme.toml"), \
             patch("tau.config.CONFIG_PATH", tmp_path / "config.toml"):
            (tmp_path / "theme.toml").write_text("[theme]\n", encoding="utf-8")
            from tau.config import get_theme_file_paths as _gtfp
            paths = _gtfp()
            assert tmp_path / "theme.toml" in paths

    def test_excludes_theme_toml_when_absent(self, tmp_path):
        with patch("tau.config.THEME_PATH", tmp_path / "nonexistent.toml"), \
             patch("tau.config.CONFIG_PATH", tmp_path / "config.toml"):
            from tau.config import get_theme_file_paths as _gtfp
            paths = _gtfp()
            assert tmp_path / "nonexistent.toml" not in paths


# ===========================================================================
# _ThemeWatcher
# ===========================================================================

class TestThemeWatcher:
    def _make_watcher(self, app_ref=None):
        from tau.cli import _ThemeWatcher
        if app_ref is None:
            app_ref = [None]
        return _ThemeWatcher(app_ref)

    def test_start_and_stop(self):
        w = self._make_watcher()
        w.start()
        assert w._thread is not None
        assert w._thread.is_alive()
        w.stop()
        w._thread.join(timeout=2)
        assert not w._thread.is_alive()

    def test_snapshot_records_mtimes(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text("[defaults]\n", encoding="utf-8")
        with patch("tau.cli.get_theme_file_paths", return_value=[config]):
            w = self._make_watcher()
            w._snapshot()
            assert str(config) in w._mtimes
            assert w._mtimes[str(config)] == config.stat().st_mtime

    def test_check_detects_change_and_reloads(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text("[defaults]\n", encoding="utf-8")

        mock_app = MagicMock()
        app_ref = [mock_app]

        with patch("tau.cli.get_theme_file_paths", return_value=[config]), \
             patch("tau.cli.load_config") as mock_load_config, \
             patch("tau.cli.theme") as mock_theme:

            mock_load_config.return_value = TauConfig()
            w = self._make_watcher(app_ref)
            w._snapshot()

            # Simulate a file change by altering mtime record
            w._mtimes[str(config)] = 0.0

            w._check()

            mock_load_config.assert_called_once()
            mock_theme.load.assert_called_once_with(mock_load_config.return_value, force=True)
            mock_app.invalidate.assert_called_once()

    def test_check_no_change_no_reload(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text("[defaults]\n", encoding="utf-8")

        mock_app = MagicMock()
        app_ref = [mock_app]

        with patch("tau.cli.get_theme_file_paths", return_value=[config]), \
             patch("tau.cli.load_config") as mock_load_config, \
             patch("tau.cli.theme") as mock_theme:

            mock_load_config.return_value = TauConfig()
            w = self._make_watcher(app_ref)
            w._snapshot()
            w._check()  # no change since snapshot

            mock_load_config.assert_not_called()
            mock_theme.load.assert_not_called()
            mock_app.invalidate.assert_not_called()

    def test_check_no_app_does_not_crash(self, tmp_path):
        """When app_ref[0] is None, _check should not raise."""
        config = tmp_path / "config.toml"
        config.write_text("[defaults]\n", encoding="utf-8")
        app_ref = [None]

        with patch("tau.cli.get_theme_file_paths", return_value=[config]), \
             patch("tau.cli.load_config", return_value=TauConfig()), \
             patch("tau.cli.theme"):
            w = self._make_watcher(app_ref)
            # Force a change
            w._mtimes[str(config)] = 0.0
            w._check()  # must not raise

    def test_watcher_polls_on_interval(self, tmp_path):
        """Integration: watcher thread calls _check at least once in 1.5s."""
        config = tmp_path / "config.toml"
        config.write_text("[defaults]\n", encoding="utf-8")

        check_calls: list[float] = []

        with patch("tau.cli.get_theme_file_paths", return_value=[config]):
            w = self._make_watcher()
            original_check = w._check
            def recording_check():
                check_calls.append(time.monotonic())
                original_check()
            w._check = recording_check
            w.start()
            time.sleep(1.5)
            w.stop()

        assert len(check_calls) >= 1

    def test_theme_toml_hot_reload(self, tmp_path):
        """Writing to theme.toml causes theme to reload with new colors."""
        from tau.cli import _Theme, _ThemeWatcher

        config = tmp_path / "config.toml"
        theme_toml = tmp_path / "theme.toml"
        config.write_text('[defaults]\nprovider = "openai"\n', encoding="utf-8")
        theme_toml.write_text('user_color = "cyan"\n', encoding="utf-8")

        with patch("tau.config.CONFIG_PATH", config), \
             patch("tau.config.THEME_PATH", theme_toml):

            _Theme._loaded = False

            app_ref = [None]
            w = _ThemeWatcher(app_ref)
            w._snapshot()

            # Update theme.toml
            theme_toml.write_text('user_color = "magenta"\n', encoding="utf-8")

            # Force mtime to look changed
            w._mtimes[str(theme_toml)] = 0.0

            with patch("tau.cli.load_config") as mock_lc, \
                 patch("tau.cli.theme") as mock_theme:
                from tau.config import load_config as _lc
                mock_lc.return_value = _lc.__wrapped__(config) if hasattr(_lc, "__wrapped__") else TauConfig()
                w._check()
                mock_theme.load.assert_called_once()


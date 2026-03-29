"""Tests for @file standalone CLI argument support in `tau run`.

These tests cover the argument-splitting logic that enables:
  tau run @code.py "review this"
  tau run @a.py @b.py "compare these"
  tau run @only.py          (no prompt text)
  tau run "normal prompt"   (no @file args, no regression)
  tau run                   (no args → REPL mode, no regression)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import tau.cli as _cli_mod
from tau.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_config():
    cfg = MagicMock()
    cfg.provider = "openai"
    cfg.model = "gpt-4o"
    cfg.max_tokens = 8192
    cfg.max_turns = 10
    cfg.system_prompt = "You are helpful."
    cfg.thinking_budgets.model_dump.return_value = {}
    cfg.trim_strategy = "tail"
    cfg.parallel_tools = False
    cfg.parallel_tools_max_workers = 4
    cfg.max_cost = None
    cfg.shell = MagicMock(require_confirmation=False, timeout=30,
                          allowed_commands=[], use_persistent_shell=False)
    cfg.tools = MagicMock(enabled_only=[], disabled=[])
    cfg.skills = MagicMock(paths=[], disabled=[])
    cfg.extensions = MagicMock(paths=[], disabled=[])
    cfg.ollama = MagicMock(base_url="http://localhost:11434")
    cfg.theme = MagicMock(
        user_color="cyan", assistant_color="green", tool_color="yellow",
        system_color="dim", error_color="red", accent_color="cyan",
        success_color="green", warning_color="yellow", border_style="dim",
    )
    cfg.calculate_cost.return_value = 0.0
    return cfg


# ===========================================================================
# Unit tests: argument splitting logic (pure Python, no Click runner)
# ===========================================================================

class TestArgSplitting:
    """Test the @file / text arg splitting logic directly."""

    def _split(self, args: tuple[str, ...]) -> tuple[list[str], str | None]:
        """Replicate the splitting logic from run_cmd."""
        file_args = [a for a in args if a.startswith("@")]
        text_args  = [a for a in args if not a.startswith("@")]
        prompt: str | None = " ".join(text_args) if text_args else None
        if file_args:
            file_str = " ".join(file_args)
            prompt = (file_str + "\n\n" + prompt) if prompt else file_str
        return file_args, prompt

    def test_no_args_gives_none_prompt(self):
        _, prompt = self._split(())
        assert prompt is None

    def test_plain_text_no_at_files(self):
        file_args, prompt = self._split(("review the code",))
        assert file_args == []
        assert prompt == "review the code"

    def test_single_at_file_no_text(self):
        file_args, prompt = self._split(("@code.py",))
        assert file_args == ["@code.py"]
        assert prompt == "@code.py"

    def test_at_file_and_text(self):
        file_args, prompt = self._split(("@code.py", "review this"))
        assert file_args == ["@code.py"]
        assert "@code.py" in prompt
        assert "review this" in prompt
        # File comes before text
        assert prompt.index("@code.py") < prompt.index("review this")

    def test_multiple_at_files_and_text(self):
        file_args, prompt = self._split(("@a.py", "@b.py", "compare these"))
        assert file_args == ["@a.py", "@b.py"]
        assert "@a.py" in prompt
        assert "@b.py" in prompt
        assert "compare these" in prompt

    def test_multiple_at_files_no_text(self):
        file_args, prompt = self._split(("@a.py", "@b.py"))
        assert file_args == ["@a.py", "@b.py"]
        assert "@a.py" in prompt
        assert "@b.py" in prompt

    def test_at_file_and_text_separator(self):
        """@file block and text block are separated by a blank line."""
        _, prompt = self._split(("@code.py", "review this"))
        assert "\n\n" in prompt

    def test_non_at_prefix_not_treated_as_file(self):
        file_args, prompt = self._split(("some@email.com", "hello"))
        assert file_args == []
        assert "some@email.com" in prompt

    def test_multiple_text_words_joined(self):
        file_args, prompt = self._split(("hello", "world"))
        assert file_args == []
        assert prompt == "hello world"


# ===========================================================================
# Integration: expand_at_files is called on the combined prompt
# ===========================================================================

class TestAtFileExpansionEndToEnd:
    """Verify that @file args in prompt get expanded by expand_at_files."""

    def test_at_file_arg_gets_expanded(self, tmp_path: Path):
        from tau.editor import expand_at_files

        (tmp_path / "code.py").write_text("def hello(): pass\n")

        # Simulate what run_cmd produces:
        prompt = "@code.py\n\nreview this"
        expanded, inlined = expand_at_files(prompt, str(tmp_path))

        assert len(inlined) == 1
        assert "def hello(): pass" in expanded
        assert "<file" in expanded

    def test_multiple_at_file_args_get_expanded(self, tmp_path: Path):
        from tau.editor import expand_at_files

        (tmp_path / "a.py").write_text("class A: pass\n")
        (tmp_path / "b.py").write_text("class B(A): pass\n")

        prompt = "@a.py @b.py\n\ncompare these"
        expanded, inlined = expand_at_files(prompt, str(tmp_path))

        assert len(inlined) == 2
        assert "class A" in expanded
        assert "class B" in expanded

    def test_nonexistent_at_file_arg_left_as_is(self, tmp_path: Path):
        from tau.editor import expand_at_files

        prompt = "@nonexist.py\n\nreview this"
        expanded, inlined = expand_at_files(prompt, str(tmp_path))

        assert len(inlined) == 0
        assert "@nonexist.py" in expanded

    def test_no_at_file_args_no_regression(self, tmp_path: Path):
        from tau.editor import expand_at_files

        prompt = "just a normal question"
        expanded, inlined = expand_at_files(prompt, str(tmp_path))

        assert inlined == []
        assert expanded == prompt

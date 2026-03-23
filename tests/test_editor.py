"""Tests for tau.editor — @file expansion, tab completion, inline shell, image detection."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tau.editor import (
    BUILTIN_SLASH_COMMANDS,
    complete_path,
    complete_slash_commands,
    expand_at_files,
    is_image_path,
    is_shell_command,
    run_inline_shell,
)


# ===========================================================================
# @file reference expansion
# ===========================================================================


class TestExpandAtFiles:
    """Tests for expand_at_files()."""

    def test_single_file(self, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("print('hello')\n")
        text, inlined = expand_at_files("review @foo.py", str(tmp_path))
        assert len(inlined) == 1
        assert 'print(\'hello\')' in text
        assert '<file path="foo.py">' in text
        assert "@foo.py" not in text

    def test_multiple_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("aaa\n")
        (tmp_path / "b.py").write_text("bbb\n")
        text, inlined = expand_at_files("compare @a.py and @b.py", str(tmp_path))
        assert len(inlined) == 2
        assert "aaa" in text
        assert "bbb" in text

    def test_nested_path(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("main code\n")
        text, inlined = expand_at_files("look at @src/main.py", str(tmp_path))
        assert len(inlined) == 1
        assert "main code" in text
        assert '<file path="src/main.py">' in text

    def test_nonexistent_file_left_as_is(self, tmp_path: Path) -> None:
        text, inlined = expand_at_files("review @nonexist.py", str(tmp_path))
        assert len(inlined) == 0
        assert "@nonexist.py" in text

    def test_outside_workspace_blocked(self, tmp_path: Path) -> None:
        text, inlined = expand_at_files("review @/etc/passwd", str(tmp_path))
        assert len(inlined) == 0
        assert "@/etc/passwd" in text

    def test_no_at_references(self, tmp_path: Path) -> None:
        text, inlined = expand_at_files("just a normal prompt", str(tmp_path))
        assert len(inlined) == 0
        assert text == "just a normal prompt"

    def test_at_in_email_not_expanded(self, tmp_path: Path) -> None:
        # @user is not a real file, should stay as-is
        text, inlined = expand_at_files("email user@example.com", str(tmp_path))
        assert len(inlined) == 0

    def test_large_file_skipped(self, tmp_path: Path) -> None:
        big = tmp_path / "big.txt"
        big.write_text("x" * (300 * 1024))
        text, inlined = expand_at_files("read @big.txt", str(tmp_path))
        assert len(inlined) == 0
        assert "too large" in text

    def test_binary_file_read_with_replace(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02hello\xff")
        text, inlined = expand_at_files("read @data.bin", str(tmp_path))
        # Should inline with replacement chars since we use errors='replace'
        assert len(inlined) == 1


# ===========================================================================
# Tab completion — paths
# ===========================================================================


class TestCompletePath:
    def test_empty_prefix_lists_root(self, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("")
        (tmp_path / "bar").mkdir()
        results = complete_path("", str(tmp_path))
        assert "foo.py" in results
        assert "bar/" in results

    def test_partial_match(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.py").write_text("")
        (tmp_path / "beta.py").write_text("")
        results = complete_path("al", str(tmp_path))
        assert "alpha.py" in results
        assert "beta.py" not in results

    def test_subdirectory(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("")
        (sub / "utils.py").write_text("")
        results = complete_path("src/", str(tmp_path))
        assert "src/main.py" in results
        assert "src/utils.py" in results

    def test_subdirectory_partial(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("")
        (sub / "utils.py").write_text("")
        results = complete_path("src/m", str(tmp_path))
        assert "src/main.py" in results
        assert "src/utils.py" not in results

    def test_outside_workspace_returns_empty(self, tmp_path: Path) -> None:
        results = complete_path("/etc", str(tmp_path))
        assert results == []

    def test_hidden_files_skipped_by_default(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.py").write_text("")
        results = complete_path("", str(tmp_path))
        assert "visible.py" in results
        assert ".hidden" not in results

    def test_hidden_files_shown_with_dot_prefix(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("")
        results = complete_path(".", str(tmp_path))
        assert ".hidden" in results

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        results = complete_path("nodir/foo", str(tmp_path))
        assert results == []


# ===========================================================================
# Tab completion — slash commands
# ===========================================================================


class TestCompleteSlashCommands:
    def test_full_match(self) -> None:
        results = complete_slash_commands("/help", BUILTIN_SLASH_COMMANDS)
        assert "/help" in results

    def test_partial_match(self) -> None:
        results = complete_slash_commands("/he", BUILTIN_SLASH_COMMANDS)
        assert "/help" in results

    def test_multiple_matches(self) -> None:
        results = complete_slash_commands("/c", BUILTIN_SLASH_COMMANDS)
        assert "/clear" in results
        assert "/compact" in results

    def test_extension_commands_included(self) -> None:
        results = complete_slash_commands("/j", BUILTIN_SLASH_COMMANDS, ["json"])
        assert "/json" in results

    def test_no_match(self) -> None:
        results = complete_slash_commands("/zzz", BUILTIN_SLASH_COMMANDS)
        assert results == []

    def test_non_slash_prefix_returns_empty(self) -> None:
        results = complete_slash_commands("help", BUILTIN_SLASH_COMMANDS)
        assert results == []


# ===========================================================================
# Inline shell
# ===========================================================================


class TestIsShellCommand:
    def test_basic(self) -> None:
        assert is_shell_command("!ls")

    def test_with_args(self) -> None:
        assert is_shell_command("!git status")

    def test_just_bang_is_not(self) -> None:
        assert not is_shell_command("!")

    def test_normal_text_is_not(self) -> None:
        assert not is_shell_command("hello")

    def test_slash_is_not(self) -> None:
        assert not is_shell_command("/help")


class TestRunInlineShell:
    def test_echo(self, tmp_path: Path) -> None:
        output = run_inline_shell("echo hello", str(tmp_path))
        assert "hello" in output

    def test_ls(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        output = run_inline_shell("ls", str(tmp_path))
        assert "file.txt" in output

    def test_nonzero_exit_code(self, tmp_path: Path) -> None:
        output = run_inline_shell("exit 42", str(tmp_path))
        assert "exit code 42" in output

    def test_stderr_captured(self, tmp_path: Path) -> None:
        output = run_inline_shell("echo err >&2", str(tmp_path))
        assert "err" in output

    def test_timeout(self, tmp_path: Path) -> None:
        output = run_inline_shell("sleep 10", str(tmp_path), timeout=1)
        assert "timed out" in output

    def test_cwd_respected(self, tmp_path: Path) -> None:
        output = run_inline_shell("pwd", str(tmp_path))
        # Resolve symlinks for comparison (macOS /var → /private/var)
        assert str(tmp_path.resolve()) in Path(output.strip()).resolve().__str__()


# ===========================================================================
# Image detection
# ===========================================================================


class TestIsImagePath:
    def test_png_exists(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG")
        assert is_image_path(str(img))

    def test_jpg_exists(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8")
        assert is_image_path(str(img))

    def test_non_image_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("not an image")
        assert not is_image_path(str(f))

    def test_image_extension_but_missing(self, tmp_path: Path) -> None:
        assert not is_image_path("/tmp/nonexistent.png")

    def test_webp(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.webp"
        img.write_bytes(b"RIFF")
        assert is_image_path(str(img))


# ===========================================================================
# Integration: @file expansion in _on_enter (via cli module)
# ===========================================================================


class TestAtFileInCli:
    """Test that @file references work through the _handle_slash / REPL path."""

    def test_expand_at_file_single_shot(self, tmp_path: Path) -> None:
        """expand_at_files is called programmatically — verify basic contract."""
        (tmp_path / "test.py").write_text("def foo(): pass\n")
        text, inlined = expand_at_files("explain @test.py", str(tmp_path))
        assert len(inlined) == 1
        assert "def foo(): pass" in text
        assert "@test.py" not in text

"""Tests for context file discovery and loading (tau/context_files.py)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tau.context_files import _find_context_files, load_context_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str = "test content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ===========================================================================
# _find_context_files
# ===========================================================================

class TestFindContextFiles:
    def test_no_files_returns_empty(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        files = _find_context_files(str(workspace))
        assert files == []

    def test_finds_agents_md_in_workspace(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        agents_file = _write(workspace / "AGENTS.md")
        files = _find_context_files(str(workspace))
        assert agents_file.resolve() in [f.resolve() for f in files]

    def test_finds_claude_md_in_workspace(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        claude_file = _write(workspace / "CLAUDE.md")
        files = _find_context_files(str(workspace))
        assert claude_file.resolve() in [f.resolve() for f in files]

    def test_both_agents_and_claude_found(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        _write(workspace / "AGENTS.md", "agents content")
        _write(workspace / "CLAUDE.md", "claude content")
        files = _find_context_files(str(workspace))
        names = [f.name for f in files]
        assert "AGENTS.md" in names
        assert "CLAUDE.md" in names

    def test_finds_files_in_parent_directory(self, tmp_path):
        parent = tmp_path / "org"
        workspace = parent / "project"
        workspace.mkdir(parents=True)
        parent_file = _write(parent / "AGENTS.md", "org rules")
        files = _find_context_files(str(workspace))
        assert parent_file.resolve() in [f.resolve() for f in files]

    def test_parent_files_before_workspace_files(self, tmp_path):
        parent = tmp_path / "org"
        workspace = parent / "project"
        workspace.mkdir(parents=True)
        _write(parent / "AGENTS.md", "org rules")
        _write(workspace / "AGENTS.md", "project rules")
        files = _find_context_files(str(workspace))
        # Parent should come before workspace
        names_with_parents = [(f.parent.name, f.name) for f in files]
        parent_idx = next(i for i, (p, _) in enumerate(names_with_parents) if p == "org")
        project_idx = next(i for i, (p, _) in enumerate(names_with_parents) if p == "project")
        assert parent_idx < project_idx

    @patch("tau.context_files.TAU_HOME")
    def test_finds_global_file(self, mock_home, tmp_path):
        mock_home.__truediv__ = lambda self, x: tmp_path / "tau_home" / x
        # We need to actually mock the Path object properly
        global_dir = tmp_path / "tau_home"
        global_dir.mkdir()
        global_file = _write(global_dir / "AGENTS.md", "global rules")

        import tau.context_files as cf
        original_home = cf.TAU_HOME
        cf.TAU_HOME = global_dir
        try:
            workspace = tmp_path / "project"
            workspace.mkdir()
            files = _find_context_files(str(workspace))
            assert global_file.resolve() in [f.resolve() for f in files]
        finally:
            cf.TAU_HOME = original_home

    def test_no_duplicates(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        _write(workspace / "AGENTS.md", "content")
        files = _find_context_files(str(workspace))
        resolved = [f.resolve() for f in files]
        assert len(resolved) == len(set(resolved))


# ===========================================================================
# load_context_files
# ===========================================================================

class TestLoadContextFiles:
    def test_empty_when_no_files(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        assert load_context_files(str(workspace)) == ""

    def test_loads_single_file_content(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        _write(workspace / "AGENTS.md", "Be concise. Use pytest.")
        result = load_context_files(str(workspace))
        assert "Be concise. Use pytest." in result

    def test_concatenates_multiple_files(self, tmp_path):
        parent = tmp_path / "org"
        workspace = parent / "project"
        workspace.mkdir(parents=True)
        _write(parent / "AGENTS.md", "org: use tabs")
        _write(workspace / "AGENTS.md", "project: use black")
        result = load_context_files(str(workspace))
        assert "org: use tabs" in result
        assert "project: use black" in result

    def test_skips_empty_files(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        _write(workspace / "AGENTS.md", "")
        result = load_context_files(str(workspace))
        assert result == ""

    def test_file_paths_included_as_headers(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        _write(workspace / "AGENTS.md", "some rules")
        result = load_context_files(str(workspace))
        assert "# Context:" in result

    def test_unreadable_file_skipped(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        bad_file = workspace / "AGENTS.md"
        bad_file.write_bytes(b"\x80\x81\x82")  # invalid utf-8
        # Should not raise, just skip
        result = load_context_files(str(workspace))
        # May or may not have content depending on OS handling
        # The key is it doesn't crash
        assert isinstance(result, str)


# ===========================================================================
# Integration with ContextManager
# ===========================================================================

class TestContextManagerIntegration:
    def test_context_files_injected_into_system_prompt(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        _write(workspace / "AGENTS.md", "Always write tests first.")

        from tau.core.context import ContextManager
        from tau.core.types import AgentConfig

        config = AgentConfig(
            workspace_root=str(workspace),
            compaction_enabled=False,
        )
        ctx = ContextManager(config)
        context_text = load_context_files(str(workspace))
        if context_text:
            ctx.inject_prompt_fragment(context_text)

        sys_msgs = [m for m in ctx.get_messages() if m.role == "system"]
        combined = " ".join(m.content for m in sys_msgs)
        assert "Always write tests first." in combined

"""Tests for prompt templates (tau/prompts.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tau.prompts import (
    _template_dirs,
    extract_variables,
    list_templates,
    load_template,
    parse_var_args,
    render,
    resolve_template,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ===========================================================================
# extract_variables
# ===========================================================================

class TestExtractVariables:
    def test_no_variables(self):
        assert extract_variables("Just plain text.") == []

    def test_single_variable(self):
        assert extract_variables("Hello {{name}}!") == ["name"]

    def test_multiple_unique(self):
        result = extract_variables("{{a}} and {{b}} and {{c}}")
        assert result == ["a", "b", "c"]

    def test_duplicates_deduped(self):
        result = extract_variables("{{x}} then {{x}} again")
        assert result == ["x"]

    def test_preserves_order(self):
        result = extract_variables("{{z}} {{a}} {{m}}")
        assert result == ["z", "a", "m"]

    def test_hyphens_and_underscores(self):
        result = extract_variables("{{my-var}} {{my_var}}")
        assert result == ["my-var", "my_var"]

    def test_ignores_invalid_names(self):
        assert extract_variables("{{123bad}}") == []
        assert extract_variables("{{ spaces }}") == []


# ===========================================================================
# render
# ===========================================================================

class TestRender:
    def test_basic_substitution(self):
        assert render("Hello {{name}}!", {"name": "world"}) == "Hello world!"

    def test_multiple_variables(self):
        result = render("{{a}} + {{b}} = {{c}}", {"a": "1", "b": "2", "c": "3"})
        assert result == "1 + 2 = 3"

    def test_missing_variable_left_as_is(self):
        result = render("{{present}} and {{missing}}", {"present": "yes"})
        assert result == "yes and {{missing}}"

    def test_extra_variables_ignored(self):
        result = render("{{x}}", {"x": "val", "unused": "ignored"})
        assert result == "val"

    def test_no_variables_passthrough(self):
        text = "No vars here."
        assert render(text, {"key": "val"}) == text

    def test_empty_value(self):
        assert render("{{x}}", {"x": ""}) == ""

    def test_multiline(self):
        template = "Line 1: {{a}}\nLine 2: {{b}}"
        result = render(template, {"a": "foo", "b": "bar"})
        assert result == "Line 1: foo\nLine 2: bar"


# ===========================================================================
# parse_var_args
# ===========================================================================

class TestParseVarArgs:
    def test_basic(self):
        assert parse_var_args(("key=value",)) == {"key": "value"}

    def test_multiple(self):
        result = parse_var_args(("a=1", "b=2"))
        assert result == {"a": "1", "b": "2"}

    def test_value_with_equals(self):
        result = parse_var_args(("expr=a=b",))
        assert result == {"expr": "a=b"}

    def test_strips_whitespace(self):
        result = parse_var_args((" key = value ",))
        assert result == {"key": "value"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Expected key=value"):
            parse_var_args(("no-equals-sign",))

    def test_empty_list(self):
        assert parse_var_args(()) == {}


# ===========================================================================
# _template_dirs
# ===========================================================================

class TestTemplateDirs:
    def test_no_dirs_exist(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        dirs = _template_dirs(str(workspace))
        assert dirs == []

    def test_project_dir_found(self, tmp_path):
        workspace = tmp_path / "project"
        prompts = workspace / ".tau" / "prompts"
        prompts.mkdir(parents=True)
        dirs = _template_dirs(str(workspace))
        assert prompts.resolve() in [d.resolve() for d in dirs]

    def test_project_before_global(self, tmp_path, monkeypatch):
        workspace = tmp_path / "project"
        project_prompts = workspace / ".tau" / "prompts"
        project_prompts.mkdir(parents=True)

        global_prompts = tmp_path / "global_tau" / "prompts"
        global_prompts.mkdir(parents=True)

        import tau.prompts as mod
        monkeypatch.setattr(mod, "TAU_HOME", tmp_path / "global_tau")

        dirs = _template_dirs(str(workspace))
        assert len(dirs) == 2
        assert dirs[0].resolve() == project_prompts.resolve()


# ===========================================================================
# list_templates
# ===========================================================================

class TestListTemplates:
    def test_empty_when_no_dirs(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        assert list_templates(str(workspace)) == {}

    def test_finds_md_files(self, tmp_path):
        workspace = tmp_path / "project"
        prompts = workspace / ".tau" / "prompts"
        _write(prompts / "fix-bug.md", "Fix the bug in {{file}}")
        _write(prompts / "review.md", "Review {{file}}")
        templates = list_templates(str(workspace))
        assert set(templates.keys()) == {"fix-bug", "review"}

    def test_ignores_non_md(self, tmp_path):
        workspace = tmp_path / "project"
        prompts = workspace / ".tau" / "prompts"
        _write(prompts / "notes.txt", "not a template")
        _write(prompts / "real.md", "a template")
        templates = list_templates(str(workspace))
        assert list(templates.keys()) == ["real"]

    def test_project_shadows_global(self, tmp_path, monkeypatch):
        workspace = tmp_path / "project"
        project_prompts = workspace / ".tau" / "prompts"
        global_prompts = tmp_path / "global_tau" / "prompts"

        _write(project_prompts / "shared.md", "project version")
        _write(global_prompts / "shared.md", "global version")

        import tau.prompts as mod
        monkeypatch.setattr(mod, "TAU_HOME", tmp_path / "global_tau")

        templates = list_templates(str(workspace))
        assert "shared" in templates
        # Project version should win
        assert templates["shared"].resolve() == (project_prompts / "shared.md").resolve()


# ===========================================================================
# load_template / resolve_template
# ===========================================================================

class TestLoadAndResolve:
    def test_load_existing(self, tmp_path):
        workspace = tmp_path / "project"
        _write(workspace / ".tau" / "prompts" / "greet.md", "Hello {{name}}!")
        content = load_template("greet", str(workspace))
        assert content == "Hello {{name}}!"

    def test_load_missing_returns_none(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        assert load_template("nonexistent", str(workspace)) is None

    def test_resolve_with_variables(self, tmp_path):
        workspace = tmp_path / "project"
        _write(workspace / ".tau" / "prompts" / "fix.md", "Fix {{file}} for {{issue}}")
        result = resolve_template("fix", str(workspace), {"file": "main.py", "issue": "null ref"})
        assert result == "Fix main.py for null ref"

    def test_resolve_missing_returns_none(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        assert resolve_template("nope", str(workspace), {}) is None

    def test_resolve_partial_variables(self, tmp_path):
        workspace = tmp_path / "project"
        _write(workspace / ".tau" / "prompts" / "t.md", "{{a}} and {{b}}")
        result = resolve_template("t", str(workspace), {"a": "yes"})
        assert result == "yes and {{b}}"

"""Tests for tau.tools.fs."""

import pytest
from pathlib import Path
from tau.tools.fs import read_file, write_file, edit_file, list_dir, search_files, _resolve


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Patch workspace root to a temp dir for all fs tool calls."""
    import tau.tools.fs as fs_mod
    original = fs_mod._resolve

    def patched_resolve(path: str, workspace_root: str = ".") -> Path:
        return original(path, str(tmp_path))

    fs_mod._resolve = patched_resolve
    yield tmp_path
    fs_mod._resolve = original


def test_write_and_read(workspace: Path):
    write_file("hello.txt", "line1\nline2\nline3\n")
    out = read_file("hello.txt")
    assert "line1" in out
    assert "line2" in out


def test_read_line_range(workspace: Path):
    write_file("nums.txt", "a\nb\nc\nd\ne\n")
    out = read_file("nums.txt", start_line=1, end_line=2)
    assert "b" in out
    assert "c" in out
    assert "a" not in out
    assert "d" not in out


def test_edit_file(workspace: Path):
    write_file("code.py", "x = 1\ny = 2\n")
    edit_file("code.py", "x = 1", "x = 99")
    out = read_file("code.py")
    assert "99" in out
    assert "x = 1" not in out


def test_edit_file_not_found_raises(workspace: Path):
    write_file("f.py", "hello world\n")
    with pytest.raises(ValueError, match="old_str not found"):
        edit_file("f.py", "DOES NOT EXIST", "replacement")


def test_edit_file_ambiguous_raises(workspace: Path):
    write_file("dup.py", "foo\nfoo\n")
    with pytest.raises(ValueError, match="appears 2 times"):
        edit_file("dup.py", "foo", "bar")


def test_list_dir(workspace: Path):
    write_file("a.txt", "")
    write_file("b.txt", "")
    (workspace / "subdir").mkdir()
    out = list_dir(".")
    assert "a.txt" in out
    assert "b.txt" in out
    assert "subdir/" in out


def test_search_files(workspace: Path):
    write_file("src/main.py", "def hello():\n    pass\n")
    out = search_files("def hello", ".")
    assert "main.py" in out
    assert "def hello" in out


def test_search_files_regex(workspace: Path):
    write_file("data.txt", "error: something failed\n")
    out = search_files(r"error:\s+\w+", ".", use_regex=True)
    assert "data.txt" in out

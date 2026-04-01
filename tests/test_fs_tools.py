"""Tests for tau.tools.fs."""

import pytest
from pathlib import Path
from tau.tools.fs import read_file, write_file, edit_file, list_dir, search_files, grep, find, ls, _resolve


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


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------

def test_grep_basic(workspace: Path):
    write_file("app.py", "def foo():\n    return 1\n")
    out = grep("def foo", ".")
    assert "app.py" in out
    assert "def foo" in out


def test_grep_line_number(workspace: Path):
    write_file("app.py", "line1\nTARGET\nline3\n")
    out = grep("TARGET", ".")
    assert ":2:" in out


def test_grep_case_insensitive(workspace: Path):
    write_file("readme.txt", "Hello World\n")
    out = grep("hello world", ".", case_insensitive=True)
    assert "readme.txt" in out


def test_grep_case_sensitive_no_match(workspace: Path):
    write_file("readme.txt", "Hello World\n")
    out = grep("hello world", ".", case_insensitive=False)
    assert "No matches" in out


def test_grep_include_filter(workspace: Path):
    write_file("code.py", "import os\n")
    write_file("notes.txt", "import note\n")
    out = grep("import", ".", include=r"\.py$")
    assert "code.py" in out
    assert "notes.txt" not in out


def test_grep_max_results(workspace: Path):
    write_file("big.txt", ("match\n" * 50))
    out = grep("match", ".", max_results=5)
    assert "truncated at 5" in out


def test_grep_invalid_regex(workspace: Path):
    out = grep("[invalid(", ".")
    assert "Invalid regex" in out


def test_grep_no_match(workspace: Path):
    write_file("f.txt", "nothing here\n")
    out = grep("ZZZNOMATCH", ".")
    assert "No matches" in out


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------

def test_find_all(workspace: Path):
    write_file("a.py", "")
    write_file("sub/b.txt", "")
    out = find(".")
    assert "a.py" in out
    assert "b.txt" in out


def test_find_by_name(workspace: Path):
    write_file("foo.py", "")
    write_file("bar.txt", "")
    out = find(".", name=r"\.py$")
    assert "foo.py" in out
    assert "bar.txt" not in out


def test_find_files_only(workspace: Path):
    write_file("file.txt", "")
    (workspace / "mydir").mkdir()
    out = find(".", type="f")
    assert "file.txt" in out
    assert "mydir" not in out


def test_find_dirs_only(workspace: Path):
    write_file("file.txt", "")
    (workspace / "mydir").mkdir()
    out = find(".", type="d")
    assert "mydir" in out
    assert "file.txt" not in out


def test_find_max_depth(workspace: Path):
    write_file("top.txt", "")
    write_file("sub/deep.txt", "")
    out = find(".", max_depth=0)
    assert "top.txt" in out
    assert "deep.txt" not in out


def test_find_no_match(workspace: Path):
    write_file("f.py", "")
    out = find(".", name="ZZZNOMATCH")
    assert "No matches" in out


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------

def test_ls_basic(workspace: Path):
    write_file("x.txt", "")
    (workspace / "subdir").mkdir()
    out = ls(".")
    assert "x.txt" in out
    assert "subdir/" in out


def test_ls_hidden_excluded_by_default(workspace: Path):
    write_file(".hidden", "")
    write_file("visible.txt", "")
    out = ls(".")
    assert "visible.txt" in out
    assert ".hidden" not in out


def test_ls_hidden_included_with_all(workspace: Path):
    write_file(".hidden", "")
    out = ls(".", all=True)
    assert ".hidden" in out


def test_ls_long_format(workspace: Path):
    write_file("f.txt", "hello")
    out = ls(".", long=True)
    # long format includes permissions and size
    assert "f.txt" in out
    assert "5" in out  # size of "hello"


def test_ls_empty_dir(workspace: Path):
    (workspace / "empty").mkdir()
    out = ls("empty")
    assert out == "(empty)"


def test_ls_not_a_dir(workspace: Path):
    write_file("f.txt", "")
    with pytest.raises(NotADirectoryError):
        ls("f.txt")

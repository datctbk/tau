from __future__ import annotations

from tau.core.chunker import chunk_file


def test_chunk_python_ast_first():
    text = (
        "class A:\n"
        "    def m(self):\n"
        "        return 1\n\n"
        "def f(x):\n"
        "    return x + 1\n"
    )
    chunks = chunk_file("src/sample.py", text)
    names = [c.name for c in chunks]
    kinds = [c.kind for c in chunks]
    assert "A" in names
    assert "f" in names
    assert "class" in kinds
    assert "function" in kinds


def test_chunk_js_regex_decls():
    text = (
        "export function add(a, b) {\n"
        "  return a + b;\n"
        "}\n\n"
        "class C {\n"
        "  m() { return 1; }\n"
        "}\n"
    )
    chunks = chunk_file("web/mod.js", text)
    assert any(c.name == "add" for c in chunks)
    assert any(c.name == "C" and c.kind == "class" for c in chunks)


def test_chunk_ts_regex_decls():
    text = (
        "export async function run(v: number): Promise<number> {\n"
        "  return v;\n"
        "}\n"
    )
    chunks = chunk_file("web/mod.ts", text)
    assert len(chunks) >= 1
    assert chunks[0].language == "typescript"


def test_fallback_deterministic_for_unknown_ext():
    text = "\n".join([f"line {i}" for i in range(1, 20)]) + "\n"
    c1 = chunk_file("notes/readme.txt", text)
    c2 = chunk_file("notes/readme.txt", text)
    assert [x.id for x in c1] == [x.id for x in c2]
    assert [x.content_hash for x in c1] == [x.content_hash for x in c2]


def test_chunk_java_decls():
    text = (
        "public class UserService {\n"
        "  public String ping() { return \"ok\"; }\n"
        "}\n\n"
        "public static int sum(int a, int b) {\n"
        "  return a + b;\n"
        "}\n"
    )
    chunks = chunk_file("svc/UserService.java", text)
    assert any(c.name == "UserService" for c in chunks)
    assert any(c.name == "sum" for c in chunks)
    assert all(c.language == "java" for c in chunks)


def test_chunk_go_decls():
    text = (
        "package main\n\n"
        "type User struct {\n"
        "  Name string\n"
        "}\n\n"
        "func (u User) NameLen() int {\n"
        "  return len(u.Name)\n"
        "}\n"
    )
    chunks = chunk_file("main.go", text)
    assert any(c.name == "User" for c in chunks)
    assert any(c.name == "NameLen" for c in chunks)
    assert all(c.language == "go" for c in chunks)


def test_chunk_rust_decls():
    text = (
        "pub struct User { name: String }\n\n"
        "pub fn ping() -> &'static str {\n"
        "  \"ok\"\n"
        "}\n"
    )
    chunks = chunk_file("src/lib.rs", text)
    assert any(c.name == "User" for c in chunks)
    assert any(c.name == "ping" for c in chunks)
    assert all(c.language == "rust" for c in chunks)

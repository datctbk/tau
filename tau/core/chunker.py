"""Deterministic semantic chunking primitives (AST-first + text fallback).

Phase 2 scope:
- Python: stdlib AST-based function/class chunks.
- JavaScript/TypeScript: lightweight regex-based declaration chunks.
- Fallback: deterministic fixed-size line chunks.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTS = {".py", ".js", ".ts", ".java", ".go", ".rs"}


@dataclass(frozen=True)
class CodeChunk:
    id: str
    path: str
    language: str
    kind: str
    name: str
    start_line: int
    end_line: int
    content: str
    content_hash: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_slice(lines: list[str], start_line: int, end_line: int) -> str:
    start = max(1, start_line)
    end = max(start, end_line)
    return "".join(lines[start - 1 : end]).rstrip("\n")


def _mk_chunk(
    *,
    rel_path: str,
    language: str,
    kind: str,
    name: str,
    start_line: int,
    end_line: int,
    content: str,
) -> CodeChunk:
    digest = _sha256_text(content)
    cid = f"{rel_path}:{start_line}-{end_line}:{digest[:16]}"
    return CodeChunk(
        id=cid,
        path=rel_path,
        language=language,
        kind=kind,
        name=name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        content_hash=digest,
    )


def _chunk_python(rel_path: str, text: str) -> list[CodeChunk]:
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except Exception:
        return []
    chunks: list[CodeChunk] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            name = getattr(node, "name", f"{kind}@{start}")
            content = _safe_slice(lines, start, end)
            if not content.strip():
                continue
            chunks.append(
                _mk_chunk(
                    rel_path=rel_path,
                    language="python",
                    kind=kind,
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=content,
                )
            )
    chunks.sort(key=lambda c: (c.start_line, c.end_line, c.name))
    return chunks


_JS_TS_DECL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:(?:async\s+)?function\s+([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*)|const\s+([A-Za-z_$][\w$]*)\s*=\s*\(|let\s+([A-Za-z_$][\w$]*)\s*=\s*\(|var\s+([A-Za-z_$][\w$]*)\s*=\s*\()",
    re.MULTILINE,
)


def _chunk_js_ts(rel_path: str, text: str, language: str) -> list[CodeChunk]:
    lines = text.splitlines(keepends=True)
    line_starts: list[int] = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln))

    def _line_of_offset(off: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return max(1, lo)

    matches = list(_JS_TS_DECL_RE.finditer(text))
    if not matches:
        return []
    chunks: list[CodeChunk] = []
    for i, m in enumerate(matches):
        start_off = m.start()
        end_off = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        start_line = _line_of_offset(start_off)
        end_line = max(start_line, _line_of_offset(max(start_off, end_off - 1)))
        name = next((g for g in m.groups() if g), f"decl@{start_line}")
        # Clip by lines for deterministic boundaries.
        content = _safe_slice(lines, start_line, end_line)
        if not content.strip():
            continue
        kind = "class" if re.search(r"^\s*(?:export\s+)?class\b", m.group(0) or "") else "function"
        chunks.append(
            _mk_chunk(
                rel_path=rel_path,
                language=language,
                kind=kind,
                name=name,
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )
    chunks.sort(key=lambda c: (c.start_line, c.end_line, c.name))
    return chunks


_JAVA_DECL_RE = re.compile(
    r"^\s*(?:public|protected|private|static|final|abstract|sealed|non-sealed|\s)+\s*class\s+([A-Za-z_]\w*)"
    r"|^\s*(?:public|protected|private|static|final|synchronized|native|abstract|\s)+\s*[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)

_GO_DECL_RE = re.compile(
    r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\("
    r"|^\s*type\s+([A-Za-z_]\w*)\s+struct\b",
    re.MULTILINE,
)

_RUST_DECL_RE = re.compile(
    r"^\s*(?:pub\s+)?fn\s+([A-Za-z_]\w*)\s*\("
    r"|^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)


def _chunk_by_decl_regex(rel_path: str, text: str, language: str, decl_re: re.Pattern[str]) -> list[CodeChunk]:
    lines = text.splitlines(keepends=True)
    line_starts: list[int] = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln))

    def _line_of_offset(off: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return max(1, lo)

    matches = list(decl_re.finditer(text))
    if not matches:
        return []
    chunks: list[CodeChunk] = []
    for i, m in enumerate(matches):
        start_off = m.start()
        end_off = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        start_line = _line_of_offset(start_off)
        end_line = max(start_line, _line_of_offset(max(start_off, end_off - 1)))
        name = next((g for g in m.groups() if g), f"decl@{start_line}")
        content = _safe_slice(lines, start_line, end_line)
        if not content.strip():
            continue
        head = (m.group(0) or "").strip()
        kind = "class"
        if language in {"go", "rust"} and ("struct" in head or "enum" in head or "trait" in head or "impl" in head):
            kind = "class"
        elif "class" in head or "struct" in head:
            kind = "class"
        else:
            kind = "function"
        chunks.append(
            _mk_chunk(
                rel_path=rel_path,
                language=language,
                kind=kind,
                name=name,
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )
    chunks.sort(key=lambda c: (c.start_line, c.end_line, c.name))
    return chunks


def _chunk_text_fallback(rel_path: str, text: str, language: str, max_lines: int = 120) -> list[CodeChunk]:
    lines = text.splitlines(keepends=True)
    chunks: list[CodeChunk] = []
    if not lines:
        return chunks
    start = 1
    while start <= len(lines):
        end = min(len(lines), start + max_lines - 1)
        content = _safe_slice(lines, start, end)
        if content.strip():
            chunks.append(
                _mk_chunk(
                    rel_path=rel_path,
                    language=language,
                    kind="text",
                    name=f"lines_{start}_{end}",
                    start_line=start,
                    end_line=end,
                    content=content,
                )
            )
        start = end + 1
    return chunks


def chunk_file(path: str | Path, text: str) -> list[CodeChunk]:
    p = Path(path)
    rel = p.as_posix()
    ext = p.suffix.lower()
    if ext == ".py":
        chunks = _chunk_python(rel, text)
        return chunks if chunks else _chunk_text_fallback(rel, text, "python")
    if ext == ".js":
        chunks = _chunk_js_ts(rel, text, "javascript")
        return chunks if chunks else _chunk_text_fallback(rel, text, "javascript")
    if ext == ".ts":
        chunks = _chunk_js_ts(rel, text, "typescript")
        return chunks if chunks else _chunk_text_fallback(rel, text, "typescript")
    if ext == ".java":
        chunks = _chunk_by_decl_regex(rel, text, "java", _JAVA_DECL_RE)
        return chunks if chunks else _chunk_text_fallback(rel, text, "java")
    if ext == ".go":
        chunks = _chunk_by_decl_regex(rel, text, "go", _GO_DECL_RE)
        return chunks if chunks else _chunk_text_fallback(rel, text, "go")
    if ext == ".rs":
        chunks = _chunk_by_decl_regex(rel, text, "rust", _RUST_DECL_RE)
        return chunks if chunks else _chunk_text_fallback(rel, text, "rust")
    return _chunk_text_fallback(rel, text, "text")

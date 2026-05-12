from __future__ import annotations

import re
from pathlib import Path

from tau.core.chunker import SUPPORTED_EXTS, chunk_file
from tau.core.code_index import detect_workspace_changes, scan_workspace_files


def _tokenize(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(x) >= 2}


def build_lexical_rehydrate_block(
    *,
    query: str,
    workspace_root: str | Path,
    max_chunks: int = 8,
    max_chars_per_chunk: int = 1200,
    max_total_chars: int = 7000,
    max_files_scan: int = 120,
) -> str:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return ""

    root = Path(workspace_root).resolve()
    files = [p for p in scan_workspace_files(root) if p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        return ""

    changed = set()
    try:
        changes, _new_m, _old_m = detect_workspace_changes(root)
        changed = set(changes.changed)
    except Exception:
        changed = set()

    # Prioritize changed files first, then lexical path overlap.
    def _file_priority(p: Path) -> tuple[int, int, str]:
        rel = p.relative_to(root).as_posix()
        rel_tokens = _tokenize(rel)
        overlap = len(rel_tokens & q_tokens)
        is_changed = 1 if rel in changed else 0
        return (is_changed, overlap, rel)

    files = sorted(files, key=_file_priority, reverse=True)[:max_files_scan]

    candidates: list[tuple[float, str, int, int, str]] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for c in chunk_file(rel, text):
            c_tokens = _tokenize(c.content)
            overlap = len(c_tokens & q_tokens)
            if overlap <= 0:
                continue
            score = float(overlap)
            if rel in changed:
                score += 1.5
            score += min(1.0, len(_tokenize(c.name) & q_tokens) * 0.3)
            snippet = c.content.strip()
            if len(snippet) > max_chars_per_chunk:
                snippet = snippet[:max_chars_per_chunk].rstrip() + "\n...[truncated]"
            candidates.append((score, rel, c.start_line, c.end_line, snippet))

    if not candidates:
        return ""
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    lines = ["Rehydrated Code Context (lexical, post-compaction):"]
    used = len(lines[0])
    selected = 0
    for score, rel, sline, eline, snippet in candidates:
        header = f"- {rel}:{sline}-{eline} (score={score:.2f})"
        block = f"{header}\n```text\n{snippet}\n```"
        cost = len(block) + 2
        if used + cost > max_total_chars:
            break
        lines.append(block)
        used += cost
        selected += 1
        if selected >= max_chunks:
            break

    if selected == 0:
        return ""
    return "\n\n".join(lines)


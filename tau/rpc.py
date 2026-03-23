"""tau RPC — JSONL-over-stdio protocol for process integration.

Protocol
--------
Communication uses strict **LF-delimited JSONL** (one JSON object per ``\\n``).

**Client → Server (requests):**

    {"type": "prompt", "text": "...", "images": [...]}
    {"type": "steer",  "text": "..."}
    {"type": "enqueue","text": "..."}
    {"type": "session_info"}
    {"type": "exit"}

**Server → Client (responses):**

    Every agent Event is emitted as-is via ``event.to_dict()`` plus a
    ``{"type": "ready"}`` sentinel after each prompt completes so the client
    knows the agent is idle.  Session info is emitted as
    ``{"type": "session_info", ...}``.

Errors are reported as ``{"type": "error", "message": "..."}``.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from typing import IO, TextIO

from tau.core.types import ErrorEvent, Event
from tau.sdk import TauSession, create_session

logger = logging.getLogger(__name__)


def _write(out: TextIO, obj: dict) -> None:
    """Write a single JSON line, flush immediately."""
    out.write(json.dumps(obj, ensure_ascii=False))
    out.write("\n")
    out.flush()


def _read_request(inp: TextIO) -> dict | None:
    """Read one JSON line from *inp*. Returns None on EOF."""
    line = inp.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


# ---------------------------------------------------------------------------
# RPC server loop
# ---------------------------------------------------------------------------


def run_rpc(
    session: TauSession,
    *,
    inp: TextIO | None = None,
    out: TextIO | None = None,
) -> None:
    """Run the RPC event loop reading from *inp* and writing to *out*.

    Defaults to ``sys.stdin`` / ``sys.stdout``.
    """
    if inp is None:
        inp = sys.stdin
    if out is None:
        out = sys.stdout

    # Signal readiness
    _write(out, {"type": "ready"})

    while True:
        try:
            req = _read_request(inp)
        except json.JSONDecodeError as exc:
            _write(out, {"type": "error", "message": f"Invalid JSON: {exc}"})
            continue

        if req is None:
            # EOF — client disconnected
            break

        req_type = req.get("type")

        if req_type == "exit":
            _write(out, {"type": "exit", "status": "ok"})
            break

        elif req_type == "prompt":
            text = req.get("text", "")
            if not text:
                _write(out, {"type": "error", "message": "prompt requires 'text'"})
                continue
            images = req.get("images")
            try:
                for event in session.prompt(text, images=images):
                    _write(out, event.to_dict())
            except Exception as exc:  # noqa: BLE001
                _write(out, {"type": "error", "message": str(exc)})
            _write(out, {"type": "ready"})

        elif req_type == "steer":
            text = req.get("text", "")
            if text:
                session.steer(text)
                _write(out, {"type": "steer", "status": "ok"})
            else:
                _write(out, {"type": "error", "message": "steer requires 'text'"})

        elif req_type == "enqueue":
            text = req.get("text", "")
            if text:
                session.enqueue(text)
                _write(out, {"type": "enqueue", "status": "ok"})
            else:
                _write(out, {"type": "error", "message": "enqueue requires 'text'"})

        elif req_type == "session_info":
            s = session.session
            _write(out, {
                "type": "session_info",
                "id": s.id,
                "name": s.name,
                "provider": s.config.provider,
                "model": s.config.model,
                "messages": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            })

        else:
            _write(out, {
                "type": "error",
                "message": f"Unknown request type: {req_type!r}",
            })

    session.close()


# ---------------------------------------------------------------------------
# CLI entry helper
# ---------------------------------------------------------------------------


def start_rpc(
    *,
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    workspace: str = ".",
    session_name: str | None = None,
    resume_id: str | None = None,
    in_memory: bool = True,
    shell_confirm: bool = False,
) -> None:
    """Convenience wrapper: create a session and run the RPC loop on stdio."""
    session = create_session(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        workspace=workspace,
        session_name=session_name,
        resume_id=resume_id,
        in_memory=in_memory,
        shell_confirm=shell_confirm,
    )
    run_rpc(session)

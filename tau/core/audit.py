"""Append-only audit logging helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_audit_record(workspace_root: str, event_type: str, payload: dict[str, Any]) -> str:
    root = Path(workspace_root) / ".tau" / "audit"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "assistant-actions.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(target)

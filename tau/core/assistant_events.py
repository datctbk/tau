"""Unified assistant event envelope scaffold (Phase A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

Severity = Literal["debug", "info", "warning", "error"]


@dataclass
class AssistantEventEnvelope:
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = ""
    workflow_id: str | None = None
    severity: Severity = "info"
    family: str = "assistant"
    name: str = "event"
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "severity": self.severity,
            "family": self.family,
            "name": self.name,
            "payload": self.payload,
        }


def make_assistant_event(
    *,
    family: str,
    name: str,
    payload: dict[str, Any],
    session_id: str = "",
    workflow_id: str | None = None,
    severity: Severity = "info",
) -> AssistantEventEnvelope:
    return AssistantEventEnvelope(
        family=family,
        name=name,
        payload=payload,
        session_id=session_id,
        workflow_id=workflow_id,
        severity=severity,
    )


def append_assistant_event(workspace_root: str, event: AssistantEventEnvelope) -> str:
    """Append a unified assistant event envelope to the workspace event log."""
    root = Path(workspace_root) / ".tau" / "events"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "assistant-events.jsonl"
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
    return str(target)

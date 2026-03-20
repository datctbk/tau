"""Session manager — persist and resume agent conversations."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau.core.types import AgentConfig, CompactionEntry, ForkInfo

logger = logging.getLogger(__name__)

TAU_HOME = Path.home() / ".tau"
SESSIONS_DIR = TAU_HOME / "sessions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SessionMeta:
    id: str
    name: str | None
    created_at: str
    updated_at: str
    provider: str
    model: str
    parent_id: str | None = None    # set when this session was forked
    fork_index: int | None = None   # message index it was forked after

    def display(self) -> str:
        label = self.name or "(unnamed)"
        fork_marker = f"  ⎇ fork@{self.fork_index}" if self.parent_id else ""
        return f"{self.id[:8]}  {label:<24}  {self.model:<20}  {self.updated_at[:19]}{fork_marker}"


@dataclass
class Session:
    id: str
    name: str | None
    created_at: str
    updated_at: str
    config: AgentConfig
    messages: list[dict[str, Any]] = field(default_factory=list)  # raw dicts
    compactions: list[dict[str, Any]] = field(default_factory=list)  # compaction history
    parent_id: str | None = None    # set when forked
    fork_index: int | None = None   # message index forked after

    @property
    def meta(self) -> SessionMeta:
        return SessionMeta(
            id=self.id,
            name=self.name,
            created_at=self.created_at,
            updated_at=self.updated_at,
            provider=self.config.provider,
            model=self.config.model,
            parent_id=self.parent_id,
            fork_index=self.fork_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": {
                "provider": self.config.provider,
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "max_turns": self.config.max_turns,
                "system_prompt": self.config.system_prompt,
                "trim_strategy": self.config.trim_strategy,
                "workspace_root": self.config.workspace_root,
            },
            "messages": self.messages,
            "compactions": self.compactions,
            "parent_id": self.parent_id,
            "fork_index": self.fork_index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        cfg_d = d.get("config", {})
        config = AgentConfig(
            provider=cfg_d.get("provider", "openai"),
            model=cfg_d.get("model", "gpt-4o"),
            max_tokens=cfg_d.get("max_tokens", 8192),
            max_turns=cfg_d.get("max_turns", 20),
            system_prompt=cfg_d.get("system_prompt", ""),
            trim_strategy=cfg_d.get("trim_strategy", "sliding_window"),
            workspace_root=cfg_d.get("workspace_root", "."),
        )
        return cls(
            id=d["id"],
            name=d.get("name"),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            config=config,
            messages=d.get("messages", []),
            compactions=d.get("compactions", []),
            parent_id=d.get("parent_id"),
            fork_index=d.get("fork_index"),
        )


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionNotFoundError(Exception):
    pass


class SessionManager:
    def __init__(self, sessions_dir: Path = SESSIONS_DIR) -> None:
        self._dir = sessions_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    # ------------------------------------------------------------------

    def new_session(self, config: AgentConfig, name: str | None = None) -> Session:
        now = _now_iso()
        session = Session(
            id=str(uuid.uuid4()),
            name=name,
            created_at=now,
            updated_at=now,
            config=config,
        )
        self.save(session)
        logger.debug("Created session %s", session.id)
        return session

    def save(self, session: Session, messages: list[dict] | None = None) -> None:
        if messages is not None:
            session.messages = messages
        session.updated_at = _now_iso()
        self._path(session.id).write_text(
            json.dumps(session.to_dict(), indent=2), encoding="utf-8"
        )
        logger.debug("Saved session %s (%d messages)", session.id, len(session.messages))

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            # allow prefix matching (first 8 chars)
            matches = list(self._dir.glob(f"{session_id}*.json"))
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                raise SessionNotFoundError(
                    f"Ambiguous session prefix {session_id!r} — be more specific."
                )
            else:
                raise SessionNotFoundError(f"Session {session_id!r} not found.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session.from_dict(data)

    def list_sessions(self) -> list[SessionMeta]:
        metas: list[SessionMeta] = []
        for p in sorted(self._dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                metas.append(Session.from_dict(data).meta)
            except Exception:  # noqa: BLE001
                logger.warning("Could not parse session file %s", p)
        return metas

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"Session {session_id!r} not found.")
        path.unlink()
        logger.debug("Deleted session %s", session_id)

    def append_compaction(self, session: Session, entry: CompactionEntry) -> None:
        """Record a compaction entry in the session and persist to disk."""
        session.compactions.append({
            "summary": entry.summary,
            "tokens_before": entry.tokens_before,
            "timestamp": entry.timestamp,
        })
        session.updated_at = _now_iso()
        self._path(session.id).write_text(
            json.dumps(session.to_dict(), indent=2), encoding="utf-8"
        )
        logger.debug(
            "Recorded compaction in session %s (tokens_before=%d)",
            session.id, entry.tokens_before,
        )

    def fork(
        self,
        session_id: str,
        fork_index: int,
        name: str | None = None,
    ) -> Session:
        """
        Create a new session that is a copy of *session_id* up to and including
        message at *fork_index* (0-based, among all messages including system).

        The new session records parent_id and fork_index so the tree can be
        reconstructed.  Returns the new forked Session.
        """
        parent = self.load(session_id)

        # Clamp to valid range
        if fork_index < 0 or fork_index >= len(parent.messages):
            raise ValueError(
                f"fork_index {fork_index} is out of range "
                f"(session has {len(parent.messages)} messages, indices 0–{len(parent.messages)-1})."
            )

        now = _now_iso()
        forked = Session(
            id=str(uuid.uuid4()),
            name=name or f"fork of {parent.name or parent.id[:8]} @{fork_index}",
            created_at=now,
            updated_at=now,
            config=parent.config,
            messages=parent.messages[: fork_index + 1],
            compactions=[],          # compaction history is not inherited
            parent_id=parent.id,
            fork_index=fork_index,
        )
        self.save(forked)
        logger.debug(
            "Forked session %s → %s at index %d (%d messages)",
            parent.id, forked.id, fork_index, len(forked.messages),
        )
        return forked

    def get_fork_points(self, session_id: str) -> list[ForkInfo]:
        """
        Return every user message in *session_id* as a potential fork point,
        with its index and a short content preview.
        """
        session = self.load(session_id)
        points: list[ForkInfo] = []
        for i, msg in enumerate(session.messages):
            if msg.get("role") == "user":
                preview = (msg.get("content") or "")[:80].replace("\n", " ")
                points.append(ForkInfo(index=i, content=preview))
        return points

    def list_branches(self, session_id: str) -> list[SessionMeta]:
        """Return all sessions that were forked directly from *session_id*."""
        branches: list[SessionMeta] = []
        for p in self._dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("parent_id") == session_id:
                    branches.append(Session.from_dict(data).meta)
            except Exception:  # noqa: BLE001
                pass
        return sorted(branches, key=lambda m: m.created_at)

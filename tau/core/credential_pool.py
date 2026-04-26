"""Credential pool — multi-key management with failover and rotation.

Supports multiple API keys per provider with configurable selection
strategies:
  - fill_first:  Use highest-priority credential until exhausted
  - round_robin: Rotate through available credentials
  - random:      Random selection from available pool
  - least_used:  Prefer the credential with fewest requests

Exhausted credentials auto-recover after a configurable cooldown TTL.
State is persisted to ~/.tau/credentials/pool.json.
"""

from __future__ import annotations

import json
import logging
import os
import random as _random
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Status constants ────────────────────────────────────────────────────

STATUS_OK = "ok"
STATUS_EXHAUSTED = "exhausted"

SOURCE_MANUAL = "manual"
SOURCE_ENV = "env"

STRATEGY_FILL_FIRST = "fill_first"
STRATEGY_ROUND_ROBIN = "round_robin"
STRATEGY_RANDOM = "random"
STRATEGY_LEAST_USED = "least_used"
SUPPORTED_POOL_STRATEGIES = {
    STRATEGY_FILL_FIRST,
    STRATEGY_ROUND_ROBIN,
    STRATEGY_RANDOM,
    STRATEGY_LEAST_USED,
}

# Cooldown before retrying an exhausted credential (seconds).
EXHAUSTED_TTL_429_SECONDS = 60 * 60  # 1 hour for rate-limited
EXHAUSTED_TTL_DEFAULT_SECONDS = 60 * 60  # 1 hour for quota / billing


# ── Default pool file location ──────────────────────────────────────────

def _get_tau_home() -> Path:
    return Path(os.environ.get("TAU_HOME", Path.home() / ".tau"))


def _get_pool_path() -> Path:
    return _get_tau_home() / "credentials" / "pool.json"


# ── Core data structures ────────────────────────────────────────────────

@dataclass
class PooledCredential:
    """One credential in the pool."""

    provider: str
    id: str
    label: str
    priority: int
    source: str
    api_key: str
    base_url: Optional[str] = None
    last_status: Optional[str] = None
    last_status_at: Optional[float] = None
    last_error_code: Optional[int] = None
    last_error_message: Optional[str] = None
    request_count: int = 0

    @classmethod
    def from_dict(cls, provider: str, payload: Dict[str, Any]) -> "PooledCredential":
        return cls(
            provider=provider,
            id=payload.get("id", uuid.uuid4().hex[:8]),
            label=payload.get("label", provider),
            priority=int(payload.get("priority", 0)),
            source=payload.get("source", SOURCE_MANUAL),
            api_key=payload.get("api_key", ""),
            base_url=payload.get("base_url"),
            last_status=payload.get("last_status"),
            last_status_at=payload.get("last_status_at"),
            last_error_code=payload.get("last_error_code"),
            last_error_message=payload.get("last_error_message"),
            request_count=int(payload.get("request_count", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field_def in fields(self):
            if field_def.name == "provider":
                continue
            value = getattr(self, field_def.name)
            if value is not None:
                result[field_def.name] = value
        return result


def _exhausted_ttl(error_code: Optional[int]) -> int:
    """Return cooldown seconds based on the HTTP status that caused exhaustion."""
    if error_code == 429:
        return EXHAUSTED_TTL_429_SECONDS
    return EXHAUSTED_TTL_DEFAULT_SECONDS


def _exhausted_until(entry: PooledCredential) -> Optional[float]:
    """Return the timestamp when an exhausted entry can be retried."""
    if entry.last_status != STATUS_EXHAUSTED:
        return None
    if entry.last_status_at:
        return entry.last_status_at + _exhausted_ttl(entry.last_error_code)
    return None


# ── Persistence ─────────────────────────────────────────────────────────


def load_pool_data(pool_path: Optional[Path] = None) -> Dict[str, list]:
    """Load the credential pool from disk."""
    path = pool_path or _get_pool_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to load credential pool: %s", e)
        return {}


def save_pool_data(
    data: Dict[str, list], pool_path: Optional[Path] = None,
) -> None:
    """Save the credential pool to disk atomically."""
    path = pool_path or _get_pool_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".pool_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except (OSError, NotImplementedError):
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Pool class ──────────────────────────────────────────────────────────


class CredentialPool:
    """Thread-safe credential pool with strategy-based selection."""

    def __init__(
        self,
        provider: str,
        entries: List[PooledCredential],
        strategy: str = STRATEGY_FILL_FIRST,
        pool_path: Optional[Path] = None,
    ):
        self.provider = provider
        self._entries = sorted(entries, key=lambda e: e.priority)
        self._strategy = (
            strategy if strategy in SUPPORTED_POOL_STRATEGIES
            else STRATEGY_FILL_FIRST
        )
        self._lock = threading.Lock()
        self._rr_index = 0
        self._pool_path = pool_path

    def has_credentials(self) -> bool:
        return bool(self._entries)

    def entries(self) -> List[PooledCredential]:
        return list(self._entries)

    def has_available(self) -> bool:
        """True if at least one entry is not currently in exhaustion cooldown."""
        return bool(self._available_entries())

    def _available_entries(self) -> List[PooledCredential]:
        """Return entries not currently in exhaustion cooldown."""
        now = time.time()
        available: List[PooledCredential] = []
        for entry in self._entries:
            if entry.last_status == STATUS_EXHAUSTED:
                until = _exhausted_until(entry)
                if until is not None and now < until:
                    continue
                # Cooldown expired — auto-recover
                entry = replace(
                    entry,
                    last_status=STATUS_OK,
                    last_status_at=None,
                    last_error_code=None,
                    last_error_message=None,
                )
                self._replace_entry_by_id(entry.id, entry)
            available.append(entry)
        return available

    def _replace_entry_by_id(self, entry_id: str, new: PooledCredential) -> None:
        for idx, e in enumerate(self._entries):
            if e.id == entry_id:
                self._entries[idx] = new
                return

    def _persist(self) -> None:
        data = load_pool_data(self._pool_path)
        data[self.provider] = [e.to_dict() for e in self._entries]
        save_pool_data(data, self._pool_path)

    def select(self) -> Optional[PooledCredential]:
        """Select the next credential according to the pool strategy.

        Returns None if no credentials are available.
        """
        with self._lock:
            available = self._available_entries()
            if not available:
                return None

            if self._strategy == STRATEGY_FILL_FIRST:
                selected = available[0]
            elif self._strategy == STRATEGY_ROUND_ROBIN:
                idx = self._rr_index % len(available)
                selected = available[idx]
                self._rr_index = (self._rr_index + 1) % len(available)
            elif self._strategy == STRATEGY_RANDOM:
                selected = _random.choice(available)
            elif self._strategy == STRATEGY_LEAST_USED:
                selected = min(available, key=lambda e: e.request_count)
            else:
                selected = available[0]

            # Increment request count
            updated = replace(selected, request_count=selected.request_count + 1)
            self._replace_entry_by_id(selected.id, updated)
            return updated

    def mark_exhausted(
        self,
        credential_id: str,
        status_code: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        """Mark a credential as exhausted with an optional error code."""
        with self._lock:
            for idx, entry in enumerate(self._entries):
                if entry.id == credential_id:
                    updated = replace(
                        entry,
                        last_status=STATUS_EXHAUSTED,
                        last_status_at=time.time(),
                        last_error_code=status_code,
                        last_error_message=message,
                    )
                    self._entries[idx] = updated
                    self._persist()
                    return

    def mark_ok(self, credential_id: str) -> None:
        """Mark a credential as healthy after successful use."""
        with self._lock:
            for idx, entry in enumerate(self._entries):
                if entry.id == credential_id:
                    updated = replace(
                        entry,
                        last_status=STATUS_OK,
                        last_status_at=time.time(),
                        last_error_code=None,
                        last_error_message=None,
                    )
                    self._entries[idx] = updated
                    self._persist()
                    return


# ── Pool loading convenience ────────────────────────────────────────────


def load_pool(
    provider: str,
    strategy: str = STRATEGY_FILL_FIRST,
    pool_path: Optional[Path] = None,
) -> CredentialPool:
    """Load a credential pool for a provider from disk."""
    data = load_pool_data(pool_path)
    raw_entries = data.get(provider, [])
    entries = []
    for payload in raw_entries:
        if not isinstance(payload, dict):
            continue
        try:
            entries.append(PooledCredential.from_dict(provider, payload))
        except Exception as e:
            logger.warning("Skipping invalid pool entry for %s: %s", provider, e)
    return CredentialPool(
        provider, entries, strategy=strategy, pool_path=pool_path,
    )

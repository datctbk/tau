"""Tests for tau.core.credential_pool."""

import json
import time

import pytest

from tau.core.credential_pool import (
    CredentialPool,
    PooledCredential,
    STATUS_EXHAUSTED,
    STATUS_OK,
    STRATEGY_FILL_FIRST,
    STRATEGY_LEAST_USED,
    STRATEGY_RANDOM,
    STRATEGY_ROUND_ROBIN,
    load_pool,
    load_pool_data,
    save_pool_data,
)


@pytest.fixture
def pool_path(tmp_path):
    return tmp_path / "credentials" / "pool.json"


def _make_entries(provider="openai", count=3):
    return [
        PooledCredential(
            provider=provider,
            id=f"cred_{i}",
            label=f"Key {i}",
            priority=i,
            source="manual",
            api_key=f"sk-key-{i}",
        )
        for i in range(count)
    ]


class TestPooledCredential:
    def test_from_dict(self):
        c = PooledCredential.from_dict("openai", {
            "id": "abc",
            "label": "My Key",
            "api_key": "sk-123",
            "priority": 5,
        })
        assert c.provider == "openai"
        assert c.id == "abc"
        assert c.api_key == "sk-123"
        assert c.priority == 5

    def test_to_dict(self):
        c = PooledCredential(
            provider="openai", id="abc", label="Key",
            priority=0, source="manual", api_key="sk-123",
        )
        d = c.to_dict()
        assert "provider" not in d  # provider not serialized
        assert d["id"] == "abc"
        assert d["api_key"] == "sk-123"

    def test_roundtrip(self):
        original = PooledCredential(
            provider="openai", id="abc", label="Key",
            priority=0, source="manual", api_key="sk-123",
            base_url="https://api.openai.com",
        )
        d = original.to_dict()
        restored = PooledCredential.from_dict("openai", d)
        assert restored.id == original.id
        assert restored.api_key == original.api_key
        assert restored.base_url == original.base_url


class TestCredentialPoolStrategies:
    def test_fill_first(self):
        entries = _make_entries()
        pool = CredentialPool("openai", entries, strategy=STRATEGY_FILL_FIRST)
        selected = pool.select()
        assert selected.id == "cred_0"
        # Second select also returns first (highest priority)
        selected2 = pool.select()
        assert selected2.id == "cred_0"

    def test_round_robin(self):
        entries = _make_entries()
        pool = CredentialPool("openai", entries, strategy=STRATEGY_ROUND_ROBIN)
        ids = [pool.select().id for _ in range(6)]
        # Should cycle through: 0, 1, 2, 0, 1, 2
        assert ids == ["cred_0", "cred_1", "cred_2", "cred_0", "cred_1", "cred_2"]

    def test_random(self):
        entries = _make_entries(count=5)
        pool = CredentialPool("openai", entries, strategy=STRATEGY_RANDOM)
        # Just verify we get a valid credential
        for _ in range(10):
            selected = pool.select()
            assert selected is not None
            assert selected.api_key.startswith("sk-key-")

    def test_least_used(self):
        entries = _make_entries()
        pool = CredentialPool("openai", entries, strategy=STRATEGY_LEAST_USED)
        # All start at 0 — first select picks the first (min by priority)
        s1 = pool.select()
        assert s1.request_count == 1
        # Now cred_0 has count=1, others have 0 → next should pick cred_1
        s2 = pool.select()
        assert s2.id != s1.id


class TestExhaustion:
    def test_mark_exhausted_skips(self, pool_path):
        entries = _make_entries(count=2)
        pool = CredentialPool("openai", entries, strategy=STRATEGY_FILL_FIRST, pool_path=pool_path)

        # Mark first as exhausted
        pool.mark_exhausted("cred_0", status_code=429, message="Rate limited")

        # Next select should skip exhausted and return second
        selected = pool.select()
        assert selected.id == "cred_1"

    def test_all_exhausted(self, pool_path):
        entries = _make_entries(count=2)
        pool = CredentialPool("openai", entries, strategy=STRATEGY_FILL_FIRST, pool_path=pool_path)
        pool.mark_exhausted("cred_0", status_code=429)
        pool.mark_exhausted("cred_1", status_code=429)

        assert not pool.has_available()
        assert pool.select() is None

    def test_cooldown_recovery(self, pool_path):
        entries = _make_entries(count=1)
        pool = CredentialPool("openai", entries, strategy=STRATEGY_FILL_FIRST, pool_path=pool_path)

        # Mark as exhausted with a timestamp in the past (expired cooldown)
        pool._entries[0] = PooledCredential(
            provider="openai", id="cred_0", label="Key 0",
            priority=0, source="manual", api_key="sk-key-0",
            last_status=STATUS_EXHAUSTED,
            last_status_at=time.time() - 7200,  # 2 hours ago (past 1h TTL)
            last_error_code=429,
        )

        # Should auto-recover
        assert pool.has_available()
        selected = pool.select()
        assert selected is not None
        assert selected.id == "cred_0"

    def test_mark_ok(self, pool_path):
        entries = _make_entries(count=1)
        pool = CredentialPool("openai", entries, strategy=STRATEGY_FILL_FIRST, pool_path=pool_path)
        pool.mark_exhausted("cred_0", status_code=429)
        pool.mark_ok("cred_0")

        selected = pool.select()
        assert selected is not None
        assert selected.id == "cred_0"


class TestPersistence:
    def test_save_and_load(self, pool_path):
        data = {
            "openai": [
                {"id": "a", "label": "Key A", "api_key": "sk-a", "priority": 0, "source": "manual"},
                {"id": "b", "label": "Key B", "api_key": "sk-b", "priority": 1, "source": "manual"},
            ],
        }
        save_pool_data(data, pool_path)
        loaded = load_pool_data(pool_path)
        assert len(loaded["openai"]) == 2
        assert loaded["openai"][0]["api_key"] == "sk-a"

    def test_load_pool_convenience(self, pool_path):
        data = {
            "anthropic": [
                {"id": "x", "label": "My Key", "api_key": "sk-ant", "priority": 0, "source": "manual"},
            ],
        }
        save_pool_data(data, pool_path)
        pool = load_pool("anthropic", pool_path=pool_path)
        assert pool.has_credentials()
        selected = pool.select()
        assert selected.api_key == "sk-ant"

    def test_load_nonexistent(self, pool_path):
        pool = load_pool("openai", pool_path=pool_path)
        assert not pool.has_credentials()
        assert pool.select() is None

    def test_file_permissions(self, pool_path):
        data = {"test": []}
        save_pool_data(data, pool_path)
        import stat
        mode = stat.S_IMODE(pool_path.stat().st_mode)
        assert mode == 0o600


class TestRequestCounting:
    def test_increments_on_select(self):
        entries = _make_entries(count=1)
        pool = CredentialPool("openai", entries, strategy=STRATEGY_FILL_FIRST)
        s1 = pool.select()
        assert s1.request_count == 1
        s2 = pool.select()
        assert s2.request_count == 2

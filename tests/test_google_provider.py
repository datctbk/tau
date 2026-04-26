from __future__ import annotations

from tau.core.types import Message
from tau.providers.google_provider import _split_messages


class _FakePart:
    @staticmethod
    def from_bytes(data, mime_type):
        return {"bytes": len(data), "mime_type": mime_type}


class _FakeGTypes:
    Part = _FakePart


def test_split_messages_does_not_duplicate_last_user_turn():
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="First question"),
        Message(role="assistant", content="First answer"),
        Message(role="user", content="Second question"),
    ]

    _, history, last_user, _ = _split_messages(messages, _FakeGTypes)

    user_parts = [entry for entry in history if entry["role"] == "user"]
    assert len(user_parts) == 1
    assert user_parts[0]["parts"][0]["text"] == "First question"
    assert last_user == "Second question"


def test_split_messages_keeps_single_user_turn_as_last_request():
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Only question"),
    ]

    _, history, last_user, _ = _split_messages(messages, _FakeGTypes)

    assert history == []
    assert last_user == "Only question"
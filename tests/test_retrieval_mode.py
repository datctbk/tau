from __future__ import annotations

from tau.core.retrieval_mode import retrieval_mode_label, semantic_retrieval_enabled


def test_retrieval_mode_defaults_to_lexical(monkeypatch):
    monkeypatch.delenv("TAU_SEMANTIC_RETRIEVAL", raising=False)
    assert semantic_retrieval_enabled() is False
    assert retrieval_mode_label() == "lexical"


def test_retrieval_mode_semantic_when_enabled(monkeypatch):
    monkeypatch.setenv("TAU_SEMANTIC_RETRIEVAL", "1")
    assert semantic_retrieval_enabled() is True
    assert retrieval_mode_label() == "semantic"


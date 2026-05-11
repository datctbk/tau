from __future__ import annotations

import os


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def semantic_retrieval_enabled() -> bool:
    """Master switch for semantic retrieval path.

    Default path remains lexical unless TAU_SEMANTIC_RETRIEVAL=1.
    """
    return _truthy("TAU_SEMANTIC_RETRIEVAL")


def retrieval_mode_label() -> str:
    return "semantic" if semantic_retrieval_enabled() else "lexical"


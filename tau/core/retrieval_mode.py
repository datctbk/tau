from __future__ import annotations

import os


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def semantic_retrieval_enabled() -> bool:
    """Master switch for semantic retrieval path.

    Default path remains lexical unless custom rehydrate providers are registered
    or TAU_SEMANTIC_RETRIEVAL=1 is set.
    """
    from tau.core.rehydrate import _rehydrate_providers
    if len(_rehydrate_providers) > 0:
        return True
    return _truthy("TAU_SEMANTIC_RETRIEVAL")


def retrieval_mode_label() -> str:
    return "semantic" if semantic_retrieval_enabled() else "lexical"


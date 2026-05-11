from __future__ import annotations

import os

from tau.core.retrieval_mode import semantic_retrieval_enabled

def semantic_embeddings_enabled() -> bool:
    """Feature gate for Phase 4 semantic embedding pipeline."""
    # Phase 4 master switch: semantic retrieval must be explicitly enabled.
    if not semantic_retrieval_enabled():
        return False
    return os.getenv("TAU_SEMANTIC_EMBEDDINGS_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class Embedder:
    """Phase 4 placeholder. Explicitly disabled unless feature flag is enabled."""

    def __init__(self, model: str):
        self.model = model
        if not semantic_embeddings_enabled():
            raise RuntimeError(
                "Semantic embeddings are disabled. "
                "Set TAU_SEMANTIC_EMBEDDINGS_ENABLED=1 to enable."
            )

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embedder implementation is planned for Phase 4.")

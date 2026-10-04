"""Embedding-provider contract shared by ingestion (chunk embedding) and
retrieval (query embedding). `app.embeddings.sentence_transformer` is the
only implementation. Kept separate from both `app.retrieval.reranker` and
`app.generation.service`'s LLM provider protocols -- embeddings, reranking,
and generation are three independently configured model classes (see
`docs/PROJECT_DOCUMENTATION.md` section 21 for the full parameter table).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Provides text embeddings for ingestion and retrieval."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""

    @abstractmethod
    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Embed multiple texts."""

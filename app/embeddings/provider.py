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
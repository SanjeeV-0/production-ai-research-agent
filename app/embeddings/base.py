from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for converting text into embedding vectors."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for one piece of text."""

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
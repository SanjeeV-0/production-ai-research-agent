import hashlib
import math
from collections.abc import Sequence

from app.embeddings.provider import EmbeddingProvider


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for unit tests."""

    def __init__(self, dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        self.dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding."""
        values: list[float] = []

        for index in range(self.dimensions):
            digest = hashlib.sha256(
                f"{index}:{text}".encode()
            ).digest()

            value = int.from_bytes(
                digest[:8],
                byteorder="big",
            )

            values.append(
                (value / 2**64) * 2 - 1
            )

        norm = math.sqrt(
            sum(value * value for value in values)
        )

        if norm == 0:
            return [0.0] * self.dimensions

        return [
            value / norm
            for value in values
        ]

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [
            self.embed_text(text)
            for text in texts
        ]
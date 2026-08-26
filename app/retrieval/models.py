from dataclasses import dataclass

from app.core.models import DocumentChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk returned by vector retrieval."""

    chunk: DocumentChunk
    distance: float

    @property
    def similarity(self) -> float:
        """Return cosine similarity derived from cosine distance."""
        return 1.0 - self.distance
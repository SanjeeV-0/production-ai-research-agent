from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk returned by vector retrieval."""

    document_id: UUID
    chunk_id: UUID
    section_id: UUID
    section_path: str
    page_numbers: list[int]
    content: str
    distance: float

    @property
    def similarity(self) -> float:
        """Return cosine similarity derived from cosine distance."""
        return 1.0 - self.distance
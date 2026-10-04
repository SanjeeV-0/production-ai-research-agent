from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk returned by vector retrieval.

    `section_id`/`section_path` are `None` for a headingless document's
    chunks (no matching DocumentSection row; the repository's outer join
    keeps such chunks retrievable rather than excluding them).
    """

    document_id: UUID
    chunk_id: UUID
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    content: str
    distance: float
    rerank_score: float | None = None

    @property
    def similarity(self) -> float:
        """Return cosine similarity derived from cosine distance."""
        return 1.0 - self.distance

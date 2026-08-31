from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class RetrievalTraceCandidate:
    """Trace information for a retrieved candidate."""

    chunk_id: UUID
    document_id: UUID
    section_id: UUID
    section_path: str
    page_numbers: list[int]
    content: str
    distance: float
    rerank_score: float | None = None


@dataclass
class RetrievalTrace:
    """Debug trace for a complete retrieval operation."""

    query: str

    candidate_limit: int
    candidates: list[RetrievalTraceCandidate] = field(
        default_factory=list
    )

    final_results: list[RetrievalTraceCandidate] = field(
        default_factory=list
    )
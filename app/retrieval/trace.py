from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class RetrievalTraceCandidate:
    """Trace information for a retrieved candidate.

    `section_id`/`section_path` are `None` for a headingless document's
    chunks -- see `app.retrieval.models.RetrievedChunk`.
    """

    chunk_id: UUID
    document_id: UUID
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    content: str
    distance: float
    rerank_score: float | None = None


@dataclass
class RetrievalTraceContext:
    """Trace information for context selected for generation."""

    text: str
    sources: list[RetrievalTraceCandidate] = field(default_factory=list)


@dataclass
class RetrievalTrace:
    """Debug trace for a complete retrieval operation."""

    query: str

    candidate_limit: int

    candidates: list[RetrievalTraceCandidate] = field(default_factory=list)

    final_results: list[RetrievalTraceCandidate] = field(default_factory=list)

    context: RetrievalTraceContext | None = None

    original_query: str = ""
    sub_queries: list[str] = field(default_factory=list)
    raw_candidate_count: int = 0
    deduplicated_candidate_count: int = 0

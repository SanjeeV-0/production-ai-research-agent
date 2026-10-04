from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalSearchRequest(BaseModel):
    """Request payload for vector retrieval."""

    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    document_id: UUID | None = None
    section_id: UUID | None = None


class RetrievedChunkResponse(BaseModel):
    """API representation of a retrieved chunk.

    `section_id`/`section_path` are nullable: a headingless document's
    chunks have no section (the repository's outer join to DocumentSection
    keeps them retrievable rather than excluding them), so this must accept
    and serialize that as JSON null rather than rejecting the response.
    """

    document_id: UUID
    chunk_id: UUID
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    content: str
    distance: float
    similarity: float
    rerank_score: float | None


class RetrievalTraceCandidateResponse(BaseModel):
    """API representation of a traced retrieval candidate.

    See `RetrievedChunkResponse` -- `section_id`/`section_path` are
    nullable for the same headingless-document reason.
    """

    document_id: UUID
    chunk_id: UUID
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    content: str
    distance: float
    rerank_score: float | None


class RetrievalTraceContextResponse(BaseModel):
    """API representation of context prepared for generation."""

    text: str
    sources: list[RetrievalTraceCandidateResponse]


class RetrievalTraceResponse(BaseModel):
    """Debug trace for a retrieval operation."""

    query: str
    original_query: str
    sub_queries: list[str]
    candidate_limit: int
    raw_candidate_count: int
    deduplicated_candidate_count: int
    candidates: list[RetrievalTraceCandidateResponse]
    final_results: list[RetrievalTraceCandidateResponse]
    context: RetrievalTraceContextResponse | None = None


class RetrievalSearchResponse(BaseModel):
    """Response payload for vector retrieval."""

    results: list[RetrievedChunkResponse]
    trace: RetrievalTraceResponse | None = None

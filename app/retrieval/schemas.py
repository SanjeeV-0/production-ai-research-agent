from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalSearchRequest(BaseModel):
    """Request payload for vector retrieval."""

    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    document_id: UUID | None = None
    section_id: UUID | None = None


class RetrievedChunkResponse(BaseModel):
    """API representation of a retrieved chunk."""

    document_id: UUID
    chunk_id: UUID
    section_id: UUID
    section_path: str
    page_numbers: list[int]
    content: str
    distance: float
    similarity: float
    rerank_score: float | None


class RetrievalTraceCandidateResponse(BaseModel):
    """API representation of a traced retrieval candidate."""

    document_id: UUID
    chunk_id: UUID
    section_id: UUID
    section_path: str
    page_numbers: list[int]
    content: str
    distance: float
    rerank_score: float | None


class RetrievalTraceResponse(BaseModel):
    """Debug trace for a retrieval operation."""

    query: str
    candidate_limit: int
    candidates: list[RetrievalTraceCandidateResponse]
    final_results: list[RetrievalTraceCandidateResponse]


class RetrievalSearchResponse(BaseModel):
    """Response payload for vector retrieval."""

    results: list[RetrievedChunkResponse]
    trace: RetrievalTraceResponse | None = None
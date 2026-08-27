from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.dependencies import get_retrieval_service
from app.retrieval.schemas import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievedChunkResponse,
)
from app.retrieval.service import RetrievalService

router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
)


@router.post(
    "/search",
    response_model=RetrievalSearchResponse,
)
async def search(
    request: RetrievalSearchRequest,
    retrieval_service: Annotated[
        RetrievalService,
        Depends(get_retrieval_service),
    ],
) -> RetrievalSearchResponse:
    """Search for relevant document chunks."""

    results = await retrieval_service.search(
    query=request.query,
    limit=request.limit,
    document_id=request.document_id,
    section_id=request.section_id,
)

    return RetrievalSearchResponse(
        results=[
            RetrievedChunkResponse(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                section_id=result.section_id,
                section_path=result.section_path,
                page_numbers=result.page_numbers,
                content=result.content,
                distance=result.distance,
                similarity=result.similarity,
                rerank_score=result.rerank_score,
            )
            for result in results
        ],
    )
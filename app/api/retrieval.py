from typing import Annotated

from fastapi import APIRouter, Depends

from app.config.settings import Settings
from app.core.dependencies import get_app_settings, get_retrieval_service
from app.retrieval.schemas import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievalTraceCandidateResponse,
    RetrievalTraceResponse,
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
    ],settings: Annotated[
    Settings,
    Depends(get_app_settings),
],
) -> RetrievalSearchResponse:
    """Search for relevant document chunks."""
    
    results = await retrieval_service.search(
        query=request.query,
        limit=request.limit,
        document_id=request.document_id,
        section_id=request.section_id,
        trace=settings.trace_enabled,
    )

    trace_response = None

    if retrieval_service.last_trace is not None:
        trace_response = RetrievalTraceResponse(
            query=retrieval_service.last_trace.query,
            candidate_limit=retrieval_service.last_trace.candidate_limit,
            candidates=[
                RetrievalTraceCandidateResponse(
                    document_id=candidate.document_id,
                    chunk_id=candidate.chunk_id,
                    section_id=candidate.section_id,
                    section_path=candidate.section_path,
                    page_numbers=candidate.page_numbers,
                    content=candidate.content,
                    distance=candidate.distance,
                    rerank_score=candidate.rerank_score,
                )
                for candidate in retrieval_service.last_trace.candidates
            ],
            final_results=[
                RetrievalTraceCandidateResponse(
                    document_id=candidate.document_id,
                    chunk_id=candidate.chunk_id,
                    section_id=candidate.section_id,
                    section_path=candidate.section_path,
                    page_numbers=candidate.page_numbers,
                    content=candidate.content,
                    distance=candidate.distance,
                    rerank_score=candidate.rerank_score,
                )
                for candidate in retrieval_service.last_trace.final_results
            ],
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
        trace=trace_response,
    )
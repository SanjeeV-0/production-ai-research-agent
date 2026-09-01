from typing import Annotated

from fastapi import APIRouter, Depends

from app.config.settings import Settings
from app.core.dependencies import (
    get_app_settings,
    get_retrieval_service,
)
from app.retrieval.schemas import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievalTraceCandidateResponse,
    RetrievalTraceContextResponse,
    RetrievalTraceResponse,
    RetrievedChunkResponse,
)
from app.retrieval.service import RetrievalService

router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
)


def _trace_candidate_response(
    candidate,
) -> RetrievalTraceCandidateResponse:
    """Convert a trace candidate into an API response."""

    return RetrievalTraceCandidateResponse(
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        section_id=candidate.section_id,
        section_path=candidate.section_path,
        page_numbers=candidate.page_numbers,
        content=candidate.content,
        distance=candidate.distance,
        rerank_score=candidate.rerank_score,
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
    settings: Annotated[
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

    trace = retrieval_service.last_trace

    if trace is not None:
        context_response = None

        if trace.context is not None:
            context_response = RetrievalTraceContextResponse(
                text=trace.context.text,
                sources=[
                    _trace_candidate_response(source)
                    for source in trace.context.sources
                ],
            )

        trace_response = RetrievalTraceResponse(
            query=trace.query,
            candidate_limit=trace.candidate_limit,
            candidates=[
                _trace_candidate_response(candidate)
                for candidate in trace.candidates
            ],
            final_results=[
                _trace_candidate_response(result)
                for result in trace.final_results
            ],
            context=context_response,
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
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.dependencies import (
    get_generation_service,
    get_retrieval_service,
)
from app.generation.context import ContextAssembler
from app.generation.generation_service import GenerationService
from app.observability.langfuse import get_langfuse
from app.retrieval.service import RetrievalService

router = APIRouter(
    prefix="/research",
    tags=["research"],
)


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1)
    trace: bool = False


@router.post("/ask")
async def ask(
    request: ResearchRequest,
    retrieval_service: Annotated[
        RetrievalService,
        Depends(get_retrieval_service),
    ],
    generation_service: Annotated[
        GenerationService,
        Depends(get_generation_service),
    ],
) -> dict[str, object]:
    """Answer a research question using retrieved context."""

    query = request.query
    trace = request.trace
    langfuse = get_langfuse()

    if langfuse is None:
        return await _execute_research(
            query=query,
            retrieval_service=retrieval_service,
            generation_service=generation_service,
            trace=trace,
        )

    with langfuse.start_as_current_observation(
        as_type="span",
        name="research-request",
        input={
            "query": query,
            "trace": trace,
        },
    ) as observation:
        try:
            response = await _execute_research(
                query=query,
                retrieval_service=retrieval_service,
                generation_service=generation_service,
                trace=trace,
            )

            observation.update(
                output={
                    "answer": response["answer"],
                    "model": response["model"],
                    "source_count": len(response["sources"]),
                },
            )

            return response

        except Exception as exc:
            observation.update(
                level="ERROR",
                status_message=str(exc),
            )
            raise


async def _execute_research(
    query: str,
    retrieval_service: RetrievalService,
    generation_service: GenerationService,
    trace: bool = False,
) -> dict[str, object]:
    """Execute retrieval, context assembly, and generation."""

    results = await retrieval_service.search(
        query=query,
        limit=10,
        trace=trace,
    )

    context = ContextAssembler().assemble(results)

    generation_result = await generation_service.generate(
        query=query,
        context=context,
    )

    response: dict[str, object] = {
        "answer": generation_result.text,
        "model": generation_result.model,
        "sources": [
            {
                "document_id": source.document_id,
                "chunk_id": source.chunk_id,
                "section_id": source.section_id,
                "section_path": source.section_path,
                "page_numbers": source.page_numbers,
            }
            for source in context.sources
        ],
    }

    if trace and retrieval_service.last_trace is not None:
        response["trace"] = {
            "query": retrieval_service.last_trace.query,
            "candidate_limit": retrieval_service.last_trace.candidate_limit,
            "candidates": [
                {
                    "document_id": candidate.document_id,
                    "chunk_id": candidate.chunk_id,
                    "section_id": candidate.section_id,
                    "section_path": candidate.section_path,
                    "page_numbers": candidate.page_numbers,
                    "content": candidate.content,
                    "distance": candidate.distance,
                    "rerank_score": candidate.rerank_score,
                }
                for candidate in retrieval_service.last_trace.candidates
            ],
            "final_results": [
                {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "section_id": result.section_id,
                    "section_path": result.section_path,
                    "page_numbers": result.page_numbers,
                    "content": result.content,
                    "distance": result.distance,
                    "rerank_score": result.rerank_score,
                }
                for result in retrieval_service.last_trace.final_results
            ],
            "context": (
                {
                    "text": retrieval_service.last_trace.context.text,
                    "sources": [
                        {
                            "document_id": source.document_id,
                            "chunk_id": source.chunk_id,
                            "section_id": source.section_id,
                            "section_path": source.section_path,
                            "page_numbers": source.page_numbers,
                            "content": source.content,
                            "distance": source.distance,
                            "rerank_score": source.rerank_score,
                        }
                        for source in retrieval_service.last_trace.context.sources
                    ],
                }
                if retrieval_service.last_trace.context is not None
                else None
            ),
        }

    return response
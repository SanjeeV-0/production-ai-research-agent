from typing import Annotated

from fastapi import APIRouter, Depends

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


@router.post("/ask")
async def ask(
    query: str,
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

    langfuse = get_langfuse()

    if langfuse is None:
        return await _execute_research(
            query=query,
            retrieval_service=retrieval_service,
            generation_service=generation_service,
        )

    with langfuse.start_as_current_observation(
        as_type="span",
        name="research-request",
        input={"query": query},
    ) as observation:
        try:
            response = await _execute_research(
                query=query,
                retrieval_service=retrieval_service,
                generation_service=generation_service,
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
) -> dict[str, object]:
    """Execute retrieval, context assembly, and generation."""

    results = await retrieval_service.search(
        query=query,
        limit=10,
    )

    context = ContextAssembler().assemble(results)

    generation_result = await generation_service.generate(
        query=query,
        context=context,
    )

    return {
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
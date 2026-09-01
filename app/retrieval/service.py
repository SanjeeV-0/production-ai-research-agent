from uuid import UUID

from app.core.repositories.document import DocumentRepository
from app.embeddings.provider import EmbeddingProvider
from app.generation.context import ContextAssembler
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import Reranker
from app.retrieval.trace import (
    RetrievalTrace,
    RetrievalTraceCandidate,
    RetrievalTraceContext,
)


class RetrievalService:
    """Retrieves and reranks document chunks."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker | None = None,
        context_assembler: ContextAssembler | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.context_assembler = (
            context_assembler or ContextAssembler()
        )
        self.last_trace: RetrievalTrace | None = None

    async def search(
        self,
        query: str,
        limit: int = 10,
        max_distance: float | None = None,
        document_id: UUID | None = None,
        section_id: UUID | None = None,
        candidate_limit: int | None = None,
        trace: bool = False,
    ) -> list[RetrievedChunk]:
        """Retrieve and optionally rerank document chunks."""

        self.last_trace = None

        if candidate_limit is None:
            candidate_limit = max(limit, 50)

        if candidate_limit < limit:
            raise ValueError(
                "candidate_limit must be greater than or equal to limit."
            )

        query_embedding = self.embedding_provider.embed_text(
            query
        )

        candidates = await self.repository.search_similar_chunks(
            query_embedding=query_embedding,
            limit=candidate_limit,
            max_distance=max_distance,
            document_id=document_id,
            section_id=section_id,
        )

        if self.reranker is None:
            final_results = candidates[:limit]
        else:
            reranked = self.reranker.rerank(
                query,
                candidates,
            )

            final_results = reranked[:limit]

        if trace:
            self.last_trace = RetrievalTrace(
                query=query,
                candidate_limit=candidate_limit,
                candidates=[
                    self._to_trace_candidate(chunk)
                    for chunk in candidates
                ],
                final_results=[
                    self._to_trace_candidate(chunk)
                    for chunk in final_results
                ],
            )

            generation_context = (
                self.context_assembler.assemble(
                    final_results
                )
            )

            self.last_trace.context = RetrievalTraceContext(
                text=generation_context.text,
                sources=[
                    self._to_trace_candidate(
                        chunk
                    )
                    for chunk in final_results
                ],
            )

        return final_results

    @staticmethod
    def _to_trace_candidate(
        chunk: RetrievedChunk,
    ) -> RetrievalTraceCandidate:
        """Convert a retrieved chunk into trace metadata."""

        return RetrievalTraceCandidate(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            page_numbers=chunk.page_numbers,
            content=chunk.content,
            distance=chunk.distance,
            rerank_score=chunk.rerank_score,
        )
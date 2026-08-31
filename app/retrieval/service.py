from uuid import UUID

from app.core.repositories.document import DocumentRepository
from app.embeddings.provider import EmbeddingProvider
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import Reranker
from app.retrieval.trace import (
    RetrievalTrace,
    RetrievalTraceCandidate,
)


class RetrievalService:
    """Retrieves and reranks document chunks."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.reranker = reranker
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
        """Retrieve candidates and optionally rerank them."""

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
            results = candidates[:limit]
        else:
            reranked = self.reranker.rerank(
                query,
                candidates,
            )

            results = reranked[:limit]

        if trace:
            self.last_trace = RetrievalTrace(
                query=query,
                candidate_limit=candidate_limit,
                candidates=[
                    RetrievalTraceCandidate(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        section_id=chunk.section_id,
                        section_path=chunk.section_path,
                        page_numbers=chunk.page_numbers,
                        content=chunk.content,
                        distance=chunk.distance,
                        rerank_score=chunk.rerank_score,
                    )
                    for chunk in candidates
                ],
                final_results=[
                    RetrievalTraceCandidate(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        section_id=chunk.section_id,
                        section_path=chunk.section_path,
                        page_numbers=chunk.page_numbers,
                        content=chunk.content,
                        distance=chunk.distance,
                        rerank_score=chunk.rerank_score,
                    )
                    for chunk in results
                ],
            )

        return results
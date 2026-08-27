from uuid import UUID

from app.core.repositories.document import DocumentRepository
from app.embeddings.provider import EmbeddingProvider
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import Reranker


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

    async def search(
        self,
        query: str,
        limit: int = 10,
        max_distance: float | None = None,
        document_id: UUID | None = None,
        section_id: UUID | None = None,
        candidate_limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve candidates and optionally rerank them."""

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
            return candidates[:limit]

        reranked = self.reranker.rerank(
            query,
            candidates,
        )

        return reranked[:limit]
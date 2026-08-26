from app.core.repositories.document import DocumentRepository
from app.embeddings.provider import EmbeddingProvider
from app.retrieval.models import RetrievedChunk


class RetrievalService:
    """Retrieves document chunks using vector similarity."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider

    async def search(
    self,
    query: str,
    limit: int = 10,
) -> list[RetrievedChunk]:
        """Return chunks most similar to the query."""
        query_embedding = self.embedding_provider.embed_text(
        query
    )

        return await self.repository.search_similar_chunks(
        query_embedding=query_embedding,
        limit=limit,
    )
from uuid import uuid4

import pytest

from app.retrieval.models import RetrievedChunk
from app.retrieval.service import RetrievalService


class FakeEmbeddingProvider:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeRepository:
    def __init__(self, chunks):
        self.chunks = chunks
        self.received_limit = None

    async def search_similar_chunks(
        self,
        query_embedding,
        limit,
        max_distance=None,
        document_id=None,
        section_id=None,
    ):
        self.received_limit = limit
        return self.chunks


class FakeReranker:
    def __init__(self):
        self.received_query = None
        self.received_chunks = None

    def rerank(self, query, chunks):
        self.received_query = query
        self.received_chunks = list(chunks)

        return list(reversed(chunks))


def _chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content=f"chunk {index}",
        distance=0.1,
    )


@pytest.mark.asyncio
async def test_retrieval_service_reranks_candidates() -> None:
    chunks = [
        _chunk(1),
        _chunk(2),
        _chunk(3),
    ]

    repository = FakeRepository(chunks)
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
    )

    results = await service.search(
        "research query",
        limit=2,
        candidate_limit=3,
    )

    assert repository.received_limit == 3

    assert reranker.received_query == "research query"
    assert reranker.received_chunks == chunks

    assert [result.content for result in results] == [
        "chunk 3",
        "chunk 2",
    ]


@pytest.mark.asyncio
async def test_candidate_limit_cannot_be_smaller_than_limit() -> None:
    service = RetrievalService(
        repository=FakeRepository([]),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=FakeReranker(),
    )

    with pytest.raises(ValueError):
        await service.search(
            "research query",
            limit=10,
            candidate_limit=5,
        )
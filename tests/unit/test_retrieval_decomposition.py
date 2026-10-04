from uuid import uuid4

import pytest

from app.retrieval.models import RetrievedChunk
from app.retrieval.service import RetrievalService


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeRepository:
    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self.responses = responses
        self.queries: list[tuple[float, float]] = []

    async def search_similar_chunks(
        self,
        query_embedding,
        limit,
        max_distance=None,
        document_id=None,
        section_id=None,
    ):
        self.queries.append(tuple(query_embedding))

        if tuple(query_embedding) == (1.0, 0.0):
            # The fake embedding is intentionally simple; responses
            # are supplied through the test-specific repository below.
            return []

        return []


class FakeDecomposer:
    def __init__(self, sub_queries: tuple[str, ...]) -> None:
        self.sub_queries = sub_queries
        self.received_query: str | None = None

    async def decompose(self, query: str):
        self.received_query = query

        return type(
            "Decomposition",
            (),
            {"sub_queries": self.sub_queries},
        )()


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def rerank(self, query, chunks):
        self.calls.append((query, list(chunks)))
        return list(reversed(chunks))


def _chunk(content: str, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content=content,
        distance=distance,
    )


class QueryRepository:
    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def search_similar_chunks(
        self,
        query_embedding,
        limit,
        max_distance=None,
        document_id=None,
        section_id=None,
    ):
        query = query_embedding[0]
        self.calls.append(query)
        return self.responses[query]


class QueryEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        return [text, 0.0]


@pytest.mark.asyncio
async def test_retrieval_runs_once_per_sub_query() -> None:
    first = _chunk("first", 0.2)
    second = _chunk("second", 0.1)

    repository = QueryRepository(
        {
            "query one": [first],
            "query two": [second],
        }
    )
    embedding_provider = QueryEmbeddingProvider()
    decomposer = FakeDecomposer(("query one", "query two"))
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        query_decomposer=decomposer,
    )

    await service.search(
        "original complex query",
        limit=2,
        candidate_limit=5,
    )

    assert embedding_provider.queries == [
        "query one",
        "query two",
    ]

    assert repository.calls == [
        "query one",
        "query two",
    ]


@pytest.mark.asyncio
async def test_duplicate_chunks_are_deduplicated_and_best_distance_is_kept() -> None:
    shared_chunk_id = uuid4()

    first = RetrievedChunk(
        document_id=uuid4(),
        chunk_id=shared_chunk_id,
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content="shared",
        distance=0.4,
    )

    better = RetrievedChunk(
        document_id=first.document_id,
        chunk_id=shared_chunk_id,
        section_id=first.section_id,
        section_path="Results",
        page_numbers=[1],
        content="shared",
        distance=0.1,
    )

    other = _chunk("other", 0.2)

    repository = QueryRepository(
        {
            "query one": [first],
            "query two": [better, other],
        }
    )

    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        reranker=reranker,
        query_decomposer=FakeDecomposer(("query one", "query two")),
    )

    await service.search(
        "original query",
        limit=10,
        candidate_limit=10,
        trace=True,
    )

    assert len(reranker.calls) == 1

    _, reranked_candidates = reranker.calls[0]

    assert len(reranked_candidates) == 2

    shared = next(chunk for chunk in reranked_candidates if chunk.chunk_id == shared_chunk_id)

    assert shared.distance == 0.1
    assert service.last_trace is not None
    assert service.last_trace.original_query == "original query"
    assert service.last_trace.sub_queries == ["query one", "query two"]
    assert service.last_trace.raw_candidate_count == 3
    assert service.last_trace.deduplicated_candidate_count == 2


@pytest.mark.asyncio
async def test_global_reranker_runs_once_with_original_query() -> None:
    first = _chunk("first", 0.2)
    second = _chunk("second", 0.1)

    repository = QueryRepository(
        {
            "query one": [first],
            "query two": [second],
        }
    )

    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        reranker=reranker,
        query_decomposer=FakeDecomposer(("query one", "query two")),
    )

    results = await service.search(
        "original query",
        limit=1,
        candidate_limit=10,
    )

    assert len(reranker.calls) == 1

    rerank_query, rerank_candidates = reranker.calls[0]

    assert rerank_query == "original query"
    assert len(rerank_candidates) == 2
    assert {chunk.chunk_id for chunk in rerank_candidates} == {
        first.chunk_id,
        second.chunk_id,
    }

    assert len(results) == 1
    assert results[0].chunk_id in {
        first.chunk_id,
        second.chunk_id,
    }


@pytest.mark.asyncio
async def test_decomposed_retrieval_merges_candidates_and_reranks_globally() -> None:
    first = _chunk("first", 0.2)
    shared = _chunk("shared", 0.4)
    better_shared = RetrievedChunk(
        document_id=shared.document_id,
        chunk_id=shared.chunk_id,
        section_id=shared.section_id,
        section_path=shared.section_path,
        page_numbers=shared.page_numbers,
        content=shared.content,
        distance=0.1,
    )
    third = _chunk("third", 0.3)

    repository = QueryRepository(
        {
            "query one": [first, shared],
            "query two": [better_shared, third],
        }
    )

    embedding_provider = QueryEmbeddingProvider()
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        query_decomposer=FakeDecomposer(("query one", "query two")),
    )

    results = await service.search(
        "original complex query",
        limit=2,
        candidate_limit=10,
        trace=True,
    )

    # Each sub-query gets its own embedding and retrieval call.
    assert embedding_provider.queries == [
        "query one",
        "query two",
    ]
    assert repository.calls == [
        "query one",
        "query two",
    ]

    # Both retrieval results are merged and duplicate chunks are removed.
    assert len(reranker.calls) == 1

    rerank_query, rerank_candidates = reranker.calls[0]

    assert rerank_query == "original complex query"
    assert len(rerank_candidates) == 3

    shared_candidate = next(
        chunk for chunk in rerank_candidates if chunk.chunk_id == shared.chunk_id
    )
    assert shared_candidate.distance == 0.1

    # The single global reranker determines the final Top K.
    assert len(results) == 2
    assert results == list(reversed(rerank_candidates))[:2]

    # Retrieval trace exposes the decomposition and candidate accounting.
    assert service.last_trace is not None
    assert service.last_trace.original_query == "original complex query"
    assert service.last_trace.sub_queries == [
        "query one",
        "query two",
    ]
    assert service.last_trace.raw_candidate_count == 4
    assert service.last_trace.deduplicated_candidate_count == 3

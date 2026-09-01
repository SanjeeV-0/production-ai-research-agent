from uuid import uuid4

import pytest

from app.retrieval.models import RetrievedChunk
from app.retrieval.service import RetrievalService


class FakeEmbeddingProvider:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeRepository:
    async def search_similar_chunks(
        self,
        query_embedding,
        limit,
        max_distance=None,
        document_id=None,
        section_id=None,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="Results",
                page_numbers=[1],
                content="Relevant research content.",
                distance=0.1,
                rerank_score=4.0,
            )
        ]


class FakeReranker:
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        return chunks


class FakeObservation:
    def __init__(
        self,
        name: str,
        observation_type: str,
        input_data: dict,
    ) -> None:
        self.name = name
        self.observation_type = observation_type
        self.input_data = input_data
        self.output = None
        self.level = None
        self.status_message = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def update(
        self,
        *,
        output=None,
        level=None,
        status_message=None,
    ) -> None:
        if output is not None:
            self.output = output

        if level is not None:
            self.level = level

        if status_message is not None:
            self.status_message = status_message


class FakeLangfuse:
    def __init__(self) -> None:
        self.observations = []

    def start_as_current_observation(
        self,
        *,
        as_type,
        name,
        input,
    ):
        observation = FakeObservation(
            name=name,
            observation_type=as_type,
            input_data=input,
        )

        self.observations.append(observation)

        return observation


@pytest.mark.asyncio
async def test_retrieval_creates_langfuse_observation(
    monkeypatch,
) -> None:
    fake_langfuse = FakeLangfuse()

    monkeypatch.setattr(
        "app.retrieval.service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = RetrievalService(
        repository=FakeRepository(),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=FakeReranker(),
    )

    results = await service.search(
        query="research query",
        limit=2,
        candidate_limit=5,
    )

    assert len(results) == 1
    assert len(fake_langfuse.observations) == 1

    observation = fake_langfuse.observations[0]

    assert observation.name == "document-retrieval"
    assert observation.observation_type == "retriever"

    assert observation.input_data == {
        "query": "research query",
        "limit": 2,
        "candidate_limit": 5,
        "max_distance": None,
        "document_id": None,
        "section_id": None,
    }

    assert observation.output is not None
    assert observation.output["result_count"] == 1

    assert (
        observation.output["results"][0]["section_path"]
        == "Results"
    )

    assert (
        observation.output["results"][0]["distance"]
        == 0.1
    )


@pytest.mark.asyncio
async def test_retrieval_without_langfuse_still_works(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.retrieval.service.get_langfuse",
        lambda: None,
    )

    service = RetrievalService(
        repository=FakeRepository(),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=FakeReranker(),
    )

    results = await service.search(
        query="research query",
        limit=2,
    )

    assert len(results) == 1
    assert results[0].content == (
        "Relevant research content."
    )
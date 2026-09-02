from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_generation_service,
    get_retrieval_service,
)
from app.generation.context import GenerationContext
from app.generation.generation_service import GenerationResult
from app.main import app
from app.retrieval.models import RetrievedChunk


class FakeRetrievalService:
    async def search(
        self,
        query: str,
        limit: int = 10,
        max_distance=None,
        document_id=None,
        section_id=None,
        candidate_limit=None,
        trace=False,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="Results",
                page_numbers=[1],
                content="RAG retrieves relevant information before generation.",
                distance=0.1,
            )
        ]


class FakeGenerationService:
    def __init__(self) -> None:
        self.received_query = None
        self.received_context = None

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        self.received_query = query
        self.received_context = context

        return GenerationResult(
            text="RAG retrieves relevant information before generation.",
            model="fake-model",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )


@pytest.mark.asyncio
async def test_research_ask_endpoint() -> None:
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = (
        lambda: FakeRetrievalService()
    )

    app.dependency_overrides[get_generation_service] = (
        lambda: generation_service
    )

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/research/ask",
                json={
                    "query": "What is RAG?",
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["answer"] == (
            "RAG retrieves relevant information before generation."
        )

        assert body["model"] == "fake-model"

        assert len(body["sources"]) == 1
        assert body["sources"][0]["section_path"] == "Results"

        assert generation_service.received_query == "What is RAG?"

        assert (
            "RAG retrieves relevant information"
            in generation_service.received_context.text
        )

    finally:
        app.dependency_overrides.clear()
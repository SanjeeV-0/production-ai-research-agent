from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.core.database import async_session_factory
from app.core.dependencies import (
    get_embedding_provider,
    get_retrieval_service,
)
from app.core.models import (
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
)
from app.main import app
from app.retrieval.models import RetrievedChunk


class FakeRetrievalService:
    """Fake retrieval service for API contract tests."""

    async def search(
        self,
        query: str,
        limit: int = 10,
        max_distance=None,
        document_id=None,
        section_id=None,
        candidate_limit=None,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="Results",
                page_numbers=[1, 2],
                content=(
                    "Retrieval augmented generation combines "
                    "retrieval with generation."
                ),
                distance=0.15,
                rerank_score=4.2,
            )
        ]




def test_retrieval_search_endpoint() -> None:
    """Test the retrieval API response contract."""

    app.dependency_overrides[get_retrieval_service] = (
        lambda: FakeRetrievalService()
    )

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={
                "query": "retrieval augmented generation",
                "limit": 2,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert len(body["results"]) == 1

        result = body["results"][0]

        assert result["section_path"] == "Results"
        assert result["page_numbers"] == [1, 2]
        assert (
            result["content"]
            == "Retrieval augmented generation combines "
            "retrieval with generation."
        )
        assert result["distance"] == 0.15
        assert result["similarity"] == 0.85
        assert result["rerank_score"] == 4.2

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_rejects_empty_query() -> None:
    """Test validation of an empty retrieval query."""

    app.dependency_overrides[get_retrieval_service] = (
        lambda: FakeRetrievalService()
    )

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={
                "query": "",
                "limit": 5,
            },
        )

        assert response.status_code == 422

    finally:
        app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_retrieval_search_real_database() -> None:
    """Test retrieval API against real PostgreSQL, pgvector, and reranking."""

    async with async_session_factory() as session:
        document = Document(
            title=f"API Retrieval Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"api-retrieval-test-{uuid4()}",
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        section = DocumentSection(
            document_id=document.id,
            title="Results",
            section_path="Results",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add(section)
        await session.flush()

        page_one = DocumentPage(
            document_id=document.id,
            page_number=1,
            content=(
                "Retrieval augmented generation combines "
                "information retrieval with language generation."
            ),
        )

        page_two = DocumentPage(
            document_id=document.id,
            page_number=2,
            content=(
                "The weather forecast predicts "
                "heavy rain tomorrow."
            ),
        )

        session.add_all([page_one, page_two])
        await session.flush()

        embedding_provider = get_embedding_provider()

        relevant_content = (
            "Retrieval augmented generation combines "
            "information retrieval with language generation."
        )

        unrelated_content = (
            "The weather forecast predicts heavy rain tomorrow."
        )

        relevant_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=relevant_content,
            chunk_metadata={},
            embedding=embedding_provider.embed_text(
                relevant_content
            ),
        )

        unrelated_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=1,
            content=unrelated_content,
            chunk_metadata={},
            embedding=embedding_provider.embed_text(
                unrelated_content
            ),
        )

        session.add_all(
            [
                relevant_chunk,
                unrelated_chunk,
            ]
        )
        await session.flush()

        session.add_all(
            [
                ChunkPageMap(
                    chunk_id=relevant_chunk.id,
                    document_page_id=page_one.id,
                ),
                ChunkPageMap(
                    chunk_id=unrelated_chunk.id,
                    document_page_id=page_two.id,
                ),
            ]
        )
        await session.flush()

        # Commit so the API's separate database session
        # can see the test data.
        await session.commit()

        try:
            transport = ASGITransport(app=app)

            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/retrieval/search",
                    json={
                        "query": "retrieval augmented generation",
                        "limit": 2,
                        "document_id": str(document.id),
                    },
                )

            assert response.status_code == 200

            body = response.json()

            assert len(body["results"]) == 2

            first_result = body["results"][0]
            second_result = body["results"][1]

            # Vector retrieval + cross-encoder should put
            # the relevant chunk first.
            assert (
                first_result["content"]
                == relevant_content
            )

            assert (
                first_result["section_path"]
                == "Results"
            )

            assert first_result["page_numbers"] == [1]

            # Similarity comes from pgvector cosine distance.
            assert 0.0 <= first_result["similarity"] <= 1.0

            # These scores come from the REAL cross-encoder.
            assert first_result["rerank_score"] is not None
            assert second_result["rerank_score"] is not None

            assert (
                first_result["rerank_score"]
                >= second_result["rerank_score"]
            )

        finally:
            # Delete the test document. Relationships use
            # cascade delete for dependent records.
            await session.delete(document)
            await session.commit()
from uuid import uuid4

import pytest

from app.core.database import async_session_factory
from app.core.models import Document, DocumentChunk, DocumentSection
from app.core.repositories.document import DocumentRepository
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.retrieval.service import RetrievalService


@pytest.mark.asyncio
async def test_similar_chunks_are_retrieved() -> None:
    provider = DeterministicEmbeddingProvider(
        dimensions=384,
    )

    async with async_session_factory() as session:
        document = Document(
            title=f"Retrieval Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"retrieval-test-{uuid4()}",
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

        similar_content = "retrieval augmented generation"
        unrelated_content = "weather forecast tomorrow"

        similar_embedding = provider.embed_text(
            similar_content
        )
        unrelated_embedding = provider.embed_text(
            unrelated_content
        )

        similar_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=similar_content,
            chunk_metadata={},
            embedding=similar_embedding,
        )

        unrelated_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=1,
            content=unrelated_content,
            chunk_metadata={},
            embedding=unrelated_embedding,
        )

        session.add_all(
            [
                similar_chunk,
                unrelated_chunk,
            ]
        )

        await session.flush()

        repository = DocumentRepository(session)

        retrieval = RetrievalService(
            repository=repository,
            embedding_provider=provider,
        )

        results = await retrieval.search(
            "retrieval augmented generation",
            limit=2,
        )

        assert len(results) == 2
        assert results[0].content == similar_content

        await session.delete(document)
        await session.commit()
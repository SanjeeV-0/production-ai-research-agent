from uuid import uuid4

import pytest
from sqlalchemy import select

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

        similar_embedding = provider.embed_text(similar_content)
        unrelated_embedding = provider.embed_text(unrelated_content)

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

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        results = await retrieval.search(
            "retrieval augmented generation",
            limit=2,
        )

        assert len(results) == 2
        assert results[0].chunk.content == similar_content
        assert results[0].distance <= results[1].distance
        assert results[0].similarity >= results[1].similarity
        assert 0.0 <= results[0].similarity <= 1.0
        assert 0.0 <= results[1].similarity <= 1.0

        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_similarity_threshold_filters_distant_chunks() -> None:
    provider = DeterministicEmbeddingProvider(
        dimensions=384,
    )

    async with async_session_factory() as session:
        document = Document(
            title=f"Retrieval Threshold Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"retrieval-threshold-test-{uuid4()}",
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

        similar_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=similar_content,
            chunk_metadata={},
            embedding=provider.embed_text(similar_content),
        )

        unrelated_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=1,
            content=unrelated_content,
            chunk_metadata={},
            embedding=provider.embed_text(unrelated_content),
        )

        session.add_all([similar_chunk, unrelated_chunk])
        await session.flush()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        results = await retrieval.search(
            "retrieval augmented generation",
            limit=10,
            max_distance=0.5,
            section_id=section.id,
        )

        assert all(
            result.chunk.document_id == document.id
            for result in results
        )

        assert all(
            result.distance <= 0.5
            for result in results
        )

        assert all(
            result.chunk.content != unrelated_content
            for result in results
        )

        await session.delete(document)
        await session.commit()

@pytest.mark.asyncio
async def test_section_filter_limits_results_to_section() -> None:
    provider = DeterministicEmbeddingProvider(
        dimensions=384,
    )

    async with async_session_factory() as session:
        document = Document(
            title=f"Retrieval Section Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"retrieval-section-test-{uuid4()}",
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

        other_section = DocumentSection(
            document_id=document.id,
            title="Other Section",
            section_path="Other Section",
            section_level=1,
            section_index=1,
            section_metadata={},
        )

        session.add_all([section, other_section])
        await session.flush()


        similar_content = "retrieval augmented generation"
        unrelated_content = "weather forecast tomorrow"

        similar_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=similar_content,
            chunk_metadata={},
            embedding=provider.embed_text(similar_content),
        )

        unrelated_chunk = DocumentChunk(
            document_id=document.id,
            section_id=other_section.id,
            chunk_index=1,
            content=unrelated_content,
            chunk_metadata={},
            embedding=provider.embed_text(unrelated_content),
        )

        session.add_all(
            [
                similar_chunk,
                unrelated_chunk,
            ]
        )
        await session.flush()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        # Search only the Results section.
        results = await retrieval.search(
            "retrieval augmented generation",
            limit=10,
            section_id=section.id,
        )

        assert results
        assert all(
            result.chunk.section_id == section.id
            for result in results
        )

        assert all(
            result.chunk.content != unrelated_content
            for result in results
        )

        # Search only the Other Section.
        results = await retrieval.search(
            "retrieval augmented generation",
            limit=10,
            section_id=other_section.id,
        )

        assert results
        assert all(
            result.chunk.section_id == other_section.id
            for result in results
        )

        assert all(
            result.chunk.content != similar_content
            for result in results
        )

        await session.delete(document)
        await session.commit()
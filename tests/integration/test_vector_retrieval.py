from uuid import uuid4

import pytest

from app.core.database import async_session_factory
from app.core.models import (
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentStatus,
)
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
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
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
            document_id=document.id,
        )

        assert len(results) == 2
        assert results[0].document_id == document.id
        assert results[0].chunk_id == similar_chunk.id
        assert results[0].section_id == section.id
        assert results[0].section_path == "Results"
        assert results[0].page_numbers == []
        assert results[0].content == similar_content
        assert results[0].distance <= results[1].distance
        assert results[0].similarity >= results[1].similarity
        assert 0.0 <= results[0].similarity <= 1.0
        assert 0.0 <= results[1].similarity <= 1.0

        await session.rollback()


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
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
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
            document_id=document.id,
        )

        assert all(result.document_id == document.id for result in results)

        assert all(result.distance <= 0.5 for result in results)

        assert all(result.content != unrelated_content for result in results)

        await session.rollback()


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
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
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

        results = await retrieval.search(
            "retrieval augmented generation",
            limit=10,
            document_id=document.id,
            section_id=section.id,
        )

        assert results
        assert all(result.section_id == section.id for result in results)

        assert all(result.content != unrelated_content for result in results)

        results = await retrieval.search(
            "retrieval augmented generation",
            limit=10,
            document_id=document.id,
            section_id=other_section.id,
        )

        assert results
        assert all(result.section_id == other_section.id for result in results)

        assert all(result.content != similar_content for result in results)

        await session.rollback()


@pytest.mark.asyncio
async def test_retrieved_chunk_preserves_multiple_page_numbers() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=384)

    async with async_session_factory() as session:
        document = Document(
            title=f"Retrieval Page Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"retrieval-page-test-{uuid4()}",
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
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
            content="Results begin on page one.",
        )

        page_two = DocumentPage(
            document_id=document.id,
            page_number=2,
            content="Results continue on page two.",
        )

        session.add_all([page_one, page_two])
        await session.flush()

        content = "retrieval augmented generation across two pages"

        chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=content,
            chunk_metadata={},
            embedding=provider.embed_text(content),
        )

        session.add(chunk)
        await session.flush()

        session.add_all(
            [
                ChunkPageMap(
                    chunk_id=chunk.id,
                    document_page_id=page_one.id,
                ),
                ChunkPageMap(
                    chunk_id=chunk.id,
                    document_page_id=page_two.id,
                ),
            ]
        )

        await session.flush()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        results = await retrieval.search(
            "retrieval augmented generation across two pages",
            limit=1,
            document_id=document.id,
        )

        assert len(results) == 1
        assert results[0].chunk_id == chunk.id
        assert results[0].page_numbers == [1, 2]

        await session.rollback()


@pytest.mark.asyncio
async def test_retrieval_only_returns_ready_current_versions() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=384)
    logical_document_id = uuid4()

    async with async_session_factory() as session:
        documents = [
            Document(
                title=f"Ready Current {uuid4()}",
                document_type="research_paper",
                content_hash=f"ready-current-{uuid4()}",
                logical_document_id=logical_document_id,
                version_number=1,
                is_current=True,
                status=DocumentStatus.READY,
                document_metadata={},
            ),
            Document(
                title=f"Ready Historical {uuid4()}",
                document_type="research_paper",
                content_hash=f"ready-historical-{uuid4()}",
                logical_document_id=logical_document_id,
                version_number=2,
                is_current=False,
                status=DocumentStatus.READY,
                document_metadata={},
            ),
            Document(
                title=f"Failed Version {uuid4()}",
                document_type="research_paper",
                content_hash=f"failed-{uuid4()}",
                logical_document_id=logical_document_id,
                version_number=3,
                is_current=False,
                status=DocumentStatus.FAILED,
                document_metadata={},
            ),
            Document(
                title=f"Processing Version {uuid4()}",
                document_type="research_paper",
                content_hash=f"processing-{uuid4()}",
                logical_document_id=logical_document_id,
                version_number=4,
                is_current=False,
                status=DocumentStatus.PROCESSING,
                document_metadata={},
            ),
        ]

        session.add_all(documents)
        await session.flush()

        sections = [
            DocumentSection(
                document_id=document.id,
                title="Results",
                section_path="Results",
                section_level=1,
                section_index=0,
                section_metadata={},
            )
            for document in documents
        ]

        session.add_all(sections)
        await session.flush()

        query_text = "retrieval augmented generation"
        query_embedding = provider.embed_text(query_text)

        chunks = [
            DocumentChunk(
                document_id=document.id,
                section_id=section.id,
                chunk_index=0,
                content=f"{document.title} retrieval augmented generation",
                chunk_metadata={},
                embedding=query_embedding,
            )
            for document, section in zip(
                documents,
                sections,
                strict=True,
            )
        ]

        session.add_all(chunks)
        await session.flush()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        results = await retrieval.search(
            query_text,
            limit=10,
            document_id=documents[0].id,
        )

        assert len(results) == 1
        assert results[0].document_id == documents[0].id

        await session.rollback()


@pytest.mark.asyncio
async def test_current_ready_version_remains_retrievable_during_new_version_processing() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=384)
    logical_document_id = uuid4()

    async with async_session_factory() as session:
        version_one = Document(
            title=f"Version One {uuid4()}",
            document_type="research_paper",
            content_hash=f"version-one-{uuid4()}",
            logical_document_id=logical_document_id,
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        version_two = Document(
            title=f"Version Two {uuid4()}",
            document_type="research_paper",
            content_hash=f"version-two-{uuid4()}",
            logical_document_id=logical_document_id,
            version_number=2,
            is_current=False,
            status=DocumentStatus.PROCESSING,
            document_metadata={},
        )

        session.add_all([version_one, version_two])
        await session.flush()

        sections = [
            DocumentSection(
                document_id=document.id,
                title="Results",
                section_path="Results",
                section_level=1,
                section_index=0,
                section_metadata={},
            )
            for document in [version_one, version_two]
        ]

        session.add_all(sections)
        await session.flush()

        query_text = "retrieval augmented generation"

        chunks = [
            DocumentChunk(
                document_id=document.id,
                section_id=section.id,
                chunk_index=0,
                content=f"version {document.version_number} retrieval content",
                chunk_metadata={},
                embedding=provider.embed_text(query_text),
            )
            for document, section in zip(
                [version_one, version_two],
                sections,
                strict=True,
            )
        ]

        session.add_all(chunks)
        await session.flush()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        results = await retrieval.search(
            query_text,
            limit=10,
            document_id=version_one.id,
        )

        assert len(results) == 1
        assert results[0].document_id == version_one.id

        version_one.is_current = False

        await session.flush()

        version_two.is_current = True
        version_two.status = DocumentStatus.READY

        await session.flush()

        results = await retrieval.search(
            query_text,
            limit=10,
            document_id=version_two.id,
        )

        assert len(results) == 1
        assert results[0].document_id == version_two.id

        await session.rollback()

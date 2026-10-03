from uuid import uuid4

import pytest

from app.core.database import async_session_factory
from app.core.models import (
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentStatus,
)
from app.core.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_headingless_current_ready_chunk_is_retrievable() -> None:
    logical_document_id = uuid4()

    async with async_session_factory() as session:
        document = Document(
            title="Headingless Research Paper",
            authors="Test Author",
            source="integration-test",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"headingless-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )
        session.add(document)
        await session.flush()

        page = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="This page contains the searchable content.",
        )
        session.add(page)
        await session.flush()

        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content="This is a headingless searchable chunk.",
            chunk_metadata={},
            section_id=None,
            embedding=[0.1] * 384,
        )
        session.add(chunk)
        await session.flush()

        mapping = ChunkPageMap(
            chunk_id=chunk.id,
            document_page_id=page.id,
        )
        session.add(mapping)

        await session.commit()

        repository = DocumentRepository(session)

        results = await repository.search_similar_chunks(
            query_embedding=[0.1] * 384,
            limit=10,
            document_id=document.id,
        )

        assert len(results) == 1

        result = results[0]

        assert result.document_id == document.id
        assert result.chunk_id == chunk.id
        assert result.section_id is None
        assert result.section_path is None
        assert result.page_numbers == [1]
        assert result.content == "This is a headingless searchable chunk."

        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_only_current_ready_version_is_retrievable() -> None:
    logical_document_id = uuid4()

    async with async_session_factory() as session:
        old_document = Document(
            logical_document_id=logical_document_id,
            content_hash="a" * 64,
            version_number=1,
            is_current=False,
            status=DocumentStatus.READY,
            title="Version 1",
            document_type="pdf",
            document_metadata={},
        )

        current_document = Document(
            logical_document_id=logical_document_id,
            content_hash="b" * 64,
            version_number=2,
            is_current=True,
            status=DocumentStatus.READY,
            title="Version 2",
            document_type="pdf",
            document_metadata={},
        )
        session.add_all([old_document, current_document])
        await session.flush()

        old_page = DocumentPage(
            document_id=old_document.id,
            page_number=1,
            content="Old version page",
        )
        current_page = DocumentPage(
            document_id=current_document.id,
            page_number=1,
            content="Current version page",
        )
        session.add_all([old_page, current_page])
        await session.flush()

        old_chunk = DocumentChunk(
            document_id=old_document.id,
            section_id=None,
            content="Old version searchable content.",
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        current_chunk = DocumentChunk(
            document_id=current_document.id,
            section_id=None,
            content="Current version searchable content.",
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        session.add_all([old_chunk, current_chunk])
        await session.flush()

        session.add_all(
            [
                ChunkPageMap(
                    chunk_id=old_chunk.id,
                    document_page_id=old_page.id,
                ),
                ChunkPageMap(
                    chunk_id=current_chunk.id,
                    document_page_id=current_page.id,
                ),
            ]
        )

        await session.commit()

        repository = DocumentRepository(session)

        results = await repository.search_similar_chunks(
            query_embedding=[0.1] * 384,
            limit=10,
        )

        result_document_ids = {result.document_id for result in results}

        assert current_document.id in result_document_ids
        assert old_document.id not in result_document_ids

        await session.delete(old_document)
        await session.delete(current_document)
        await session.commit()


@pytest.mark.asyncio
async def test_only_ready_documents_are_retrievable() -> None:
    logical_document_id = uuid4()

    async with async_session_factory() as session:
        failed_document = Document(
            logical_document_id=logical_document_id,
            content_hash="c" * 64,
            version_number=1,
            is_current=False,
            status=DocumentStatus.FAILED,
            title="Failed version",
            document_type="pdf",
            document_metadata={},
        )

        processing_document = Document(
            logical_document_id=logical_document_id,
            content_hash="d" * 64,
            version_number=2,
            is_current=False,
            status=DocumentStatus.PROCESSING,
            title="Processing version",
            document_type="pdf",
            document_metadata={},
        )

        ready_document = Document(
            logical_document_id=logical_document_id,
            content_hash="e" * 64,
            version_number=3,
            is_current=True,
            status=DocumentStatus.READY,
            title="Ready version",
            document_type="pdf",
            document_metadata={},
        )

        session.add_all(
            [
                failed_document,
                processing_document,
                ready_document,
            ]
        )
        await session.flush()

        failed_page = DocumentPage(
            document_id=failed_document.id,
            page_number=1,
            content="Failed page",
        )
        processing_page = DocumentPage(
            document_id=processing_document.id,
            page_number=1,
            content="Processing page",
        )
        ready_page = DocumentPage(
            document_id=ready_document.id,
            page_number=1,
            content="Ready page",
        )
        session.add_all(
            [
                failed_page,
                processing_page,
                ready_page,
            ]
        )
        await session.flush()

        failed_chunk = DocumentChunk(
            document_id=failed_document.id,
            section_id=None,
            content="Failed searchable content.",
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        processing_chunk = DocumentChunk(
            document_id=processing_document.id,
            section_id=None,
            content="Processing searchable content.",
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        ready_chunk = DocumentChunk(
            document_id=ready_document.id,
            section_id=None,
            content="Ready searchable content.",
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        session.add_all(
            [
                failed_chunk,
                processing_chunk,
                ready_chunk,
            ]
        )
        await session.flush()

        session.add_all(
            [
                ChunkPageMap(
                    chunk_id=failed_chunk.id,
                    document_page_id=failed_page.id,
                ),
                ChunkPageMap(
                    chunk_id=processing_chunk.id,
                    document_page_id=processing_page.id,
                ),
                ChunkPageMap(
                    chunk_id=ready_chunk.id,
                    document_page_id=ready_page.id,
                ),
            ]
        )

        await session.commit()

        repository = DocumentRepository(session)

        results = await repository.search_similar_chunks(
            query_embedding=[0.1] * 384,
            limit=10,
        )

        result_document_ids = {result.document_id for result in results}

        assert ready_document.id in result_document_ids
        assert failed_document.id not in result_document_ids
        assert processing_document.id not in result_document_ids

        await session.delete(failed_document)
        await session.delete(processing_document)
        await session.delete(ready_document)
        await session.commit()


@pytest.mark.asyncio
async def test_deleted_document_version_is_not_retrievable() -> None:
    async with async_session_factory() as session:
        document = Document(
            logical_document_id=uuid4(),
            content_hash="f" * 64,
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            title="Deleted version",
            document_type="pdf",
            document_metadata={},
        )
        session.add(document)
        await session.flush()

        page = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Deleted page",
        )
        session.add(page)
        await session.flush()

        chunk = DocumentChunk(
            document_id=document.id,
            section_id=None,
            content="This content will be deleted.",
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        session.add(chunk)
        await session.flush()

        session.add(
            ChunkPageMap(
                chunk_id=chunk.id,
                document_page_id=page.id,
            )
        )
        await session.commit()

        repository = DocumentRepository(session)

        results_before_delete = await repository.search_similar_chunks(
            query_embedding=[0.1] * 384,
            limit=10,
            document_id=document.id,
        )

        assert len(results_before_delete) == 1

        await session.delete(document)
        await session.commit()

        results_after_delete = await repository.search_similar_chunks(
            query_embedding=[0.1] * 384,
            limit=10,
            document_id=document.id,
        )

        assert results_after_delete == []


@pytest.mark.asyncio
async def test_similar_chunks_are_ordered_by_vector_distance() -> None:
    async with async_session_factory() as session:
        document = Document(
            logical_document_id=uuid4(),
            content_hash="1" * 64,
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            title="Ranking test",
            document_type="pdf",
            document_metadata={},
        )
        session.add(document)
        await session.flush()

        page = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Ranking test page",
        )
        session.add(page)
        await session.flush()

        close_chunk = DocumentChunk(
            document_id=document.id,
            section_id=None,
            content="Close match",
            chunk_index=0,
            embedding=[0.1] * 384,
        )

        far_chunk = DocumentChunk(
            document_id=document.id,
            section_id=None,
            content="Far match",
            chunk_index=1,
            embedding=[0.9] * 384,
        )

        session.add_all([close_chunk, far_chunk])
        await session.flush()

        session.add_all(
            [
                ChunkPageMap(
                    chunk_id=close_chunk.id,
                    document_page_id=page.id,
                ),
                ChunkPageMap(
                    chunk_id=far_chunk.id,
                    document_page_id=page.id,
                ),
            ]
        )
        await session.commit()

        repository = DocumentRepository(session)

        results = await repository.search_similar_chunks(
            query_embedding=[0.1] * 384,
            limit=2,
            document_id=document.id,
        )

        assert len(results) == 2
        assert results[0].chunk_id == close_chunk.id
        assert results[1].chunk_id == far_chunk.id

        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_similar_chunks_respects_limit() -> None:
    async with async_session_factory() as session:
        document = Document(
            logical_document_id=uuid4(),
            content_hash="2" * 64,
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            title="Limit test",
            document_type="pdf",
            document_metadata={},
        )
        session.add(document)
        await session.flush()

        page = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Limit test page",
        )
        session.add(page)
        await session.flush()

        chunks = [
            DocumentChunk(
                document_id=document.id,
                section_id=None,
                content=f"Match {index}",
                chunk_index=index,
                embedding=[0.1 + index * 0.01] * 384,
            )
            for index in range(3)
        ]

        session.add_all(chunks)
        await session.flush()

        session.add_all(
            [
                ChunkPageMap(
                    chunk_id=chunk.id,
                    document_page_id=page.id,
                )
                for chunk in chunks
            ]
        )
        await session.commit()

        repository = DocumentRepository(session)

        results = await repository.search_similar_chunks(
            query_embedding=[0.1] * 384,
            limit=2,
            document_id=document.id,
        )

        assert len(results) == 2

        await session.delete(document)
        await session.commit()

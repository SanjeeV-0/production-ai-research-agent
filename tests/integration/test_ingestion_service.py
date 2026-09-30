import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import (
    ChunkPageMap,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentStatus,
    StoredFile,
)
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.normalizer import calculate_content_hash
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage


async def get_stored_file(
    session,
    document_id,
) -> StoredFile:
    result = await session.execute(
        select(StoredFile).where(
            StoredFile.document_id == document_id,
        )
    )
    stored_file = result.scalar_one()
    return stored_file


@pytest.mark.asyncio
async def test_ingestion_service_deduplicates_content(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "research.md"

    document_path.write_text(
        f"# RAG Research\n\nRetrieval-Augmented Generation content {uuid4()}",
        encoding="utf-8",
    )

    file_storage = LocalFileStorage(tmp_path / "storage")

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
            file_storage=file_storage,
        )

        logical_document_id = uuid4()

        first_document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
            logical_document_id=logical_document_id,
        )

        await session.commit()

        second_document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
            logical_document_id=logical_document_id,
        )

        await session.commit()

        assert first_document.id == second_document.id

        stored_file = await get_stored_file(
            session,
            first_document.id,
        )

        await session.delete(stored_file)
        await session.delete(first_document)
        await session.commit()


@pytest.mark.asyncio
async def test_ingestion_service_marks_document_ready(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "research.md"

    document_path.write_text(
        f"# RAG Research\n\nRetrieval content {uuid4()}",
        encoding="utf-8",
    )

    file_storage = LocalFileStorage(tmp_path / "storage")

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
            file_storage=file_storage,
        )

        document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
        )

        assert document.status == DocumentStatus.READY
        assert document.processing_attempt == 1
        assert document.processing_started_at is not None
        assert document.processing_completed_at is not None
        assert document.failed_at is None
        assert document.last_error is None

        stored_file = await get_stored_file(
            session,
            document.id,
        )

        await session.delete(stored_file)
        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_ingestion_service_marks_failed_and_rolls_back_processing(
    tmp_path: Path,
) -> None:
    content = f"# RAG Research\n\nRetrieval content {uuid4()}"

    document_path = tmp_path / "research.md"

    document_path.write_text(
        content,
        encoding="utf-8",
    )

    class FailingEmbeddingProvider(
        DeterministicEmbeddingProvider,
    ):
        def embed_batch(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            raise RuntimeError("embedding failure")

    file_storage = LocalFileStorage(tmp_path / "storage")

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=FailingEmbeddingProvider(
                dimensions=384,
            ),
            file_storage=file_storage,
        )

        with pytest.raises(RuntimeError, match="embedding failure"):
            await service.ingest_file(
                path=document_path,
                loader=MarkdownLoader(),
                title="Failed Research Paper",
                document_type="research_paper",
                source="integration-test",
            )

        content_hash = calculate_content_hash(content)

        document = await service.repository.get_by_content_hash(content_hash)

        assert document is not None
        assert document.status == DocumentStatus.FAILED
        assert document.processing_attempt == 1
        assert document.processing_started_at is not None
        assert document.processing_completed_at is not None
        assert document.failed_at is not None
        assert document.last_error == "embedding failure"

        page_result = await session.execute(
            select(DocumentPage).where(
                DocumentPage.document_id == document.id,
            )
        )

        assert page_result.scalars().all() == []

        stored_file = await get_stored_file(
            session,
            document.id,
        )

        await session.delete(stored_file)
        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_ingestion_preserves_original_file_bytes(
    tmp_path: Path,
) -> None:
    original_bytes = b"# Original Bytes Test\n\noriginal uploaded bytes 01 02"
    document_path = tmp_path / "source.txt"
    document_path.write_bytes(original_bytes)

    file_storage = LocalFileStorage(tmp_path / "storage")

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
            file_storage=file_storage,
        )

        document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Original Bytes Test",
            document_type="research_paper",
            source="integration-test",
        )

        stored_file = await get_stored_file(
            session,
            document.id,
        )

        retrieved_bytes = await file_storage.retrieve(
            stored_file.storage_key,
        )

        assert retrieved_bytes == original_bytes

        await session.delete(stored_file)
        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_ingestion_persists_correct_stored_file_metadata(
    tmp_path: Path,
) -> None:
    original_bytes = b"# Metadata Test\n\nmetadata test content"
    document_path = tmp_path / "example.txt"
    document_path.write_bytes(original_bytes)

    file_storage = LocalFileStorage(tmp_path / "storage")

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
            file_storage=file_storage,
        )

        document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Stored File Metadata Test",
            document_type="research_paper",
            source="integration-test",
        )

        stored_file = await get_stored_file(
            session,
            document.id,
        )

        assert stored_file.original_filename == "example.txt"
        assert (
            stored_file.content_hash
            == hashlib.sha256(
                original_bytes,
            ).hexdigest()
        )
        assert stored_file.size_bytes == len(original_bytes)
        assert stored_file.storage_key == (
            f"documents/{document.logical_document_id}/{document.id}/{stored_file.id}"
        )

        await session.delete(stored_file)
        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_failed_ingestion_retains_original_file(
    tmp_path: Path,
) -> None:
    content = f"# Failed Research\n\nFailure content {uuid4()}"
    document_path = tmp_path / "failed.md"
    document_path.write_text(
        content,
        encoding="utf-8",
    )

    original_bytes = document_path.read_bytes()

    class FailingEmbeddingProvider(
        DeterministicEmbeddingProvider,
    ):
        def embed_batch(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            raise RuntimeError("embedding failure")

    file_storage = LocalFileStorage(tmp_path / "storage")

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=FailingEmbeddingProvider(
                dimensions=384,
            ),
            file_storage=file_storage,
        )

        with pytest.raises(RuntimeError, match="embedding failure"):
            await service.ingest_file(
                path=document_path,
                loader=MarkdownLoader(),
                title="Failed Research Paper",
                document_type="research_paper",
                source="integration-test",
            )

        content_hash = calculate_content_hash(content)

        document = await service.repository.get_by_content_hash(
            content_hash,
        )

        assert document is not None
        assert document.status == DocumentStatus.FAILED

        stored_file = await get_stored_file(
            session,
            document.id,
        )

        retrieved_bytes = await file_storage.retrieve(
            stored_file.storage_key,
        )

        assert retrieved_bytes == original_bytes

        await session.delete(stored_file)
        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_headingless_document_ingests_with_page_provenance(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "headingless.md"

    document_path.write_text(
        """Retrieval systems improve document search by finding relevant passages.

Semantic retrieval can improve recall by representing related passages in vector space.

Evaluation compares retrieval quality across different search strategies.
""",
        encoding="utf-8",
    )

    file_storage = LocalFileStorage(tmp_path / "storage")

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
            file_storage=file_storage,
        )

        document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Headingless Document",
            document_type="research_paper",
            source="integration-test",
        )

        assert document.status == DocumentStatus.READY

        section_result = await session.execute(
            select(DocumentSection).where(
                DocumentSection.document_id == document.id,
            )
        )
        sections = section_result.scalars().all()

        assert sections == []

        chunk_result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = chunk_result.scalars().all()

        assert chunks
        assert all(chunk.section_id is None for chunk in chunks)

        page_mapping_result = await session.execute(
            select(ChunkPageMap)
            .join(
                DocumentChunk,
                DocumentChunk.id == ChunkPageMap.chunk_id,
            )
            .where(DocumentChunk.document_id == document.id)
        )
        mappings = page_mapping_result.scalars().all()

        assert mappings

        page_result = await session.execute(
            select(DocumentPage).where(
                DocumentPage.document_id == document.id,
            )
        )
        pages = page_result.scalars().all()

        page_ids = {page.id for page in pages}

        assert all(mapping.document_page_id in page_ids for mapping in mappings)

        stored_file = await get_stored_file(
            session,
            document.id,
        )

        await session.delete(stored_file)
        await session.delete(document)
        await session.commit()

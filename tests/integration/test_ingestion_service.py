import hashlib
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import (
    BlobStatus,
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentStatus,
    FileBlob,
    StoredFile,
)
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.normalizer import calculate_content_hash
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage


def create_ingestion_service(
    session,
    storage_root: Path,
    embedding_provider=None,
) -> IngestionService:
    if embedding_provider is None:
        embedding_provider = DeterministicEmbeddingProvider(
            dimensions=384,
        )

    return IngestionService(
        session,
        embedding_provider=embedding_provider,
        file_storage=LocalFileStorage(storage_root),
    )


class RecordingFileStorage(LocalFileStorage):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.stored_keys: list[str] = []
        self.deleted_keys: list[str] = []

    async def store(self, content: bytes, storage_key: str) -> None:
        await super().store(content, storage_key)
        self.stored_keys.append(storage_key)

    async def delete(self, storage_key: str) -> None:
        self.deleted_keys.append(storage_key)
        await super().delete(storage_key)


class FailingDeleteStorage(RecordingFileStorage):
    async def delete(self, storage_key: str) -> None:
        self.deleted_keys.append(storage_key)
        raise RuntimeError("cleanup failure")


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

        first_result = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
            logical_document_id=logical_document_id,
        )
        first_document = first_result.document

        second_result = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
            logical_document_id=logical_document_id,
        )
        second_document = second_result.document

        assert first_result.outcome == "new_version"
        assert second_result.outcome == "duplicate"
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

        result = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
        )
        document = result.document

        assert result.outcome == "created"
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

        result = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Original Bytes Test",
            document_type="research_paper",
            source="integration-test",
        )
        document = result.document

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

        result = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Stored File Metadata Test",
            document_type="research_paper",
            source="integration-test",
        )
        document = result.document

        stored_file = await get_stored_file(
            session,
            document.id,
        )

        expected_hash = hashlib.sha256(original_bytes).hexdigest()

        assert stored_file.original_filename == "example.txt"
        assert stored_file.content_hash == expected_hash
        assert stored_file.size_bytes == len(original_bytes)
        # Storage key is now content-addressed (shared-blob model), not
        # derived from this version's UUIDs -- see
        # app.storage.keys.build_blob_storage_key.
        assert stored_file.storage_key == f"blobs/{expected_hash[:2]}/{expected_hash}"

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
async def test_ingestion_never_deletes_blob_when_stored_file_commit_fails(
    tmp_path: Path,
) -> None:
    """Blob writes are now a SHARED resource (see `app.ingestion.
    blob_service.BlobService`), not owned by any one version -- so unlike
    the pre-blob-model behavior (which deleted the just-written file on any
    downstream commit failure), a failure committing THIS version's
    `StoredFile` reference must never delete the blob's physical bytes: it
    might already be a reference target for another version/request. The
    document-creation and blob-publish/ready commits are allowed to succeed
    (side_effect only fails the 4th commit -- the StoredFile reference
    commit); the real `session.rollback()` that follows means none of this
    session's work is actually durable in Postgres afterward, so this test
    asserts only the file-storage-level contract, not persisted DB state.
    """

    document_path = tmp_path / "commit-failure.md"
    document_path.write_text(
        f"# Commit Failure\n\nContent {uuid4()}",
        encoding="utf-8",
    )

    file_storage = RecordingFileStorage(tmp_path / "storage")

    async with async_session_factory() as session:
        original_commit = session.commit
        commit_mock = AsyncMock(
            side_effect=[
                None,  # document creation commit
                None,  # blob PENDING publish commit
                None,  # blob READY commit
                RuntimeError("database commit failure"),  # StoredFile reference commit
            ],
        )
        session.commit = commit_mock

        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
            file_storage=file_storage,
        )

        with pytest.raises(
            RuntimeError,
            match="database commit failure",
        ):
            await service.ingest_file(
                path=document_path,
                loader=MarkdownLoader(),
                title="Commit Failure Test",
                document_type="research_paper",
                source="integration-test",
            )

        assert len(file_storage.stored_keys) == 1
        assert file_storage.deleted_keys == []
        assert await file_storage.exists(file_storage.stored_keys[0])

        session.commit = original_commit
        await session.rollback()


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

        result = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Headingless Document",
            document_type="research_paper",
            source="integration-test",
        )
        document = result.document

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


@pytest.mark.asyncio
async def test_retry_failed_document_succeeds(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "research.md"
    storage_root = tmp_path / "storage"

    source_path.write_text(
        f"# Retry Research\n\nRetrieval content {uuid4()}",
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

    async with async_session_factory() as session:
        service = create_ingestion_service(
            session,
            storage_root,
            embedding_provider=FailingEmbeddingProvider(
                dimensions=384,
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="embedding failure",
        ):
            await service.ingest_file(
                path=source_path,
                loader=MarkdownLoader(),
                title="Retry Research",
                document_type="research_paper",
                source="integration-test",
            )

        result = await session.execute(
            select(Document)
            .where(Document.title == "Retry Research")
            .order_by(Document.created_at.desc())
        )
        document = result.scalars().first()

        assert document is not None

        stored_file_result = await session.execute(
            select(StoredFile).where(StoredFile.document_id == document.id)
        )
        stored_file = stored_file_result.scalar_one()

        original_document_id = document.id
        original_file_id = stored_file.id
        original_storage_key = stored_file.storage_key
        original_attempt = document.processing_attempt

        # Retry using a working embedding provider.
        retry_service = create_ingestion_service(
            session,
            storage_root,
        )

        retried_document = await retry_service.retry_document(
            document.id,
            loader=MarkdownLoader(),
        )

        assert retried_document.id == original_document_id
        assert retried_document.status == DocumentStatus.READY
        assert retried_document.processing_attempt == (original_attempt + 1)
        assert retried_document.last_error is None
        assert retried_document.failed_at is None

        stored_file_result = await session.execute(
            select(StoredFile).where(StoredFile.document_id == retried_document.id)
        )
        retried_file = stored_file_result.scalar_one()

        assert retried_file.id == original_file_id
        assert retried_file.storage_key == original_storage_key

        chunk_result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == retried_document.id)
        )
        chunks = chunk_result.scalars().all()

        assert chunks
        assert all(chunk.embedding is not None for chunk in chunks)


@pytest.mark.asyncio
async def test_retry_failed_document_can_fail_again(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "research.md"
    storage_root = tmp_path / "storage"

    source_path.write_text(
        f"# Retry Failure\n\nRetrieval content {uuid4()}",
        encoding="utf-8",
    )

    class FailingEmbeddingProvider(
        DeterministicEmbeddingProvider,
    ):
        def embed_batch(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            raise RuntimeError("retry embedding failure")

    async with async_session_factory() as session:
        initial_service = create_ingestion_service(
            session,
            storage_root,
            embedding_provider=FailingEmbeddingProvider(
                dimensions=384,
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="retry embedding failure",
        ):
            await initial_service.ingest_file(
                path=source_path,
                loader=MarkdownLoader(),
                title="Retry Failure",
                document_type="research_paper",
                source="integration-test",
            )

        result = await session.execute(
            select(Document)
            .where(Document.title == "Retry Failure")
            .order_by(Document.created_at.desc())
        )
        document = result.scalars().first()

        assert document is not None

        original_attempt = document.processing_attempt

        retry_service = create_ingestion_service(
            session,
            storage_root,
            embedding_provider=FailingEmbeddingProvider(
                dimensions=384,
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="retry embedding failure",
        ):
            await retry_service.retry_document(
                document.id,
                loader=MarkdownLoader(),
            )

        result = await session.execute(select(Document).where(Document.id == document.id))
        document = result.scalar_one()

        assert document.status == DocumentStatus.FAILED
        assert document.processing_attempt == (original_attempt + 1)
        assert document.last_error == ("retry embedding failure")
        assert document.failed_at is not None


@pytest.mark.asyncio
async def test_failed_retry_does_not_affect_current_ready_version(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage"

    async with async_session_factory() as session:
        logical_document_id = uuid4()

        current_document = Document(
            title="Current Version",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"current-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        failed_document = Document(
            title="Failed Version",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"failed-{uuid4()}",
            version_number=2,
            is_current=False,
            status=DocumentStatus.FAILED,
            processing_attempt=1,
            last_error="original failure",
            document_metadata={},
        )

        session.add_all(
            [
                current_document,
                failed_document,
            ]
        )
        await session.flush()

        current_document_id = current_document.id
        failed_document_id = failed_document.id

        # Give the failed version an existing stored file (blob + reference
        # -- StoredFile.content_hash now has a foreign key into file_blobs).
        content_hash = f"stored-file-test-{uuid4()}"
        storage_key = f"blobs/{content_hash[:2]}/{content_hash}"

        file_bytes = b"# Failed Version\n\nretry content"

        file_storage = LocalFileStorage(storage_root)

        await file_storage.store(
            file_bytes,
            storage_key,
        )

        blob = FileBlob(
            content_hash=content_hash,
            storage_key=storage_key,
            size_bytes=len(file_bytes),
            status=BlobStatus.READY,
        )
        session.add(blob)
        await session.flush()

        stored_file = StoredFile(
            document_id=failed_document_id,
            original_filename="failed.md",
            content_hash=content_hash,
            size_bytes=len(file_bytes),
            storage_key=storage_key,
        )

        session.add(stored_file)
        await session.commit()

        class FailingEmbeddingProvider(
            DeterministicEmbeddingProvider,
        ):
            def embed_batch(
                self,
                texts: list[str],
            ) -> list[list[float]]:
                raise RuntimeError("retry failed again")

        service = create_ingestion_service(
            session,
            storage_root,
            embedding_provider=FailingEmbeddingProvider(
                dimensions=384,
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="retry failed again",
        ):
            await service.retry_document(
                failed_document_id,
                loader=MarkdownLoader(),
            )

        result = await session.execute(select(Document).where(Document.id == current_document_id))
        current_document = result.scalar_one()

        result = await session.execute(select(Document).where(Document.id == failed_document_id))
        failed_document = result.scalar_one()

        assert current_document.status == DocumentStatus.READY
        assert current_document.is_current is True

        assert failed_document.status == DocumentStatus.FAILED
        assert failed_document.is_current is False


@pytest.mark.asyncio
async def test_retry_rejects_non_failed_document(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage"

    async with async_session_factory() as session:
        document = Document(
            title=f"Ready Document {uuid4()}",
            document_type="research_paper",
            logical_document_id=uuid4(),
            content_hash=f"ready-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        session.add(document)
        await session.commit()

        service = create_ingestion_service(
            session,
            storage_root,
        )

        with pytest.raises(
            ValueError,
            match="Only FAILED documents can be retried",
        ):
            await service.retry_document(
                document.id,
                loader=MarkdownLoader(),
            )


@pytest.mark.asyncio
async def test_retry_requires_stored_file(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage"

    async with async_session_factory() as session:
        document = Document(
            title=f"Missing File {uuid4()}",
            document_type="research_paper",
            logical_document_id=uuid4(),
            content_hash=f"missing-file-{uuid4()}",
            version_number=1,
            is_current=False,
            status=DocumentStatus.FAILED,
            document_metadata={},
        )

        session.add(document)
        await session.commit()

        service = create_ingestion_service(
            session,
            storage_root,
        )

        with pytest.raises(
            ValueError,
            match="No stored file found",
        ):
            await service.retry_document(
                document.id,
                loader=MarkdownLoader(),
            )

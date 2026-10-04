from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import (
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentStatus,
    StoredFile,
)
from app.core.repositories.document import DocumentRepository
from app.core.services.document_deletion import DocumentDeletionService
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.retrieval.service import RetrievalService
from app.storage.interface import FileStorage


class LocalTestStorage(FileStorage):
    def __init__(self, root: Path) -> None:
        self.root = root

    async def store(self, content: bytes, storage_key: str) -> None:
        path = self.root / storage_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    async def retrieve(self, storage_key: str) -> bytes:
        return (self.root / storage_key).read_bytes()

    async def delete(self, storage_key: str) -> None:
        (self.root / storage_key).unlink()

    async def exists(self, storage_key: str) -> bool:
        return (self.root / storage_key).is_file()


class FailingDeleteStorage(FileStorage):
    """File storage test double that fails during physical deletion."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def store(self, content: bytes, storage_key: str) -> None:
        path = self.root / storage_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    async def retrieve(self, storage_key: str) -> bytes:
        return (self.root / storage_key).read_bytes()

    async def delete(self, storage_key: str) -> None:
        raise OSError("Simulated physical file deletion failure")

    async def exists(self, storage_key: str) -> bool:
        return (self.root / storage_key).is_file()


@pytest.mark.asyncio
async def test_delete_version_removes_database_records_and_physical_file(
    tmp_path: Path,
) -> None:
    storage = FailingDeleteStorage(tmp_path)

    # We need successful physical deletion for this test, so override the
    # failing implementation with a small concrete storage implementation.
    class SuccessfulDeleteStorage(FailingDeleteStorage):
        async def delete(self, storage_key: str) -> None:
            (self.root / storage_key).unlink()

    storage = SuccessfulDeleteStorage(tmp_path)

    logical_document_id = uuid4()
    document_id = uuid4()
    stored_file_id = uuid4()
    storage_key = f"documents/{logical_document_id}/{document_id}/{stored_file_id}"

    file_path = tmp_path / storage_key
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"original document bytes")

    async with async_session_factory() as session:
        document = Document(
            id=document_id,
            title="Deletable Document",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"content-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        stored_file = StoredFile(
            id=stored_file_id,
            document_id=document_id,
            original_filename="research.pdf",
            content_hash=f"file-{uuid4()}",
            size_bytes=len(b"original document bytes"),
            storage_key=storage_key,
        )

        session.add_all([document, stored_file])
        await session.commit()

        service = DocumentDeletionService(
            session=session,
            file_storage=storage,
        )

        await service.delete_version(document_id)

        deleted_document = await session.get(Document, document_id)
        deleted_stored_file = await session.get(StoredFile, stored_file_id)

        assert deleted_document is None
        assert deleted_stored_file is None
        assert not file_path.exists()


@pytest.mark.asyncio
async def test_delete_version_keeps_database_clean_when_physical_delete_fails(
    tmp_path: Path,
) -> None:
    storage = FailingDeleteStorage(tmp_path)

    logical_document_id = uuid4()
    document_id = uuid4()
    stored_file_id = uuid4()
    storage_key = f"documents/{logical_document_id}/{document_id}/{stored_file_id}"

    file_path = tmp_path / storage_key
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"original document bytes")

    async with async_session_factory() as session:
        document = Document(
            id=document_id,
            title="Deletion Failure Document",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"content-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        stored_file = StoredFile(
            id=stored_file_id,
            document_id=document_id,
            original_filename="research.pdf",
            content_hash=f"file-{uuid4()}",
            size_bytes=len(b"original document bytes"),
            storage_key=storage_key,
        )

        session.add_all([document, stored_file])
        await session.commit()

        service = DocumentDeletionService(
            session=session,
            file_storage=storage,
        )

        with pytest.raises(OSError, match="Simulated physical file deletion failure"):
            await service.delete_version(document_id)

        deleted_document = await session.get(Document, document_id)
        deleted_stored_file = await session.get(StoredFile, stored_file_id)

        assert deleted_document is None
        assert deleted_stored_file is None
        assert file_path.exists()


@pytest.mark.asyncio
async def test_delete_logical_document_removes_all_versions_and_files(
    tmp_path: Path,
) -> None:
    storage = LocalTestStorage(tmp_path)

    logical_document_id = uuid4()

    async with async_session_factory() as session:
        documents: list[Document] = []
        stored_files: list[StoredFile] = []

        for version_number in (1, 2):
            document = Document(
                title=f"Version {version_number}",
                document_type="research_paper",
                logical_document_id=logical_document_id,
                content_hash=f"content-{uuid4()}",
                version_number=version_number,
                is_current=version_number == 2,
                status=DocumentStatus.READY,
                document_metadata={},
            )
            session.add(document)
            await session.flush()

            storage_key = f"documents/{logical_document_id}/{document.id}/{uuid4()}"

            file_content = f"version {version_number}".encode()
            file_path = tmp_path / storage_key
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(file_content)

            stored_file = StoredFile(
                document_id=document.id,
                original_filename=f"version-{version_number}.pdf",
                content_hash=f"file-{uuid4()}",
                size_bytes=len(file_content),
                storage_key=storage_key,
            )

            session.add(stored_file)

            documents.append(document)
            stored_files.append(stored_file)

        await session.commit()

        document_ids = [document.id for document in documents]
        storage_keys = [stored_file.storage_key for stored_file in stored_files]

        service = DocumentDeletionService(
            session=session,
            file_storage=storage,
        )

        await service.delete_logical_document(logical_document_id)

        for document_id in document_ids:
            assert await session.get(Document, document_id) is None

        for stored_file in stored_files:
            assert await session.get(StoredFile, stored_file.id) is None

        for storage_key in storage_keys:
            assert not (tmp_path / storage_key).exists()


@pytest.mark.asyncio
async def test_delete_logical_document_rejects_unknown_logical_document() -> None:
    logical_document_id = uuid4()

    async with async_session_factory() as session:
        service = DocumentDeletionService(
            session=session,
            file_storage=LocalTestStorage(Path.cwd()),
        )

        with pytest.raises(
            ValueError,
            match=f"Logical document not found: {logical_document_id}",
        ):
            await service.delete_logical_document(logical_document_id)


@pytest.mark.asyncio
async def test_delete_logical_document_keeps_database_clean_when_file_cleanup_fails(
    tmp_path: Path,
) -> None:
    storage = FailingDeleteStorage(tmp_path)

    logical_document_id = uuid4()

    async with async_session_factory() as session:
        document = Document(
            title="Cleanup Failure Document",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"content-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        storage_key = f"documents/{logical_document_id}/{document.id}/{uuid4()}"

        file_content = b"original document bytes"

        file_path = tmp_path / storage_key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(file_content)

        stored_file = StoredFile(
            document_id=document.id,
            original_filename="research.pdf",
            content_hash=f"file-{uuid4()}",
            size_bytes=len(file_content),
            storage_key=storage_key,
        )

        session.add(stored_file)
        await session.commit()

        document_id = document.id
        stored_file_id = stored_file.id

        service = DocumentDeletionService(
            session=session,
            file_storage=storage,
        )

        with pytest.raises(
            OSError,
            match="Simulated physical file deletion failure",
        ):
            await service.delete_logical_document(logical_document_id)

        assert await session.get(Document, document_id) is None
        assert await session.get(StoredFile, stored_file_id) is None

        # The physical file is intentionally left behind because
        # FileStorage.delete() failed after the DB commit.
        assert file_path.exists()


@pytest.mark.asyncio
async def test_delete_logical_document_removes_rag_records(
    tmp_path: Path,
) -> None:
    storage = LocalTestStorage(tmp_path)

    logical_document_id = uuid4()

    async with async_session_factory() as session:
        document = Document(
            title="RAG Cleanup Document",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"content-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )
        session.add(document)
        await session.flush()

        stored_file = StoredFile(
            document_id=document.id,
            original_filename="research.pdf",
            content_hash=f"file-{uuid4()}",
            size_bytes=10,
            storage_key=(f"documents/{logical_document_id}/{document.id}/{uuid4()}"),
        )

        page = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Page content",
        )

        section = DocumentSection(
            document_id=document.id,
            title="Introduction",
            section_path="1",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add_all([stored_file, page, section])
        await session.flush()

        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content="Searchable chunk content",
            chunk_metadata={},
            section_id=section.id,
            embedding=[0.0] * 384,
        )

        session.add(chunk)
        await session.flush()

        chunk_page_map = ChunkPageMap(
            chunk_id=chunk.id,
            document_page_id=page.id,
        )

        session.add(chunk_page_map)

        file_path = tmp_path / stored_file.storage_key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"original file")

        await session.commit()

        document_id = document.id
        stored_file_id = stored_file.id
        page_id = page.id
        section_id = section.id
        chunk_id = chunk.id

        service = DocumentDeletionService(
            session=session,
            file_storage=storage,
        )

        await service.delete_logical_document(logical_document_id)

        assert await session.get(Document, document_id) is None
        assert await session.get(StoredFile, stored_file_id) is None
        assert await session.get(DocumentPage, page_id) is None
        assert await session.get(DocumentSection, section_id) is None
        assert await session.get(DocumentChunk, chunk_id) is None

        mapping_result = await session.execute(
            select(ChunkPageMap).where(
                ChunkPageMap.chunk_id == chunk_id,
            )
        )
        assert mapping_result.scalar_one_or_none() is None

        assert not file_path.exists()


@pytest.mark.asyncio
async def test_delete_logical_document_removes_versions_in_every_status(
    tmp_path: Path,
) -> None:
    """The entire logical document is removed regardless of each version's
    lifecycle status -- READY, FAILED, and PROCESSING versions are all
    deleted together."""

    storage = LocalTestStorage(tmp_path)

    logical_document_id = uuid4()

    async with async_session_factory() as session:
        documents: list[Document] = []
        stored_files: list[StoredFile] = []

        for version_number, document_status in enumerate(
            (DocumentStatus.READY, DocumentStatus.FAILED, DocumentStatus.PROCESSING),
            start=1,
        ):
            document = Document(
                title=f"Version {version_number} ({document_status})",
                document_type="research_paper",
                logical_document_id=logical_document_id,
                content_hash=f"content-{uuid4()}",
                version_number=version_number,
                is_current=document_status == DocumentStatus.READY,
                status=document_status,
                document_metadata={},
            )
            session.add(document)
            await session.flush()

            storage_key = f"documents/{logical_document_id}/{document.id}/{uuid4()}"

            file_content = f"version {version_number}".encode()
            file_path = tmp_path / storage_key
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(file_content)

            stored_file = StoredFile(
                document_id=document.id,
                original_filename=f"version-{version_number}.pdf",
                content_hash=f"file-{uuid4()}",
                size_bytes=len(file_content),
                storage_key=storage_key,
            )

            session.add(stored_file)

            documents.append(document)
            stored_files.append(stored_file)

        await session.commit()

        document_ids = [document.id for document in documents]
        storage_keys = [stored_file.storage_key for stored_file in stored_files]

        service = DocumentDeletionService(
            session=session,
            file_storage=storage,
        )

        await service.delete_logical_document(logical_document_id)

        for document_id in document_ids:
            assert await session.get(Document, document_id) is None

        for stored_file in stored_files:
            assert await session.get(StoredFile, stored_file.id) is None

        for storage_key in storage_keys:
            assert not (tmp_path / storage_key).exists()


@pytest.mark.asyncio
async def test_delete_logical_document_does_not_affect_another_logical_document(
    tmp_path: Path,
) -> None:
    storage = LocalTestStorage(tmp_path)

    deleted_logical_document_id = uuid4()
    untouched_logical_document_id = uuid4()

    async with async_session_factory() as session:
        deleted_document = Document(
            title="Deleted Document",
            document_type="research_paper",
            logical_document_id=deleted_logical_document_id,
            content_hash=f"content-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )
        untouched_document = Document(
            title="Untouched Document",
            document_type="research_paper",
            logical_document_id=untouched_logical_document_id,
            content_hash=f"content-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        session.add_all([deleted_document, untouched_document])
        await session.flush()

        deleted_storage_key = (
            f"documents/{deleted_logical_document_id}/{deleted_document.id}/{uuid4()}"
        )
        untouched_storage_key = (
            f"documents/{untouched_logical_document_id}/{untouched_document.id}/{uuid4()}"
        )

        for storage_key in (deleted_storage_key, untouched_storage_key):
            file_path = tmp_path / storage_key
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"content")

        deleted_stored_file = StoredFile(
            document_id=deleted_document.id,
            original_filename="deleted.pdf",
            content_hash=f"file-{uuid4()}",
            size_bytes=7,
            storage_key=deleted_storage_key,
        )
        untouched_stored_file = StoredFile(
            document_id=untouched_document.id,
            original_filename="untouched.pdf",
            content_hash=f"file-{uuid4()}",
            size_bytes=7,
            storage_key=untouched_storage_key,
        )

        session.add_all([deleted_stored_file, untouched_stored_file])
        await session.commit()

        untouched_document_id = untouched_document.id
        untouched_stored_file_id = untouched_stored_file.id

        service = DocumentDeletionService(
            session=session,
            file_storage=storage,
        )

        await service.delete_logical_document(deleted_logical_document_id)

        assert await session.get(Document, deleted_document.id) is None
        assert await session.get(StoredFile, deleted_stored_file.id) is None
        assert not (tmp_path / deleted_storage_key).exists()

        # The other logical document is completely untouched.
        assert await session.get(Document, untouched_document_id) is not None
        assert await session.get(StoredFile, untouched_stored_file_id) is not None
        assert (tmp_path / untouched_storage_key).exists()


@pytest.mark.asyncio
async def test_delete_logical_document_chunks_are_not_retrievable_after_deletion(
    tmp_path: Path,
) -> None:
    """Integration-level assertion using the existing retrieval path (not
    just raw row counts): after deletion, no chunk from any version of the
    logical document is returned by RetrievalService.search."""

    storage = LocalTestStorage(tmp_path)
    provider = DeterministicEmbeddingProvider(dimensions=384)

    logical_document_id = uuid4()
    searchable_content = f"retrieval augmented generation {uuid4()}"

    async with async_session_factory() as session:
        document = Document(
            title="Searchable Document",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"content-{uuid4()}",
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

        chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=searchable_content,
            chunk_metadata={},
            embedding=provider.embed_text(searchable_content),
        )
        session.add(chunk)
        await session.flush()

        storage_key = f"documents/{logical_document_id}/{document.id}/{uuid4()}"
        file_path = tmp_path / storage_key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"original file")

        stored_file = StoredFile(
            document_id=document.id,
            original_filename="research.pdf",
            content_hash=f"file-{uuid4()}",
            size_bytes=13,
            storage_key=storage_key,
        )
        session.add(stored_file)

        await session.commit()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        # Sanity check: the chunk is retrievable before deletion.
        results_before = await retrieval.search(searchable_content, limit=5)
        assert any(result.chunk_id == chunk.id for result in results_before)

        service = DocumentDeletionService(
            session=session,
            file_storage=storage,
        )

        await service.delete_logical_document(logical_document_id)

        # No chunk from this logical document is retrievable after deletion.
        results_after = await retrieval.search(searchable_content, limit=5)
        assert all(result.chunk_id != chunk.id for result in results_after)
        assert all(result.document_id != document.id for result in results_after)

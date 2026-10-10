"""Tests for the end-of-session document-data cleanup helper in
`tests/conftest.py`.

These call `cleanup_document_test_data()` directly, as a plain async
function -- they do not trigger `pytest_sessionfinish` and do not start a
nested pytest session. `pytest_sessionfinish` itself only runs once, for
real, after the whole suite (including these tests) has finished.

`cleanup_document_test_data()` wipes ALL document-lifecycle rows globally
(it has no per-test scoping -- that's the point, it's the same mechanism
used as the final session-wide backstop). This is safe to call mid-suite
because no test in this project relies on another test's previously
committed data still being present.
"""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.config.settings import get_settings
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
from tests.conftest import cleanup_document_test_data


@pytest.mark.asyncio
async def test_cleanup_is_a_noop_when_nothing_to_clean() -> None:
    """A bulk DELETE matching zero rows, and a storage root with nothing to
    remove, must both succeed without raising."""

    await cleanup_document_test_data()


@pytest.mark.asyncio
async def test_cleanup_removes_all_document_lifecycle_rows_and_files() -> None:
    """One row of every document-lifecycle type, plus one physical file,
    must all be removed by a single cleanup call."""

    document_id = uuid4()
    logical_document_id = uuid4()
    stored_file_id = uuid4()
    blob_content_hash = f"session-cleanup-blob-{uuid4()}"

    storage_root = Path(get_settings().storage_root)
    storage_key = f"blobs/{blob_content_hash[:2]}/{blob_content_hash}"
    file_path = storage_root / storage_key
    file_content = b"session cleanup helper test fixture"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(file_content)

    async with async_session_factory() as session:
        document = Document(
            id=document_id,
            title="Session Cleanup Helper Test Fixture",
            document_type="research_paper",
            logical_document_id=logical_document_id,
            content_hash=f"session-cleanup-test-{uuid4()}",
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )
        session.add(document)
        await session.flush()

        blob = FileBlob(
            content_hash=blob_content_hash,
            storage_key=storage_key,
            size_bytes=len(file_content),
            status=BlobStatus.READY,
        )
        session.add(blob)
        await session.flush()

        stored_file = StoredFile(
            id=stored_file_id,
            document_id=document.id,
            original_filename="session-cleanup-test.md",
            content_hash=blob_content_hash,
            size_bytes=len(file_content),
            storage_key=storage_key,
        )

        section = DocumentSection(
            document_id=document.id,
            title="Section",
            section_path="1",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        page = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Page content",
        )

        session.add_all([stored_file, section, page])
        await session.flush()

        chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content="Chunk content",
            chunk_metadata={},
            embedding=[0.0] * 384,
        )
        session.add(chunk)
        await session.flush()

        session.add(ChunkPageMap(chunk_id=chunk.id, document_page_id=page.id))
        await session.commit()

        chunk_id = chunk.id
        page_id = page.id
        section_id = section.id

    assert file_path.exists()

    await cleanup_document_test_data()

    assert not file_path.exists()
    assert storage_root.is_dir()
    assert (storage_root / ".gitkeep").exists()

    async with async_session_factory() as verification_session:
        assert await verification_session.get(Document, document_id) is None
        assert await verification_session.get(StoredFile, stored_file_id) is None
        assert await verification_session.get(FileBlob, blob_content_hash) is None
        assert await verification_session.get(DocumentPage, page_id) is None
        assert await verification_session.get(DocumentSection, section_id) is None
        assert await verification_session.get(DocumentChunk, chunk_id) is None

        mapping = (
            await verification_session.execute(
                select(ChunkPageMap).where(ChunkPageMap.chunk_id == chunk_id)
            )
        ).scalar_one_or_none()
        assert mapping is None


@pytest.mark.asyncio
async def test_cleanup_called_twice_in_a_row_is_still_a_noop() -> None:
    """Idempotency: calling cleanup when the previous call already removed
    everything must not raise."""

    await cleanup_document_test_data()
    await cleanup_document_test_data()

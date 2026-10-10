"""Targeted tests for: global content-duplicate detection across filenames
and categories, duplicate-upload metadata handling, and the BlobService
PENDING-recovery state machine. Runs against the real configured database
(no separate test database), with uniquely-suffixed content on every test
and explicit cleanup, consistent with the rest of this suite.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.database import async_session_factory
from app.core.models import BlobStatus, Document, FileBlob
from app.core.services.document import DocumentService
from app.ingestion.blob_service import BlobService
from app.ingestion.schemas import DocumentInput
from app.storage.local import LocalFileStorage


async def _cleanup_document(document: Document) -> None:
    async with async_session_factory() as session:
        fresh = await session.get(Document, document.id)
        if fresh is not None:
            await session.delete(fresh)
            await session.commit()


@pytest.mark.asyncio
async def test_duplicate_detected_across_different_filename_and_category() -> None:
    """Same normalized content, uploaded under a different title AND a
    different document_type, with NO shared logical_document_id -- must
    still resolve to the same existing Document (global dedup, not scoped
    by filename or category)."""

    unique_text = f"Shared normalized content across categories {uuid4()}."

    async with async_session_factory() as session:
        service = DocumentService(session)

        first_input = DocumentInput(
            title="Research Write-Up",
            document_type="research_paper",
            content=unique_text,
        )
        first_document, first_match = await service.ingest_as_new_or_duplicate(first_input)

        second_input = DocumentInput(
            title="Completely Different Title",
            document_type="manual",
            content=unique_text,
        )
        second_document, second_match = await service.ingest_as_new_or_duplicate(second_input)

        assert first_match.matched is False
        assert second_match.matched is True
        assert second_document.id == first_document.id
        # The second call's different title/document_type are metadata,
        # applied to the SAME existing row -- not a new logical document.
        assert second_document.logical_document_id == first_document.logical_document_id

    await _cleanup_document(first_document)


@pytest.mark.asyncio
async def test_duplicate_upload_applies_submitted_metadata_without_clearing_omitted() -> None:
    """Approved behavior: a duplicate upload's explicitly submitted,
    non-empty metadata is applied to the existing matched document; fields
    the second upload did not (meaningfully) submit are left untouched."""

    unique_text = f"Metadata application test content {uuid4()}."

    async with async_session_factory() as session:
        service = DocumentService(session)

        first_input = DocumentInput(
            title="Original Title",
            source="original-source",
            document_type="research_paper",
            content=unique_text,
        )
        first_document, _ = await service.ingest_as_new_or_duplicate(first_input)

        assert first_document.title == "Original Title"
        assert first_document.source == "original-source"

        # Second upload: new title, SAME document_type, no source at all
        # (None -- not submitted).
        second_input = DocumentInput(
            title="Updated Title",
            source=None,
            document_type="research_paper",
            content=unique_text,
        )
        second_document, second_match = await service.ingest_as_new_or_duplicate(second_input)

        assert second_match.matched is True
        assert second_document.id == first_document.id
        assert second_document.title == "Updated Title"
        assert "title" in second_match.updated_metadata_fields
        # document_type submitted but unchanged -- not reported as updated.
        assert "document_type" not in second_match.updated_metadata_fields
        # source was not submitted (None) -- original value preserved, not
        # cleared.
        assert second_document.source == "original-source"
        assert "source" not in second_match.updated_metadata_fields

    await _cleanup_document(first_document)


@pytest.mark.asyncio
async def test_duplicate_upload_with_no_changed_metadata_reports_no_updates() -> None:
    unique_text = f"No metadata change test {uuid4()}."

    async with async_session_factory() as session:
        service = DocumentService(session)

        document_input = DocumentInput(
            title="Stable Title",
            source="stable-source",
            document_type="research_paper",
            content=unique_text,
        )
        first_document, _ = await service.ingest_as_new_or_duplicate(document_input)

        second_document, second_match = await service.ingest_as_new_or_duplicate(document_input)

        assert second_match.matched is True
        assert second_document.id == first_document.id
        assert second_match.updated_metadata_fields == []

    await _cleanup_document(first_document)


@pytest.mark.asyncio
async def test_reconcile_pending_blobs_finishes_a_blob_whose_bytes_already_exist(
    tmp_path: Path,
) -> None:
    """Simulates a crash AFTER the physical write but BEFORE the DB commit
    that would have marked the blob READY: reconciliation must detect the
    existing bytes and mark the row READY, never re-triggering a write."""

    file_storage = LocalFileStorage(tmp_path / "storage")
    content = f"Stale pending blob with bytes already written {uuid4()}".encode()

    import hashlib

    content_hash = hashlib.sha256(content).hexdigest()
    from app.storage.keys import build_blob_storage_key

    storage_key = build_blob_storage_key(content_hash)

    await file_storage.store(content, storage_key)

    stale_created_at = datetime.now(UTC) - timedelta(hours=1)

    async with async_session_factory() as session:
        blob = FileBlob(
            content_hash=content_hash,
            storage_key=storage_key,
            size_bytes=len(content),
            status=BlobStatus.PENDING,
            created_at=stale_created_at,
        )
        session.add(blob)
        await session.commit()

        blob_service = BlobService(session, file_storage=file_storage)
        acted_on = await blob_service.reconcile_pending_blobs(stale_after=timedelta(minutes=15))

        assert content_hash in acted_on

        refreshed = await session.get(FileBlob, content_hash)
        assert refreshed.status == BlobStatus.READY
        assert refreshed.ready_at is not None

        await session.delete(refreshed)
        await session.commit()


@pytest.mark.asyncio
async def test_reconcile_pending_blobs_removes_a_blob_with_no_bytes(
    tmp_path: Path,
) -> None:
    """Simulates a crash BEFORE the physical write ever happened:
    reconciliation must delete the orphaned PENDING row (nothing to
    recover), never leave it reusable by a future caller."""

    file_storage = LocalFileStorage(tmp_path / "storage")
    content_hash = f"never-written-{uuid4()}"
    storage_key = f"blobs/{content_hash[:2]}/{content_hash}"

    stale_created_at = datetime.now(UTC) - timedelta(hours=1)

    async with async_session_factory() as session:
        blob = FileBlob(
            content_hash=content_hash,
            storage_key=storage_key,
            size_bytes=0,
            status=BlobStatus.PENDING,
            created_at=stale_created_at,
        )
        session.add(blob)
        await session.commit()

        blob_service = BlobService(session, file_storage=file_storage)
        acted_on = await blob_service.reconcile_pending_blobs(stale_after=timedelta(minutes=15))

        assert content_hash in acted_on

        refreshed = await session.get(FileBlob, content_hash)
        assert refreshed is None


@pytest.mark.asyncio
async def test_reconcile_pending_blobs_leaves_recent_pending_rows_alone(
    tmp_path: Path,
) -> None:
    """A PENDING row younger than the staleness cutoff is presumed to be a
    legitimate in-flight write -- reconciliation must not touch it."""

    file_storage = LocalFileStorage(tmp_path / "storage")
    content_hash = f"recent-pending-{uuid4()}"
    storage_key = f"blobs/{content_hash[:2]}/{content_hash}"

    async with async_session_factory() as session:
        blob = FileBlob(
            content_hash=content_hash,
            storage_key=storage_key,
            size_bytes=0,
            status=BlobStatus.PENDING,
        )
        session.add(blob)
        await session.commit()

        blob_service = BlobService(session, file_storage=file_storage)
        acted_on = await blob_service.reconcile_pending_blobs(stale_after=timedelta(minutes=15))

        assert content_hash not in acted_on

        refreshed = await session.get(FileBlob, content_hash)
        assert refreshed is not None
        assert refreshed.status == BlobStatus.PENDING

        await session.delete(refreshed)
        await session.commit()


@pytest.mark.asyncio
async def test_get_or_create_ready_blob_self_heals_someone_elses_pending_row(
    tmp_path: Path,
) -> None:
    """If a PENDING row for this exact content already exists (e.g. left by
    another, possibly crashed, request) and its bytes are missing,
    `get_or_create_ready_blob` must self-heal by writing them itself and
    marking the row READY, rather than failing or returning an unusable
    blob."""

    file_storage = LocalFileStorage(tmp_path / "storage")
    content = f"Self-heal test content {uuid4()}".encode()

    import hashlib

    content_hash = hashlib.sha256(content).hexdigest()
    from app.storage.keys import build_blob_storage_key

    storage_key = build_blob_storage_key(content_hash)

    async with async_session_factory() as session:
        # Simulate another request's abandoned PENDING reservation --
        # created, but the physical write never happened.
        session.add(
            FileBlob(
                content_hash=content_hash,
                storage_key=storage_key,
                size_bytes=len(content),
                status=BlobStatus.PENDING,
            )
        )
        await session.commit()

        blob_service = BlobService(session, file_storage=file_storage)
        result = await blob_service.get_or_create_ready_blob(content)

        assert result.content_hash == content_hash

        retrieved = await file_storage.retrieve(result.storage_key)
        assert retrieved == content

        refreshed = await session.get(FileBlob, content_hash)
        assert refreshed.status == BlobStatus.READY

        await session.delete(refreshed)
        await session.commit()

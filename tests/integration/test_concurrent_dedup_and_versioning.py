"""Concurrency tests for global content deduplication, shared blob storage,
and safe version-number allocation.

These run genuinely concurrent coroutines (`asyncio.gather`), each with its
own `AsyncSession` (a single `AsyncSession` is not safe for concurrent use
from multiple coroutines), against the REAL configured database -- there is
no separate test database, per project convention (see `tests/conftest.py`'s
own documented investigation of this). This is not a fully deterministic,
barrier-synchronized race (the codebase has no injectable hook to pause a
request mid-transaction without adding test-only instrumentation to
production code, which would be scope creep for this task) -- it is two
real concurrent requests racing against a real Postgres instance, which is
sufficient to exercise the `ON CONFLICT`/unique-constraint recovery paths
this task added. Every row created here is cleaned up by the test itself
(no reliance on the session-wide `conftest.py` backstop alone), and a unique
UUID-suffixed content string is used in every case so no test can collide
with another test's or a previous run's data.
"""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import Document, FileBlob, StoredFile
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.normalizer import calculate_content_hash
from app.ingestion.schemas import IngestOutcome
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage


async def _cleanup_logical_document(logical_document_id) -> None:
    async with async_session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.logical_document_id == logical_document_id)
        )
        documents = result.scalars().all()

        for document in documents:
            stored_file_result = await session.execute(
                select(StoredFile).where(StoredFile.document_id == document.id)
            )
            for stored_file in stored_file_result.scalars().all():
                await session.delete(stored_file)

        await session.flush()

        for document in documents:
            await session.delete(document)

        await session.commit()


async def _cleanup_blob(content_hash: str) -> None:
    async with async_session_factory() as session:
        blob = await session.get(FileBlob, content_hash)
        if blob is not None:
            await session.delete(blob)
            await session.commit()


@pytest.mark.asyncio
async def test_concurrent_identical_uploads_converge_on_one_document(
    tmp_path: Path,
) -> None:
    """Two concurrent uploads of byte-identical content, with no caller-
    supplied logical_document_id, must converge on exactly one Document row
    -- never two, and never an unhandled exception from either request."""

    unique_text = f"Concurrent duplicate upload test {uuid4()}"
    document_path = tmp_path / "concurrent.md"
    document_path.write_text(f"# Concurrency\n\n{unique_text}", encoding="utf-8")

    file_storage = LocalFileStorage(tmp_path / "storage")

    async def run_upload():
        async with async_session_factory() as session:
            service = IngestionService(
                session,
                embedding_provider=DeterministicEmbeddingProvider(dimensions=384),
                file_storage=file_storage,
            )
            return await service.ingest_file(
                path=document_path,
                loader=MarkdownLoader(),
                title="Concurrent Duplicate Test",
                document_type="research_paper",
                source="integration-test",
            )

    results = await asyncio.gather(
        run_upload(),
        run_upload(),
        return_exceptions=True,
    )

    # Neither request may surface an unhandled exception.
    for result in results:
        assert not isinstance(result, BaseException), f"Unhandled exception: {result!r}"

    outcomes = [result.outcome for result in results]
    document_ids = {result.document.id for result in results}

    # Exactly one request created the document; the other recognized it as
    # a duplicate. Order between the two is not guaranteed.
    assert sorted(outcomes) == sorted([IngestOutcome.CREATED, IngestOutcome.DUPLICATE])
    assert len(document_ids) == 1

    content_hash = calculate_content_hash(f"# Concurrency\n\n{unique_text}")

    async with async_session_factory() as session:
        db_result = await session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        documents = db_result.scalars().all()
        assert len(documents) == 1

    logical_document_id = documents[0].logical_document_id
    stored_file_content_hash = None

    async with async_session_factory() as session:
        stored_file_result = await session.execute(
            select(StoredFile).where(StoredFile.document_id == documents[0].id)
        )
        stored_file = stored_file_result.scalar_one()
        stored_file_content_hash = stored_file.content_hash

    await _cleanup_logical_document(logical_document_id)
    await _cleanup_blob(stored_file_content_hash)


@pytest.mark.asyncio
async def test_concurrent_version_creation_never_duplicates_version_numbers(
    tmp_path: Path,
) -> None:
    """Two concurrent uploads of DIFFERENT content under the SAME explicit
    logical_document_id must end up with two distinct version numbers --
    never a duplicate, never an unhandled constraint violation."""

    logical_document_id = uuid4()
    unique_suffix = uuid4()

    file_storage = LocalFileStorage(tmp_path / "storage")

    async def run_upload(label: str):
        document_path = tmp_path / f"version-{label}.md"
        document_path.write_text(
            f"# Version {label}\n\nContent {unique_suffix} variant {label}",
            encoding="utf-8",
        )

        async with async_session_factory() as session:
            service = IngestionService(
                session,
                embedding_provider=DeterministicEmbeddingProvider(dimensions=384),
                file_storage=file_storage,
            )
            return await service.ingest_file(
                path=document_path,
                loader=MarkdownLoader(),
                title=f"Concurrent Version Test {label}",
                document_type="research_paper",
                source="integration-test",
                logical_document_id=logical_document_id,
            )

    results = await asyncio.gather(
        run_upload("A"),
        run_upload("B"),
        return_exceptions=True,
    )

    for result in results:
        assert not isinstance(result, BaseException), f"Unhandled exception: {result!r}"

    version_numbers = sorted(result.document.version_number for result in results)
    outcomes = [result.outcome for result in results]

    assert version_numbers == [1, 2]
    assert all(outcome == IngestOutcome.NEW_VERSION for outcome in outcomes)

    async with async_session_factory() as session:
        db_result = await session.execute(
            select(Document).where(Document.logical_document_id == logical_document_id)
        )
        documents = db_result.scalars().all()
        assert sorted(document.version_number for document in documents) == [1, 2]
        assert len({document.version_number for document in documents}) == 2

        stored_file_hashes = []
        for document in documents:
            stored_file_result = await session.execute(
                select(StoredFile).where(StoredFile.document_id == document.id)
            )
            stored_file_hashes.append(stored_file_result.scalar_one().content_hash)

    await _cleanup_logical_document(logical_document_id)

    for content_hash in stored_file_hashes:
        await _cleanup_blob(content_hash)


@pytest.mark.asyncio
async def test_concurrent_identical_raw_bytes_produce_one_physical_blob(
    tmp_path: Path,
) -> None:
    """Two concurrent ingestions of byte-identical content (as two DIFFERENT
    logical documents, via two distinct explicit logical_document_ids
    reached indirectly through two 'new document' uploads of non-matching
    titles but IDENTICAL file bytes) must still result in exactly one
    physical FileBlob row/file -- blob identity is global regardless of
    logical document identity.

    Note: because normalized-CONTENT dedup is global too, two uploads of
    byte-identical files always collapse to one Document as well (this is
    expected and asserted here) -- a scenario with two independently-owned
    logical documents sharing one blob but DIFFERENT normalized content
    would require different raw bytes with a shared... no such scenario
    exists for a single file upload, so this test documents the blob-layer
    guarantee using the same concurrent-duplicate path as the test above,
    with explicit blob-row assertions.
    """

    unique_text = f"Shared blob concurrency test {uuid4()}"
    document_path = tmp_path / "shared-blob.md"
    document_path.write_text(f"# Shared Blob\n\n{unique_text}", encoding="utf-8")
    raw_bytes = document_path.read_bytes()

    file_storage = LocalFileStorage(tmp_path / "storage")

    async def run_upload():
        async with async_session_factory() as session:
            service = IngestionService(
                session,
                embedding_provider=DeterministicEmbeddingProvider(dimensions=384),
                file_storage=file_storage,
            )
            return await service.ingest_file(
                path=document_path,
                loader=MarkdownLoader(),
                title="Shared Blob Test",
                document_type="research_paper",
                source="integration-test",
            )

    results = await asyncio.gather(
        run_upload(),
        run_upload(),
        return_exceptions=True,
    )

    for result in results:
        assert not isinstance(result, BaseException), f"Unhandled exception: {result!r}"

    import hashlib

    expected_hash = hashlib.sha256(raw_bytes).hexdigest()

    async with async_session_factory() as session:
        blob = await session.get(FileBlob, expected_hash)
        assert blob is not None
        assert blob.status == "READY"

        retrieved_bytes = await file_storage.retrieve(blob.storage_key)
        assert retrieved_bytes == raw_bytes

    document_ids = {result.document.id for result in results}
    assert len(document_ids) == 1

    logical_document_id = None
    async with async_session_factory() as session:
        document = await session.get(Document, next(iter(document_ids)))
        logical_document_id = document.logical_document_id

    await _cleanup_logical_document(logical_document_id)
    await _cleanup_blob(expected_hash)

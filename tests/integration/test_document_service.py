from uuid import uuid4

import pytest

from app.core.database import async_session_factory
from app.core.models import DocumentStatus
from app.core.services.document import DocumentService
from app.ingestion.schemas import DocumentInput


@pytest.mark.asyncio
async def test_ingest_document_deduplicates_content() -> None:
    document_input = DocumentInput(
        title="Test Research Paper",
        authors="Test Author",
        source="integration-test",
        document_type="research_paper",
        content="This is   a test research document.",
    )

    async with async_session_factory() as session:
        service = DocumentService(session)

        logical_document_id = uuid4()

        first_document, first_created = (
            await service.ingest_document(
                document_input,
                logical_document_id=logical_document_id,
            )
        )

        await session.commit()

        second_document, second_created = (
            await service.ingest_document(
                document_input,
                logical_document_id=logical_document_id,
            )
        )

        await session.commit()

        assert first_created is True
        assert second_created is False
        assert first_document.id == second_document.id

        await session.delete(first_document)
        await session.commit()


@pytest.mark.asyncio
async def test_changed_content_creates_new_version() -> None:
    logical_document_id = uuid4()

    first_input = DocumentInput(
        title="Versioned Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Original document content.",
    )

    second_input = DocumentInput(
        title="Versioned Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Updated document content.",
    )

    async with async_session_factory() as session:
        service = DocumentService(session)

        first_document, first_created = await service.ingest_document(
            first_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        await service.mark_processing(first_document)
        await service.mark_ready(first_document)
        await session.commit()

        second_document, second_created = await service.ingest_document(
            second_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        assert first_created is True
        assert second_created is True

        assert second_document.id != first_document.id
        assert second_document.logical_document_id == logical_document_id

        assert first_document.version_number == 1
        assert second_document.version_number == 2

        assert first_document.is_current is True
        assert second_document.is_current is False

        assert second_document.status == DocumentStatus.UPLOADED

        await session.delete(second_document)
        await session.delete(first_document)
        await session.commit()


@pytest.mark.asyncio
async def test_new_ready_version_replaces_current_version() -> None:
    logical_document_id = uuid4()

    first_input = DocumentInput(
        title="Versioned Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Original document content.",
    )

    second_input = DocumentInput(
        title="Versioned Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Updated document content.",
    )

    async with async_session_factory() as session:
        service = DocumentService(session)

        first_document, _ = await service.ingest_document(
            first_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        await service.mark_processing(first_document)
        await service.mark_ready(first_document)
        await session.commit()

        second_document, second_created = await service.ingest_document(
            second_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        assert second_created is True
        assert first_document.is_current is True
        assert second_document.is_current is False

        await service.mark_processing(second_document)
        await session.commit()

        await service.mark_ready(second_document)
        await session.commit()

        assert first_document.is_current is False
        assert second_document.is_current is True

        assert first_document.status == DocumentStatus.READY
        assert second_document.status == DocumentStatus.READY

        await session.delete(second_document)
        await session.delete(first_document)
        await session.commit()


@pytest.mark.asyncio
async def test_failed_new_version_preserves_current_version() -> None:
    logical_document_id = uuid4()

    first_input = DocumentInput(
        title="Versioned Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Original document content.",
    )

    second_input = DocumentInput(
        title="Versioned Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Updated document content.",
    )

    async with async_session_factory() as session:
        service = DocumentService(session)

        first_document, _ = await service.ingest_document(
            first_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        await service.mark_processing(first_document)
        await service.mark_ready(first_document)
        await session.commit()

        second_document, second_created = await service.ingest_document(
            second_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        assert second_created is True
        assert first_document.is_current is True
        assert second_document.is_current is False

        await service.mark_processing(second_document)
        await session.commit()

        await service.mark_failed(
            second_document,
            "Simulated processing failure",
        )
        await session.commit()

        assert first_document.status == DocumentStatus.READY
        assert first_document.is_current is True

        assert second_document.status == DocumentStatus.FAILED
        assert second_document.is_current is False
        assert second_document.last_error == "Simulated processing failure"

        await session.delete(second_document)
        await session.delete(first_document)
        await session.commit()



@pytest.mark.asyncio
async def test_failed_version_can_be_retried_without_creating_new_version() -> None:
    logical_document_id = uuid4()

    document_input = DocumentInput(
        title="Retryable Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Document content that initially fails.",
    )

    async with async_session_factory() as session:
        service = DocumentService(session)

        document, created = await service.ingest_document(
            document_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        assert created is True
        assert document.version_number == 1
        assert document.processing_attempt == 0

        await service.mark_processing(document)
        await session.commit()

        assert document.processing_attempt == 1
        assert document.status == DocumentStatus.PROCESSING

        await service.mark_failed(
            document,
            "Simulated processing failure",
        )
        await session.commit()

        assert document.status == DocumentStatus.FAILED
        assert document.is_current is False
        assert document.processing_attempt == 1

        retried_document, retry_created = await service.ingest_document(
            document_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        assert retry_created is False
        assert retried_document.id == document.id
        assert retried_document.version_number == 1
        assert retried_document.logical_document_id == logical_document_id

        await session.delete(document)
        await session.commit()



@pytest.mark.asyncio
async def test_processing_version_is_reused_without_creating_new_version() -> None:
    logical_document_id = uuid4()

    first_input = DocumentInput(
        title="Versioned Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Original document content.",
    )

    second_input = DocumentInput(
        title="Versioned Document",
        authors="Test Author",
        source="integration-test",
        publication_date=None,
        document_type="research_paper",
        content="Updated document content.",
    )

    async with async_session_factory() as session:
        service = DocumentService(session)

        first_document, _ = await service.ingest_document(
            first_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        await service.mark_processing(first_document)
        await service.mark_ready(first_document)
        await session.commit()

        second_document, second_created = await service.ingest_document(
            second_input,
            logical_document_id=logical_document_id,
        )
        await session.commit()

        assert second_created is True
        assert second_document.version_number == 2
        assert second_document.is_current is False

        await service.mark_processing(second_document)
        await session.commit()

        assert second_document.status == DocumentStatus.PROCESSING
        assert second_document.processing_attempt == 1

        duplicate_document, duplicate_created = (
            await service.ingest_document(
                second_input,
                logical_document_id=logical_document_id,
            )
        )
        await session.commit()

        assert duplicate_created is False
        assert duplicate_document.id == second_document.id
        assert duplicate_document.version_number == 2
        assert duplicate_document.status == DocumentStatus.PROCESSING
        assert duplicate_document.processing_attempt == 1

        await session.delete(second_document)
        await session.delete(first_document)
        await session.commit()
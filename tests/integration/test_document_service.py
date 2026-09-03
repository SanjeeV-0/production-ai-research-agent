from uuid import uuid4

import pytest

from app.core.database import async_session_factory
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
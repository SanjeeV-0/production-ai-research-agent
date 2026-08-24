import pytest

from app.core.database import async_session_factory
from app.ingestion.schemas import DocumentInput
from app.ingestion.service import IngestionService


@pytest.mark.asyncio
async def test_ingestion_service_deduplicates_content() -> None:
    document_input = DocumentInput(
        title="Test Research Paper",
        authors="Test Author",
        source="integration-test",
        document_type="research_paper",
        content="This is   a test research document.",
    )

    async with async_session_factory() as session:
        service = IngestionService(session)

        first_document = await service.ingest(document_input)
        await session.commit()

        second_document = await service.ingest(document_input)
        await session.commit()

        assert first_document.id == second_document.id

        await session.delete(first_document)
        await session.commit()
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.database import async_session_factory
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.service import IngestionService


@pytest.mark.asyncio
async def test_ingestion_service_deduplicates_content(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "research.md"

    document_path.write_text(
        f"# RAG Research\n\n"
        f"Retrieval-Augmented Generation content {uuid4()}",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(),
        )

        first_document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
        )

        await session.commit()

        second_document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
        )

        await session.commit()

        assert first_document.id == second_document.id

        await session.delete(first_document)
        await session.commit()
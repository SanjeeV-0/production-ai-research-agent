from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import DocumentPage, StoredFile
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage


@pytest.mark.asyncio
async def test_ingest_file_persists_pages(tmp_path: Path) -> None:
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

        document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="RAG Research",
            document_type="research_paper",
            source="integration-test",
        )

        await session.commit()

        result = await session.execute(
            select(DocumentPage).where(DocumentPage.document_id == document.id)
        )

        page = result.scalar_one()

        assert page is not None
        assert page.document_id == document.id
        assert page.page_number == 1
        assert "Retrieval-Augmented Generation" in page.content

        result = await session.execute(
            select(StoredFile).where(
                StoredFile.document_id == document.id,
            )
        )
        stored_file = result.scalar_one()

        await session.delete(stored_file)
        await session.delete(document)
        await session.commit()

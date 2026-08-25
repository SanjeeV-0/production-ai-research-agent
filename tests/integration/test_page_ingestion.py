from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import DocumentPage
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.service import IngestionService


@pytest.mark.asyncio
async def test_ingest_file_persists_pages(tmp_path: Path) -> None:
    document_path = tmp_path / "research.md"

    document_path.write_text(
    f"# RAG Research\n\nRetrieval-Augmented Generation content {uuid4()}",
    encoding="utf-8",
)

    async with async_session_factory() as session:
        service = IngestionService(
    session,
    embedding_provider=DeterministicEmbeddingProvider(),
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
            select(DocumentPage).where(
                DocumentPage.document_id == document.id
            )
        )

        page = result.scalar_one()
        

        assert page is not None
        assert page.document_id == document.id
        assert page.page_number == 1
        assert "Retrieval-Augmented Generation" in page.content

        await session.delete(document)
        await session.commit()
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import DocumentPage, DocumentStatus
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.normalizer import calculate_content_hash
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
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
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


@pytest.mark.asyncio
async def test_ingestion_service_marks_document_ready(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "research.md"

    document_path.write_text(
        "# RAG Research\n\n"
        f"Retrieval content {uuid4()}",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
        )

        document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Test Research Paper",
            document_type="research_paper",
            source="integration-test",
        )

        assert document.status == DocumentStatus.READY
        assert document.processing_attempt == 1
        assert document.processing_started_at is not None
        assert document.processing_completed_at is not None
        assert document.failed_at is None
        assert document.last_error is None

        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_ingestion_service_marks_failed_and_rolls_back_processing(
    tmp_path: Path,
) -> None:
    content = (
        "# RAG Research\n\n"
        f"Retrieval content {uuid4()}"
    )

    document_path = tmp_path / "research.md"

    document_path.write_text(
        content,
        encoding="utf-8",
    )
    class FailingEmbeddingProvider(
        DeterministicEmbeddingProvider,
    ):
        def embed_batch(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            raise RuntimeError("embedding failure")

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=FailingEmbeddingProvider(
                dimensions=384,
            ),
        )

        with pytest.raises(RuntimeError, match="embedding failure"):
            await service.ingest_file(
                path=document_path,
                loader=MarkdownLoader(),
                title="Failed Research Paper",
                document_type="research_paper",
                source="integration-test",
            )

        content_hash = calculate_content_hash(content)

        document = await service.repository.get_by_content_hash(
            content_hash
        )

        assert document is not None
        assert document.status == DocumentStatus.FAILED
        assert document.processing_attempt == 1
        assert document.processing_started_at is not None
        assert document.processing_completed_at is not None
        assert document.failed_at is not None
        assert document.last_error == "embedding failure"

        page_result = await session.execute(
            select(DocumentPage).where(
                DocumentPage.document_id == document.id
            )
        )

        assert page_result.scalars().all() == []

        await session.delete(document)
        await session.commit()
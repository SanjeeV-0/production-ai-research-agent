from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.core.database import async_session_factory
from app.core.models import (
    ChunkPageMap,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
)
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.service import IngestionService


@pytest.mark.asyncio
async def test_duplicate_ingestion_does_not_duplicate_artifacts(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "duplicate.md"

    document_path.write_text(
        """# Results

Retrieval improves document search.

## Retrieval

The retrieval system improves recall.

# Discussion

The results are useful for future research.
""",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(dimensions=384,),
        )
        logical_document_id = uuid4()
        first_document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Duplicate Test",
            document_type="research_paper",
            source="integration-test",
            logical_document_id=logical_document_id,
        )

        await session.commit()

        # Capture the artifact counts after the first ingestion.
        page_count_result = await session.execute(
            select(func.count())
            .select_from(DocumentPage)
            .where(
                DocumentPage.document_id
                == first_document.id
            )
        )

        section_count_result = await session.execute(
            select(func.count())
            .select_from(DocumentSection)
            .where(
                DocumentSection.document_id
                == first_document.id
            )
        )

        chunk_count_result = await session.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.document_id
                == first_document.id
            )
        )

        mapping_count_result = await session.execute(
            select(func.count())
            .select_from(ChunkPageMap)
            .join(
                DocumentChunk,
                DocumentChunk.id
                == ChunkPageMap.chunk_id,
            )
            .where(
                DocumentChunk.document_id
                == first_document.id
            )
        )

        page_count = page_count_result.scalar_one()
        section_count = section_count_result.scalar_one()
        chunk_count = chunk_count_result.scalar_one()
        mapping_count = mapping_count_result.scalar_one()

        # Ingest exactly the same file again.
        second_document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Duplicate Test",
            document_type="research_paper",
            source="integration-test",
            logical_document_id=logical_document_id,
        )

        await session.commit()

        assert second_document.id == first_document.id

        # Counts must remain unchanged.
        page_count_after = await session.execute(
            select(func.count())
            .select_from(DocumentPage)
            .where(
                DocumentPage.document_id
                == first_document.id
            )
        )

        section_count_after = await session.execute(
            select(func.count())
            .select_from(DocumentSection)
            .where(
                DocumentSection.document_id
                == first_document.id
            )
        )

        chunk_count_after = await session.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.document_id
                == first_document.id
            )
        )

        mapping_count_after = await session.execute(
            select(func.count())
            .select_from(ChunkPageMap)
            .join(
                DocumentChunk,
                DocumentChunk.id
                == ChunkPageMap.chunk_id,
            )
            .where(
                DocumentChunk.document_id
                == first_document.id
            )
        )

        assert page_count_after.scalar_one() == page_count
        assert (
            section_count_after.scalar_one()
            == section_count
        )
        assert chunk_count_after.scalar_one() == chunk_count
        assert (
            mapping_count_after.scalar_one()
            == mapping_count
        )

        await session.delete(first_document)
        await session.commit()
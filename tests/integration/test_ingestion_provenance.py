from pathlib import Path

import pytest
from sqlalchemy import select

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
async def test_ingestion_preserves_section_and_page_provenance(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "provenance.md"

    document_path.write_text(
        """# Results

Retrieval improves document search.

## Retrieval

The retrieval system improves recall.

The second retrieval paragraph provides additional context.

## Evaluation

Evaluation confirms the improvement.

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

        document = await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Provenance Test",
            document_type="research_paper",
            source="integration-test",
        )

        await session.commit()

        # --------------------------------------------------
        # Sections
        # --------------------------------------------------
        section_result = await session.execute(
            select(DocumentSection)
            .where(
                DocumentSection.document_id == document.id
            )
            .order_by(
                DocumentSection.section_level,
                DocumentSection.section_index,
            )
        )

        sections = section_result.scalars().all()

        assert len(sections) == 4

        section_by_path = {
            section.section_path: section
            for section in sections
        }

        assert set(section_by_path) == {
            "Results",
            "Results > Retrieval",
            "Results > Evaluation",
            "Discussion",
        }

        results_section = section_by_path["Results"]
        retrieval_section = section_by_path[
            "Results > Retrieval"
        ]
        evaluation_section = section_by_path[
            "Results > Evaluation"
        ]
        discussion_section = section_by_path["Discussion"]

        assert results_section.parent_section_id is None
        assert retrieval_section.parent_section_id == (
            results_section.id
        )
        assert evaluation_section.parent_section_id == (
            results_section.id
        )
        assert discussion_section.parent_section_id is None

        # --------------------------------------------------
        # Pages
        # --------------------------------------------------
        page_result = await session.execute(
            select(DocumentPage)
            .where(
                DocumentPage.document_id == document.id
            )
            .order_by(DocumentPage.page_number)
        )

        pages = page_result.scalars().all()

        assert len(pages) == 1
        page = pages[0]

        # --------------------------------------------------
        # Chunks
        # --------------------------------------------------
        chunk_result = await session.execute(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == document.id
            )
            .order_by(DocumentChunk.chunk_index)
        )

        chunks = chunk_result.scalars().all()

        assert chunks

        # Every chunk must point to a real section
        section_ids = {
            section.id
            for section in sections
        }

        assert all(
            chunk.section_id in section_ids
            for chunk in chunks
        )

        # --------------------------------------------------
        # Section provenance
        # --------------------------------------------------
        for chunk in chunks:
            assert chunk.section_id in section_ids

        result_chunks = [
            chunk
            for chunk in chunks
            if chunk.section_id == results_section.id
        ]

        retrieval_chunks = [
            chunk
            for chunk in chunks
            if chunk.section_id == retrieval_section.id
        ]

        evaluation_chunks = [
            chunk
            for chunk in chunks
            if chunk.section_id == evaluation_section.id
        ]

        discussion_chunks = [
            chunk
            for chunk in chunks
            if chunk.section_id == discussion_section.id
        ]

        assert result_chunks
        assert retrieval_chunks
        assert evaluation_chunks
        assert discussion_chunks

        # --------------------------------------------------
        # Page provenance
        # --------------------------------------------------
        mapping_result = await session.execute(
            select(ChunkPageMap)
            .join(
                DocumentChunk,
                DocumentChunk.id == ChunkPageMap.chunk_id,
            )
            .where(
                DocumentChunk.document_id == document.id
            )
        )

        mappings = mapping_result.scalars().all()

        assert mappings

        assert all(
            mapping.document_page_id == page.id
            for mapping in mappings
        )

        # --------------------------------------------------
        # Relationship consistency
        # --------------------------------------------------
        chunk_ids = {
            chunk.id
            for chunk in chunks
        }

        assert all(
            mapping.chunk_id in chunk_ids
            for mapping in mappings
        )

        await session.delete(document)
        await session.commit()
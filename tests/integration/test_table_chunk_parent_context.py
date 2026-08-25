from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import Document, DocumentChunk, DocumentSection
from app.ingestion.structure import TableData
from app.ingestion.table_chunker import split_table


@pytest.mark.asyncio
async def test_table_fragments_recover_parent_section() -> None:
    async with async_session_factory() as session:
        document = Document(
            title=f"Table Parent Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"table-parent-test-{uuid4()}",
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        section = DocumentSection(
            document_id=document.id,
            title="Results",
            section_path="3 Results",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add(section)
        await session.flush()

        table = TableData(
            table_id="table_1",
            title="Retrieval Results",
            headers=["Model", "Recall@10"],
            rows=[
                ["Model A", "0.82"],
                ["Model B", "0.85"],
                ["Model C", "0.88"],
                ["Model D", "0.91"],
                ["Model E", "0.93"],
            ],
            page_numbers=[10, 11],
        )

        fragments = split_table(
            table,
            max_tokens=15,
        )

        assert len(fragments) > 1

        persisted_chunks = []

        for fragment in fragments:
            chunk = DocumentChunk(
                document_id=document.id,
                section_id=section.id,
                chunk_index=len(persisted_chunks),
                content=fragment.content,
                chunk_metadata={
                    "content_type": "table",
                    "table_id": fragment.table_id,
                    "table_fragment_index": fragment.fragment_index,
                    "table_fragment_count": fragment.fragment_count,
                },
            )

            session.add(chunk)
            persisted_chunks.append(chunk)

        await session.commit()

        result = await session.execute(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == document.id,
            )
            .order_by(DocumentChunk.chunk_index)
        )

        chunks = result.scalars().all()

        assert len(chunks) == len(fragments)

        for chunk in chunks:
            assert chunk.section_id == section.id
            assert chunk.chunk_metadata["content_type"] == "table"
            assert chunk.chunk_metadata["table_id"] == "table_1"

        section_result = await session.execute(
            select(DocumentSection).where(
                DocumentSection.id == chunks[0].section_id,
            )
        )

        parent_section = section_result.scalar_one()

        assert parent_section.id == section.id
        assert parent_section.title == "Results"
        assert parent_section.section_path == "3 Results"

        await session.delete(document)
        await session.commit()
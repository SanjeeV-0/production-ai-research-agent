from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import (
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
)


@pytest.mark.asyncio
async def test_chunk_has_section_and_page_provenance() -> None:
    async with async_session_factory() as session:
        document = Document(
            title=f"Provenance Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"provenance-test-{uuid4()}",
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        section = DocumentSection(
            document_id=document.id,
            title="Results",
            section_path="Results",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add(section)
        await session.flush()

        page_one = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Results begin on this page.",
        )

        page_two = DocumentPage(
            document_id=document.id,
            page_number=2,
            content="Results continue on this page.",
        )

        session.add_all([page_one, page_two])
        await session.flush()

        chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content="Results begin and continue across pages.",
            chunk_metadata={},
        )

        session.add(chunk)
        await session.flush()

        session.add_all(
            [
                ChunkPageMap(
                    chunk_id=chunk.id,
                    document_page_id=page_one.id,
                ),
                ChunkPageMap(
                    chunk_id=chunk.id,
                    document_page_id=page_two.id,
                ),
            ]
        )

        await session.commit()

        result = await session.execute(
            select(ChunkPageMap).where(
                ChunkPageMap.chunk_id == chunk.id
            )
        )

        mappings = result.scalars().all()

        assert len(mappings) == 2
        assert {
            mapping.document_page_id
            for mapping in mappings
        } == {
            page_one.id,
            page_two.id,
        }

        assert chunk.section_id == section.id

        await session.delete(document)
        await session.commit()
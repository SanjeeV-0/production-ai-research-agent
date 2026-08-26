from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import ChunkPageMap, Document, DocumentPage, DocumentSection
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.chunk_service import ChunkService
from app.ingestion.size_guard import ChildChunk


@pytest.mark.asyncio
async def test_persist_chunks_maps_chunks_to_multiple_pages() -> None:
    async with async_session_factory() as session:
        document = Document(
            title=f"Chunk Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"chunk-test-{uuid4()}",
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        page_one = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Page one content.",
        )

        page_two = DocumentPage(
            document_id=document.id,
            page_number=2,
            content="Page two content.",
        )
        section = DocumentSection(
            document_id=document.id,
            title="Test Section",
            section_path="Test Section",
            section_level=1,
            section_index=0,
            section_metadata={},
)

        session.add(section)
        await session.flush()

        session.add_all([page_one, page_two])
        await session.flush()

        page_ids = {
            1: page_one.id,
            2: page_two.id,
        }

        chunks = [
            ChildChunk(
                index=0,
                content="Chunk from page one.",
                page_numbers=[1],
                section_id=section.id,
                section_path="Test Section",
                section_level=1,
                source_units=(),
            ),
            ChildChunk(
                index=1,
                content="Chunk crossing pages.",
                page_numbers=[1, 2],
                section_id=section.id,
                section_path="Test Section",
                section_level=1,
                source_units=(),
            ),
                    ]

        service = ChunkService(
    session,
    embedding_provider=DeterministicEmbeddingProvider(
        dimensions=384,
    ),
)

        persisted_chunks = await service.persist_chunks(
            document_id=document.id,
            page_ids=page_ids,
            chunks=chunks,
        )

        await session.commit()

        assert len(persisted_chunks) == 2

        result = await session.execute(
            select(ChunkPageMap).where(
                ChunkPageMap.chunk_id == persisted_chunks[1].id
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

        await session.delete(document)
        await session.commit()
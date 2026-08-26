from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import (
    Document,
    DocumentChunk,
    DocumentSection,
)
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)


@pytest.mark.asyncio
async def test_chunk_embedding_can_be_persisted_and_read() -> None:
    provider = SentenceTransformerEmbeddingProvider()

    embedding = provider.embed_text(
        "Retrieval augmented generation research."
    )

    assert len(embedding) == 384

    async with async_session_factory() as session:
        document = Document(
            title=f"Embedding Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"embedding-test-{uuid4()}",
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

        chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content="Retrieval augmented generation research.",
            chunk_metadata={},
            embedding=embedding,
        )

        session.add(chunk)
        await session.flush()

        result = await session.execute(
            select(DocumentChunk).where(
                DocumentChunk.id == chunk.id
            )
        )

        stored_chunk = result.scalar_one()

        assert stored_chunk.embedding is not None
        assert len(stored_chunk.embedding) == 384

        # Verify the chunk still points to the expected section.
        assert stored_chunk.section_id == section.id

        # Verify the stored vector values.
        assert stored_chunk.embedding == pytest.approx(
            embedding,
            abs=1e-6,
        )

        await session.delete(document)
        await session.commit()
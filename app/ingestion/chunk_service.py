from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import ChunkPageMap, DocumentChunk
from app.core.repositories.document import DocumentRepository
from app.ingestion.size_guard import ChildChunk


class ChunkService:
    """Persists chunks and their source-page mappings."""

    def __init__(self, session: AsyncSession) -> None:
        self.repository = DocumentRepository(session)

    async def persist_chunks(
    self,
    document_id: UUID,
    page_ids: dict[int, UUID],
    chunks: list[ChildChunk],
) -> list[DocumentChunk]:
        """Persist chunks and map them to source pages."""
        persisted_chunks: list[DocumentChunk] = []

        for chunk in chunks:
            document_chunk = DocumentChunk(
    document_id=document_id,
    section_id=chunk.section_id,
    chunk_index=chunk.index,
    content=chunk.content,
    chunk_metadata={
        "page_numbers": chunk.page_numbers,
        "section_path": chunk.section_path,
    },
)

            await self.repository.create_chunk(document_chunk)

            for page_number in chunk.page_numbers:
                page_id = page_ids[page_number]

                mapping = ChunkPageMap(
                    chunk_id=document_chunk.id,
                    document_page_id=page_id,
                )

                await self.repository.create_chunk_page_mapping(mapping)

            persisted_chunks.append(document_chunk)

        return persisted_chunks
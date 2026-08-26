from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import (
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
)
from app.retrieval.models import RetrievedChunk


class DocumentRepository:
    """Database access operations for research documents."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, document_id: UUID) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )

        return result.scalar_one_or_none()

    async def get_by_content_hash(self, content_hash: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )

        return result.scalar_one_or_none()

    async def create(self, document: Document) -> Document:
        self.session.add(document)
        await self.session.flush()

        return document

    async def create_page(self, page: DocumentPage) -> DocumentPage:
        """Persist an extracted document page."""
        self.session.add(page)
        await self.session.flush()

        return page

    async def create_chunk(
        self,
        chunk: DocumentChunk,
    ) -> DocumentChunk:
        """Persist a document chunk."""
        self.session.add(chunk)
        await self.session.flush()

        return chunk

    async def create_chunk_page_mapping(
        self,
        mapping: ChunkPageMap,
    ) -> ChunkPageMap:
        """Persist a chunk-to-page mapping."""
        self.session.add(mapping)
        await self.session.flush()

        return mapping

    async def create_section(
        self,
        section: DocumentSection,
    ) -> DocumentSection:
        """Persist a document section."""
        self.session.add(section)
        await self.session.flush()

        return section

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        limit: int = 10,
        max_distance: float | None = None,
        document_id: UUID | None = None,
        section_id: UUID | None = None,
    ) -> list[RetrievedChunk]:
        """Return chunks ranked by cosine distance."""

        distance = DocumentChunk.embedding.cosine_distance(
            query_embedding
        )

        query = (
            select(
                DocumentChunk,
                DocumentSection.section_path,
                distance.label("distance"),
            )
            .join(
                DocumentSection,
                DocumentChunk.section_id == DocumentSection.id,
            )
            .where(DocumentChunk.embedding.is_not(None))
        )

        if document_id is not None:
            query = query.where(
                DocumentChunk.document_id == document_id
            )
        if section_id is not None:
            query = query.where(
                DocumentChunk.section_id == section_id
            )

        if max_distance is not None:
            query = query.where(
                distance <= max_distance
            )

        query = (
            query
            .order_by(distance)
            .limit(limit)
        )

        result = await self.session.execute(query)
        rows = result.all()

        retrieved_chunks = []

        for chunk, section_path, chunk_distance in rows:
            page_result = await self.session.execute(
                select(DocumentPage.page_number)
                .join(
                    ChunkPageMap,
                    ChunkPageMap.document_page_id == DocumentPage.id,
                )
                .where(
                    ChunkPageMap.chunk_id == chunk.id
                )
                .order_by(DocumentPage.page_number)
            )

            page_numbers = list(page_result.scalars().all())

            retrieved_chunks.append(
                RetrievedChunk(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    section_id=chunk.section_id,
                    section_path=section_path,
                    page_numbers=page_numbers,
                    content=chunk.content,
                    distance=float(chunk_distance),
                )
            )

        return retrieved_chunks
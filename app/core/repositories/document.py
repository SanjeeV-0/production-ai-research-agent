from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import (
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentStatus,
    StoredFile,
)
from app.retrieval.models import RetrievedChunk


class DocumentRepository:
    """All SQL for Document/StoredFile-adjacent/page/section/chunk data.

    Owns every query in the application -- services (DocumentService,
    IngestionService, DocumentDeletionService) and RetrievalService call
    into this rather than building their own queries, so there is exactly
    one place that knows the actual table/column shapes.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, document_id: UUID) -> Document | None:
        result = await self.session.execute(select(Document).where(Document.id == document_id))

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

    async def create_stored_file(self, stored_file: StoredFile) -> StoredFile:
        """Persist the original file metadata."""
        self.session.add(stored_file)
        await self.session.flush()

        return stored_file

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
        """Return chunks ranked by cosine distance.

        This WHERE clause is the ONLY place "only the current, READY
        version is searchable" is enforced -- it is a live query filter, not
        a data-deletion rule. A non-current or non-READY version's chunks
        and embeddings are never touched by this method; they simply don't
        match this filter until/unless that version becomes current again
        (see DocumentService.set_current). This is why switching current
        versions is cheap (no re-embedding) and why historical versions stay
        fully queryable the moment they're promoted back to current.

        The join to DocumentSection is an OUTER join specifically so a chunk
        with section_id IS NULL (a headingless document's chunk -- see
        app.ingestion.structure_extractor) is still returned, with
        section_path reported as NULL, rather than being silently excluded
        by an inner join.
        """

        distance = DocumentChunk.embedding.cosine_distance(query_embedding)

        query = (
            select(
                DocumentChunk,
                DocumentSection.section_path,
                distance.label("distance"),
            )
            .join(
                Document,
                DocumentChunk.document_id == Document.id,
            )
            .outerjoin(
                DocumentSection,
                DocumentChunk.section_id == DocumentSection.id,
            )
            .where(
                DocumentChunk.embedding.is_not(None),
                Document.status == DocumentStatus.READY,
                Document.is_current.is_(True),
            )
        )

        if document_id is not None:
            query = query.where(DocumentChunk.document_id == document_id)

        if section_id is not None:
            query = query.where(DocumentChunk.section_id == section_id)

        if max_distance is not None:
            query = query.where(distance <= max_distance)

        query = query.order_by(distance).limit(limit)

        result = await self.session.execute(query)
        rows = result.all()

        if not rows:
            return []

        chunk_ids = [chunk.id for chunk, _, _ in rows]

        page_result = await self.session.execute(
            select(
                ChunkPageMap.chunk_id,
                DocumentPage.page_number,
            )
            .join(
                DocumentPage,
                ChunkPageMap.document_page_id == DocumentPage.id,
            )
            .where(ChunkPageMap.chunk_id.in_(chunk_ids))
            .order_by(
                ChunkPageMap.chunk_id,
                DocumentPage.page_number,
            )
        )

        page_numbers_by_chunk: dict[UUID, list[int]] = {chunk_id: [] for chunk_id in chunk_ids}

        for chunk_id, page_number in page_result.all():
            page_numbers_by_chunk[chunk_id].append(page_number)

        return [
            RetrievedChunk(
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                section_id=chunk.section_id,
                section_path=section_path,
                page_numbers=page_numbers_by_chunk[chunk.id],
                content=chunk.content,
                distance=float(chunk_distance),
            )
            for chunk, section_path, chunk_distance in rows
        ]

    async def update(self, document: Document) -> Document:
        """Persist changes to a document."""

        self.session.add(document)
        await self.session.flush()
        await self.session.refresh(document)

        return document

    async def get_by_logical_and_content_hash(
        self,
        logical_document_id: UUID,
        content_hash: str,
    ) -> Document | None:
        result = await self.session.execute(
            select(Document).where(
                Document.logical_document_id == logical_document_id,
                Document.content_hash == content_hash,
            )
        )

        return result.scalar_one_or_none()

    async def get_current_version(
        self,
        logical_document_id: UUID,
    ) -> Document | None:
        result = await self.session.execute(
            select(Document).where(
                Document.logical_document_id == logical_document_id,
                Document.is_current.is_(True),
            )
        )

        return result.scalar_one_or_none()

    async def get_latest_version_number(
        self,
        logical_document_id: UUID,
    ) -> int:
        result = await self.session.execute(
            select(func.max(Document.version_number)).where(
                Document.logical_document_id == logical_document_id,
            )
        )

        return result.scalar_one() or 0

    async def delete(self, document: Document) -> None:
        """Delete a document within the current database transaction."""

        await self.session.delete(document)
        await self.session.flush()

    async def get_newest_ready_version(
        self,
        logical_document_id: UUID,
        exclude_document_id: UUID | None = None,
    ) -> Document | None:
        """Return the newest READY version for a logical document."""

        query = (
            select(Document)
            .where(
                Document.logical_document_id == logical_document_id,
                Document.status == DocumentStatus.READY,
            )
            .order_by(Document.version_number.desc())
            .limit(1)
        )

        if exclude_document_id is not None:
            query = query.where(Document.id != exclude_document_id)

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    async def get_by_logical_document_id(
        self,
        logical_document_id: UUID,
    ) -> list[Document]:
        """Return every version of a logical document, newest first."""

        result = await self.session.execute(
            select(Document)
            .where(
                Document.logical_document_id == logical_document_id,
            )
            .order_by(Document.version_number.desc())
        )

        return list(result.scalars().all())

    async def list_logical_documents(self) -> list[tuple[UUID, Document | None]]:
        """Return every distinct logical document paired with its current
        version, or None when no version is currently marked current."""

        logical_id_result = await self.session.execute(
            select(Document.logical_document_id).distinct().order_by(Document.logical_document_id)
        )

        logical_document_ids = [row[0] for row in logical_id_result.all()]

        if not logical_document_ids:
            return []

        current_version_result = await self.session.execute(
            select(Document).where(
                Document.logical_document_id.in_(logical_document_ids),
                Document.is_current.is_(True),
            )
        )

        current_version_by_logical_id = {
            document.logical_document_id: document
            for document in current_version_result.scalars().all()
        }

        return [
            (logical_document_id, current_version_by_logical_id.get(logical_document_id))
            for logical_document_id in logical_document_ids
        ]

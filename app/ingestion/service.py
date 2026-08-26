from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document, DocumentPage
from app.core.repositories.document import DocumentRepository
from app.core.services.document import DocumentService
from app.embeddings.provider import EmbeddingProvider
from app.ingestion.chunk_service import ChunkService
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.schemas import DocumentInput
from app.ingestion.section_builder import SectionBuilder
from app.ingestion.section_service import SectionService
from app.ingestion.semantic_shredder import shred_semantically
from app.ingestion.size_guard import apply_size_guard
from app.ingestion.structure_extractor import StructureExtractor


class IngestionService:
    """Orchestrates the document ingestion workflow."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.session = session
        self.repository = DocumentRepository(session)
        self.document_service = DocumentService(session)
        self.structure_extractor = StructureExtractor()
        self.section_builder = SectionBuilder()
        self.section_service = SectionService(session)
        self.chunk_service = ChunkService(session,
         embedding_provider=embedding_provider,
)
        self.embedding_provider = embedding_provider

    async def ingest_file(
        self,
        path: Path,
        loader: DocumentLoader,
        title: str,
        document_type: str,
        source: str | None = None,
    ) -> Document:
        pages = loader.load(path)

        combined_content = "\n\n".join(
            page.content for page in pages
        )

        document, created = (
            await self.document_service.ingest_document(
                DocumentInput(
                    title=title,
                    source=source,
                    document_type=document_type,
                    content=combined_content,
                )
            )
        )

        if not created:
            return document

        page_ids: dict[int, UUID] = {}

        for page in pages:
            document_page = DocumentPage(
                document_id=document.id,
                page_number=page.page_number,
                content=page.content,
            )

            await self.repository.create_page(
                document_page
            )

            page_ids[page.page_number] = document_page.id

        structural_units = self.structure_extractor.extract(
            pages
        )

        section_nodes = self.section_builder.build(
            structural_units
        )

        section_map = await self.section_service.persist_sections(
            document.id,
            section_nodes,
        )

        semantic_units = shred_semantically(
            structural_units,
            embedding_provider=self.embedding_provider,
            threshold=0.7,
        )

        child_chunks = apply_size_guard(
            semantic_units,
            section_map=section_map,
            max_tokens=500,
        )

        await self.chunk_service.persist_chunks(
            document_id=document.id,
            page_ids=page_ids,
            chunks=child_chunks,
        )

        return document
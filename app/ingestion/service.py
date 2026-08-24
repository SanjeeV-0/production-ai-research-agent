from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document, DocumentPage
from app.core.repositories.document import DocumentRepository
from app.core.services.document import DocumentService
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.schemas import DocumentInput


class IngestionService:
    """Orchestrates the document ingestion workflow."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = DocumentRepository(session)
        self.document_service = DocumentService(session)

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

        document, created = await self.document_service.ingest_document(
            DocumentInput(
                title=title,
                source=source,
                document_type=document_type,
                content=combined_content,
            )
        )

        if not created:
            return document

        for page in pages:
            document_page = DocumentPage(
                document_id=document.id,
                page_number=page.page_number,
                content=page.content,
            )

            await self.repository.create_page(document_page)

        return document
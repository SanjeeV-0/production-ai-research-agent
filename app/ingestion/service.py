from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document
from app.core.services.document import DocumentService
from app.ingestion.schemas import DocumentInput


class IngestionService:
    """Orchestrates the document ingestion workflow."""

    def __init__(self, session: AsyncSession) -> None:
        self.document_service = DocumentService(session)

    async def ingest(self, document: DocumentInput) -> Document:
        return await self.document_service.ingest_document(document)
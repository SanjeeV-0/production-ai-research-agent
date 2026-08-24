from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document
from app.core.repositories.document import DocumentRepository
from app.ingestion.normalizer import calculate_content_hash
from app.ingestion.schemas import DocumentInput


class DocumentService:
    """Application-level operations for research documents."""

    def __init__(self, session: AsyncSession) -> None:
        self.repository = DocumentRepository(session)

    async def ingest_document(
    self,
    document_input: DocumentInput,
    ) -> tuple[Document, bool]:
        content_hash = calculate_content_hash(document_input.content)

        existing_document = await self.repository.get_by_content_hash(
            content_hash
        )

        if existing_document is not None:
            return existing_document, False
        document = Document(
            title=document_input.title,
            authors=document_input.authors,
            source=document_input.source,
            publication_date=document_input.publication_date,
            document_type=document_input.document_type,
            content_hash=content_hash,
            document_metadata={
                "content_length": len(document_input.content),
            },
        )

        document = await self.repository.create(document)

        return document, True
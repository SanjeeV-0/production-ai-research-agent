from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document
from app.core.repositories.document import DocumentRepository


class DocumentService:
    """Application-level operations for research documents."""

    def __init__(self, session: AsyncSession) -> None:
        self.repository = DocumentRepository(session)

    async def get_document(self, document_id: UUID) -> Document | None:
        return await self.repository.get_by_id(document_id)

    async def document_exists(self, content_hash: str) -> bool:
        document = await self.repository.get_by_content_hash(content_hash)

        return document is not None
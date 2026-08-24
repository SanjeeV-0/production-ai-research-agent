from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document


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
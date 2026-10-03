from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import StoredFile


class StoredFileRepository:
    """Data-access operations for stored original files."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_document_id(
        self,
        document_id: UUID,
    ) -> StoredFile | None:
        result = await self.session.execute(
            select(StoredFile).where(StoredFile.document_id == document_id)
        )
        return result.scalar_one_or_none()

    async def delete(self, stored_file: StoredFile) -> None:
        await self.session.delete(stored_file)

    async def get_by_document_ids(
        self,
        document_ids: list[UUID],
    ) -> list[StoredFile]:
        if not document_ids:
            return []

        result = await self.session.execute(
            select(StoredFile).where(
                StoredFile.document_id.in_(document_ids),
            )
        )

        return list(result.scalars().all())

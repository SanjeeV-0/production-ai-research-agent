from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.repositories.document import DocumentRepository
from app.core.repositories.stored_file import StoredFileRepository
from app.core.services.document import DocumentService
from app.storage.interface import FileStorage


class DocumentDeletionService:
    """Coordinate database and physical-file deletion."""

    def __init__(
        self,
        session: AsyncSession,
        file_storage: FileStorage,
    ) -> None:
        self.session = session
        self.file_storage = file_storage
        self.document_repository = DocumentRepository(session)
        self.stored_file_repository = StoredFileRepository(session)
        self.document_service = DocumentService(session)

    async def delete_version(self, document_id: UUID) -> None:
        """Delete a document version and then remove its physical file."""

        document = await self.document_repository.get_by_id(document_id)

        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        stored_file = await self.stored_file_repository.get_by_document_id(document.id)

        if stored_file is None:
            raise ValueError(f"Stored file not found for document: {document_id}")

        storage_key = stored_file.storage_key

        # Delete the StoredFile row first because Document deletion may
        # otherwise violate the StoredFile foreign-key constraint.
        await self.stored_file_repository.delete(stored_file)

        await self.document_service.delete_version(document)

        # DB state is committed before touching physical storage.
        await self.session.commit()

        # A failure here leaves an orphaned physical file, but the database
        # remains clean and contains no reference to it.
        await self.file_storage.delete(storage_key)

    async def delete_logical_document(
        self,
        logical_document_id: UUID,
    ) -> None:
        """Delete all versions of a logical document and their physical files."""

        documents = await self.document_repository.get_by_logical_document_id(logical_document_id)

        if not documents:
            raise ValueError(f"Logical document not found: {logical_document_id}")

        document_ids = [document.id for document in documents]

        stored_files = await self.stored_file_repository.get_by_document_ids(document_ids)

        storage_keys = [stored_file.storage_key for stored_file in stored_files]

        for stored_file in stored_files:
            await self.stored_file_repository.delete(stored_file)

        for document in documents:
            await self.document_repository.delete(document)

        await self.session.commit()

        for storage_key in storage_keys:
            await self.file_storage.delete(storage_key)

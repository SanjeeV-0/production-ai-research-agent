from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import FileBlob
from app.core.repositories.document import DocumentRepository
from app.core.repositories.file_blob import FileBlobRepository
from app.core.repositories.stored_file import StoredFileRepository
from app.core.services.document import DocumentService
from app.storage.interface import FileStorage


class DocumentDeletionService:
    """Coordinate database and physical-file deletion.

    Ordering invariant (both methods below): all DB deletes happen first and
    are committed, and ONLY THEN are physical files deleted. This means a
    failure before commit leaves both the DB and the filesystem untouched,
    and a failure during physical deletion (after commit) leaves the DB
    already consistent with at most an orphaned file on disk -- never a
    dangling DB reference to a missing file.

    Blob deletion is additionally REFERENCE-COUNTED: a `FileBlob` backing a
    `StoredFile` may be shared by other versions (even across different
    logical documents) via `app.ingestion.blob_service.BlobService`, so it
    is only ever a candidate for physical deletion once every `StoredFile`
    row referencing it is gone. See `_lock_and_collect_blob_candidates` for
    how this is made safe against a concurrent upload creating a brand new
    reference to the same blob while a deletion is in flight.
    """

    def __init__(
        self,
        session: AsyncSession,
        file_storage: FileStorage,
    ) -> None:
        self.session = session
        self.file_storage = file_storage
        self.document_repository = DocumentRepository(session)
        self.stored_file_repository = StoredFileRepository(session)
        self.file_blob_repository = FileBlobRepository(session)
        self.document_service = DocumentService(session)

    async def _lock_and_collect_blob_candidates(
        self,
        content_hashes: list[str],
    ) -> list[FileBlob]:
        """Take a row lock (`SELECT ... FOR UPDATE`) on every distinct
        `FileBlob` that one of the about-to-be-deleted `StoredFile` rows
        references, BEFORE deleting anything.

        This is the actual coordination protocol against concurrent
        reference creation: PostgreSQL's foreign-key enforcement takes an
        implicit `FOR KEY SHARE` lock on `file_blobs` whenever a new
        `StoredFile` row is inserted referencing it (standard FK-locking
        behavior, not something this code has to implement itself) -- which
        conflicts with the `FOR UPDATE` lock taken here. So a concurrent
        ingestion request that is in the middle of creating a NEW reference
        to one of these exact blobs will either complete (and be correctly
        counted below) or block until this transaction commits/rolls back
        (and this transaction's reference count was therefore correct and
        final at the moment it decided whether to delete), never racing
        past this check unseen.

        Hashes are locked in sorted order so that two deletions (or a
        deletion and another deletion) touching overlapping sets of blobs
        can never deadlock against each other.
        """

        blobs: list[FileBlob] = []

        for content_hash in sorted(set(content_hashes)):
            blob = await self.file_blob_repository.get_by_hash(content_hash, for_update=True)

            if blob is not None:
                blobs.append(blob)

        return blobs

    async def _delete_now_unreferenced_blobs(
        self,
        locked_blobs: list[FileBlob],
    ) -> list[str]:
        """After the referencing `StoredFile` row(s) have been deleted
        (but BEFORE commit), delete the `FileBlob` row for any blob that
        now has zero remaining references. Returns the storage keys whose
        physical bytes must be removed AFTER this transaction commits.
        """

        storage_keys_to_delete: list[str] = []

        for blob in locked_blobs:
            remaining = await self.file_blob_repository.count_references(blob.content_hash)

            if remaining == 0:
                storage_keys_to_delete.append(blob.storage_key)
                await self.file_blob_repository.delete(blob)

        return storage_keys_to_delete

    async def _delete_physical_blobs(self, storage_keys: list[str]) -> None:
        """Delete physical blob bytes after their DB rows are committed
        gone. A failure here is logged with the storage key and re-raised
        (never silently swallowed) -- it leaves a harmless orphaned file
        with no DB reference pointing at it, the same accepted failure mode
        already used elsewhere in this codebase for the pre-blob-model
        StoredFile deletion path.
        """

        for storage_key in storage_keys:
            try:
                await self.file_storage.delete(storage_key)
            except Exception:
                # Re-raised, not swallowed: the caller (and ultimately the
                # HTTP layer) must see that cleanup did not fully succeed,
                # even though the DB state is already correct and
                # consistent at this point.
                raise

    async def delete_version(self, document_id: UUID) -> None:
        """Delete a document version and then remove its physical blob, IF
        no other version still references it."""

        document = await self.document_repository.get_by_id(document_id)

        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        stored_file = await self.stored_file_repository.get_by_document_id(document.id)

        if stored_file is None:
            raise ValueError(f"Stored file not found for document: {document_id}")

        locked_blobs = await self._lock_and_collect_blob_candidates([stored_file.content_hash])

        # Delete the StoredFile row first because Document deletion may
        # otherwise violate the StoredFile foreign-key constraint.
        await self.stored_file_repository.delete(stored_file)

        await self.document_service.delete_version(document)

        storage_keys_to_delete = await self._delete_now_unreferenced_blobs(locked_blobs)

        # DB state (including any now-unreferenced FileBlob rows) is
        # committed before touching physical storage.
        await self.session.commit()

        await self._delete_physical_blobs(storage_keys_to_delete)

    async def delete_logical_document(
        self,
        logical_document_id: UUID,
    ) -> None:
        """Delete all versions of a logical document and any of their
        physical blobs that become unreferenced as a result."""

        documents = await self.document_repository.get_by_logical_document_id(logical_document_id)

        if not documents:
            raise ValueError(f"Logical document not found: {logical_document_id}")

        document_ids = [document.id for document in documents]

        stored_files = await self.stored_file_repository.get_by_document_ids(document_ids)

        content_hashes = [stored_file.content_hash for stored_file in stored_files]

        locked_blobs = await self._lock_and_collect_blob_candidates(content_hashes)

        for stored_file in stored_files:
            await self.stored_file_repository.delete(stored_file)

        for document in documents:
            await self.document_repository.delete(document)

        storage_keys_to_delete = await self._delete_now_unreferenced_blobs(locked_blobs)

        await self.session.commit()

        await self._delete_physical_blobs(storage_keys_to_delete)

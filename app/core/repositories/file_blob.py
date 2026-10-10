"""All SQL for `FileBlob` rows (shared, content-addressed physical blobs).

Kept separate from `StoredFileRepository` since a `FileBlob` is NOT owned by
any single `Document`/`StoredFile` -- it is a shared resource that may be
referenced by many `StoredFile` rows across different logical documents.
See `app.ingestion.blob_service.BlobService` for the creation/reconciliation
state machine built on top of these methods.
"""

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import BlobStatus, FileBlob, StoredFile


class FileBlobRepository:
    """Data-access operations for shared physical blobs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def try_create_pending(
        self,
        content_hash: str,
        storage_key: str,
        size_bytes: int,
    ) -> tuple[FileBlob, bool]:
        """Attempt to reserve a new PENDING blob row for `content_hash`.

        Uses `INSERT ... ON CONFLICT (content_hash) DO NOTHING` -- a single
        atomic, database-coordinated statement -- so that when two requests
        race to create a blob for identical bytes, exactly one `INSERT`
        succeeds (returns `True`) and the other sees zero rows affected
        (returns `False`) with no unprotected check-then-write window and no
        application-level lock required. The loser must then call
        `get_by_hash` to fetch the row the winner created (which may still
        be PENDING if the winner hasn't finished writing yet -- see
        `BlobService` for how callers handle that).
        """

        statement = (
            pg_insert(FileBlob)
            .values(
                content_hash=content_hash,
                storage_key=storage_key,
                size_bytes=size_bytes,
                status=BlobStatus.PENDING,
            )
            .on_conflict_do_nothing(index_elements=["content_hash"])
            .returning(FileBlob)
        )

        result = await self.session.execute(statement)
        row = result.scalar_one_or_none()

        if row is not None:
            return row, True

        existing = await self.get_by_hash(content_hash)

        if existing is None:
            # Should be unreachable: ON CONFLICT means a row already exists,
            # so a None here would mean it was deleted between the INSERT
            # and this SELECT by a concurrent cleanup -- surface loudly
            # rather than silently treating it as "created".
            raise RuntimeError(
                f"FileBlob {content_hash} conflicted on insert but could not be re-fetched."
            )

        return existing, False

    async def get_by_hash(
        self,
        content_hash: str,
        for_update: bool = False,
    ) -> FileBlob | None:
        """Fetch a blob by its content hash.

        `for_update=True` takes a row-level lock (`SELECT ... FOR UPDATE`),
        used only by the reference-counted delete path
        (`DocumentDeletionService`) to serialize "is this blob still
        referenced?" against a concurrent reference creation on the same
        blob -- see that service for why this is the one place locking is
        actually necessary in this codebase.
        """

        query = select(FileBlob).where(FileBlob.content_hash == content_hash)

        if for_update:
            query = query.with_for_update()

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    async def mark_ready(self, blob: FileBlob, ready_at) -> FileBlob:
        """Transition a blob from PENDING to READY once its bytes are
        durably written. Never call this before the physical write
        (`FileStorage.store`) has actually completed."""

        blob.status = BlobStatus.READY
        blob.ready_at = ready_at

        self.session.add(blob)
        await self.session.flush()

        return blob

    async def count_references(self, content_hash: str) -> int:
        """Count how many `StoredFile` rows currently reference this blob.

        Used by the reference-counted delete path to decide whether a blob
        is safe to physically remove -- call this (and act on it) only
        while holding the `for_update` lock from `get_by_hash`, inside the
        same transaction, so no concurrent reference creation can slip in
        between the count and the delete.
        """

        result = await self.session.execute(
            select(func.count())
            .select_from(StoredFile)
            .where(StoredFile.content_hash == content_hash)
        )

        return result.scalar_one()

    async def delete(self, blob: FileBlob) -> None:
        """Delete a blob's database row within the current transaction.

        Callers MUST have already verified (via `count_references`, under a
        `for_update` lock on this same row) that no `StoredFile` references
        it, and must delete the physical file only AFTER this row's
        deletion is committed -- mirroring the existing
        commit-before-physical-delete ordering used elsewhere in this
        codebase (see `DocumentDeletionService`).
        """

        await self.session.delete(blob)
        await self.session.flush()

    async def find_stale_pending(self, older_than) -> list[FileBlob]:
        """Return PENDING blobs created before `older_than` -- candidates
        for reconciliation (see `BlobService.reconcile_pending_blobs`).

        A PENDING row younger than this cutoff is assumed to still be an
        in-flight, legitimate write and is left alone.
        """

        result = await self.session.execute(
            select(FileBlob).where(
                FileBlob.status == BlobStatus.PENDING,
                FileBlob.created_at < older_than,
            )
        )

        return list(result.scalars().all())

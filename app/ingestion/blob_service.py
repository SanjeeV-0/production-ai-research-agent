"""Creation and reconciliation of shared, content-addressed `FileBlob`s.

This is the ONLY code path that writes raw file bytes to `FileStorage` for
ingestion -- `IngestionService` no longer writes a version's bytes directly;
it asks `BlobService.get_or_create_ready_blob` for an already-verified-ready
blob and only ever stores a `StoredFile` *reference* to it.

State machine (see `app.core.models.BlobStatus`):

    try_create_pending() -- INSERT ... ON CONFLICT DO NOTHING, committed
    immediately so the PENDING row is visible to concurrent requests racing
    on the same content hash
        |
        | (we won the insert)              | (someone else already has a row)
        v                                   v
    write bytes atomically            inspect existing row's status
    (FileStorage.store)                     |
        |                              READY -> reuse immediately
        v                                   |
    mark_ready() + commit              PENDING -> self-heal (see below)

Self-healing a PENDING row we did not create: we have no way to know
whether its creator is still actively writing or crashed mid-write. Since
the storage key is content-addressed and keyed by `content_hash`, writing
the SAME bytes to the SAME key is always safe and idempotent (the atomic
temp-file+rename in `LocalFileStorage.store` means a concurrent write from
two different processes targeting the same key can never corrupt the file
--  whichever write lands last simply replaces the file with byte-identical
content). So a PENDING row we did not create is handled by re-attempting
the write ourselves and then marking it READY -- this is also exactly the
recovery action `reconcile_pending_blobs` takes for a stale PENDING row left
by a crashed process, so there is only one code path to reason about for
"finish a blob that might already be half-written".
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import BlobStatus, FileBlob
from app.core.repositories.file_blob import FileBlobRepository
from app.storage.interface import FileStorage
from app.storage.keys import build_blob_storage_key


@dataclass(frozen=True)
class StoredBlob:
    """A blob guaranteed READY at the moment it is returned."""

    content_hash: str
    storage_key: str
    size_bytes: int


class BlobService:
    """Coordinates shared blob creation, reuse, and reconciliation."""

    def __init__(
        self,
        session: AsyncSession,
        file_storage: FileStorage,
    ) -> None:
        self.session = session
        self.file_storage = file_storage
        self.repository = FileBlobRepository(session)

    async def get_or_create_ready_blob(self, content: bytes) -> StoredBlob:
        """Return a READY blob for `content`, creating it if necessary.

        Never returns before the blob's bytes are durably, atomically
        written and its row is committed as READY -- callers (StoredFile
        creation) must never construct a reference to a blob that might
        still be PENDING.
        """

        content_hash = sha256(content).hexdigest()
        storage_key = build_blob_storage_key(content_hash)

        blob, created = await self.repository.try_create_pending(
            content_hash=content_hash,
            storage_key=storage_key,
            size_bytes=len(content),
        )

        # Publish the PENDING row immediately so a concurrent racer's
        # `try_create_pending` call can actually see it (and so a crash
        # between here and `mark_ready` below leaves a discoverable,
        # reconcilable PENDING row rather than nothing at all).
        await self.session.commit()

        if created:
            return await self._finish_write(blob, content)

        if blob.status == BlobStatus.READY:
            return StoredBlob(
                content_hash=blob.content_hash,
                storage_key=blob.storage_key,
                size_bytes=blob.size_bytes,
            )

        # Someone else's PENDING row (possibly abandoned by a crash) --
        # self-heal by (re-)writing the same bytes ourselves; safe because
        # the storage key is content-addressed and the write is atomic.
        return await self._finish_write(blob, content)

    async def _finish_write(self, blob: FileBlob, content: bytes) -> StoredBlob:
        """Write `content` to `blob.storage_key` and mark it READY.

        If the physical write itself fails, the blob row is left PENDING
        (not deleted, not marked READY) -- it becomes a normal
        reconciliation candidate for `reconcile_pending_blobs` rather than a
        silently swallowed failure. The exception is re-raised so the
        caller's ingestion attempt fails loudly, exactly like any other
        ingestion-step failure.
        """

        await self.file_storage.store(content, blob.storage_key)

        blob = await self.repository.mark_ready(blob, ready_at=datetime.now(UTC))
        await self.session.commit()

        return StoredBlob(
            content_hash=blob.content_hash,
            storage_key=blob.storage_key,
            size_bytes=blob.size_bytes,
        )

    async def reconcile_pending_blobs(
        self,
        stale_after: timedelta = timedelta(minutes=15),
    ) -> list[str]:
        """Recover PENDING blobs abandoned by a crashed/killed process.

        For each PENDING row older than `stale_after`:
          - if the physical file already exists at its storage key (the
            crash happened AFTER the write but BEFORE the DB commit that
            would have marked it READY), mark it READY -- no re-write
            needed, the bytes are already there and correct by construction
            (content-addressed key).
          - if the physical file does NOT exist (the crash happened before
            or during the write), the row is deleted outright -- there is
            nothing to self-heal into, and a PENDING row with no bytes
            behind it must never be reused by a future caller. This is safe
            specifically because `try_create_pending`'s `ON CONFLICT DO
            NOTHING` means a FRESH upload racing against this exact
            content_hash would simply re-win the insert once this stale row
            is gone, rather than being silently blocked forever.

        This is a plain, synchronous, callable operation (not a background
        job/queue) -- intended to be invoked explicitly (e.g. from an
        operator script or a scheduled task external to this codebase), not
        wired into the request path. Returns the list of content hashes
        that were acted on, for logging/visibility; never swallows a
        filesystem error silently -- an `exists()`/`delete()` failure
        propagates.
        """

        cutoff = datetime.now(UTC) - stale_after
        stale = await self.repository.find_stale_pending(cutoff)

        acted_on: list[str] = []

        for blob in stale:
            if await self.file_storage.exists(blob.storage_key):
                await self.repository.mark_ready(blob, ready_at=datetime.now(UTC))
            else:
                await self.repository.delete(blob)

            acted_on.append(blob.content_hash)

        await self.session.commit()

        return acted_on

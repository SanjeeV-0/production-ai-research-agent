import os
import tempfile
from pathlib import Path

from app.storage.interface import FileStorage


class LocalFileStorage(FileStorage):
    """Filesystem-backed implementation of FileStorage, rooted at `root`."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve_path(self, storage_key: str) -> Path:
        """Resolve a storage key and prevent path traversal.

        Security boundary: every storage operation goes through this method,
        which resolves `storage_key` relative to `self.root` and then
        verifies the resolved absolute path is still inside `self.root`
        before any file I/O happens. Storage keys are always server-generated
        from UUIDs (`app.storage.keys.build_storage_key`), never taken
        directly from user input, so this is defense in depth rather than a
        mitigation for a specific observed attack -- but it means a
        malformed or unexpected key (e.g. containing `..`) fails loudly here
        instead of silently reading/writing outside the storage root.
        """
        path = (self.root / storage_key).resolve()

        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("storage_key resolves outside the storage root") from exc

        return path

    async def store(
        self,
        content: bytes,
        storage_key: str,
    ) -> None:
        """Store bytes at the specified storage key, atomically.

        Writes to a temporary file in the SAME target directory, then
        `os.replace`s it onto `storage_key` -- `os.replace` is atomic on both
        POSIX and Windows when source and destination are on the same
        filesystem (guaranteed here, since the temp file is created inside
        `path.parent`). This means any reader that calls `retrieve`/`exists`
        on `storage_key` either sees the complete prior content (nothing
        written yet) or the complete new content -- never a partially
        written file, even if this process crashes mid-write. This matters
        most for content-addressed blobs (`app.ingestion.blob_service`),
        where two concurrent callers can legitimately both attempt to write
        the exact same `storage_key` for the exact same bytes.
        """
        path = self._resolve_path(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_file = tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )

        try:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_file.close()

            os.replace(temporary_file.name, path)
        except Exception:
            temporary_file.close()
            Path(temporary_file.name).unlink(missing_ok=True)
            raise

    async def retrieve(
        self,
        storage_key: str,
    ) -> bytes:
        """Retrieve bytes using the storage key."""
        path = self._resolve_path(storage_key)
        return path.read_bytes()

    async def delete(
        self,
        storage_key: str,
    ) -> None:
        """Delete a stored file."""
        path = self._resolve_path(storage_key)
        path.unlink()

    async def exists(
        self,
        storage_key: str,
    ) -> bool:
        """Check whether a stored file exists."""
        path = self._resolve_path(storage_key)
        return path.is_file()

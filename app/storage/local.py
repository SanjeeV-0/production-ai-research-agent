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
        """Store bytes at the specified storage key."""
        path = self._resolve_path(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

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

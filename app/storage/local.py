from pathlib import Path

from app.storage.interface import FileStorage


class LocalFileStorage(FileStorage):
    """Filesystem-backed implementation of FileStorage."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve_path(self, storage_key: str) -> Path:
        """Resolve a storage key and prevent path traversal."""
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

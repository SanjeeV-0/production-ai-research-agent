from abc import ABC, abstractmethod


class FileStorage(ABC):
    """Abstract interface for storing and managing original files."""

    @abstractmethod
    async def store(
        self,
        content: bytes,
        storage_key: str,
    ) -> None:
        """Store file bytes at the specified storage key."""
        raise NotImplementedError

    @abstractmethod
    async def retrieve(
        self,
        storage_key: str,
    ) -> bytes:
        """Retrieve file bytes using the storage key."""
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        storage_key: str,
    ) -> None:
        """Delete a stored file using the storage key."""
        raise NotImplementedError

    @abstractmethod
    async def exists(
        self,
        storage_key: str,
    ) -> bool:
        """Check whether a stored file exists."""
        raise NotImplementedError

from pathlib import Path

import pytest

from app.storage.local import LocalFileStorage


@pytest.mark.asyncio
async def test_store_and_retrieve_preserves_bytes(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(tmp_path)
    content = b"\x00\x01original\xff\xfe bytes"
    storage_key = "documents/test/file.bin"

    await storage.store(content, storage_key)

    assert await storage.exists(storage_key) is True
    assert await storage.retrieve(storage_key) == content


@pytest.mark.asyncio
async def test_store_creates_nested_directories(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(tmp_path)
    content = b"nested content"
    storage_key = "a/b/c/test.txt"

    await storage.store(content, storage_key)

    assert (tmp_path / "a" / "b" / "c" / "test.txt").is_file()
    assert await storage.retrieve(storage_key) == content


@pytest.mark.asyncio
async def test_exists_returns_false_for_missing_file(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(tmp_path)

    assert await storage.exists("missing/file.txt") is False


@pytest.mark.asyncio
async def test_delete_removes_file(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(tmp_path)
    storage_key = "documents/test/file.txt"

    await storage.store(b"content", storage_key)

    assert await storage.exists(storage_key) is True

    await storage.delete(storage_key)

    assert await storage.exists(storage_key) is False


@pytest.mark.asyncio
async def test_storage_key_cannot_escape_root(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(tmp_path)

    with pytest.raises(ValueError, match="outside the storage root"):
        await storage.store(
            b"malicious content",
            "../outside.txt",
        )


@pytest.mark.asyncio
async def test_retrieve_missing_file_raises_error(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(tmp_path)

    with pytest.raises(FileNotFoundError):
        await storage.retrieve("missing.txt")


@pytest.mark.asyncio
async def test_delete_missing_file_raises_error(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(tmp_path)

    with pytest.raises(FileNotFoundError):
        await storage.delete("missing.txt")

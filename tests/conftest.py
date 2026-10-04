import asyncio
import shutil
import sys
from pathlib import Path

from sqlalchemy import delete


def pytest_sessionstart() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def cleanup_document_test_data() -> None:
    """Remove every document-lifecycle row from the configured database, and
    every physical file under the configured storage root except
    `.gitkeep`, leaving the development environment exactly as it was before
    the test session ran.

    This exists because the 17 real-database integration tests commit
    directly into whatever database `DATABASE_URL` points at (there is no
    separate test database and no per-test transaction/SAVEPOINT isolation
    -- see the project's own investigation of this). Most of those tests
    clean up after themselves by hand, but that cleanup is just ordinary
    code at the end of each test function: if an assertion fails first, the
    cleanup lines never run and the committed rows persist. This function is
    the single, guaranteed backstop for that gap -- it does not change how
    any existing test behaves or require them to stop calling `commit()`.

    Deletes children before parents in FK-safe order. `StoredFile` has no
    `ON DELETE CASCADE` from `Document` at the database level, so it must be
    removed explicitly before `Document`; `ChunkPageMap`, `DocumentChunk`,
    `DocumentSection`, and `DocumentPage` do have `ON DELETE CASCADE` from
    `Document`, but are deleted explicitly here too for clarity and
    defensiveness. A bulk `DELETE` that matches zero rows is a successful
    no-op, so this is safe to call unconditionally, including when there is
    nothing to clean.
    """

    # Imported lazily (inside the function, not at module import time) so
    # that collecting this conftest never has a side effect of importing the
    # full application/database stack before pytest has even decided which
    # tests to run.
    from app.core.database import async_session_factory
    from app.core.models import (
        ChunkPageMap,
        Document,
        DocumentChunk,
        DocumentPage,
        DocumentSection,
        StoredFile,
    )

    async with async_session_factory() as session:
        await session.execute(delete(ChunkPageMap))
        await session.execute(delete(DocumentChunk))
        await session.execute(delete(DocumentSection))
        await session.execute(delete(DocumentPage))
        await session.execute(delete(StoredFile))
        await session.execute(delete(Document))
        await session.commit()

    _cleanup_storage_root()


def _cleanup_storage_root() -> None:
    """Remove every file/directory under the configured storage root except
    `.gitkeep`, preserving the root directory itself.

    A no-op if the storage root doesn't exist, or already contains nothing
    but `.gitkeep` -- e.g. if a file was already removed by hand, or the
    directory was never created at all.
    """

    from app.config.settings import get_settings

    storage_root = Path(get_settings().storage_root)

    if not storage_root.is_dir():
        return

    for entry in storage_root.iterdir():
        if entry.name == ".gitkeep":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Guaranteed end-of-session cleanup.

    pytest calls this hook once after the whole test run finishes --
    regardless of how many tests failed, or whether any assertion raised --
    so document test fixtures never carry forward into the next run or leak
    into the currently-running dev server's view of the database. This is
    the ONLY place cleanup happens; no existing test's behavior changes.
    """

    try:
        asyncio.run(cleanup_document_test_data())
    except Exception as exc:
        message = (
            "\n"
            + "=" * 70
            + "\nPOST-TEST DOCUMENT DATA CLEANUP FAILED\n"
            + f"{type(exc).__name__}: {exc}\n"
            + "The development database and/or storage root may still "
            "contain test fixture data.\n" + "=" * 70 + "\n"
        )

        terminal_reporter = session.config.pluginmanager.getplugin("terminalreporter")
        if terminal_reporter is not None:
            terminal_reporter.write_line(message, red=True, bold=True)
        else:
            print(message, file=sys.stderr)

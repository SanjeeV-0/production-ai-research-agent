"""Proves table chunks participate in the SAME retrieval system as ordinary
text chunks -- no separate table-retrieval code path exists, and none is
created here. Follows the exact conventions already established in
tests/integration/test_vector_retrieval.py (DeterministicEmbeddingProvider,
real DocumentRepository/RetrievalService, real Postgres/pgvector) and the
current-version-switching pattern from
tests/integration/test_retrieval_api.py.

Covers Checkpoint A, step A7, items 9-13.
"""

from uuid import uuid4

import pytest

from app.core.database import async_session_factory
from app.core.models import (
    Document,
    DocumentChunk,
    DocumentSection,
    DocumentStatus,
)
from app.core.repositories.document import DocumentRepository
from app.core.services.document import DocumentService
from app.core.services.document_deletion import DocumentDeletionService
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.generation.context import ContextAssembler
from app.retrieval.service import RetrievalService
from app.storage.local import LocalFileStorage


def _table_chunk_content() -> str:
    return "| Model | Score |\n| --- | --- |\n| A | 0.90 |\n| B | 0.85 |"


@pytest.mark.asyncio
async def test_table_chunk_is_retrieved_through_the_normal_retrieval_pipeline() -> None:
    """A5/A7 item 9: a table chunk must be returned by the exact same
    RetrievalService.search used for prose -- not a separate code path."""

    provider = DeterministicEmbeddingProvider(dimensions=384)
    table_content = _table_chunk_content()

    async with async_session_factory() as session:
        document = Document(
            title=f"Table Retrieval Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"table-retrieval-{uuid4()}",
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        section = DocumentSection(
            document_id=document.id,
            title="Results",
            section_path="Results",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add(section)
        await session.flush()

        table_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=table_content,
            chunk_metadata={
                "content_type": "table",
                "table_id": "table-1",
                "table_fragment_index": 0,
                "table_fragment_count": 1,
            },
            embedding=provider.embed_text(table_content),
        )

        session.add(table_chunk)
        await session.flush()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        results = await retrieval.search(
            table_content,
            limit=5,
            document_id=document.id,
        )

        assert len(results) == 1
        assert results[0].chunk_id == table_chunk.id
        assert results[0].content == table_content
        assert results[0].section_id == section.id
        assert results[0].section_path == "Results"

        await session.rollback()


@pytest.mark.asyncio
async def test_table_retrieval_respects_current_version_filtering() -> None:
    """A5/A7 item 10: a table chunk belonging to a non-current (or
    non-READY) version must never be retrieved -- same
    status=READY AND is_current=true rule as every other chunk."""

    provider = DeterministicEmbeddingProvider(dimensions=384)
    logical_document_id = uuid4()
    table_content = _table_chunk_content()

    async with async_session_factory() as session:
        current_version = Document(
            title=f"Current {uuid4()}",
            document_type="research_paper",
            content_hash=f"current-{uuid4()}",
            logical_document_id=logical_document_id,
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        historical_version = Document(
            title=f"Historical {uuid4()}",
            document_type="research_paper",
            content_hash=f"historical-{uuid4()}",
            logical_document_id=logical_document_id,
            version_number=2,
            is_current=False,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        session.add_all([current_version, historical_version])
        await session.flush()

        current_chunk = DocumentChunk(
            document_id=current_version.id,
            section_id=None,
            chunk_index=0,
            content=table_content,
            chunk_metadata={"content_type": "table", "table_id": "table-1"},
            embedding=provider.embed_text(table_content),
        )

        historical_chunk = DocumentChunk(
            document_id=historical_version.id,
            section_id=None,
            chunk_index=0,
            content=table_content,
            chunk_metadata={"content_type": "table", "table_id": "table-1"},
            embedding=provider.embed_text(table_content),
        )

        session.add_all([current_chunk, historical_chunk])
        await session.flush()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        # No document_id filter: both chunks are embedding-identical, so
        # without current-version filtering BOTH would come back.
        results = await retrieval.search(table_content, limit=10)

        matching = [r for r in results if r.chunk_id in {current_chunk.id, historical_chunk.id}]

        assert len(matching) == 1
        assert matching[0].chunk_id == current_chunk.id

        await session.rollback()


@pytest.mark.asyncio
async def test_table_becomes_retrievable_again_after_switching_current_version() -> None:
    """A5/A7 item 11 + Checkpoint A rule 7 (do NOT delete historical
    vectors): a historical version's table chunk is never deleted when it
    stops being current, and becomes retrievable again, unchanged, the
    moment it is promoted back to current via the existing
    DocumentService.set_current -- no re-ingestion, no re-embedding."""

    provider = DeterministicEmbeddingProvider(dimensions=384)
    logical_document_id = uuid4()
    table_content = _table_chunk_content()

    async with async_session_factory() as session:
        version_one = Document(
            title="Version One",
            document_type="research_paper",
            content_hash=f"v1-{uuid4()}",
            logical_document_id=logical_document_id,
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        version_two = Document(
            title="Version Two",
            document_type="research_paper",
            content_hash=f"v2-{uuid4()}",
            logical_document_id=logical_document_id,
            version_number=2,
            is_current=False,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        session.add_all([version_one, version_two])
        await session.flush()

        chunk_one = DocumentChunk(
            document_id=version_one.id,
            section_id=None,
            chunk_index=0,
            content=f"{table_content} (version one)",
            chunk_metadata={"content_type": "table", "table_id": "table-1"},
            embedding=provider.embed_text(f"{table_content} (version one)"),
        )

        chunk_two = DocumentChunk(
            document_id=version_two.id,
            section_id=None,
            chunk_index=0,
            content=f"{table_content} (version two)",
            chunk_metadata={"content_type": "table", "table_id": "table-1"},
            embedding=provider.embed_text(f"{table_content} (version two)"),
        )

        session.add_all([chunk_one, chunk_two])
        await session.commit()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        # Only version one's table chunk is retrievable while it is current.
        results = await retrieval.search(f"{table_content} (version one)", limit=10)
        v1_hits = [r for r in results if r.chunk_id == chunk_one.id]
        v2_hits = [r for r in results if r.chunk_id == chunk_two.id]
        assert v1_hits
        assert not v2_hits

        document_service = DocumentService(session)
        await document_service.set_current(version_two)
        await session.commit()

        # Switching current must not have deleted either chunk.
        stored_chunk_one = await session.get(DocumentChunk, chunk_one.id)
        stored_chunk_two = await session.get(DocumentChunk, chunk_two.id)
        assert stored_chunk_one is not None
        assert stored_chunk_one.embedding is not None
        assert stored_chunk_two is not None
        assert stored_chunk_two.embedding is not None

        # Now only version two's (originally historical) table chunk is
        # retrievable, with NO re-ingestion or re-embedding having occurred.
        results = await retrieval.search(f"{table_content} (version two)", limit=10)
        v1_hits = [r for r in results if r.chunk_id == chunk_one.id]
        v2_hits = [r for r in results if r.chunk_id == chunk_two.id]
        assert not v1_hits
        assert v2_hits

        await session.delete(version_one)
        await session.delete(version_two)
        await session.commit()


@pytest.mark.asyncio
async def test_deleting_a_version_removes_its_table_chunks(tmp_path) -> None:
    """A7 item 12: deleting a version via the existing DocumentDeletionService
    must remove its table chunks exactly as it does for prose chunks -- no
    special-cased table cleanup, and no table chunk left orphaned."""

    async with async_session_factory() as session:
        document = Document(
            title=f"Table Deletion Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"table-deletion-{uuid4()}",
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        table_chunk = DocumentChunk(
            document_id=document.id,
            section_id=None,
            chunk_index=0,
            content=_table_chunk_content(),
            chunk_metadata={"content_type": "table", "table_id": "table-1"},
            embedding=DeterministicEmbeddingProvider(dimensions=384).embed_text(
                _table_chunk_content()
            ),
        )

        session.add(table_chunk)
        await session.commit()

        document_id = document.id
        table_chunk_id = table_chunk.id

        deletion_service = DocumentDeletionService(
            session=session,
            file_storage=LocalFileStorage(tmp_path / "storage"),
        )

        # No StoredFile exists for this manually-constructed document, so
        # delete_logical_document (which tolerates zero StoredFiles) is used
        # rather than delete_version (which requires one).
        await deletion_service.delete_logical_document(document.logical_document_id)

        remaining_chunk = await session.get(DocumentChunk, table_chunk_id)
        remaining_document = await session.get(Document, document_id)

        assert remaining_chunk is None
        assert remaining_document is None


@pytest.mark.asyncio
async def test_retrieved_table_chunk_flows_into_rag_context_unchanged() -> None:
    """A6/A7 item 13: a table retrieved by RetrievalService must pass through
    ContextAssembler exactly like any other chunk -- no special table
    generation path. The LLM sees the table's markdown exactly as stored."""

    provider = DeterministicEmbeddingProvider(dimensions=384)
    table_content = _table_chunk_content()

    async with async_session_factory() as session:
        document = Document(
            title=f"Table RAG Context Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"table-rag-{uuid4()}",
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        section = DocumentSection(
            document_id=document.id,
            title="Results",
            section_path="Results",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add(section)
        await session.flush()

        table_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=table_content,
            chunk_metadata={"content_type": "table", "table_id": "table-1"},
            embedding=provider.embed_text(table_content),
        )

        session.add(table_chunk)
        await session.flush()

        retrieval = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=provider,
        )

        results = await retrieval.search(table_content, limit=1, document_id=document.id)

        assert len(results) == 1

        context = ContextAssembler().assemble(results)

        assert "| Model | Score |" in context.text
        assert "| A | 0.90 |" in context.text
        assert context.text.startswith("[Source 1]\n")
        assert len(context.sources) == 1
        assert context.sources[0].chunk_id == table_chunk.id

        await session.rollback()

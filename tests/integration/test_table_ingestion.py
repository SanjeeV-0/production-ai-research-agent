"""Integration tests proving documents containing tables flow through the
REAL production ingestion pipeline (IngestionService -> StructureExtractor ->
fragment_table_units -> shred_semantically -> apply_size_guard ->
ChunkService) end-to-end: detection, chunking, embedding, persistence, and
provenance -- using the same real-database conventions as the rest of
tests/integration (see tests/integration/test_ingestion_provenance.py).

Covers Checkpoint A, step A7, items 1-8 and 14-15 (retrieval-specific items
9-13 live in tests/integration/test_table_retrieval.py).
"""

from pathlib import Path

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import (
    ChunkPageMap,
    DocumentChunk,
    DocumentSection,
    DocumentStatus,
    StoredFile,
)
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage


def create_ingestion_service(session, storage_root: Path) -> IngestionService:
    return IngestionService(
        session,
        embedding_provider=DeterministicEmbeddingProvider(dimensions=384),
        file_storage=LocalFileStorage(storage_root),
    )


async def _cleanup(session, document) -> None:
    stored_file_result = await session.execute(
        select(StoredFile).where(StoredFile.document_id == document.id)
    )
    stored_file = stored_file_result.scalar_one_or_none()
    if stored_file is not None:
        await session.delete(stored_file)
    await session.delete(document)
    await session.commit()


@pytest.mark.asyncio
async def test_document_containing_only_a_table_ingests_as_ready(tmp_path: Path) -> None:
    """A1/A7 item 1: a document that is nothing but a table must still
    ingest successfully and produce a retrievable, embedded table chunk."""

    document_path = tmp_path / "table_only.md"
    document_path.write_text(
        "# Results\n\n| Model | Score |\n| --- | --- |\n| A | 0.90 |\n| B | 0.85 |\n",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = create_ingestion_service(session, tmp_path / "storage")

        document = (await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Table Only",
            document_type="research_paper",
            source="integration-test",
        )).document

        assert document.status == DocumentStatus.READY

        chunk_result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        chunks = chunk_result.scalars().all()

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_metadata["content_type"] == "table"
        assert "| Model | Score |" in chunk.content
        assert "| A | 0.90 |" in chunk.content
        # A7 item 7: table embedding.
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384

        await _cleanup(session, document)


@pytest.mark.asyncio
async def test_document_with_text_and_table_produces_both_chunk_kinds(tmp_path: Path) -> None:
    """A7 item 2: mixed text + table content must not cause the table to
    swallow or corrupt the surrounding prose, and both must be independently
    persisted and distinguishable via chunk_metadata."""

    document_path = tmp_path / "mixed.md"
    document_path.write_text(
        "# Results\n\n"
        "The experiment produced the following outcomes.\n\n"
        "| Model | Score |\n"
        "| --- | --- |\n"
        "| A | 0.90 |\n"
        "| B | 0.85 |\n\n"
        "The table demonstrates a clear improvement over the baseline.\n",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = create_ingestion_service(session, tmp_path / "storage")

        document = (await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Mixed Content",
            document_type="research_paper",
            source="integration-test",
        )).document

        assert document.status == DocumentStatus.READY

        chunk_result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = chunk_result.scalars().all()

        table_chunks = [c for c in chunks if c.chunk_metadata.get("content_type") == "table"]
        text_chunks = [c for c in chunks if c.chunk_metadata.get("content_type") != "table"]

        assert len(table_chunks) == 1
        assert text_chunks  # at least the two surrounding paragraphs

        assert "| Model | Score |" in table_chunks[0].content
        assert all("|" not in c.content for c in text_chunks)

        await _cleanup(session, document)


@pytest.mark.asyncio
async def test_multiple_tables_persist_with_distinct_table_ids(tmp_path: Path) -> None:
    """A7 item 3: multiple tables in one document must each persist as their
    own chunk(s) with distinct table_id provenance, not be merged together."""

    document_path = tmp_path / "multi_table.md"
    document_path.write_text(
        "# Results\n\n"
        "| Model | Score |\n"
        "| --- | --- |\n"
        "| A | 0.90 |\n\n"
        "# Baselines\n\n"
        "| Baseline | Score |\n"
        "| --- | --- |\n"
        "| B | 0.50 |\n",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = create_ingestion_service(session, tmp_path / "storage")

        document = (await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Multiple Tables",
            document_type="research_paper",
            source="integration-test",
        )).document

        chunk_result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        chunks = chunk_result.scalars().all()

        table_chunks = [c for c in chunks if c.chunk_metadata.get("content_type") == "table"]

        assert len(table_chunks) == 2

        table_ids = {c.chunk_metadata["table_id"] for c in table_chunks}
        assert len(table_ids) == 2

        await _cleanup(session, document)


@pytest.mark.asyncio
async def test_table_chunk_preserves_section_and_page_provenance(tmp_path: Path) -> None:
    """A4/A7 item 6: a table chunk under a real heading must point to that
    section (not None, not a different section), and its page_numbers must
    reflect the page it was actually extracted from via ChunkPageMap."""

    document_path = tmp_path / "provenance.md"
    document_path.write_text(
        "# Introduction\n\nIntroductory text.\n\n"
        "# Results\n\n"
        "| Model | Score |\n"
        "| --- | --- |\n"
        "| A | 0.90 |\n",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = create_ingestion_service(session, tmp_path / "storage")

        document = (await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Provenance",
            document_type="research_paper",
            source="integration-test",
        )).document

        section_result = await session.execute(
            select(DocumentSection).where(
                DocumentSection.document_id == document.id,
                DocumentSection.section_path == "Results",
            )
        )
        results_section = section_result.scalar_one()

        chunk_result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        table_chunks = [
            c
            for c in chunk_result.scalars().all()
            if c.chunk_metadata.get("content_type") == "table"
        ]
        assert len(table_chunks) == 1
        table_chunk = table_chunks[0]

        assert table_chunk.section_id == results_section.id

        mapping_result = await session.execute(
            select(ChunkPageMap).where(ChunkPageMap.chunk_id == table_chunk.id)
        )
        mappings = mapping_result.scalars().all()
        assert mappings

        await _cleanup(session, document)


@pytest.mark.asyncio
async def test_headingless_table_has_no_fabricated_section(tmp_path: Path) -> None:
    """A4/A7 item 14: a table with no preceding heading must persist with
    section_id=None -- never a fabricated or nearest-guessed section."""

    document_path = tmp_path / "headingless_table.md"
    document_path.write_text(
        "| Model | Score |\n| --- | --- |\n| A | 0.90 |\n",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = create_ingestion_service(session, tmp_path / "storage")

        document = (await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Headingless Table",
            document_type="research_paper",
            source="integration-test",
        )).document

        section_result = await session.execute(
            select(DocumentSection).where(DocumentSection.document_id == document.id)
        )
        assert section_result.scalars().all() == []

        chunk_result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        chunks = chunk_result.scalars().all()

        assert len(chunks) == 1
        assert chunks[0].chunk_metadata["content_type"] == "table"
        assert chunks[0].section_id is None

        await _cleanup(session, document)


@pytest.mark.asyncio
async def test_docx_table_ingests_through_the_same_pipeline(tmp_path: Path) -> None:
    """A1/A7: .docx is the other format with genuinely structured table data
    (python-docx's document.tables) -- it must flow through the identical
    detection/fragmentation/chunking path as a native Markdown table, not a
    separate code path. PDF is deliberately not tested here: pypdf's
    extract_text() has no table structure to extract (see
    StructureExtractor's module docstring), so PDF table support does not
    exist and is not claimed by any test."""

    from docx import Document as DocxDocument

    docx_path = tmp_path / "table.docx"
    docx_document = DocxDocument()
    docx_document.add_paragraph("Results summary.")
    table = docx_document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Model"
    table.cell(0, 1).text = "Score"
    table.cell(1, 0).text = "A"
    table.cell(1, 1).text = "0.90"
    docx_document.save(docx_path)

    async with async_session_factory() as session:
        service = create_ingestion_service(session, tmp_path / "storage")

        document = (await service.ingest_file(
            path=docx_path,
            loader=DocxLoader(),
            title="Docx Table",
            document_type="research_paper",
            source="integration-test",
        )).document

        assert document.status == DocumentStatus.READY

        chunk_result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        chunks = chunk_result.scalars().all()

        table_chunks = [c for c in chunks if c.chunk_metadata.get("content_type") == "table"]

        assert len(table_chunks) == 1
        assert "| Model | Score |" in table_chunks[0].content
        assert "| A | 0.90 |" in table_chunks[0].content
        assert table_chunks[0].embedding is not None

        await _cleanup(session, document)


@pytest.mark.asyncio
async def test_large_table_fragments_into_multiple_chunks_with_shared_table_id(
    tmp_path: Path,
) -> None:
    """A7 items 5/8: a table large enough to exceed the chunk token budget
    must persist as multiple DocumentChunk rows (not one oversized chunk,
    and not a crash), each tagged with the same table_id and the correct
    fragment index/count."""

    # The production chunk budget is 500 whitespace-tokens (IngestionService's
    # hard-coded max_chunk_tokens); each row costs ~5 tokens, so 150 rows
    # reliably exceeds it and forces fragment_table_units to split.
    rows = "\n".join(f"| Model{i} | 0.{i % 100:02d} |" for i in range(150))

    document_path = tmp_path / "large_table.md"
    document_path.write_text(
        f"# Results\n\n| Model | Score |\n| --- | --- |\n{rows}\n",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = create_ingestion_service(session, tmp_path / "storage")

        document = (await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Large Table",
            document_type="research_paper",
            source="integration-test",
        )).document

        assert document.status == DocumentStatus.READY

        chunk_result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = chunk_result.scalars().all()

        table_chunks = [c for c in chunks if c.chunk_metadata.get("content_type") == "table"]

        assert len(table_chunks) > 1

        table_ids = {c.chunk_metadata["table_id"] for c in table_chunks}
        assert table_ids == {table_chunks[0].chunk_metadata["table_id"]}

        fragment_counts = {c.chunk_metadata["table_fragment_count"] for c in table_chunks}
        assert fragment_counts == {len(table_chunks)}

        fragment_indexes = sorted(c.chunk_metadata["table_fragment_index"] for c in table_chunks)
        assert fragment_indexes == list(range(len(table_chunks)))

        assert all(c.embedding is not None for c in table_chunks)

        await _cleanup(session, document)


@pytest.mark.asyncio
async def test_normal_text_only_ingestion_is_unaffected(tmp_path: Path) -> None:
    """A7 item 15: plain documents with no table syntax at all must ingest
    exactly as before -- no spurious TABLE chunks, no behavior change."""

    document_path = tmp_path / "plain.md"
    document_path.write_text(
        "# Introduction\n\nThis document has no tables at all, just prose.\n",
        encoding="utf-8",
    )

    async with async_session_factory() as session:
        service = create_ingestion_service(session, tmp_path / "storage")

        document = (await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Plain Text",
            document_type="research_paper",
            source="integration-test",
        )).document

        assert document.status == DocumentStatus.READY

        chunk_result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        chunks = chunk_result.scalars().all()

        assert chunks
        assert all("content_type" not in c.chunk_metadata for c in chunks)

        await _cleanup(session, document)

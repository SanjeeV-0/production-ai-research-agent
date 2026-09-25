import asyncio
import selectors
from pathlib import Path

from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import DocumentChunk, DocumentSection
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.service import IngestionService


async def main() -> None:
    path = Path("tmp/retrieval_test.md")

    provider = SentenceTransformerEmbeddingProvider()

    async with async_session_factory() as session:
        service = IngestionService(
            session=session,
            embedding_provider=provider,
        )

        document = await service.ingest_file(
            path=path,
            loader=MarkdownLoader(),
            title="Chunk Inspection Sample",
            document_type="research_note",
        )

        print("\n" + "=" * 80)
        print("DOCUMENT")
        print("=" * 80)
        print(f"ID:              {document.id}")
        print(f"Logical ID:      {document.logical_document_id}")
        print(f"Version:         {document.version_number}")
        print(f"Status:          {document.status}")
        print(f"Current:         {document.is_current}")

        sections_result = await session.execute(
            select(DocumentSection)
            .where(DocumentSection.document_id == document.id)
            .order_by(DocumentSection.section_index)
        )
        sections = sections_result.scalars().all()

        chunks_result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = chunks_result.scalars().all()

        print("\n" + "=" * 80)
        print(f"SECTIONS ({len(sections)})")
        print("=" * 80)

        for section in sections:
            print(
                f"[{section.section_index}] "
                f"level={section.section_level} "
                f"path={section.section_path!r}"
            )

        print("\n" + "=" * 80)
        print(f"CHUNKS ({len(chunks)})")
        print("=" * 80)

        for chunk in chunks:
            print("\n" + "-" * 80)
            print(f"Chunk index:    {chunk.chunk_index}")
            print(f"Chunk ID:       {chunk.id}")
            print(f"Section ID:     {chunk.section_id}")
            print(f"Content length: {len(chunk.content)} characters")
            print(f"Embedding:      {len(chunk.embedding) if chunk.embedding else 0} dimensions")
            print(f"Metadata:       {chunk.chunk_metadata}")
            print("\nCONTENT:")
            print(chunk.content)

            if chunk.embedding:
                print("\nEMBEDDING (first 10 values):")
                print(chunk.embedding[:10])

        print("\n" + "=" * 80)
        print("INSPECTION COMPLETE")
        print("=" * 80)

        # Clean up the temporary database records.
        print("\nTemporary database records retained for retrieval inspection.")


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )

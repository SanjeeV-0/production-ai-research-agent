import asyncio
from pathlib import Path

from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import DocumentChunk
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage

BENCHMARK_DIR = Path("tests/evaluation/fixtures/benchmark")
STORAGE_DIR = Path("storage")


async def main() -> None:
    embedding_provider = SentenceTransformerEmbeddingProvider()
    loader = DocxLoader()
    file_storage = LocalFileStorage(STORAGE_DIR)

    total_chunks = 0

    async with async_session_factory() as session:
        service = IngestionService(
            session=session,
            embedding_provider=embedding_provider,
            file_storage=file_storage,
        )

        for path in sorted(BENCHMARK_DIR.glob("*.docx")):
            print(f"\n{path.name}")

            document = await service.ingest_file(
                path=path,
                loader=loader,
                title=path.stem,
                document_type="research_paper",
                source="benchmark-corpus",
            )

            print(f"  status: {document.status}")

            chunks = list(
                (
                    await session.scalars(
                        select(DocumentChunk).where(DocumentChunk.document_id == document.id)
                    )
                ).all()
            )

            total_chunks += len(chunks)

            table_chunks = [
                chunk
                for chunk in chunks
                if chunk.chunk_metadata and chunk.chunk_metadata.get("content_type") == "table"
            ]

            print(f"  chunks: {len(chunks)}")
            print(f"  table chunks: {len(table_chunks)}")

            if table_chunks:
                fragment_counts: dict[str, int] = {}
                headers_present = True

                for chunk in table_chunks:
                    metadata = chunk.chunk_metadata or {}
                    table_id = str(metadata.get("table_id", "unknown"))

                    fragment_counts[table_id] = fragment_counts.get(table_id, 0) + 1

                    content = chunk.content or ""

                    if "|" not in content:
                        headers_present = False

                print(
                    "  table fragment counts:",
                    sorted(fragment_counts.values()),
                )
                print(f"  headers present: {headers_present}")

        await session.commit()

    print(f"\nTOTAL CHUNKS: {total_chunks}")


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=asyncio.SelectorEventLoop,
    )

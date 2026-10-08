from pathlib import Path
from statistics import mean, median

from sqlalchemy import select

from app.config.settings import get_settings
from app.core.database import async_session_factory
from app.core.models import DocumentChunk
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage

BENCHMARK_DIR = Path(__file__).parent / "benchmark"


async def main() -> None:
    settings = get_settings()

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model,
    )
    file_storage = LocalFileStorage(Path(settings.storage_root))

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=embedding_provider,
            file_storage=file_storage,
        )

        for path in sorted(BENCHMARK_DIR.glob("*.docx")):
            document = await service.ingest_file(
                path=path,
                loader=DocxLoader(),
                title=path.stem.replace("_", " ").title(),
                document_type="benchmark",
                source="benchmark-measurement",
            )

            result = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index)
            )
            chunks = result.scalars().all()

            sizes = [len(chunk.content.split()) for chunk in chunks]

            print(f"\n{path.name}")
            print(f"  status:  {document.status}")
            print(f"  chunks:  {len(sizes)}")
            print(f"  min:     {min(sizes)} words")
            print(f"  max:     {max(sizes)} words")
            print(f"  average: {mean(sizes):.1f} words")
            print(f"  median:  {median(sizes):.1f} words")


if __name__ == "__main__":
    import asyncio
    import selectors

    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        ),
    )
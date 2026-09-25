import asyncio
import selectors

from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import Document
from app.core.repositories.document import DocumentRepository
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.service import RetrievalService


async def main() -> None:
    query = (
        "How does semantic chunking differ from fixed-size chunking, "
        "and why can it improve retrieval?"
    )

    async with async_session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.title == "Chunk Inspection Sample")
        )
        document = result.scalar_one()

        embedding_provider = SentenceTransformerEmbeddingProvider()
        reranker = CrossEncoderReranker()

        service = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=embedding_provider,
            reranker=reranker,
        )

        results = await service.search(
            query=query,
            limit=5,
            candidate_limit=50,
            document_id=document.id,
            trace=True,
        )

        trace = service.last_trace

        print("=" * 100)
        print("QUERY")
        print("=" * 100)
        print(query)

        print("\n" + "=" * 100)
        print("VECTOR CANDIDATES")
        print("=" * 100)

        for i, candidate in enumerate(trace.candidates, start=1):
            print(f"\n[{i}]")
            print(f"chunk_id     : {candidate.chunk_id}")
            print(f"section_path: {candidate.section_path}")
            print(f"distance     : {candidate.distance}")
            print(f"rerank_score : {candidate.rerank_score}")
            print(f"content      : {candidate.content[:500]}")

        print("\n" + "=" * 100)
        print("FINAL RERANKED RESULTS")
        print("=" * 100)

        for i, result in enumerate(results, start=1):
            print(f"\n[{i}]")
            print(f"chunk_id     : {result.chunk_id}")
            print(f"section_path: {result.section_path}")
            print(f"distance     : {result.distance}")
            print(f"similarity   : {result.similarity}")
            print(f"rerank_score : {result.rerank_score}")
            print(f"content      : {result.content[:800]}")

        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"Vector candidates : {len(trace.candidates)}")
        print(f"Final results     : {len(results)}")


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )

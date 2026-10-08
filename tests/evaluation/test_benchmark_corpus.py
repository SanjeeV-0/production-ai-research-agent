from sqlalchemy import func, select

from app.core.database import async_session_factory
from app.core.models import DocumentChunk
from tests.evaluation.retrieval_cases import (
    RETRIEVAL_BENCHMARK_CASE_SPECS,
)
from tests.evaluation.benchmark_corpus import (
    BENCHMARK_DOCUMENTS,
    ensure_benchmark_corpus,
    resolve_benchmark_evaluation_cases,
)
def test_benchmark_gold_resolution() -> None:
    import asyncio

    asyncio.run(_resolve_benchmark_gold())


async def _resolve_benchmark_gold() -> None:
    from app.core.database import async_session_factory

    async with async_session_factory() as session:
        cases = await resolve_benchmark_evaluation_cases(
            session,
            tuple(RETRIEVAL_BENCHMARK_CASE_SPECS),
        )

        assert len(cases) == 46

        for case in cases:
            assert case.relevant_chunk_ids
            assert case.relevance_by_chunk_id

        print(f"Resolved benchmark cases: {len(cases)}")
        print(
            "Resolved gold chunk references: "
            f"{sum(len(case.relevant_chunk_ids) for case in cases)}"
        )

def test_benchmark_corpus_measurement() -> None:
    import asyncio

    asyncio.run(_measure_benchmark_corpus())


async def _measure_benchmark_corpus() -> None:
    async with async_session_factory() as session:
        documents = await ensure_benchmark_corpus(session)

        assert len(documents) == len(BENCHMARK_DOCUMENTS)

        total_chunks = 0

        for _, title, logical_document_id in BENCHMARK_DOCUMENTS:
            document = documents[title]

            assert document.logical_document_id == logical_document_id
            assert document.title == title

            result = await session.execute(
                select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.document_id == document.id,
                )
            )
            chunk_count = result.scalar_one()

            print(f"{title}: {chunk_count} chunks")

            assert chunk_count > 0
            total_chunks += chunk_count

        print(f"Total benchmark chunks: {total_chunks}")
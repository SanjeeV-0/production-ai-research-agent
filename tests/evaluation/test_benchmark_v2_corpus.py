from sqlalchemy import func, select

from app.core.database import async_session_factory
from app.core.models import DocumentChunk
from tests.evaluation.benchmark_corpus_v2 import (
    BENCHMARK_V2_DOCUMENTS,
    ensure_benchmark_v2_corpus,
    resolve_benchmark_v2_evaluation_cases,
)
from tests.evaluation.retrieval_cases_v2 import (
    RETRIEVAL_BENCHMARK_V2_CASE_SPECS,
)


def test_benchmark_v2_gold_resolution() -> None:
    import asyncio

    asyncio.run(_resolve_benchmark_v2_gold())


async def _resolve_benchmark_v2_gold() -> None:
    async with async_session_factory() as session:
        cases = await resolve_benchmark_v2_evaluation_cases(
            session,
            tuple(RETRIEVAL_BENCHMARK_V2_CASE_SPECS),
        )

        assert len(cases) == len(RETRIEVAL_BENCHMARK_V2_CASE_SPECS)

        unresolved: list[str] = []

        for spec, case in zip(RETRIEVAL_BENCHMARK_V2_CASE_SPECS, cases, strict=True):
            if not case.relevant_chunk_ids or not case.relevance_by_chunk_id:
                unresolved.append(spec.query)

        total_gold_fragments = sum(
            len(spec.relevant_chunks) for spec in RETRIEVAL_BENCHMARK_V2_CASE_SPECS
        )
        resolved_chunk_refs = sum(len(case.relevant_chunk_ids) for case in cases)

        print(f"Resolved benchmark V2 cases: {len(cases)}")
        print(f"Total gold fragment references: {total_gold_fragments}")
        print(f"Resolved gold chunk references: {resolved_chunk_refs}")
        print(f"Unresolved cases: {len(unresolved)}")

        # resolve_benchmark_v2_evaluation_cases already raises loudly on any
        # missing/ambiguous fragment (see its docstring) -- this is a second,
        # independent guarantee that every case ended up with at least one
        # resolved relevant chunk, per the "do not silently ignore failures"
        # requirement.
        assert not unresolved, f"Cases with no resolved gold chunks: {unresolved}"


def test_benchmark_v2_corpus_measurement() -> None:
    import asyncio

    asyncio.run(_measure_benchmark_v2_corpus())


async def _measure_benchmark_v2_corpus() -> None:
    async with async_session_factory() as session:
        documents = await ensure_benchmark_v2_corpus(session)

        assert len(documents) == len(BENCHMARK_V2_DOCUMENTS)

        total_chunks = 0
        chunks_by_format: dict[str, int] = {}

        for _, title, logical_document_id, source_format in BENCHMARK_V2_DOCUMENTS:
            document = documents[title]

            assert document.logical_document_id == logical_document_id
            assert document.title == title

            result = await session.execute(
                select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.document_id == document.id,
                )
            )
            chunk_count = result.scalar_one()

            print(f"[{source_format:>8}] {title}: {chunk_count} chunks")

            assert chunk_count > 0
            total_chunks += chunk_count
            chunks_by_format[source_format] = (
                chunks_by_format.get(source_format, 0) + chunk_count
            )

        print(f"Total benchmark V2 chunks: {total_chunks}")
        print(f"Chunks by source format: {chunks_by_format}")

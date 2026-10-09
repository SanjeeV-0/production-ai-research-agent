"""Scratch tool (not part of the test suite): runs Benchmark V2's full real
retrieval pipeline (decomposition -> embedding -> HNSW search -> merge ->
cross-encoder reranking) across all 102 queries at ONE configuration
(candidate_limit=100, final_limit=10), with overall + per-category +
per-source-format breakdowns.

This exists because the full 3x3 (candidate_limit x final_limit) sweep in
test_benchmark_v2_retrieval.py makes roughly 918 real retrieval calls,
each invoking a paid OpenRouter decomposition call plus CPU cross-encoder
reranking, taking on the order of hours. This script runs the single
representative configuration the project chose to validate now.
"""

import asyncio
import selectors
import time
from collections import defaultdict

from app.core.database import async_session_factory
from app.core.dependencies import (
    get_embedding_provider,
    get_query_decomposer,
    get_reranker,
    get_retrieval_service,
)
from tests.evaluation.benchmark_corpus_v2 import (
    BENCHMARK_V2_FORMAT_BY_TITLE,
    resolve_benchmark_v2_evaluation_cases,
)
from tests.evaluation.retrieval_cases_v2 import RETRIEVAL_BENCHMARK_V2_CASE_SPECS
from tests.evaluation.retrieval_metrics import (
    RetrievalCaseMetrics,
    aggregate_case_metrics,
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
)

CANDIDATE_LIMIT = 100
FINAL_LIMIT = 10


def _case_source_format(spec) -> str:
    formats = {
        BENCHMARK_V2_FORMAT_BY_TITLE[rc.document_title] for rc in spec.relevant_chunks
    }
    return next(iter(formats)) if len(formats) == 1 else "mixed"


async def main() -> None:
    async with async_session_factory() as session:
        cases = await resolve_benchmark_v2_evaluation_cases(
            session,
            tuple(RETRIEVAL_BENCHMARK_V2_CASE_SPECS),
        )
        case_formats = [_case_source_format(spec) for spec in RETRIEVAL_BENCHMARK_V2_CASE_SPECS]

        retrieval_service = await get_retrieval_service(
            session=session,
            embedding_provider=get_embedding_provider(),
            reranker=get_reranker(),
            query_decomposer=get_query_decomposer(),
        )

        print(
            f"Running {len(cases)} queries at candidate_limit={CANDIDATE_LIMIT}, "
            f"final_limit={FINAL_LIMIT} ...",
            flush=True,
        )

        all_metrics: list[RetrievalCaseMetrics] = []
        by_category: dict[str, list[RetrievalCaseMetrics]] = defaultdict(list)
        by_format: dict[str, list[RetrievalCaseMetrics]] = defaultdict(list)

        start = time.time()

        for i, (case, case_format) in enumerate(zip(cases, case_formats, strict=True), start=1):
            query_start = time.time()
            results = await retrieval_service.search(
                query=case.query,
                limit=FINAL_LIMIT,
                candidate_limit=CANDIDATE_LIMIT,
            )
            retrieved_ids = [r.chunk_id for r in results]

            metrics = RetrievalCaseMetrics(
                query=case.query,
                k=FINAL_LIMIT,
                recall=recall_at_k(retrieved_ids, case.relevant_chunk_ids, FINAL_LIMIT),
                precision=precision_at_k(retrieved_ids, case.relevant_chunk_ids, FINAL_LIMIT),
                mrr=mrr_at_k(retrieved_ids, case.relevant_chunk_ids, FINAL_LIMIT),
                hit_rate=hit_rate_at_k(retrieved_ids, case.relevant_chunk_ids, FINAL_LIMIT),
            )

            all_metrics.append(metrics)
            by_category[case.category].append(metrics)
            by_format[case_format].append(metrics)

            elapsed_query = time.time() - query_start
            print(
                f"[{i:>3}/{len(cases)}] ({elapsed_query:5.1f}s) "
                f"recall={metrics.recall:.2f} precision={metrics.precision:.2f} "
                f"mrr={metrics.mrr:.2f} hit={metrics.hit_rate:.0f} | {case.query[:70]}",
                flush=True,
            )

        total_elapsed = time.time() - start

        print()
        print("=" * 88)
        print(f"OVERALL (candidate={CANDIDATE_LIMIT}, final={FINAL_LIMIT}, n={len(all_metrics)})")
        print("=" * 88)
        overall = aggregate_case_metrics(all_metrics, k=FINAL_LIMIT)
        print(
            f"Recall={overall.mean_recall:.3f} | Precision={overall.mean_precision:.3f} | "
            f"MRR={overall.mean_mrr:.3f} | HitRate={overall.mean_hit_rate:.3f}"
        )
        print(f"Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

        print()
        print("-" * 88)
        print("BY CATEGORY")
        print("-" * 88)
        for category in sorted(by_category):
            m = by_category[category]
            agg = aggregate_case_metrics(m, k=FINAL_LIMIT)
            print(
                f"{category:>20} (n={len(m):>3}) | Recall={agg.mean_recall:.3f} | "
                f"Precision={agg.mean_precision:.3f} | MRR={agg.mean_mrr:.3f} | "
                f"HitRate={agg.mean_hit_rate:.3f}"
            )

        print()
        print("-" * 88)
        print("BY SOURCE FORMAT")
        print("-" * 88)
        for fmt in ("markdown", "docx", "pdf", "mixed"):
            m = by_format.get(fmt)
            if not m:
                continue
            agg = aggregate_case_metrics(m, k=FINAL_LIMIT)
            print(
                f"{fmt:>20} (n={len(m):>3}) | Recall={agg.mean_recall:.3f} | "
                f"Precision={agg.mean_precision:.3f} | MRR={agg.mean_mrr:.3f} | "
                f"HitRate={agg.mean_hit_rate:.3f}"
            )


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )

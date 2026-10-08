import asyncio

from app.core.database import async_session_factory
from app.core.dependencies import (
    get_embedding_provider,
    get_query_decomposer,
    get_reranker,
    get_retrieval_service,
)
from app.retrieval.service import RetrievalService
from tests.evaluation.benchmark_corpus import (
    resolve_benchmark_evaluation_cases,
)
from tests.evaluation.retrieval_cases import RETRIEVAL_BENCHMARK_CASE_SPECS
from tests.evaluation.retrieval_metrics import (
    RetrievalCaseMetrics,
    aggregate_case_metrics,
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
)


CANDIDATE_LIMITS = (20, 50, 100)
FINAL_LIMITS = (3, 5, 10)


def _calculate_case_metrics(
    *,
    case,
    retrieved_ids,
    k: int,
) -> RetrievalCaseMetrics:
    return RetrievalCaseMetrics(
        query=case.query,
        k=k,
        recall=recall_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            k,
        ),
        precision=precision_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            k,
        ),
        mrr=mrr_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            k,
        ),
        hit_rate=hit_rate_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            k,
        ),
    )


async def _run_benchmark() -> None:
    async with async_session_factory() as session:
        cases = await resolve_benchmark_evaluation_cases(
            session,
            RETRIEVAL_BENCHMARK_CASE_SPECS,
        )

        retrieval_service: RetrievalService = await get_retrieval_service(
            session=session,
            embedding_provider=get_embedding_provider(),
            reranker=get_reranker(),
            query_decomposer=get_query_decomposer(),
        )

        print()
        print("=" * 88)
        print("RETRIEVAL BENCHMARK")
        print("=" * 88)
        print(f"Evaluation cases : {len(cases)}")
        print(f"Candidate limits : {CANDIDATE_LIMITS}")
        print(f"Final limits     : {FINAL_LIMITS}")
        print()

        for candidate_limit in CANDIDATE_LIMITS:
            for final_limit in FINAL_LIMITS:
                case_metrics: list[RetrievalCaseMetrics] = []

                for case in cases:
                    results = await retrieval_service.search(
                        query=case.query,
                        limit=final_limit,
                        candidate_limit=candidate_limit,
                    )

                    retrieved_ids = [
                        result.chunk_id
                        for result in results
                    ]

                    case_metrics.append(
                        _calculate_case_metrics(
                            case=case,
                            retrieved_ids=retrieved_ids,
                            k=final_limit,
                        )
                    )

                aggregate = aggregate_case_metrics(
                    case_metrics,
                    k=final_limit,
                )

                print(
                    f"candidate={candidate_limit:>3} "
                    f"final={final_limit:>2} | "
                    f"Recall={aggregate.mean_recall:.3f} | "
                    f"Precision={aggregate.mean_precision:.3f} | "
                    f"MRR={aggregate.mean_mrr:.3f} | "
                    f"HitRate={aggregate.mean_hit_rate:.3f}"
                )


def test_retrieval_benchmark() -> None:
    asyncio.run(_run_benchmark())
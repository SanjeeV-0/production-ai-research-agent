"""Checkpoint D retrieval baseline: makes the retrieval evaluation dataset
genuinely runnable against the REAL retrieval path, now across recall,
precision, MRR, and hit rate at multiple K values.

Exercises the unmodified `RetrievalService` (real `DocumentRepository`
pgvector/HNSW query, real embeddings, real cross-encoder reranker) against
a small, reproducible corpus ingested through the unmodified production
`IngestionService`, and scores the results with the existing, unmodified
`recall_at_k`/`mrr_at_k`/`precision_at_k`/`hit_rate_at_k` metrics. No new
retrieval method, no RRF, and no change to retrieval/document-lifecycle
behavior are introduced here.

Query decomposition is intentionally not exercised in this checkpoint -- it
requires a configured OpenRouter API key and a live network call, which
this evaluation test does not depend on.

K_VALUES = (1, 3): the corpus has exactly 3 chunks total (1 relevant chunk
per query), so any K >= 3 is equivalent to K=3 (the whole corpus is
visible) -- K=1 is the stricter, more discriminating signal for a corpus
this size, and K=3 is the pre-existing value. Results are macro-averaged
(simple unweighted mean) across the 3 hand-written cases and PRINTED as a
measured baseline -- assertions stay structural (valid range, every case
found, corpus size sane), never a hardcoded numeric target.
"""

import pytest

from app.config.settings import get_settings
from app.core.database import async_session_factory
from app.core.repositories.document import DocumentRepository
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.service import RetrievalService
from tests.evaluation.corpus import (
    EVALUATION_CORPUS_LOGICAL_DOCUMENT_ID,
    ensure_evaluation_corpus,
)
from tests.evaluation.retrieval_cases import RETRIEVAL_EVALUATION_CASE_SPECS
from tests.evaluation.retrieval_dataset import resolve_evaluation_cases
from tests.evaluation.retrieval_metrics import (
    RetrievalCaseMetrics,
    aggregate_case_metrics,
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
)

K_VALUES = (1, 3)


def _print_baseline_report(
    case_metrics_by_k: dict[int, list[RetrievalCaseMetrics]],
) -> None:
    """Print a concise, human-readable retrieval baseline report.

    Purely a presentation step -- no persistent storage, no Langfuse. The
    numbers are whatever was actually measured this run, not a target to
    match.
    """

    print("\n" + "=" * 60)
    print("RETRIEVAL BASELINE (measured, not a fixed target)")
    print("=" * 60)

    for k in K_VALUES:
        case_metrics = case_metrics_by_k[k]
        aggregate = aggregate_case_metrics(case_metrics, k)

        print(f"\n--- K={k} ({aggregate.case_count} cases) ---")
        for metrics in case_metrics:
            print(
                f"  query={metrics.query!r}: "
                f"recall={metrics.recall:.3f} "
                f"precision={metrics.precision:.3f} "
                f"mrr={metrics.mrr:.3f} "
                f"hit_rate={metrics.hit_rate:.3f}"
            )
        print(
            f"  MEAN: recall={aggregate.mean_recall:.3f} "
            f"precision={aggregate.mean_precision:.3f} "
            f"mrr={aggregate.mean_mrr:.3f} "
            f"hit_rate={aggregate.mean_hit_rate:.3f}"
        )

    print("=" * 60)


@pytest.mark.asyncio
async def test_retrieval_evaluation_corpus_queries_find_their_known_chunk() -> None:
    async with async_session_factory() as session:
        corpus_version = await ensure_evaluation_corpus(session)

        cases = await resolve_evaluation_cases(
            session,
            EVALUATION_CORPUS_LOGICAL_DOCUMENT_ID,
            RETRIEVAL_EVALUATION_CASE_SPECS,
        )

        settings = get_settings()

        retrieval_service = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=SentenceTransformerEmbeddingProvider(
                model_name=settings.embedding_model,
            ),
            reranker=CrossEncoderReranker(model_name=settings.reranker_model),
        )

        case_metrics_by_k: dict[int, list[RetrievalCaseMetrics]] = {k: [] for k in K_VALUES}

        for case in cases:
            # Scoped to the corpus's own version so this is never affected
            # by whatever else happens to exist in the shared dev database.
            # limit=max(K_VALUES) retrieves enough results to compute every
            # K from one search call per case.
            results = await retrieval_service.search(
                case.query,
                limit=max(K_VALUES),
                document_id=corpus_version.id,
            )

            retrieved_ids = [result.chunk_id for result in results]

            for k in K_VALUES:
                case_metrics_by_k[k].append(
                    RetrievalCaseMetrics(
                        query=case.query,
                        k=k,
                        recall=recall_at_k(retrieved_ids, case.relevant_chunk_ids, k),
                        precision=precision_at_k(retrieved_ids, case.relevant_chunk_ids, k),
                        mrr=mrr_at_k(retrieved_ids, case.relevant_chunk_ids, k),
                        hit_rate=hit_rate_at_k(retrieved_ids, case.relevant_chunk_ids, k),
                    )
                )

        _print_baseline_report(case_metrics_by_k)

        # Structural correctness checks, not tuned/fabricated benchmark
        # targets: every metric is in its valid [0, 1] range, every case
        # was actually scored, and -- at K=3, where the whole 3-chunk
        # corpus is visible -- each hand-written query must find its own
        # known source chunk at least once via the live retrieval path.
        for k in K_VALUES:
            aggregate = aggregate_case_metrics(case_metrics_by_k[k], k)

            assert aggregate.case_count == len(cases)
            assert 0.0 <= aggregate.mean_recall <= 1.0
            assert 0.0 <= aggregate.mean_precision <= 1.0
            assert 0.0 <= aggregate.mean_mrr <= 1.0
            assert 0.0 <= aggregate.mean_hit_rate <= 1.0

        final_k_metrics = case_metrics_by_k[K_VALUES[-1]]
        assert all(metrics.hit_rate == 1.0 for metrics in final_k_metrics), final_k_metrics

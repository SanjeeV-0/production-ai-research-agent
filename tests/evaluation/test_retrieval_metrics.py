from uuid import uuid4

import pytest

from tests.evaluation.retrieval_metrics import (
    RetrievalCaseMetrics,
    aggregate_case_metrics,
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()
    chunk_c = uuid4()
    chunk_d = uuid4()

    retrieved = [chunk_a, chunk_b, chunk_c]
    relevant = frozenset({chunk_b, chunk_d})

    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_at_k_with_all_relevant_chunks() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()

    retrieved = [chunk_a, chunk_b]
    relevant = frozenset({chunk_a, chunk_b})

    assert recall_at_k(retrieved, relevant, k=2) == 1.0


def test_recall_at_k_respects_k() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()
    chunk_c = uuid4()

    retrieved = [chunk_a, chunk_b, chunk_c]
    relevant = frozenset({chunk_c})

    assert recall_at_k(retrieved, relevant, k=2) == 0.0


def test_mrr_at_k() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()
    chunk_c = uuid4()

    retrieved = [chunk_a, chunk_b, chunk_c]
    relevant = frozenset({chunk_b})

    assert mrr_at_k(retrieved, relevant, k=3) == 0.5


def test_mrr_at_k_returns_first_relevant_rank() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()
    chunk_c = uuid4()

    retrieved = [chunk_a, chunk_b, chunk_c]
    relevant = frozenset({chunk_b, chunk_c})

    assert mrr_at_k(retrieved, relevant, k=3) == 0.5


def test_mrr_at_k_returns_zero_when_no_relevant_chunk_is_found() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()

    retrieved = [chunk_a]
    relevant = frozenset({chunk_b})

    assert mrr_at_k(retrieved, relevant, k=1) == 0.0


def test_precision_at_k() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()
    chunk_c = uuid4()

    retrieved = [chunk_a, chunk_b, chunk_c]
    relevant = frozenset({chunk_b, chunk_c})

    # 2 of the top 3 are relevant -> 2 / 3.
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_divides_by_k_not_by_results_returned() -> None:
    chunk_a = uuid4()

    retrieved = [chunk_a]
    relevant = frozenset({chunk_a})

    # Only 1 result was ever retrieved, but k=5 -- precision is penalized
    # for the missing results rather than rescaled to 1.0.
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(1 / 5)


def test_precision_at_k_with_no_relevant_chunks_in_top_k() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()

    retrieved = [chunk_a]
    relevant = frozenset({chunk_b})

    assert precision_at_k(retrieved, relevant, k=1) == 0.0


def test_precision_at_k_returns_zero_for_empty_relevant_ids() -> None:
    chunk_a = uuid4()

    assert precision_at_k([chunk_a], frozenset(), k=1) == 0.0


def test_hit_rate_at_k_returns_one_when_any_relevant_chunk_is_in_top_k() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()
    chunk_c = uuid4()

    retrieved = [chunk_a, chunk_b, chunk_c]
    relevant = frozenset({chunk_c})

    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0


def test_hit_rate_at_k_returns_zero_when_no_relevant_chunk_is_in_top_k() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()

    retrieved = [chunk_a]
    relevant = frozenset({chunk_b})

    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0


def test_hit_rate_at_k_respects_k() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()

    retrieved = [chunk_a, chunk_b]
    relevant = frozenset({chunk_b})

    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 1.0


def test_hit_rate_at_k_returns_zero_for_empty_relevant_ids() -> None:
    chunk_a = uuid4()

    assert hit_rate_at_k([chunk_a], frozenset(), k=1) == 0.0


def test_aggregate_case_metrics_computes_macro_average() -> None:
    case_metrics = [
        RetrievalCaseMetrics(
            query="query one",
            k=3,
            recall=1.0,
            precision=1.0,
            mrr=1.0,
            hit_rate=1.0,
        ),
        RetrievalCaseMetrics(
            query="query two",
            k=3,
            recall=0.0,
            precision=0.0,
            mrr=0.0,
            hit_rate=0.0,
        ),
    ]

    aggregate = aggregate_case_metrics(case_metrics, k=3)

    assert aggregate.k == 3
    assert aggregate.case_count == 2
    assert aggregate.mean_recall == pytest.approx(0.5)
    assert aggregate.mean_precision == pytest.approx(0.5)
    assert aggregate.mean_mrr == pytest.approx(0.5)
    assert aggregate.mean_hit_rate == pytest.approx(0.5)


def test_aggregate_case_metrics_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate_case_metrics([], k=3)


def test_aggregate_case_metrics_rejects_mismatched_k() -> None:
    case_metrics = [
        RetrievalCaseMetrics(
            query="query one",
            k=1,
            recall=1.0,
            precision=1.0,
            mrr=1.0,
            hit_rate=1.0,
        ),
    ]

    with pytest.raises(ValueError, match="k=3"):
        aggregate_case_metrics(case_metrics, k=3)

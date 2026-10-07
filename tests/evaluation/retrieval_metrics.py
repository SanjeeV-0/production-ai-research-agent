from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID


def recall_at_k(
    retrieved_ids: Sequence[UUID],
    relevant_ids: frozenset[UUID],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k.intersection(relevant_ids)

    return len(relevant_retrieved) / len(relevant_ids)


def mrr_at_k(
    retrieved_ids: Sequence[UUID],
    relevant_ids: frozenset[UUID],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def precision_at_k(
    retrieved_ids: Sequence[UUID],
    relevant_ids: frozenset[UUID],
    k: int,
) -> float:
    """Fraction of the top-K results that are relevant: relevant retrieved
    in top K / K. Divides by K (not by however many items were actually
    retrieved), so returning fewer than K items is correctly penalized
    rather than silently rescaled -- standard Precision@K semantics.
    """
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k.intersection(relevant_ids)

    return len(relevant_retrieved) / k


def hit_rate_at_k(
    retrieved_ids: Sequence[UUID],
    relevant_ids: frozenset[UUID],
    k: int,
) -> float:
    """1.0 if at least one relevant chunk appears in the top K, else 0.0.

    Returns `float` (not `bool`) so it can be averaged/printed uniformly
    alongside the other three metrics here.
    """
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])

    return 1.0 if retrieved_at_k.intersection(relevant_ids) else 0.0


@dataclass(frozen=True)
class RetrievalCaseMetrics:
    """All four metrics for one evaluation case at one K."""

    query: str
    k: int
    recall: float
    precision: float
    mrr: float
    hit_rate: float


@dataclass(frozen=True)
class RetrievalAggregateMetrics:
    """Macro-average (simple unweighted mean) of `RetrievalCaseMetrics`
    across every evaluation case, at one K."""

    k: int
    mean_recall: float
    mean_precision: float
    mean_mrr: float
    mean_hit_rate: float
    case_count: int


def aggregate_case_metrics(
    case_metrics: Sequence[RetrievalCaseMetrics],
    k: int,
) -> RetrievalAggregateMetrics:
    """Macro-average `case_metrics` into one `RetrievalAggregateMetrics`.

    All entries must share the given `k` -- this does not filter or mix
    results from different K values itself, so the caller groups by K
    before calling this.
    """
    if not case_metrics:
        raise ValueError("case_metrics must not be empty")

    if any(metrics.k != k for metrics in case_metrics):
        raise ValueError(f"All case_metrics must have k={k}")

    count = len(case_metrics)

    return RetrievalAggregateMetrics(
        k=k,
        mean_recall=sum(metrics.recall for metrics in case_metrics) / count,
        mean_precision=sum(metrics.precision for metrics in case_metrics) / count,
        mean_mrr=sum(metrics.mrr for metrics in case_metrics) / count,
        mean_hit_rate=sum(metrics.hit_rate for metrics in case_metrics) / count,
        case_count=count,
    )

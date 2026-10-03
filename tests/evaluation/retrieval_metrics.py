from collections.abc import Sequence
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
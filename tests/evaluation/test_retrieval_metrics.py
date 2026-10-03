from uuid import uuid4

from tests.evaluation.retrieval_metrics import mrr_at_k, recall_at_k


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
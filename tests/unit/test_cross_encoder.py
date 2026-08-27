from uuid import uuid4

from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.models import RetrievedChunk


def _chunk(content: str, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content=content,
        distance=distance,
    )


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.2, 0.9, 0.5]


def test_cross_encoder_reranks_by_score() -> None:
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model = FakeCrossEncoder()

    first = _chunk(
        "first candidate",
        distance=0.1,
    )

    second = _chunk(
        "second candidate",
        distance=0.2,
    )

    third = _chunk(
        "third candidate",
        distance=0.3,
    )

    results = reranker.rerank(
        "research query",
        [first, second, third],
    )

    assert [result.content for result in results] == [
        "second candidate",
        "third candidate",
        "first candidate",
    ]

    assert results[0].rerank_score == 0.9
    assert results[1].rerank_score == 0.5
    assert results[2].rerank_score == 0.2


def test_cross_encoder_returns_empty_for_no_candidates() -> None:
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model = FakeCrossEncoder()

    results = reranker.rerank(
        "research query",
        [],
    )

    assert results == []
from uuid import uuid4

import pytest

from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.models import RetrievedChunk


def _chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content=content,
        distance=0.5,
    )


@pytest.mark.integration
def test_real_cross_encoder_reranks_candidates() -> None:
    reranker = CrossEncoderReranker()

    relevant = _chunk(
        "Retrieval augmented generation retrieves relevant "
        "documents before generating an answer."
    )

    unrelated = _chunk(
        "The weather forecast predicts rain tomorrow."
    )

    results = reranker.rerank(
        "retrieval augmented generation",
        [unrelated, relevant],
    )

    assert len(results) == 2
    assert results[0].content == relevant.content
    assert results[0].rerank_score is not None
    assert results[1].rerank_score is not None
    assert (
        results[0].rerank_score
        > results[1].rerank_score
    )
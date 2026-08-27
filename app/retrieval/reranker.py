from collections.abc import Sequence
from typing import Protocol

from app.retrieval.models import RetrievedChunk


class Reranker(Protocol):
    """Ranks retrieved chunks against a query."""

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Return chunks ordered by reranker relevance."""
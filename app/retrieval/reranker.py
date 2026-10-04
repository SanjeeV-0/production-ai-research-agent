"""Reranker contract. `app.retrieval.cross_encoder.CrossEncoderReranker` is
the only implementation; this Protocol exists so `RetrievalService` and unit
tests can swap in a fake without depending on sentence-transformers.
"""

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

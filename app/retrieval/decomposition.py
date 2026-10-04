"""Shared data types and provider contract for query decomposition.

Decomposition is query PLANNING for retrieval -- splitting a multi-intent
question into focused sub-queries -- not generic rewriting/paraphrasing.
`QueryDecompositionProvider` is deliberately a thin `Protocol` (just "given a
query, return candidate strings") so the normalization/validation/fallback
rules in `app.retrieval.query_decomposer.QueryDecomposer` are provider-
agnostic; `app.retrieval.openrouter_decomposition` is the only provider
implementation today.
"""

from dataclasses import dataclass
from typing import Protocol

# Hard-coded ceiling on how many sub-queries (including the original query)
# QueryDecomposer._normalize_queries will ever keep. Not a Settings field --
# changing it requires a code change here.
MAX_SUB_QUERIES = 5


@dataclass(frozen=True)
class QueryDecompositionResult:
    """Normalized result of query decomposition.

    `was_decomposed`/`fallback_reason` are computed by QueryDecomposer but
    are NOT currently surfaced through RetrievalTrace or either HTTP API --
    callers only consume `sub_queries`. The only externally visible signal
    that decomposition didn't happen (or failed) is `len(sub_queries) == 1`.
    """

    original_query: str
    sub_queries: tuple[str, ...]
    was_decomposed: bool
    fallback_reason: str | None = None


class QueryDecompositionProvider(Protocol):
    """Provider capable of decomposing a query."""

    async def decompose(self, query: str) -> list[str]:
        """Return candidate focused retrieval queries."""

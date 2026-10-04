from dataclasses import dataclass
from typing import Protocol

MAX_SUB_QUERIES = 5


@dataclass(frozen=True)
class QueryDecompositionResult:
    """Normalized result of query decomposition."""

    original_query: str
    sub_queries: tuple[str, ...]
    was_decomposed: bool
    fallback_reason: str | None = None


class QueryDecompositionProvider(Protocol):
    """Provider capable of decomposing a query."""

    async def decompose(self, query: str) -> list[str]:
        """Return candidate focused retrieval queries."""

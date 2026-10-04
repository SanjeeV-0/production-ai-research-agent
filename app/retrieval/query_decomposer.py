from app.retrieval.decomposition import (
    MAX_SUB_QUERIES,
    QueryDecompositionProvider,
    QueryDecompositionResult,
)


class QueryDecomposer:
    """Normalize and validate LLM-generated query decomposition."""

    def __init__(self, provider: QueryDecompositionProvider) -> None:
        self.provider = provider

    async def decompose(self, query: str) -> QueryDecompositionResult:
        original_query = query.strip()

        if not original_query:
            raise ValueError("query must not be empty")

        try:
            candidates = await self.provider.decompose(original_query)
            sub_queries = self._normalize_queries(
                original_query,
                candidates,
            )

            return QueryDecompositionResult(
                original_query=original_query,
                sub_queries=tuple(sub_queries),
                was_decomposed=len(sub_queries) > 1,
            )
        except Exception as exc:
            return QueryDecompositionResult(
                original_query=original_query,
                sub_queries=(original_query,),
                was_decomposed=False,
                fallback_reason=str(exc),
            )

    @staticmethod
    def _normalize_queries(
        original_query: str,
        candidates: list[str],
    ) -> list[str]:
        queries: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized = value.strip()

            if not normalized:
                return

            key = normalized.casefold()

            if key in seen:
                return

            seen.add(key)
            queries.append(normalized)

        # Original query is always first and can never be removed.
        add(original_query)

        for candidate in candidates:
            add(candidate)

            if len(queries) >= MAX_SUB_QUERIES:
                break

        return queries

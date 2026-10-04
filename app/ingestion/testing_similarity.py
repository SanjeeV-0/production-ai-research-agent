"""Test double for `app.ingestion.similarity.SimilarityProvider`. Not used
by any production code path -- exists so tests can supply a deterministic
or stubbed similarity function without depending on a real embedding model.
"""

from collections.abc import Callable


class CallableSimilarityProvider:
    """Similarity provider backed by a callable."""

    def __init__(
        self,
        function: Callable[[str, str], float],
    ) -> None:
        self.function = function

    def similarity(
        self,
        left: str,
        right: str,
    ) -> float:
        """Return similarity from the configured callable."""
        return self.function(left, right)

from collections.abc import Sequence
from math import sqrt
from typing import Protocol


class SimilarityProvider(Protocol):
    """Provides semantic similarity between two texts."""

    def similarity(self, left: str, right: str) -> float:
        """Return a similarity score for two texts."""


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions.")

    if not left:
        raise ValueError("Vectors must not be empty.")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left,
                                            right,
                                            strict=True)
    )

    left_norm = sqrt(
        sum(value * value for value in left)
    )

    right_norm = sqrt(
        sum(value * value for value in right)
    )

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
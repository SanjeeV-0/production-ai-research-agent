import pytest

from app.ingestion.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one() -> None:
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    assert result == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert result == pytest.approx(-1.0)


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 2.0],
            [1.0],
        )


def test_empty_vectors_raise() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_zero_vector_returns_zero() -> None:
    result = cosine_similarity(
        [0.0, 0.0],
        [1.0, 2.0],
    )

    assert result == 0.0
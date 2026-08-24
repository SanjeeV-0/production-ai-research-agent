from app.ingestion.normalizer import (
    calculate_content_hash,
    normalize_content,
)


def test_normalize_content_collapses_whitespace() -> None:
    content = "This   is\n a   research   paper."

    assert normalize_content(content) == "This is a research paper."


def test_identical_normalized_content_has_same_hash() -> None:
    content_a = "This is a research paper."
    content_b = "This   is\n a research paper."

    assert calculate_content_hash(content_a) == calculate_content_hash(content_b)


def test_different_content_has_different_hash() -> None:
    content_a = "This is paper A."
    content_b = "This is paper B."

    assert calculate_content_hash(content_a) != calculate_content_hash(content_b)
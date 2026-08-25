import pytest

from app.ingestion.chunker import chunk_pages


def test_chunk_pages_creates_chunks() -> None:
    pages = [
        (1, "This is page one content."),
        (2, "This is page two content."),
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert chunks[0].index == 0


def test_chunk_preserves_page_provenance() -> None:
    pages = [
        (10, "A " * 20),
        (11, "B " * 20),
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=30,
        chunk_overlap=5,
    )

    assert any(
        10 in chunk.page_numbers and 11 in chunk.page_numbers
        for chunk in chunks
    )


def test_invalid_chunk_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=0)


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=100, chunk_overlap=100)
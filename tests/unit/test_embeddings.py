from app.embeddings.testing import (
    DeterministicEmbeddingProvider,
)


def test_embed_text_returns_expected_dimensions() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)

    embedding = provider.embed_text(
        "retrieval augmented generation"
    )

    assert len(embedding) == 16


def test_embed_text_is_deterministic() -> None:
    provider = DeterministicEmbeddingProvider()

    first = provider.embed_text("research document")
    second = provider.embed_text("research document")

    assert first == second


def test_different_texts_produce_embeddings() -> None:
    provider = DeterministicEmbeddingProvider()

    first = provider.embed_text("research")
    second = provider.embed_text("database")

    assert first != second


def test_batch_embedding_preserves_order() -> None:
    provider = DeterministicEmbeddingProvider()

    texts = [
        "first document",
        "second document",
        "third document",
    ]

    result = provider.embed_batch(texts)

    assert len(result) == 3
    assert result[0] == provider.embed_text(texts[0])
    assert result[1] == provider.embed_text(texts[1])
    assert result[2] == provider.embed_text(texts[2])
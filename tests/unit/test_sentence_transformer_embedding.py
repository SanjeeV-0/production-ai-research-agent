from unittest.mock import MagicMock

from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)


def test_embed_text_returns_embedding() -> None:
    model = MagicMock()

    embedding = MagicMock()
    embedding.tolist.return_value = [0.1, 0.2, 0.3]

    model.encode.return_value = embedding

    provider = SentenceTransformerEmbeddingProvider.__new__(
        SentenceTransformerEmbeddingProvider
    )
    provider.model = model

    result = provider.embed_text("retrieval")

    assert result == [0.1, 0.2, 0.3]

    model.encode.assert_called_once_with(
        "retrieval",
        convert_to_numpy=True,
    )


def test_embed_batch_preserves_order() -> None:
    model = MagicMock()

    first = MagicMock()
    first.tolist.return_value = [1.0, 0.0]

    second = MagicMock()
    second.tolist.return_value = [0.0, 1.0]

    model.encode.return_value = [
        first,
        second,
    ]

    provider = SentenceTransformerEmbeddingProvider.__new__(
        SentenceTransformerEmbeddingProvider
    )
    provider.model = model

    result = provider.embed_batch(
        ["first", "second"]
    )

    assert result == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    model.encode.assert_called_once_with(
        ["first", "second"],
        convert_to_numpy=True,
    )
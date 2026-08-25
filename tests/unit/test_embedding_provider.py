import pytest

from app.embeddings.base import EmbeddingProvider


def test_embedding_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()
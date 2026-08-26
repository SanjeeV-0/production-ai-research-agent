from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingProvider:
    """Embedding provider backed by a Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings while preserving input order."""
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
        )

        return [
            embedding.tolist()
            for embedding in embeddings
        ]
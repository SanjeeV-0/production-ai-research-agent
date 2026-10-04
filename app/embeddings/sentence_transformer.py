"""The only `EmbeddingProvider` implementation, used for both ingestion
(embedding chunks) and retrieval (embedding queries) -- the same model must
back both sides, since a query embedding is only meaningful compared against
chunk embeddings from the same model/space.

Default model's output dimension (384) is hard-coded into the database
schema (`DocumentChunk.embedding: Vector(384)` in `app.core.models`).
Changing `model_name` to a model with a different dimension requires a
schema migration; changing it to ANY different model -- same dimension or
not -- requires re-embedding every existing chunk, since vectors from
different models are not comparable even if same-sized. Nothing in this
codebase automates that re-embedding.
"""

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
        """Generate an embedding for a single text.

        No `normalize_embeddings=True` is passed -- the raw model output is
        returned as-is. Cosine distance is still computed correctly at query
        time by pgvector's `cosine_distance` operator regardless of whether
        stored vectors are pre-normalized.
        """
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

        return [embedding.tolist() for embedding in embeddings]

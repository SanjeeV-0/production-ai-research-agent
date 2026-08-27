from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from app.retrieval.models import RetrievedChunk


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Return chunks ordered by cross-encoder relevance."""

        if not chunks:
            return []

        pairs = [
            (query, chunk.content)
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            RetrievedChunk(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                section_path=chunk.section_path,
                page_numbers=chunk.page_numbers,
                content=chunk.content,
                distance=chunk.distance,
                rerank_score=float(score),
            )
            for chunk, score in ranked
        ]
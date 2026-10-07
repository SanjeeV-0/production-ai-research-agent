"""First retrieval-evaluation checkpoint: makes the retrieval evaluation
dataset genuinely runnable against the REAL retrieval path.

Exercises the unmodified `RetrievalService` (real `DocumentRepository`
pgvector/HNSW query, real embeddings, real cross-encoder reranker) against
a small, reproducible corpus ingested through the unmodified production
`IngestionService`, and scores the results with the existing, unmodified
`recall_at_k`/`mrr_at_k` metrics. No new retrieval method, no RRF, and no
change to retrieval/document-lifecycle behavior are introduced here.

Query decomposition is intentionally not exercised in this first checkpoint
-- it requires a configured OpenRouter API key and a live network call,
which this evaluation test does not depend on.
"""

import pytest

from app.config.settings import get_settings
from app.core.database import async_session_factory
from app.core.repositories.document import DocumentRepository
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.service import RetrievalService
from tests.evaluation.corpus import (
    EVALUATION_CORPUS_LOGICAL_DOCUMENT_ID,
    ensure_evaluation_corpus,
)
from tests.evaluation.retrieval_cases import RETRIEVAL_EVALUATION_CASE_SPECS
from tests.evaluation.retrieval_dataset import resolve_evaluation_cases
from tests.evaluation.retrieval_metrics import mrr_at_k, recall_at_k

TOP_K = 3


@pytest.mark.asyncio
async def test_retrieval_evaluation_corpus_queries_find_their_known_chunk() -> None:
    async with async_session_factory() as session:
        corpus_version = await ensure_evaluation_corpus(session)

        cases = await resolve_evaluation_cases(
            session,
            EVALUATION_CORPUS_LOGICAL_DOCUMENT_ID,
            RETRIEVAL_EVALUATION_CASE_SPECS,
        )

        settings = get_settings()

        retrieval_service = RetrievalService(
            repository=DocumentRepository(session),
            embedding_provider=SentenceTransformerEmbeddingProvider(
                model_name=settings.embedding_model,
            ),
            reranker=CrossEncoderReranker(model_name=settings.reranker_model),
        )

        recall_scores: list[float] = []
        mrr_scores: list[float] = []

        for case in cases:
            # Scoped to the corpus's own version so this is never affected
            # by whatever else happens to exist in the shared dev database.
            results = await retrieval_service.search(
                case.query,
                limit=TOP_K,
                document_id=corpus_version.id,
            )

            retrieved_ids = [result.chunk_id for result in results]

            recall_scores.append(recall_at_k(retrieved_ids, case.relevant_chunk_ids, TOP_K))
            mrr_scores.append(mrr_at_k(retrieved_ids, case.relevant_chunk_ids, TOP_K))

        # Structural correctness check, not a tuned/fabricated benchmark:
        # each hand-written query must find its own known source chunk
        # somewhere in the real top-K produced by the live retrieval path.
        assert all(score > 0 for score in recall_scores), recall_scores
        assert all(score > 0 for score in mrr_scores), mrr_scores

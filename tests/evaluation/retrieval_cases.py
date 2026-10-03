from uuid import UUID

from tests.evaluation.retrieval_dataset import RetrievalEvaluationCase

RETRIEVAL_EVALUATION_CASES = [
    RetrievalEvaluationCase(
        query="What were the experimental results?",
        relevant_chunk_ids=frozenset(
            {
                UUID("..."),
                UUID("..."),
            }
        ),
    ),
]

"""Hand-written evaluation queries for the retrieval-evaluation corpus
(see tests/evaluation/corpus.py), each paired with a stable content
fragment identifying its one known-relevant chunk. These specs are resolved
into real `RetrievalEvaluationCase`s (with real, currently-persisted chunk
UUIDs) at test run time by `retrieval_dataset.resolve_evaluation_cases` --
never hardcoded UUIDs, since the corpus is reproducible, not permanent.
"""

from tests.evaluation.retrieval_dataset import RetrievalEvaluationCaseSpec

RETRIEVAL_EVALUATION_CASE_SPECS = [
    RetrievalEvaluationCaseSpec(
        query="How does dense vector search find semantically similar text?",
        relevant_chunk_content_fragments=frozenset(
            {"Dense vector retrieval uses embeddings"},
        ),
    ),
    RetrievalEvaluationCaseSpec(
        query="What does cross-encoder reranking do?",
        relevant_chunk_content_fragments=frozenset(
            {"Cross-encoder reranking scores each candidate chunk"},
        ),
    ),
    RetrievalEvaluationCaseSpec(
        query="What is query decomposition used for?",
        relevant_chunk_content_fragments=frozenset(
            {"Query decomposition splits a complex multi-part question"},
        ),
    ),
]

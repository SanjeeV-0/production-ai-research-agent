"""Hand-written retrieval evaluation queries.

The small smoke-test cases remain in ``RETRIEVAL_EVALUATION_CASE_SPECS``.
The larger Q01-Q46 benchmark cases are kept separately in
``RETRIEVAL_BENCHMARK_CASE_SPECS`` so the existing fast evaluation corpus
continues to work unchanged.

Gold references use stable document titles and content fragments. They are
resolved to real persisted ``DocumentChunk`` UUIDs at test time; synthetic
mapping IDs are intentionally not used by the application.
"""

from tests.evaluation.retrieval_dataset import (
    RelevantChunkSpec,
    RetrievalEvaluationCaseSpec,
)

RETRIEVAL_EVALUATION_CASE_SPECS = [
    RetrievalEvaluationCaseSpec(
        query="How does dense vector search find semantically similar text?",
        relevant_chunks=(
            RelevantChunkSpec(
                document_title="Retrieval Evaluation Corpus",
                content_fragment="Dense vector retrieval uses embeddings",
                relevance=3,
            ),
        ),
        category="simple_factual",
    ),
    RetrievalEvaluationCaseSpec(
        query="What does cross-encoder reranking do?",
        relevant_chunks=(
            RelevantChunkSpec(
                document_title="Retrieval Evaluation Corpus",
                content_fragment=("Cross-encoder reranking scores each candidate chunk"),
                relevance=3,
            ),
        ),
        category="simple_factual",
    ),
    RetrievalEvaluationCaseSpec(
        query="What is query decomposition used for?",
        relevant_chunks=(
            RelevantChunkSpec(
                document_title="Retrieval Evaluation Corpus",
                content_fragment=("Query decomposition splits a complex multi-part question"),
                relevance=3,
            ),
        ),
        category="simple_factual",
    ),
]


RETRIEVAL_BENCHMARK_CASE_SPECS = [
    RetrievalEvaluationCaseSpec(
        query='What problem is HNSW designed to solve?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. Given a query vector q and a corpus D of vectors, exact nearest-neighbor search',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. An HNSW index can be viewed as a sequence of nested proximity graphs. Each vector is',
                relevance=3,
            ),
        ),
        category='simple factual',
    ),
    RetrievalEvaluationCaseSpec(
        query='What role does `efSearch` play during HNSW search?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. efSearch represents the size of the dynamic candidate exploration set used during',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Larger efSearch generally increases the probability of finding true neighbors but also increases',
                relevance=3,
            ),
        ),
        category='simple factual',
    ),
    RetrievalEvaluationCaseSpec(
        query='What does an embedding represent in a vector-search system?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='An embedding model maps an input sequence into a fixed-dimensional vector. The dimension is part of',
                relevance=3,
            ),
        ),
        category='simple factual',
    ),
    RetrievalEvaluationCaseSpec(
        query='What does cosine similarity compare?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Cosine similarity compares angular direction, while dot product incorporates both direction and',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='For normalized vectors, cosine similarity and dot product are equivalent: cos(q,d) = q.d when ||q||',
                relevance=2,
            ),
        ),
        category='simple factual',
    ),
    RetrievalEvaluationCaseSpec(
        query='What is the purpose of the candidate limit in vector search?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='A candidate limit controls how much material is initially returned. The source vector document uses',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Candidate count interacts with chunk density, corpus size, query ambiguity, and reranker capacity.',
                relevance=3,
            ),
        ),
        category='simple factual',
    ),
    RetrievalEvaluationCaseSpec(
        query='What does a cross-encoder do after the initial candidates have been retrieved?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='A reranker receives an initial candidate set and scores each candidate against the query. The key distinction from independent embedding retrieval is that the model can process the query and document jointly, allowing token-level interactions to influence relevance. The document is deliberately organized to avoid repeating the same paragraph under different headings',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The source draft correctly states that reranking cannot recover a document excluded during',
                relevance=3,
            ),
        ),
        category='simple factual',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why does HNSW use multiple graph layers instead of searching the entire graph in the same way?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. An HNSW index can be viewed as a sequence of nested proximity graphs. Each vector is',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. Search begins from an entry point at the highest available layer and performs a',
                relevance=3,
            ),
        ),
        category='conceptual',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why can increasing search effort improve retrieval quality while making the system slower?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Larger efSearch generally increases the probability of finding true neighbors but also increases',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Parameter studies should report recall against efSearch, not only recall at one arbitrary setting.',
                relevance=3,
            ),
        ),
        category='conceptual',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why can the way a document is chunked affect whether the retrieved passage is actually useful?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Chunking determines the units the index can retrieve. The source draft emphasizes semantic',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Small chunks improve precision but can separate prerequisites from conclusions. Large chunks',
                relevance=3,
            ),
        ),
        category='conceptual',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why can two documents that look similarly relevant by embedding similarity still need different final rankings?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Semantic similarity and relevance overlap but are not identical. A passage may discuss the same',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Consider a query asking "which protocol replaced X in version 3?" A topically similar passage about',
                relevance=3,
            ),
        ),
        category='conceptual',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why is it important to evaluate candidate generation separately from reranking?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Evaluation should separate the first-stage candidate pool from the final ranking. A relevant chunk',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The quality ceiling of reranking is set by candidate recall. If the initial retriever does not',
                relevance=3,
            ),
        ),
        category='conceptual',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why can a larger candidate pool create a trade-off between retrieval quality and efficiency?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Candidate count interacts with chunk density, corpus size, query ambiguity, and reranker capacity.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The source material includes a recurring focus on candidate pool size and retrieval quality. In',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Sweep candidate sizes such as 20, 50, 100, and 200. Measure candidate recall, final NDCG or MRR,',
                relevance=2,
            ),
        ),
        category='conceptual',
    ),
    RetrievalEvaluationCaseSpec(
        query='How does HNSW move from broad regions of the search space toward nearby vectors?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. Search begins from an entry point at the highest available layer and performs a',
                relevance=3,
            ),
        ),
        category='paraphrase',
    ),
    RetrievalEvaluationCaseSpec(
        query='How does the hierarchical structure help HNSW narrow its search to a local neighborhood?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. An HNSW index can be viewed as a sequence of nested proximity graphs. Each vector is',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. Search begins from an entry point at the highest available layer and performs a',
                relevance=3,
            ),
        ),
        category='paraphrase',
    ),
    RetrievalEvaluationCaseSpec(
        query='What happens when an important passage is missing from the initial retrieval results?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Evaluation should separate the first-stage candidate pool from the final ranking. A relevant chunk',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The quality ceiling of reranking is set by candidate recall. If the initial retriever does not',
                relevance=3,
            ),
        ),
        category='paraphrase',
    ),
    RetrievalEvaluationCaseSpec(
        query='Can later ranking recover evidence that the first retrieval stage failed to return?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The quality ceiling of reranking is set by candidate recall. If the initial retriever does not',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Evaluation should separate the first-stage candidate pool from the final ranking. A relevant chunk',
                relevance=3,
            ),
        ),
        category='paraphrase',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why do semantic chunk boundaries matter for retrieval?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Chunking determines the units the index can retrieve. The source draft emphasizes semantic',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Small chunks improve precision but can separate prerequisites from conclusions. Large chunks',
                relevance=3,
            ),
        ),
        category='paraphrase',
    ),
    RetrievalEvaluationCaseSpec(
        query='How can splitting documents at the wrong places hurt the usefulness of retrieved context?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Small chunks improve precision but can separate prerequisites from conclusions. Large chunks',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='A practical chunk record should include stable chunk ID, parent document ID, section path, ordinal',
                relevance=2,
            ),
        ),
        category='paraphrase',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why are cross-encoders useful when embedding similarity alone is not enough?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Semantic embeddings compress each text into a single vector before comparison. Cross-encoders',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='This distinction motivates multi-stage retrieval. Vector search provides a broad semantic net; a',
                relevance=3,
            ),
        ),
        category='paraphrase',
    ),
    RetrievalEvaluationCaseSpec(
        query='What additional signal does query-document interaction provide during reranking?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The query and candidate passage are packed into one transformer sequence, allowing self-attention',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Consider a query asking "which protocol replaced X in version 3?" A topically similar passage about',
                relevance=3,
            ),
        ),
        category='paraphrase',
    ),
    RetrievalEvaluationCaseSpec(
        query='How does HNSW balance search quality, index size, construction effort, and latency?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. Given a query vector q and a corpus D of vectors, exact nearest-neighbor search',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Construction effort increases with connectivity and search breadth during insertion. A more',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='The cost is primarily memory and construction work. Because each edge must be stored and traversed,',
                relevance=2,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Larger efSearch generally increases the probability of finding true neighbors but also increases',
                relevance=2,
            ),
        ),
        category='multi-part',
    ),
    RetrievalEvaluationCaseSpec(
        query='What is the relationship between embeddings, similarity search, candidate limits, and chunking in a vector retrieval pipeline?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='An embedding model maps an input sequence into a fixed-dimensional vector. The dimension is part of',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Cosine similarity compares angular direction, while dot product incorporates both direction and',
                relevance=2,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Chunking determines the units the index can retrieve. The source draft emphasizes semantic',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='A candidate limit controls how much material is initially returned. The source vector document uses',
                relevance=3,
            ),
        ),
        category='multi-part',
    ),
    RetrievalEvaluationCaseSpec(
        query='How does a cross-encoder reranker improve the ordering of candidates, and what limits its ability to fix retrieval errors?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='A reranker receives an initial candidate set and scores each candidate against the query. The key distinction from independent embedding retrieval is that the model can process the query and document jointly, allowing token-level interactions to influence relevance. The source draft correctly states that reranking cannot recover a document excluded during candidate generation.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Consider a query asking "which protocol replaced X in version 3?" A topically similar passage about',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The quality ceiling of reranking is set by candidate recall. If the initial retriever does not',
                relevance=3,
            ),
        ),
        category='multi-part',
    ),
    RetrievalEvaluationCaseSpec(
        query='How should retrieval quality be assessed separately for candidate recall, final ranking, and overall evidence quality?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Evaluation should separate the first-stage candidate pool from the final ranking. A relevant chunk',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Recall@k asks how much of the relevant set is recovered within the first k results. It is a direct',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='To quantify the reranker contribution, hold the candidate pool constant and compare pre-rank order',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Protocol. This is a useful dual representation: binary metrics measure coverage, while graded',
                relevance=2,
            ),
        ),
        category='multi-part',
    ),
    RetrievalEvaluationCaseSpec(
        query='What happens when a complex question is broken into subqueries, and what can go wrong if important constraints are lost during decomposition?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Intent decomposition maps the parsed query to minimal evidence units. For example, a',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Subquery generation produces executable retrieval strings or structured retrieval',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. The most dangerous decomposition error is losing a restriction that changes the',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Define a constraint ledger for each query. Every subquery is checked against the ledger before',
                relevance=3,
            ),
        ),
        category='multi-part',
    ),
    RetrievalEvaluationCaseSpec(
        query='How do query planning, candidate generation, reranking, and answer generation interact when the original question contains several information needs?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Let Q be a natural-language query and I(Q) the set of information needs required to',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Multi-hop retrieval collects evidence across documents or stages. The challenge is',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The source draft correctly states that reranking cannot recover a document excluded during',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Retrieval-augmented generation couples a retriever with a generative model. The retriever supplies',
                relevance=2,
            ),
        ),
        category='multi-part',
    ),
    RetrievalEvaluationCaseSpec(
        query='How is approximate nearest-neighbor search with HNSW different from simply comparing a query against every vector?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Exact nearest-neighbor search compares a query with every indexed vector and provides a useful',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='This distinction is fundamental for evaluation. Exact search can provide the ground truth set for a',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. Given a query vector q and a corpus D of vectors, exact nearest-neighbor search',
                relevance=2,
            ),
        ),
        category='comparative',
    ),
    RetrievalEvaluationCaseSpec(
        query='What is the difference between vector similarity and cross-encoder relevance scoring?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Semantic similarity and relevance overlap but are not identical. A passage may discuss the same',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='A bi-encoder embeds queries and documents separately, allowing document vectors to be precomputed',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The query and candidate passage are packed into one transformer sequence, allowing self-attention',
                relevance=3,
            ),
        ),
        category='comparative',
    ),
    RetrievalEvaluationCaseSpec(
        query='How is candidate recall different from the quality of the final ranked list?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Evaluation should separate the first-stage candidate pool from the final ranking. A relevant chunk',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='When ranking quality drops, inspect whether the relevant evidence was present before reranking.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='To quantify the reranker contribution, hold the candidate pool constant and compare pre-rank order',
                relevance=2,
            ),
        ),
        category='comparative',
    ),
    RetrievalEvaluationCaseSpec(
        query='How does query decomposition differ from ordinary single-query retrieval?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Let Q be a natural-language query and I(Q) the set of information needs required to',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Intent decomposition maps the parsed query to minimal evidence units. For example, a',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Multi-hop retrieval collects evidence across documents or stages. The challenge is',
                relevance=2,
            ),
        ),
        category='comparative',
    ),
    RetrievalEvaluationCaseSpec(
        query='If a retrieval system returns many passages about the same topic, how can we tell whether it actually found the evidence needed to answer the question?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Dense retrieval often returns several chunks from the same document or overlapping windows. High',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Semantic similarity and relevance overlap but are not identical. A passage may discuss the same',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='The source draft uses a graded scale in which 3 indicates direct or essential evidence, 2 strong',
                relevance=2,
            ),
        ),
        category='distractor-heavy',
    ),
    RetrievalEvaluationCaseSpec(
        query='A document has very high embedding similarity to the query. Does that alone mean it should be ranked first? Explain.',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Semantic similarity and relevance overlap but are not identical. A passage may discuss the same',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Consider a query asking "which protocol replaced X in version 3?" A topically similar passage about',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='A reranker score is only meaningful relative to its training and calibration. Some models produce',
                relevance=2,
            ),
        ),
        category='distractor-heavy',
    ),
    RetrievalEvaluationCaseSpec(
        query='If increasing the number of retrieved candidates raises recall but also sends more irrelevant passages to the reranker, how should that change be evaluated?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Candidate count interacts with chunk density, corpus size, query ambiguity, and reranker capacity.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Treat candidate size as an experimental variable. Evaluate several values across query classes,',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The source material includes a recurring focus on candidate pool size and retrieval quality. In',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Sweep candidate sizes such as 20, 50, 100, and 200. Measure candidate recall, final NDCG or MRR,',
                relevance=3,
            ),
        ),
        category='distractor-heavy',
    ),
    RetrievalEvaluationCaseSpec(
        query='When several passages contain the same terminology, what should the retrieval system preserve so that the most useful evidence is selected?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='This distinction motivates multi-stage retrieval. Vector search provides a broad semantic net; a',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Dense retrieval often returns several chunks from the same document or overlapping windows. High',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Consider a query asking "which protocol replaced X in version 3?" A topically similar passage about',
                relevance=2,
            ),
        ),
        category='distractor-heavy',
    ),
    RetrievalEvaluationCaseSpec(
        query='If HNSW determines which candidates enter retrieval, why does a reranker still matter after those candidates have been selected?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='HNSW Indexing Deep Dive',
                content_fragment='Section focus. A candidate generator succeeds when it retrieves the evidence that downstream stages',
                relevance=3,
            ),
            RelevantChunkSpec(
    document_title="Reranking with Cross-Encoders",
    content_fragment=(
        "The source draft correctly states that reranking cannot recover a document "
        "excluded during candidate generation. This creates a two-stage contract: "
        "the retriever should be recall-oriented, while the reranker should be "
        "precision-oriented."
    ),
    relevance=3,
),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The source draft correctly states that reranking cannot recover a document excluded during',
                relevance=3,
            ),
        ),
        category='multi-hop',
    ),
    RetrievalEvaluationCaseSpec(
        query='How can the candidate limit in vector search influence both retrieval recall and the amount of work performed by a reranker?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='A candidate limit controls how much material is initially returned. The source vector document uses',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Candidate count interacts with chunk density, corpus size, query ambiguity, and reranker capacity.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The source material includes a recurring focus on candidate pool size and retrieval quality. In',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Because every candidate requires joint inference, cost scales approximately with the number of',
                relevance=2,
            ),
        ),
        category='multi-hop',
    ),
    RetrievalEvaluationCaseSpec(
        query='Why does query decomposition become more important when the answer depends on evidence distributed across different parts of the corpus?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Intent decomposition maps the parsed query to minimal evidence units. For example, a',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Multi-hop retrieval collects evidence across documents or stages. The challenge is',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Evaluation sets should represent the real distribution of retrieval difficulty. Include direct',
                relevance=2,
            ),
        ),
        category='multi-hop',
    ),
    RetrievalEvaluationCaseSpec(
        query='If a relevant document is not retrieved for one subquery but other subqueries succeed, what does that tell us about the retrieval pipeline?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Section focus. Planning failures fall into at least four classes: omission, over-decomposition,',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Query Decomposition and Planning',
                content_fragment='Omission lowers recall for a required information need. Over-decomposition increases noise and',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='When ranking quality drops, inspect whether the relevant evidence was present before reranking.',
                relevance=3,
            ),
        ),
        category='multi-hop',
    ),
    RetrievalEvaluationCaseSpec(
        query='How should we distinguish a failure to retrieve the right evidence from a failure to place that evidence near the top of the final results?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Evaluation should separate the first-stage candidate pool from the final ranking. A relevant chunk',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='When ranking quality drops, inspect whether the relevant evidence was present before reranking.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='To quantify the reranker contribution, hold the candidate pool constant and compare pre-rank order',
                relevance=3,
            ),
        ),
        category='multi-hop',
    ),
    RetrievalEvaluationCaseSpec(
        query='How do chunking decisions, candidate generation, and reranking collectively affect whether the final context contains useful evidence?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Chunking determines the units the index can retrieve. The source draft emphasizes semantic',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Small chunks improve precision but can separate prerequisites from conclusions. Large chunks',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The design goal is not to replace vector search but to spend expensive model capacity on a much',
                relevance=2,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The quality ceiling of reranking is set by candidate recall. If the initial retriever does not',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='RAG introduces additional constraints: context-window budget, evidence ordering, citation coverage,',
                relevance=2,
            ),
        ),
        category='multi-hop',
    ),
    RetrievalEvaluationCaseSpec(
        query='Does retrieving more candidates automatically mean that the retrieval system is better?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Candidate count interacts with chunk density, corpus size, query ambiguity, and reranker capacity.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The source material includes a recurring focus on candidate pool size and retrieval quality. In',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Protocol. Coverage should also be reported across query classes. A high aggregate hit rate can',
                relevance=2,
            ),
        ),
        category='contradiction-sensitive',
    ),
    RetrievalEvaluationCaseSpec(
        query='Can a system have good candidate recall but still produce a poor final ranking?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='When ranking quality drops, inspect whether the relevant evidence was present before reranking.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='To quantify the reranker contribution, hold the candidate pool constant and compare pre-rank order',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The source draft correctly states that reranking cannot recover a document excluded during',
                relevance=3,
            ),
        ),
        category='contradiction-sensitive',
    ),
    RetrievalEvaluationCaseSpec(
        query='If a reranker performs badly, should we immediately assume that the reranker itself is the problem?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='When ranking quality drops, inspect whether the relevant evidence was present before reranking.',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='The quality ceiling of reranking is set by candidate recall. If the initial retriever does not',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Cross-encoders can fail on negation, numerical comparisons, coreference, temporal scope, entity',
                relevance=2,
            ),
        ),
        category='contradiction-sensitive',
    ),
    RetrievalEvaluationCaseSpec(
        query='Is high similarity between a query and document enough to establish that the document contains the best evidence for the question?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Semantic similarity and relevance overlap but are not identical. A passage may discuss the same',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='This distinction motivates multi-stage retrieval. Vector search provides a broad semantic net; a',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Consider a query asking "which protocol replaced X in version 3?" A topically similar passage about',
                relevance=3,
            ),
        ),
        category='contradiction-sensitive',
    ),
    RetrievalEvaluationCaseSpec(
        query='Can improving one retrieval metric introduce a negative effect elsewhere in the pipeline?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='Protocol. The objective is not to crown one metric. It is to make failure location observable: did',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Retrieval systems live under resource constraints. Embedding inference, ANN traversal, filters,',
                relevance=2,
            ),
            RelevantChunkSpec(
                document_title='Vector Search Fundamentals',
                content_fragment='Candidate count interacts with chunk density, corpus size, query ambiguity, and reranker capacity.',
                relevance=2,
            ),
        ),
        category='contradiction-sensitive',
    ),
    RetrievalEvaluationCaseSpec(
        query='If two passages are both related to the query but one is direct evidence and the other is only supporting background, should they receive the same relevance judgment?',
        relevant_chunks=(
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='The source draft uses a graded scale in which 3 indicates direct or essential evidence, 2 strong',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Retrieval Evaluation Methodology',
                content_fragment='NDCG is useful when relevance is graded. A directly answer-bearing passage can receive more credit',
                relevance=3,
            ),
            RelevantChunkSpec(
                document_title='Reranking with Cross-Encoders',
                content_fragment='Binary labels are simple, but retrieval often benefits from graded relevance. A passage may be',
                relevance=2,
            ),
        ),
        category='contradiction-sensitive',
    ),
]

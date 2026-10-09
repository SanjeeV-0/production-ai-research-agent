"""Benchmark V2 evaluation queries (Q001-Q100).

Unlike `retrieval_cases.RETRIEVAL_BENCHMARK_CASE_SPECS` (the V1, 46-query
benchmark), V2's queries and gold evidence are authored as data in
`tests/evaluation/fixtures/benchmark_v2/questions_v2.json` rather than as
Python literals, so the question set can be inspected, diffed, and audited
independently of this loader.

Gold references use stable document titles and content fragments, resolved
to real persisted `DocumentChunk` UUIDs at test time via
`tests.evaluation.benchmark_corpus_v2.resolve_benchmark_v2_evaluation_cases`;
synthetic mapping IDs are intentionally not used by the application.
"""

import json
from pathlib import Path

from tests.evaluation.retrieval_dataset import (
    RelevantChunkSpec,
    RetrievalEvaluationCaseSpec,
)

QUESTIONS_V2_PATH = (
    Path(__file__).parent / "fixtures" / "benchmark_v2" / "questions_v2.json"
)


def _load_case_specs() -> list[RetrievalEvaluationCaseSpec]:
    raw_questions = json.loads(QUESTIONS_V2_PATH.read_text(encoding="utf-8"))

    specs: list[RetrievalEvaluationCaseSpec] = []

    for question in raw_questions:
        relevant_chunks = tuple(
            RelevantChunkSpec(
                document_title=relevant_chunk["document_title"],
                content_fragment=relevant_chunk["content_fragment"],
                relevance=relevant_chunk["relevance"],
            )
            for relevant_chunk in question["relevant_chunks"]
        )

        specs.append(
            RetrievalEvaluationCaseSpec(
                query=question["query"],
                relevant_chunks=relevant_chunks,
                expected_subqueries=tuple(
                    question.get("expected_subqueries", ()),
                ),
                category=question["category"],
            )
        )

    return specs


RETRIEVAL_BENCHMARK_V2_CASE_SPECS = _load_case_specs()

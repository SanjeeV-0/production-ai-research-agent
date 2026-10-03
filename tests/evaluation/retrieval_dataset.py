from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query: str
    relevant_chunk_ids: frozenset[UUID]

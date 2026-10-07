from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import DocumentChunk
from app.core.repositories.document import DocumentRepository


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query: str
    relevant_chunk_ids: frozenset[UUID]


@dataclass(frozen=True)
class RetrievalEvaluationCaseSpec:
    """A query plus a STABLE way to identify its relevant chunk(s).

    Chunk UUIDs are only assigned at insert time and the evaluation corpus
    is re-ingested (not literally permanent -- see tests/evaluation/corpus.py),
    so hardcoding a UUID here would go stale across re-ingestion. A content
    fragment is a stable identifier instead; `resolve_evaluation_cases`
    turns this into a real `RetrievalEvaluationCase` using whatever chunk
    UUIDs are ACTUALLY persisted right now.
    """

    query: str
    relevant_chunk_content_fragments: frozenset[str]


async def resolve_evaluation_cases(
    session: AsyncSession,
    logical_document_id: UUID,
    specs: Sequence[RetrievalEvaluationCaseSpec],
) -> list[RetrievalEvaluationCase]:
    """Resolve specs into `RetrievalEvaluationCase`s using the REAL,
    currently persisted `DocumentChunk` ids of the given logical document's
    current version -- never fabricated UUIDs. Must be called after the
    corpus has been ingested (see `ensure_evaluation_corpus`).
    """

    repository = DocumentRepository(session)
    current_version = await repository.get_current_version(logical_document_id)

    if current_version is None:
        raise ValueError(
            f"No current version for logical_document_id={logical_document_id}; "
            "ingest the evaluation corpus before resolving cases."
        )

    chunk_result = await session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == current_version.id)
    )
    chunks = chunk_result.scalars().all()

    cases: list[RetrievalEvaluationCase] = []

    for spec in specs:
        relevant_ids = frozenset(
            chunk.id
            for chunk in chunks
            for fragment in spec.relevant_chunk_content_fragments
            if fragment in chunk.content
        )

        if not relevant_ids:
            raise ValueError(
                f"No persisted chunk matched any content fragment for query: {spec.query!r}"
            )

        cases.append(RetrievalEvaluationCase(query=spec.query, relevant_chunk_ids=relevant_ids))

    return cases

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document, DocumentChunk


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query: str
    relevant_chunk_ids: frozenset[UUID]
    relevance_by_chunk_id: tuple[tuple[UUID, int], ...] = ()
    expected_subqueries: tuple[str, ...] = ()
    category: str = ""


@dataclass(frozen=True)
class RelevantChunkSpec:
    """Stable description of a chunk relevant to an evaluation query."""

    document_title: str
    content_fragment: str
    relevance: int

    def __post_init__(self) -> None:
        if not 0 <= self.relevance <= 3:
            raise ValueError(
                f"relevance must be between 0 and 3, got {self.relevance}"
            )

        if not self.document_title.strip():
            raise ValueError("document_title must not be empty")

        if not self.content_fragment.strip():
            raise ValueError("content_fragment must not be empty")


@dataclass(frozen=True)
class RetrievalEvaluationCaseSpec:
    query: str
    relevant_chunks: tuple[RelevantChunkSpec, ...] = ()
    relevant_chunk_content_fragments: frozenset[str] = frozenset()
    expected_subqueries: tuple[str, ...] = ()
    category: str = ""

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")

        if self.relevant_chunks and self.relevant_chunk_content_fragments:
            raise ValueError(
                "Use either relevant_chunks or "
                "relevant_chunk_content_fragments, not both"
            )

        if not self.relevant_chunks and not self.relevant_chunk_content_fragments:
            raise ValueError(
                "At least one relevant chunk specification is required"
            )


async def resolve_evaluation_cases(
    session: AsyncSession,
    logical_document_id: UUID,
    specs: Sequence[RetrievalEvaluationCaseSpec],
) -> list[RetrievalEvaluationCase]:
    """Resolve stable benchmark specs to real persisted chunk UUIDs.

    Document titles and content fragments are used only to identify the
    currently persisted chunks. UUIDs are never hardcoded in the dataset.
    """

    titles = {
        chunk.document_title
        for spec in specs
        for chunk in spec.relevant_chunks
    }

    result = await session.execute(
        select(Document).where(
            Document.logical_document_id == logical_document_id,
            Document.is_current.is_(True),
        )
    )
    documents = result.scalars().all()

    documents_by_title: dict[str, Document] = {}

    for document in documents:
        if document.title in documents_by_title:
            raise ValueError(
                "Multiple current documents found with title "
                f"{document.title!r}"
            )

        documents_by_title[document.title] = document

    missing_titles = titles - documents_by_title.keys()

    if missing_titles:
        raise ValueError(
            "No current document found for benchmark title(s): "
            + ", ".join(sorted(missing_titles))
        )

    document_ids = {
        document.id for document in documents_by_title.values()
    }

    chunk_result = await session.execute(
        select(DocumentChunk).where(
            DocumentChunk.document_id.in_(document_ids)
        )
    )
    chunks = chunk_result.scalars().all()

    chunks_by_document: dict[UUID, list[DocumentChunk]] = {}

    for chunk in chunks:
        chunks_by_document.setdefault(
            chunk.document_id,
            [],
        ).append(chunk)

    cases: list[RetrievalEvaluationCase] = []

    for spec in specs:
        relevance_by_chunk_id: dict[UUID, int] = {}

        for relevant_chunk in spec.relevant_chunks:
            document = documents_by_title[relevant_chunk.document_title]

            matches = [
                chunk
                for chunk in chunks_by_document.get(
                    document.id,
                    [],
                )
                if relevant_chunk.content_fragment in chunk.content
            ]

            if not matches:
                raise ValueError(
                    "No persisted chunk matched benchmark fragment "
                    f"{relevant_chunk.content_fragment!r} "
                    f"in document {relevant_chunk.document_title!r} "
                    f"for query {spec.query!r}"
                )

            for chunk in matches:
                existing_relevance = relevance_by_chunk_id.get(chunk.id)

                if existing_relevance is None:
                    relevance_by_chunk_id[chunk.id] = (
                        relevant_chunk.relevance
                    )
                else:
                    relevance_by_chunk_id[chunk.id] = max(
                        existing_relevance,
                        relevant_chunk.relevance,
                    )

        relevant_chunk_ids = frozenset(
            chunk_id
            for chunk_id, relevance in relevance_by_chunk_id.items()
            if relevance >= 1
        )

        cases.append(
            RetrievalEvaluationCase(
                query=spec.query,
                relevant_chunk_ids=relevant_chunk_ids,
                relevance_by_chunk_id=tuple(
                    sorted(
                        relevance_by_chunk_id.items(),
                        key=lambda item: str(item[0]),
                    )
                ),
                expected_subqueries=spec.expected_subqueries,
                category=spec.category,
            )
        )

    return cases
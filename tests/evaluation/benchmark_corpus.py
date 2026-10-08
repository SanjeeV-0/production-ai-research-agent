"""Five-document retrieval benchmark corpus.

The benchmark corpus uses the real production ingestion pipeline
(`IngestionService`) for all five rewritten DOCX fixtures.

Unlike the small smoke corpus, the benchmark deliberately contains
multiple logical documents. Each document therefore has its own stable
logical-document identity.

This module is benchmark-specific and does not change the existing
single-document evaluation corpus or evaluation dataset API.
"""

from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.core.models import Document, DocumentChunk, DocumentStatus
from app.core.repositories.document import DocumentRepository
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage
from tests.evaluation.retrieval_dataset import (
    RetrievalEvaluationCase,
    RetrievalEvaluationCaseSpec,
)

BENCHMARK_CORPUS_SOURCE = "retrieval-benchmark-corpus"

BENCHMARK_FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "benchmark"
)

BENCHMARK_DOCUMENTS: tuple[tuple[str, str, UUID], ...] = (
    (
        "vector_search_fundamentals_rewritten.docx",
        "Vector Search Fundamentals",
        UUID("22222222-2222-2222-2222-222222222221"),
    ),
    (
        "hnsw_indexing_deep_dive_rewritten.docx",
        "HNSW Indexing Deep Dive",
        UUID("22222222-2222-2222-2222-222222222222"),
    ),
    (
        "reranking_with_cross_encoders_rewritten.docx",
        "Reranking with Cross-Encoders",
        UUID("22222222-2222-2222-2222-222222222223"),
    ),
    (
        "query_decomposition_and_planning_rewritten.docx",
        "Query Decomposition and Planning",
        UUID("22222222-2222-2222-2222-222222222224"),
    ),
    (
        "retrieval_evaluation_methodology_rewritten.docx",
        "Retrieval Evaluation Methodology",
        UUID("22222222-2222-2222-2222-222222222225"),
    ),
)


async def ensure_benchmark_corpus(
    session: AsyncSession,
) -> dict[str, Document]:
    """Ensure all five benchmark documents are ingested and READY.

    Returns documents keyed by their stable benchmark title.
    """
    settings = get_settings()
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model,
    )
    file_storage = LocalFileStorage(
        Path(settings.storage_root),
    )

    service = IngestionService(
        session,
        embedding_provider=embedding_provider,
        file_storage=file_storage,
    )

    documents: dict[str, Document] = {}

    for filename, title, logical_document_id in BENCHMARK_DOCUMENTS:
        repository = DocumentRepository(session)

        current = await repository.get_current_version(
            logical_document_id,
        )

        if current is not None and current.status == DocumentStatus.READY:
            documents[title] = current
            continue

        document_path = BENCHMARK_FIXTURE_DIR / filename

        if not document_path.is_file():
            raise FileNotFoundError(
                f"Benchmark fixture does not exist: {document_path}"
            )

        document = await service.ingest_file(
            path=document_path,
            loader=DocxLoader(),
            title=title,
            document_type="evaluation",
            logical_document_id=logical_document_id,
            source=BENCHMARK_CORPUS_SOURCE,
            original_filename=filename,
        )

        documents[title] = document

    await session.commit()

    return documents


async def resolve_benchmark_evaluation_cases(
    session: AsyncSession,
    specs: tuple[RetrievalEvaluationCaseSpec, ...],
) -> list[RetrievalEvaluationCase]:
    """Resolve benchmark gold fragments to real persisted chunk UUIDs.

    Unlike the existing single-document resolver, this resolver allows a
    benchmark case to reference chunks from any of the five documents.

    Every gold content fragment must match exactly one persisted chunk.
    Zero matches and ambiguous matches fail loudly.
    """
    documents = await ensure_benchmark_corpus(session)

    relevant_titles = {
        relevant_chunk.document_title
        for spec in specs
        for relevant_chunk in spec.relevant_chunks
    }

    missing_titles = relevant_titles - documents.keys()

    if missing_titles:
        raise ValueError(
            "Benchmark references unknown document title(s): "
            + ", ".join(sorted(missing_titles))
        )

    document_ids = {
        document.id
        for document in documents.values()
        if document.title in relevant_titles
    }

    result = await session.execute(
        select(DocumentChunk).where(
            DocumentChunk.document_id.in_(document_ids),
        )
    )
    chunks = result.scalars().all()

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
            document = documents[relevant_chunk.document_title]

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

            if len(matches) > 1:
                matching_details = "\n\n".join(
        (
            f"chunk_id={chunk.id}\n"
            f"chunk_index={chunk.chunk_index}\n"
            f"content={chunk.content!r}"
        )
                    for chunk in matches
    )

                raise ValueError(
        "Benchmark fragment matched multiple persisted chunks.\n"
        f"Query: {spec.query!r}\n"
        f"Document: {relevant_chunk.document_title!r}\n"
        f"Fragment: {relevant_chunk.content_fragment!r}\n"
        f"Matches:\n{matching_details}"
    )

            chunk = matches[0]
            existing_relevance = relevance_by_chunk_id.get(chunk.id)

            if existing_relevance is None:
                relevance_by_chunk_id[chunk.id] = relevant_chunk.relevance
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

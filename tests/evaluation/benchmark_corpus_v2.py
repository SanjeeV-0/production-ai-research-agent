"""Nine-document retrieval benchmark corpus (V2).

V2 is an independent, newly authored corpus distinct from the five-document
V1 benchmark (`tests.evaluation.benchmark_corpus`). Unlike V1 (DOCX-only)
and the first cut of V2 (Markdown-only), this corpus is a genuine mixture
of the three source formats the production system actually accepts --
three Markdown, three DOCX, and three PDF documents -- each ingested
through its own real loader (`MarkdownLoader`, `DocxLoader`, `PDFLoader`),
so the benchmark exercises real structural extraction per format rather
than assuming Markdown-equivalent fidelity for every document.

The DOCX and PDF fixtures are not hand-typed duplicates of the Markdown
ones: they are generated once, from the same underlying prose, via
`tests/evaluation/fixtures/convert_benchmark_v2_formats.py`, as genuinely
native documents (python-docx "Heading N" styles and real Word tables;
reportlab-flowed multi-page PDF text and real tables) and committed as
binary fixtures. No document exists in more than one format.

Unlike the V1 fixture generator, V2 documents contain no templated or
repeated paragraphs -- every paragraph is independently authored content.

This module does not change the V1 benchmark corpus or the existing
single-document evaluation corpus API.
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
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage
from tests.evaluation.retrieval_dataset import (
    RetrievalEvaluationCase,
    RetrievalEvaluationCaseSpec,
)

BENCHMARK_V2_CORPUS_SOURCE = "retrieval-benchmark-v2-corpus"

BENCHMARK_V2_FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "benchmark_v2"
)

# (filename, title, logical_document_id, source_format)
# source_format is one of "markdown", "docx", "pdf" and selects both the
# real production loader used at ingestion time and the format label
# reported by the benchmark runner's per-format breakdown.
BENCHMARK_V2_DOCUMENTS: tuple[tuple[str, str, UUID, str], ...] = (
    (
        "embeddings_for_retrieval.docx",
        "Embedding Models for Semantic Retrieval",
        UUID("33333333-3333-3333-3333-333333333331"),
        "docx",
    ),
    (
        "chunking_strategies.pdf",
        "Chunking Strategies for Retrieval-Augmented Generation",
        UUID("33333333-3333-3333-3333-333333333332"),
        "pdf",
    ),
    (
        "postgres_pgvector_operations.docx",
        "PostgreSQL and pgvector for Production Vector Search",
        UUID("33333333-3333-3333-3333-333333333333"),
        "docx",
    ),
    (
        "rag_pipeline_architecture.pdf",
        "Architecture of a Production RAG Retrieval Pipeline",
        UUID("33333333-3333-3333-3333-333333333334"),
        "pdf",
    ),
    (
        "query_decomposition_practice.md",
        "Query Decomposition in Practice",
        UUID("33333333-3333-3333-3333-333333333335"),
        "markdown",
    ),
    (
        "reranking_candidate_pool_design.md",
        "Reranking and Candidate Pool Design",
        UUID("33333333-3333-3333-3333-333333333336"),
        "markdown",
    ),
    (
        "retrieval_evaluation_metrics.docx",
        "Evaluation Metrics and Benchmark Design for Retrieval Systems",
        UUID("33333333-3333-3333-3333-333333333337"),
        "docx",
    ),
    (
        "retrieval_observability.md",
        "Observability for Production Retrieval Systems",
        UUID("33333333-3333-3333-3333-333333333338"),
        "markdown",
    ),
    (
        "document_ingestion_pipeline.pdf",
        "Document Ingestion and Structural Extraction for RAG",
        UUID("33333333-3333-3333-3333-333333333339"),
        "pdf",
    ),
)

_LOADERS_BY_FORMAT: dict[str, DocumentLoader] = {
    "markdown": MarkdownLoader(),
    "docx": DocxLoader(),
    "pdf": PDFLoader(),
}

# Stable title -> source-format lookup, used by the benchmark runner to
# break retrieval metrics down by source format without needing to touch
# the shared (V1 + V2) RetrievalEvaluationCase/-Spec dataclasses.
BENCHMARK_V2_FORMAT_BY_TITLE: dict[str, str] = {
    title: source_format
    for _filename, title, _logical_document_id, source_format in BENCHMARK_V2_DOCUMENTS
}


async def ensure_benchmark_v2_corpus(
    session: AsyncSession,
) -> dict[str, Document]:
    """Ensure all nine V2 benchmark documents are ingested and READY.

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

    for filename, title, logical_document_id, source_format in BENCHMARK_V2_DOCUMENTS:
        repository = DocumentRepository(session)

        current = await repository.get_current_version(
            logical_document_id,
        )

        if current is not None and current.status == DocumentStatus.READY:
            documents[title] = current
            continue

        document_path = BENCHMARK_V2_FIXTURE_DIR / filename

        if not document_path.is_file():
            raise FileNotFoundError(
                f"Benchmark V2 fixture does not exist: {document_path}"
            )

        document = (await service.ingest_file(
            path=document_path,
            loader=_LOADERS_BY_FORMAT[source_format],
            title=title,
            document_type="evaluation",
            logical_document_id=logical_document_id,
            source=BENCHMARK_V2_CORPUS_SOURCE,
            original_filename=filename,
        )).document

        documents[title] = document

    await session.commit()

    return documents


async def resolve_benchmark_v2_evaluation_cases(
    session: AsyncSession,
    specs: tuple[RetrievalEvaluationCaseSpec, ...],
) -> list[RetrievalEvaluationCase]:
    """Resolve V2 benchmark gold fragments to real persisted chunk UUIDs.

    A benchmark case may reference chunks from any of the nine documents.
    Every gold content fragment must match exactly one persisted chunk.
    Zero matches and ambiguous matches fail loudly.
    """
    documents = await ensure_benchmark_v2_corpus(session)

    relevant_titles = {
        relevant_chunk.document_title
        for spec in specs
        for relevant_chunk in spec.relevant_chunks
    }

    missing_titles = relevant_titles - documents.keys()

    if missing_titles:
        raise ValueError(
            "Benchmark V2 references unknown document title(s): "
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
                    "No persisted chunk matched benchmark V2 fragment "
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
                    "Benchmark V2 fragment matched multiple persisted "
                    "chunks.\n"
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

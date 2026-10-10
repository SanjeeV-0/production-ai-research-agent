"""The retrieval-evaluation corpus: a small, fixed document ingested through
the REAL, unmodified production ingestion pipeline (`IngestionService`), not
a separate/duplicated ingestion path.

The corpus is reproducible rather than literally permanent: it always
targets `EVALUATION_CORPUS_LOGICAL_DOCUMENT_ID`, a fixed, well-known logical
document identity. `tests/conftest.py`'s post-session cleanup hook wipes
every document row (including this one) after each pytest session -- that
is existing, approved test infrastructure and is deliberately NOT modified
here. Instead, `ensure_evaluation_corpus` idempotently re-ingests the same
content under the same logical identity whenever it's missing, so the
corpus survives being wiped by simply being recreated identically on the
next run, through the same production code path every real document goes
through.
"""

import tempfile
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.core.models import Document, DocumentStatus
from app.core.repositories.document import DocumentRepository
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.service import IngestionService
from app.storage.local import LocalFileStorage

# Fixed, well-known identity -- never randomly generated -- so every call to
# ensure_evaluation_corpus() targets the same logical document.
EVALUATION_CORPUS_LOGICAL_DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")

# Tags the ingested Document for identification (e.g. by a future cleanup/
# maintenance script) without introducing a new column/migration -- `source`
# is an existing free-text field already used informally this way elsewhere
# in the test suite (e.g. "integration-test").
EVALUATION_CORPUS_SOURCE = "evaluation-corpus"

# Three short, single-paragraph sections under distinct headings. Each
# paragraph becomes exactly one DocumentChunk (shred_semantically forces a
# boundary at every heading, and each paragraph here is well under the
# chunk token budget), so each evaluation query below can target one known,
# unambiguous chunk by its exact source text.
EVALUATION_CORPUS_MARKDOWN = """# Retrieval Evaluation Corpus

## Vector Search

Dense vector retrieval uses embeddings and approximate nearest neighbor
search such as HNSW to find semantically similar text.

## Reranking

Cross-encoder reranking scores each candidate chunk jointly with the query
to produce a more accurate relevance ordering than embedding distance alone.

## Query Decomposition

Query decomposition splits a complex multi-part question into focused
sub-queries that are retrieved independently and then merged.
"""


async def ensure_evaluation_corpus(session: AsyncSession) -> Document:
    """Return the evaluation corpus's current READY version, ingesting it
    through the real `IngestionService` if it isn't already present.

    Reuses production ingestion unchanged: real `MarkdownLoader`, real
    `SentenceTransformerEmbeddingProvider` (semantically meaningful
    embeddings are required for recall/MRR to mean anything -- the fake,
    hash-based `DeterministicEmbeddingProvider` used elsewhere in the test
    suite would not do), and the real configured `LocalFileStorage` root.
    Content-hash idempotency in `DocumentService.ingest_document` makes a
    call against an already-ingested, unchanged corpus a safe no-op.
    """

    repository = DocumentRepository(session)
    current = await repository.get_current_version(EVALUATION_CORPUS_LOGICAL_DOCUMENT_ID)

    if current is not None and current.status == DocumentStatus.READY:
        return current

    settings = get_settings()
    embedding_provider = SentenceTransformerEmbeddingProvider(model_name=settings.embedding_model)
    file_storage = LocalFileStorage(Path(settings.storage_root))

    service = IngestionService(
        session,
        embedding_provider=embedding_provider,
        file_storage=file_storage,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        document_path = Path(tmp_dir) / "retrieval_evaluation_corpus.md"
        document_path.write_text(EVALUATION_CORPUS_MARKDOWN, encoding="utf-8")

        document = (await service.ingest_file(
            path=document_path,
            loader=MarkdownLoader(),
            title="Retrieval Evaluation Corpus",
            document_type="evaluation",
            logical_document_id=EVALUATION_CORPUS_LOGICAL_DOCUMENT_ID,
            source=EVALUATION_CORPUS_SOURCE,
        )).document

    await session.commit()

    return document

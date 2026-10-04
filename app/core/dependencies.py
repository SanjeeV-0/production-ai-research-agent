from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.core.database import get_db_session
from app.core.repositories.document import DocumentRepository
from app.core.services.document import DocumentService
from app.core.services.document_deletion import DocumentDeletionService
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.generation.generation_service import GenerationService
from app.generation.openrouter import OpenRouterGenerationProvider
from app.ingestion.service import IngestionService
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.openrouter_decomposition import (
    OpenRouterQueryDecompositionProvider,
)
from app.retrieval.query_decomposer import QueryDecomposer
from app.retrieval.reranker import Reranker
from app.retrieval.service import RetrievalService
from app.storage.interface import FileStorage
from app.storage.local import LocalFileStorage


@lru_cache
def get_query_decomposition_provider() -> OpenRouterQueryDecompositionProvider:
    """Return the configured OpenRouter query decomposition provider."""

    settings = get_settings()

    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY must be configured.")

    return OpenRouterQueryDecompositionProvider(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        app_name=settings.openrouter_app_name,
    )


def get_query_decomposer() -> QueryDecomposer:
    """Return the application query decomposer."""

    return QueryDecomposer(
        provider=get_query_decomposition_provider(),
    )


@lru_cache
def get_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """Return the application embedding provider."""

    settings = get_settings()

    return SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model,
    )


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Return the application cross-encoder reranker."""

    settings = get_settings()

    return CrossEncoderReranker(
        model_name=settings.reranker_model,
    )


def get_app_settings() -> Settings:
    """Return application settings."""

    return get_settings()


async def get_retrieval_service(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    embedding_provider: Annotated[
        SentenceTransformerEmbeddingProvider,
        Depends(get_embedding_provider),
    ],
    reranker: Annotated[
        Reranker,
        Depends(get_reranker),
    ],
    query_decomposer: Annotated[
        QueryDecomposer,
        Depends(get_query_decomposer),
    ],
) -> RetrievalService:
    """Create a retrieval service for the current database session."""

    return RetrievalService(
        repository=DocumentRepository(session),
        embedding_provider=embedding_provider,
        reranker=reranker,
        query_decomposer=query_decomposer,
    )


@lru_cache
def get_generation_provider() -> OpenRouterGenerationProvider:
    """Return the configured OpenRouter generation provider."""

    settings = get_settings()

    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY must be configured.")

    return OpenRouterGenerationProvider(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        app_name=settings.openrouter_app_name,
    )


def get_generation_service(
    provider: Annotated[
        OpenRouterGenerationProvider,
        Depends(get_generation_provider),
    ],
) -> GenerationService:
    """Create the application generation service."""

    return GenerationService(
        provider=provider,
    )


@lru_cache
def get_file_storage() -> FileStorage:
    """Return the application file storage backend."""

    settings = get_settings()

    return LocalFileStorage(Path(settings.storage_root))


async def get_ingestion_service(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    embedding_provider: Annotated[
        SentenceTransformerEmbeddingProvider,
        Depends(get_embedding_provider),
    ],
    file_storage: Annotated[
        FileStorage,
        Depends(get_file_storage),
    ],
) -> IngestionService:
    """Create an ingestion service for the current database session."""

    return IngestionService(
        session=session,
        embedding_provider=embedding_provider,
        file_storage=file_storage,
    )


def get_document_repository(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
) -> DocumentRepository:
    """Create a document repository for the current database session."""

    return DocumentRepository(session)


def get_document_service(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
) -> DocumentService:
    """Create a document service for the current database session."""

    return DocumentService(session)


def get_document_deletion_service(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    file_storage: Annotated[
        FileStorage,
        Depends(get_file_storage),
    ],
) -> DocumentDeletionService:
    """Create a document deletion service for the current database session."""

    return DocumentDeletionService(session=session, file_storage=file_storage)

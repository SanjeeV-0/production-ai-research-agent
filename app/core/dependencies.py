from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.core.database import get_db_session
from app.core.repositories.document import DocumentRepository
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.reranker import Reranker
from app.retrieval.service import RetrievalService
from app.generation.openrouter import OpenRouterGenerationProvider

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
) -> RetrievalService:
    """Create a retrieval service for the current database session."""

    return RetrievalService(
        repository=DocumentRepository(session),
        embedding_provider=embedding_provider,
        reranker=reranker,
    )


@lru_cache
def get_generation_provider() -> OpenRouterGenerationProvider:
    """Return the configured OpenRouter generation provider."""

    settings = get_settings()

    if not settings.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY must be configured."
        )

    return OpenRouterGenerationProvider(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        app_name=settings.openrouter_app_name,
    )
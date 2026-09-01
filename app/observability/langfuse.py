from functools import lru_cache

from langfuse import Langfuse

from app.config.settings import get_settings


@lru_cache
def get_langfuse() -> Langfuse | None:
    """Return the configured Langfuse client."""

    settings = get_settings()

    if not settings.langfuse_enabled:
        return None

    if not settings.langfuse_public_key:
        raise ValueError(
            "LANGFUSE_PUBLIC_KEY must be configured "
            "when Langfuse is enabled."
        )

    if not settings.langfuse_secret_key:
        raise ValueError(
            "LANGFUSE_SECRET_KEY must be configured "
            "when Langfuse is enabled."
        )

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
    )
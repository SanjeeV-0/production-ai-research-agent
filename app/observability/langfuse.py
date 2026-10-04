"""Optional, purely-additive observability integration.

Every call site in the codebase (`RetrievalService.search`,
`GenerationService.generate`, `app.api.research.ask`) checks
`if langfuse is None` and runs the exact same logic either way -- Langfuse
can never change retrieval, decomposition, or generation behavior, only
observe it. This is independent of, and unrelated to, the retrieval trace
feature controlled by `Settings.trace_enabled` (see `app.retrieval.trace`
and `app.api.retrieval`) -- a request can have one, both, or neither
enabled at the same time.
"""

from functools import lru_cache

from langfuse import Langfuse

from app.config.settings import get_settings


@lru_cache
def get_langfuse() -> Langfuse | None:
    """Return the configured Langfuse client, or None if disabled.

    Raises `ValueError` if `LANGFUSE_ENABLED=true` but the public/secret key
    is missing -- a misconfiguration is surfaced loudly rather than silently
    falling back to "disabled".
    """

    settings = get_settings()

    if not settings.langfuse_enabled:
        return None

    if not settings.langfuse_public_key:
        raise ValueError("LANGFUSE_PUBLIC_KEY must be configured when Langfuse is enabled.")

    if not settings.langfuse_secret_key:
        raise ValueError("LANGFUSE_SECRET_KEY must be configured when Langfuse is enabled.")

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
    )

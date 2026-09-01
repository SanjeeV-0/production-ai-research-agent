from app.config.settings import get_settings
from app.observability.langfuse import get_langfuse


def test_langfuse_disabled_returns_none() -> None:
    settings = get_settings()

    original_enabled = settings.langfuse_enabled

    try:
        settings.langfuse_enabled = False
        get_langfuse.cache_clear()

        assert get_langfuse() is None

    finally:
        settings.langfuse_enabled = original_enabled
        get_langfuse.cache_clear()


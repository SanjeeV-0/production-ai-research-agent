from unittest.mock import patch

from app.config.settings import Settings
from app.core.dependencies import (
    get_query_decomposer,
    get_query_decomposition_provider,
)
from app.retrieval.openrouter_decomposition import (
    OpenRouterQueryDecompositionProvider,
)
from app.retrieval.query_decomposer import QueryDecomposer


def test_query_decomposition_provider_uses_openrouter_settings() -> None:
    settings = Settings(
        openrouter_api_key="test-key",
        openrouter_model="test-model",
        openrouter_base_url="https://example.test/v1",
        openrouter_app_name="Test App",
    )

    get_query_decomposition_provider.cache_clear()

    with patch(
        "app.core.dependencies.get_settings",
        return_value=settings,
    ):
        provider = get_query_decomposition_provider()

    assert isinstance(provider, OpenRouterQueryDecompositionProvider)
    assert provider.model == "test-model"
    assert str(provider.client.base_url) == "https://example.test/v1/"


def test_query_decomposer_uses_decomposition_provider() -> None:
    get_query_decomposition_provider.cache_clear()

    with patch(
        "app.core.dependencies.get_query_decomposition_provider",
    ) as provider_factory:
        provider = object()
        provider_factory.return_value = provider

        decomposer = get_query_decomposer()

    assert isinstance(decomposer, QueryDecomposer)
    assert decomposer.provider is provider

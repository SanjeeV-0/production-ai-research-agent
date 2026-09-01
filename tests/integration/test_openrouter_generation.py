import pytest

from app.config.settings import get_settings
from app.generation.context import ContextAssembler
from app.generation.openrouter import OpenRouterGenerationProvider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_openrouter_generation() -> None:
    settings = get_settings()

    if not settings.openrouter_api_key:
        pytest.skip(
            "OPENROUTER_API_KEY is not configured."
        )

    provider = OpenRouterGenerationProvider(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        app_name=settings.openrouter_app_name,
    )

    context = ContextAssembler().assemble([])

    result = await provider.generate(
        query="What is retrieval augmented generation?",
        context=context,
    )

    assert result.text
    assert result.model

    assert result.input_tokens is not None
    assert result.output_tokens is not None
    assert result.total_tokens is not None

    assert result.total_tokens >= (
        result.input_tokens
        + result.output_tokens
    )
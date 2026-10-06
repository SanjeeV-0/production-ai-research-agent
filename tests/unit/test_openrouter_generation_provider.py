"""Unit tests for `OpenRouterGenerationProvider`'s request construction --
mirrors the conventions in tests/unit/test_openrouter_decomposition.py
(mocked `client.chat.completions.create`, no network calls). The live,
network-dependent check lives separately in
tests/integration/test_openrouter_generation.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.generation.context import ContextAssembler
from app.generation.openrouter import OpenRouterGenerationProvider


def make_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
        model="test-model",
        usage=None,
    )


@pytest.fixture
def provider() -> OpenRouterGenerationProvider:
    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
    )
    provider.client.chat.completions.create = AsyncMock()
    return provider


@pytest.mark.asyncio
async def test_sends_configured_model_temperature_and_top_p(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["temperature"] == 0.2
    assert call.kwargs["top_p"] == 1.0


@pytest.mark.asyncio
async def test_omits_optional_fields_when_not_configured(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert "max_tokens" not in call.kwargs
    assert "response_format" not in call.kwargs
    assert "reasoning_effort" not in call.kwargs


@pytest.mark.asyncio
async def test_includes_optional_fields_when_configured() -> None:
    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
        response_format={"type": "json_object"},
        reasoning_effort="medium",
    )
    provider.client.chat.completions.create = AsyncMock(return_value=make_response("An answer."))

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["temperature"] == 0.7
    assert call.kwargs["top_p"] == 0.9
    assert call.kwargs["max_tokens"] == 256
    assert call.kwargs["response_format"] == {"type": "json_object"}
    assert call.kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_raises_on_empty_response(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(None)

    context = ContextAssembler().assemble([])

    with pytest.raises(RuntimeError, match="empty response"):
        await provider.generate(query="What is RAG?", context=context)

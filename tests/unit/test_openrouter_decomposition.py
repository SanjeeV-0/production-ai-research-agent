from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.retrieval.openrouter_decomposition import (
    OpenRouterQueryDecompositionProvider,
)


def make_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


@pytest.fixture
def provider() -> OpenRouterQueryDecompositionProvider:
    provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
    )
    provider.client.chat.completions.create = AsyncMock()
    return provider


@pytest.mark.asyncio
async def test_returns_queries_from_valid_response(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?", "How does reranking work?"]}'
    )

    result = await provider.decompose("What is RAG and how does reranking work?")

    assert result == [
        "What is RAG?",
        "How does reranking work?",
    ]

    provider.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_sends_configured_model_temperature_and_top_p(provider) -> None:
    """temperature is no longer hard-coded to 0 -- it (and model/top_p) must
    come straight from the provider's configured values."""

    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?"]}'
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["temperature"] == 0.0
    assert call.kwargs["top_p"] == 1.0


@pytest.mark.asyncio
async def test_omits_optional_fields_when_not_configured(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?"]}'
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert "max_tokens" not in call.kwargs
    assert "response_format" not in call.kwargs
    assert "reasoning_effort" not in call.kwargs


@pytest.mark.asyncio
async def test_includes_optional_fields_when_configured() -> None:
    provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.5,
        max_tokens=128,
        top_p=0.8,
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )
    provider.client.chat.completions.create = AsyncMock(
        return_value=make_response('{"queries": ["What is RAG?"]}')
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["temperature"] == 0.5
    assert call.kwargs["top_p"] == 0.8
    assert call.kwargs["max_tokens"] == 128
    assert call.kwargs["response_format"] == {"type": "json_object"}
    assert call.kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_rejects_invalid_json(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response("not valid json")

    with pytest.raises(ValueError, match="invalid JSON"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_non_object_response(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response('["What is RAG?"]')

    with pytest.raises(ValueError, match="JSON object"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_missing_queries(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"result": ["What is RAG?"]}'
    )

    with pytest.raises(ValueError, match="queries"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_non_list_queries(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": "What is RAG?"}'
    )

    with pytest.raises(ValueError, match="queries"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_non_string_queries(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?", 123]}'
    )

    with pytest.raises(ValueError, match="strings"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_empty_response(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(None)

    with pytest.raises(RuntimeError, match="empty decomposition"):
        await provider.decompose("What is RAG?")

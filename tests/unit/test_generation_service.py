import pytest

from app.generation.context import ContextAssembler
from app.generation.service import (
    GenerationProvider,
    GenerationResult,
)


class FakeGenerationProvider:
    """Deterministic generation provider for tests."""

    def __init__(self) -> None:
        self.received_query = None
        self.received_context = None

    async def generate(
        self,
        query: str,
        context,
    ) -> GenerationResult:
        self.received_query = query
        self.received_context = context

        return GenerationResult(
            text="Generated answer.",
            model="fake-model",
        )


@pytest.mark.asyncio
async def test_generation_provider_receives_query_and_context() -> None:
    provider = FakeGenerationProvider()

    context = ContextAssembler().assemble([])

    result = await provider.generate(
        query="What is RAG?",
        context=context,
    )

    assert result.text == "Generated answer."
    assert result.model == "fake-model"

    assert provider.received_query == "What is RAG?"
    assert provider.received_context is context


def test_generation_provider_protocol_is_importable() -> None:
    assert GenerationProvider is not None
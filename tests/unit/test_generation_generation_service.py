import pytest

from app.generation.context import ContextAssembler
from app.generation.generation_service import GenerationService
from app.generation.service import GenerationResult


class FakeGenerationProvider:
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
    input_tokens=100,
    output_tokens=25,
    total_tokens=125,
)


@pytest.mark.asyncio
async def test_generation_service_delegates_to_provider() -> None:
    provider = FakeGenerationProvider()
    service = GenerationService(provider)

    context = ContextAssembler().assemble([])

    result = await service.generate(
        query="What is retrieval augmented generation?",
        context=context,
    )

    assert result.text == "Generated answer."
    assert result.model == "fake-model"
    assert result.input_tokens == 100
    assert result.output_tokens == 25
    assert result.total_tokens == 125
    assert (
        provider.received_query
        == "What is retrieval augmented generation?"
    )

    assert provider.received_context is context
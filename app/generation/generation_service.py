from app.generation.context import GenerationContext
from app.generation.service import (
    GenerationProvider,
    GenerationResult,
)


class GenerationService:
    """Coordinates context and LLM generation."""

    def __init__(
        self,
        provider: GenerationProvider,
    ) -> None:
        self.provider = provider

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate an answer from the supplied context."""

        return await self.provider.generate(
            query=query,
            context=context,
        )